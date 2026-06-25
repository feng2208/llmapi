package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"sync/atomic"
	"time"
)

// ModelData represents a single model in the /v1/models response.
type ModelData struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	OwnedBy string `json:"owned_by"`
}

type ModelListResponse struct {
	Object string      `json:"object"`
	Data   []ModelData `json:"data"`
}

type ProxyRouter struct {
	cfg          *Config
	basePath     string
	clientKeys   map[string]*ClientAuthKey
	modelRouters map[string]*ModelRouter
	rateLimiter  *ClientRateLimiter
	debug        bool
}

type ModelRouter struct {
	modelID  string
	handlers []*ProviderHandler
	next     uint32
}

type ProviderHandler struct {
	config     ModelProviderConfig
	keyManager *KeyManager
	client     *http.Client
}

func writeErrorJSON(w http.ResponseWriter, statusCode int, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(statusCode)

	errResp := map[string]interface{}{
		"error": map[string]interface{}{
			"message": message,
			"type":    "invalid_request_error",
			"code":    nil,
		},
	}

	json.NewEncoder(w).Encode(errResp)
}

func NewProxyRouter(cfg *Config, keyManagers map[string]*KeyManager, debug bool) *ProxyRouter {
	pr := &ProxyRouter{
		cfg:          cfg,
		basePath:     "/v1",
		clientKeys:   make(map[string]*ClientAuthKey),
		modelRouters: make(map[string]*ModelRouter),
		rateLimiter:  NewClientRateLimiter(cfg.Clients.RateLimit, cfg.Clients.Auth),
		debug:        debug,
	}

	for i := range cfg.Clients.Auth {
		pr.clientKeys[cfg.Clients.Auth[i].Key] = &cfg.Clients.Auth[i]
	}

	for _, m := range cfg.Models {
		mr := &ModelRouter{
			modelID: m.Name,
		}
		for _, p := range m.Providers {
			km := keyManagers[p.Name]

			proxyURL := p.Proxy
			if proxyURL == "" {
				proxyURL = cfg.Proxy
			}

			mr.handlers = append(mr.handlers, &ProviderHandler{
				config:     p,
				keyManager: km,
				client: &http.Client{
					Transport: buildTransport(proxyURL),
					CheckRedirect: func(req *http.Request, via []*http.Request) error {
						return http.ErrUseLastResponse
					},
				},
			})
		}
		pr.modelRouters[m.Name] = mr
	}

	return pr
}

func extractBearerToken(r *http.Request) string {
	authHeader := r.Header.Get("Authorization")
	if strings.HasPrefix(authHeader, "Bearer ") {
		return strings.TrimSpace(authHeader[7:])
	}
	return authHeader
}

// authenticate verifies the token and checks client rate limit.
// Returns an error code and message, or 0 and "" if successful.
func (pr *ProxyRouter) authenticate(r *http.Request) (int, string) {
	token := extractBearerToken(r)
	if token == "" {
		return http.StatusUnauthorized, "Missing API key"
	}

	if _, exists := pr.clientKeys[token]; !exists {
		return http.StatusUnauthorized, "Invalid API key"
	}

	if !pr.rateLimiter.Allow(token) {
		return http.StatusTooManyRequests, "Client rate limit exceeded"
	}

	return 0, ""
}

// parseRequestBody reads and parses the JSON request body.
func (pr *ProxyRouter) parseRequestBody(r *http.Request) (map[string]interface{}, error) {
	r.Body = http.MaxBytesReader(nil, r.Body, pr.cfg.MaxBodySize)

	bodyBytes, err := io.ReadAll(r.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read body: %w", err)
	}

	if pr.debug {
		log.Printf("[DEBUG] Client Request: %s %s", r.Method, r.URL.Path)
		for k, v := range r.Header {
			log.Printf("[DEBUG] Client Header: %s: %s", k, strings.Join(v, ", "))
		}
		log.Printf("[DEBUG] Client Body: %s", truncateString(string(bodyBytes), 16384))
	}

	var body map[string]interface{}
	if err := json.Unmarshal(bodyBytes, &body); err != nil {
		return nil, fmt.Errorf("invalid JSON body: %w", err)
	}

	return body, nil
}

// selectProvider performs round-robin provider selection with failover.
func (pr *ProxyRouter) selectProvider(modelID string) (*ProviderHandler, string, int, error) {
	mr, exists := pr.modelRouters[modelID]
	if !exists {
		return nil, "", -1, fmt.Errorf("model %s not found", modelID)
	}

	n := len(mr.handlers)
	startIndex := int(atomic.AddUint32(&mr.next, 1)-1) % n

	for i := 0; i < n; i++ {
		h := mr.handlers[(startIndex+i)%n]
		key, idx, err := h.keyManager.GetKey(modelID, h.config.RateLimit)
		if err == nil {
			return h, key, idx, nil
		}
	}

	return nil, "", -1, errors.New("all upstream keys exhausted or rate limited")
}

// rewriteBody modifies the JSON map according to provider settings and marshals it back.
func (pr *ProxyRouter) rewriteBody(body map[string]interface{}, pConfig ModelProviderConfig) ([]byte, error) {
	body["model"] = pConfig.Model

	if pConfig.IncludeThoughts {
		delete(body, "reasoning_effort")

		thinkingConfig := map[string]interface{}{
			"include_thoughts": true,
		}

		if pConfig.ReasoningEffort != "" {
			thinkingConfig["thinking_level"] = pConfig.ReasoningEffort
		}

		body["extra_body"] = map[string]interface{}{
			"google": map[string]interface{}{
				"thinking_config": thinkingConfig,
			},
		}
	} else if pConfig.ReasoningEffort != "" {
		body["reasoning_effort"] = pConfig.ReasoningEffort
	}

	newBodyBytes, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}
	return newBodyBytes, nil
}

type responseWriterRecorder struct {
	http.ResponseWriter
	statusCode int
}

func (rw *responseWriterRecorder) WriteHeader(code int) {
	if rw.statusCode == http.StatusOK {
		rw.statusCode = code
	}
	rw.ResponseWriter.WriteHeader(code)
}

func (rw *responseWriterRecorder) Flush() {
	if f, ok := rw.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}

func (pr *ProxyRouter) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	// Health check: respond immediately without logging
	if r.URL.Path == "/health" {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		fmt.Fprint(w, `{"status":"ok"}`)
		return
	}

	recorder := &responseWriterRecorder{ResponseWriter: w, statusCode: http.StatusOK}
	w = recorder

	start := time.Now()
	log.Printf("[%s] %s %s started", r.RemoteAddr, r.Method, r.URL.Path)
	defer func() {
		log.Printf("[%s] %s %s completed with status %d in %v", r.RemoteAddr, r.Method, r.URL.Path, recorder.statusCode, time.Since(start))
	}()

	if !strings.HasPrefix(r.URL.Path, pr.basePath) {
		writeErrorJSON(w, http.StatusNotFound, "Not Found")
		return
	}
	pathWithoutPrefix := strings.TrimPrefix(r.URL.Path, pr.basePath)

	if pathWithoutPrefix == "/models" && r.Method == http.MethodGet {
		pr.handleModels(w, r)
		return
	}

	if pathWithoutPrefix != "/chat/completions" {
		writeErrorJSON(w, http.StatusNotFound, "Not Found")
		return
	}

	// 1. Authenticate and rate limit client
	if errCode, errMsg := pr.authenticate(r); errMsg != "" {
		writeErrorJSON(w, errCode, errMsg)
		return
	}

	// 2. Parse request body
	bodyMap, err := pr.parseRequestBody(r)
	if err != nil {
		writeErrorJSON(w, http.StatusBadRequest, err.Error())
		return
	}

	// 3. Extract and validate model
	modelInterface, ok := bodyMap["model"]
	if !ok {
		writeErrorJSON(w, http.StatusBadRequest, "Missing model field")
		return
	}
	modelID, ok := modelInterface.(string)
	if !ok {
		writeErrorJSON(w, http.StatusBadRequest, "Model field must be a string")
		return
	}

	// 4. Select provider and key
	handler, key, keyIndex, err := pr.selectProvider(modelID)
	if err != nil {
		log.Printf("provider selection failed for model=%s: %v", modelID, err)
		writeErrorJSON(w, http.StatusServiceUnavailable, "Service Unavailable: "+err.Error())
		return
	}

	// 5. Rewrite body
	newBodyBytes, err := pr.rewriteBody(bodyMap, handler.config)
	if err != nil {
		writeErrorJSON(w, http.StatusInternalServerError, "Failed to rewrite body")
		return
	}

	// 6. Forward request
	handler.forwardRequest(w, r, newBodyBytes, key, keyIndex, pr.debug)
}

func (pr *ProxyRouter) handleModels(w http.ResponseWriter, r *http.Request) {
	if errCode, errMsg := pr.authenticate(r); errMsg != "" {
		writeErrorJSON(w, errCode, errMsg)
		return
	}

	now := time.Now().Unix()
	resp := ModelListResponse{
		Object: "list",
		Data:   make([]ModelData, 0, len(pr.cfg.Models)),
	}

	for _, m := range pr.cfg.Models {
		resp.Data = append(resp.Data, ModelData{
			ID:      m.Name,
			Object:  "model",
			Created: now,
			OwnedBy: "system",
		})
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(resp)
}

func (ph *ProviderHandler) forwardRequest(w http.ResponseWriter, r *http.Request, bodyBytes []byte, key string, keyIndex int, debug bool) {
	log.Printf("using provider=%s key[%d] for request", ph.config.Name, keyIndex)

	ctx, cancel := context.WithTimeout(r.Context(), ph.config.Timeout)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, r.Method, ph.config.Upstream, bytes.NewReader(bodyBytes))
	if err != nil {
		log.Printf("failed to create upstream request: %v", err)
		writeErrorJSON(w, http.StatusInternalServerError, "Internal Server Error")
		return
	}

	copyHeaders(req.Header, r.Header)
	req.Header.Set("Authorization", "Bearer "+key)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Content-Length", fmt.Sprintf("%d", len(bodyBytes)))

	if debug {
		log.Printf("[DEBUG] Upstream Request: %s %s", req.Method, req.URL.String())
		for k, v := range req.Header {
			// Mask authorization for security
			if strings.ToLower(k) == "authorization" {
				log.Printf("[DEBUG] Upstream Header: %s: Bearer ***...***", k)
			} else {
				log.Printf("[DEBUG] Upstream Header: %s: %s", k, strings.Join(v, ", "))
			}
		}
		log.Printf("[DEBUG] Upstream Body: %s", truncateString(string(bodyBytes), 16384))
	}

	resp, err := ph.client.Do(req)
	if err != nil {
		log.Printf("upstream request failed: %v", err)
		writeErrorJSON(w, http.StatusBadGateway, "Bad Gateway: "+err.Error())
		return
	}
	defer resp.Body.Close()

	if debug {
		log.Printf("[DEBUG] Upstream Response: %s", resp.Status)
		for k, v := range resp.Header {
			log.Printf("[DEBUG] Upstream Response Header: %s: %s", k, strings.Join(v, ", "))
		}
	}

	switch {
	case resp.StatusCode == http.StatusTooManyRequests:
		ph.keyManager.Mark429(keyIndex)
	case resp.StatusCode == http.StatusUnauthorized || resp.StatusCode == http.StatusForbidden:
		ph.keyManager.MarkFailed(keyIndex)
	case resp.StatusCode >= 200 && resp.StatusCode < 300:
		ph.keyManager.ResetFailures(keyIndex)
	}

	streamResponse(w, resp, debug)
}
