package main

import (
	"bytes"
	"compress/gzip"
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

	writeJSONNoEscape(w, errResp)
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
	if authHeader != "" {
		return authHeader
	}
	// Fallback: Claude SDK uses x-api-key header
	if apiKey := r.Header.Get("x-api-key"); apiKey != "" {
		return apiKey
	}
	return ""
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

// readRequestBody reads the raw request body with size limit and debug logging.
func (pr *ProxyRouter) readRequestBody(r *http.Request, label string) ([]byte, error) {
	r.Body = http.MaxBytesReader(nil, r.Body, pr.cfg.MaxBodySize)

	bodyBytes, err := io.ReadAll(r.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read body: %w", err)
	}

	if pr.debug {
		log.Printf("[DEBUG] Client Request (%s): %s %s", label, r.Method, r.URL.Path)
		for k, v := range r.Header {
			log.Printf("[DEBUG] Client Header: %s: %s", k, strings.Join(v, ", "))
		}
		log.Printf("[DEBUG] Client Body: %s", string(bodyBytes))
	}

	return bodyBytes, nil
}

// parseRequestBody reads and parses the JSON request body.
func (pr *ProxyRouter) parseRequestBody(r *http.Request) (map[string]interface{}, error) {
	bodyBytes, err := pr.readRequestBody(r, "OpenAI")
	if err != nil {
		return nil, err
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

	// Google Gemini requires a thought_signature for tool calling to work.
	// Standard clients (like LangChain, Dify, etc.) don't keep/echo non-standard fields like thought_signature
	// in the history, which leads to "Function call is missing a thought_signature" errors.
	// We automatically patch missing thought signatures with skip_thought_signature_validator.
	if pConfig.Name == "gemini" || strings.Contains(strings.ToLower(pConfig.Name), "gemini") || strings.Contains(strings.ToLower(pConfig.Upstream), "googleapis.com") {
		if msgs, ok := body["messages"].([]interface{}); ok {
			for _, msgVal := range msgs {
				msg, ok := msgVal.(map[string]interface{})
				if !ok {
					continue
				}
				role := fmt.Sprintf("%v", msg["role"])
				if role == "assistant" {
					if toolCalls, ok := msg["tool_calls"].([]interface{}); ok {
						for _, tcVal := range toolCalls {
							tc, ok := tcVal.(map[string]interface{})
							if !ok {
								continue
							}
							hasSig := false
							if ecVal, ok := tc["extra_content"]; ok {
								if ec, ok := ecVal.(map[string]interface{}); ok {
									if gVal, ok := ec["google"]; ok {
										if g, ok := gVal.(map[string]interface{}); ok {
											if sig, ok := g["thought_signature"]; ok && sig != "" {
												hasSig = true
											}
											if sig, ok := g["thoughtSignature"]; ok && sig != "" {
												hasSig = true
											}
										}
									}
								}
							}
							if !hasSig {
								var ec map[string]interface{}
								if ecVal, ok := tc["extra_content"]; ok {
									if e, ok := ecVal.(map[string]interface{}); ok {
										ec = e
									}
								}
								if ec == nil {
									ec = make(map[string]interface{})
									tc["extra_content"] = ec
								}
								var g map[string]interface{}
								if gVal, ok := ec["google"]; ok {
									if gg, ok := gVal.(map[string]interface{}); ok {
										g = gg
									}
								}
								if g == nil {
									g = make(map[string]interface{})
									ec["google"] = g
								}
								g["thought_signature"] = "c2tpcF90aG91Z2h0X3NpZ25hdHVyZV92YWxpZGF0b3I="
								g["thoughtSignature"] = "c2tpcF90aG91Z2h0X3NpZ25hdHVyZV92YWxpZGF0b3I="
							}
						}
					}
				}
			}
		}
	}

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

	newBodyBytes, err := marshalNoEscape(body)
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

func writeClaudeErrorJSON(w http.ResponseWriter, statusCode int, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(statusCode)

	errResp := map[string]interface{}{
		"type": "error",
		"error": map[string]interface{}{
			"type":    "invalid_request_error",
			"message": message,
		},
	}

	writeJSONNoEscape(w, errResp)
}

func writeError(w http.ResponseWriter, requestType string, statusCode int, message string) {
	if requestType == "claude" {
		writeClaudeErrorJSON(w, statusCode, message)
	} else {
		writeErrorJSON(w, statusCode, message)
	}
}

func (pr *ProxyRouter) parseClaudeRequest(r *http.Request) (map[string]interface{}, error) {
	bodyBytes, err := pr.readRequestBody(r, "Claude")
	if err != nil {
		return nil, err
	}

	var claudeReq ClaudeMessagesRequest
	if err := json.Unmarshal(bodyBytes, &claudeReq); err != nil {
		return nil, fmt.Errorf("invalid Claude JSON body: %w", err)
	}

	return TranslateClaudeRequestToOpenAI(&claudeReq)
}

func (pr *ProxyRouter) parseResponsesRequest(r *http.Request) (map[string]interface{}, error) {
	bodyBytes, err := pr.readRequestBody(r, "Responses")
	if err != nil {
		return nil, err
	}

	var responsesReq OpenAIResponsesRequest
	if err := json.Unmarshal(bodyBytes, &responsesReq); err != nil {
		return nil, fmt.Errorf("invalid Responses JSON body: %w", err)
	}

	return TranslateResponsesRequestToOpenAI(&responsesReq)
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

	if pathWithoutPrefix != "/chat/completions" && pathWithoutPrefix != "/messages" && pathWithoutPrefix != "/responses" {
		writeErrorJSON(w, http.StatusNotFound, "Not Found")
		return
	}

	var requestType string
	if pathWithoutPrefix == "/messages" {
		requestType = "claude"
	} else if pathWithoutPrefix == "/responses" {
		requestType = "responses"
	} else {
		requestType = "openai"
	}

	// 1. Authenticate and rate limit client
	if errCode, errMsg := pr.authenticate(r); errMsg != "" {
		writeError(w, requestType, errCode, errMsg)
		return
	}

	// 2. Parse request body
	var bodyMap map[string]interface{}
	var err error
	if requestType == "claude" {
		bodyMap, err = pr.parseClaudeRequest(r)
	} else if requestType == "responses" {
		bodyMap, err = pr.parseResponsesRequest(r)
	} else {
		bodyMap, err = pr.parseRequestBody(r)
	}

	if err != nil {
		writeError(w, requestType, http.StatusBadRequest, err.Error())
		return
	}

	// 3. Extract and validate model
	modelInterface, ok := bodyMap["model"]
	if !ok {
		writeError(w, requestType, http.StatusBadRequest, "Missing model field")
		return
	}
	modelID, ok := modelInterface.(string)
	if !ok {
		writeError(w, requestType, http.StatusBadRequest, "Model field must be a string")
		return
	}

	// 4. Select provider and key
	handler, key, keyIndex, err := pr.selectProvider(modelID)
	if err != nil {
		log.Printf("provider selection failed for model=%s: %v", modelID, err)
		writeError(w, requestType, http.StatusServiceUnavailable, "Service Unavailable: "+err.Error())
		return
	}

	// 5. Rewrite body
	newBodyBytes, err := pr.rewriteBody(bodyMap, handler.config)
	if err != nil {
		writeError(w, requestType, http.StatusInternalServerError, "Failed to rewrite body")
		return
	}

	// 6. Forward request
	handler.forwardRequest(w, r, newBodyBytes, key, keyIndex, pr.debug, requestType)
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

	w.WriteHeader(http.StatusOK)
	writeJSONNoEscape(w, resp)
}

func (ph *ProviderHandler) forwardRequest(w http.ResponseWriter, r *http.Request, bodyBytes []byte, key string, keyIndex int, debug bool, requestType string) {
	log.Printf("using provider=%s model=%s key[%d] for request", ph.config.Name, ph.config.Model, keyIndex)

	ctx, cancel := context.WithTimeout(r.Context(), ph.config.Timeout)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, r.Method, ph.config.Upstream, bytes.NewReader(bodyBytes))
	if err != nil {
		log.Printf("failed to create upstream request: %v", err)
		writeError(w, requestType, http.StatusInternalServerError, "Internal Server Error")
		return
	}

	copyHeaders(req.Header, r.Header)
	req.Header.Set("Authorization", "Bearer "+key)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Content-Length", fmt.Sprintf("%d", len(bodyBytes)))

	// Clean up protocol-specific headers that would confuse the upstream OpenAI-compatible server
	if requestType == "claude" {
		req.Header.Del("anthropic-version")
		req.Header.Del("anthropic-beta")
		req.Header.Del("x-api-key")
	}

	if requestType != "openai" {
		req.Header.Del("Accept-Encoding")
	}

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
		log.Printf("[DEBUG] Upstream Body: %s", string(bodyBytes))
	}

	resp, err := ph.client.Do(req)
	if err != nil {
		log.Printf("upstream request failed: %v", err)
		writeError(w, requestType, http.StatusBadGateway, "Bad Gateway: "+err.Error())
		return
	}
	defer resp.Body.Close()

	if requestType != "openai" && resp.Header.Get("Content-Encoding") == "gzip" {
		gzipReader, err := gzip.NewReader(resp.Body)
		if err != nil {
			log.Printf("failed to create gzip reader: %v", err)
			writeError(w, requestType, http.StatusInternalServerError, "Internal Server Error")
			return
		}
		defer gzipReader.Close()
		resp.Body = gzipReader
	}

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

	// For adapter request types (claude/responses), translate 4xx errors to their native format.
	// For openai passthrough, let streamResponse handle it to preserve all upstream headers.
	if resp.StatusCode >= 400 && requestType != "openai" {
		body, _ := io.ReadAll(resp.Body)
		if debug {
			log.Printf("[DEBUG] Upstream Error Body: %s", string(body))
		}
		var errMap map[string]interface{}
		if json.Unmarshal(body, &errMap) == nil {
			if errMsgObj, ok := errMap["error"].(map[string]interface{}); ok {
				if msg, ok := errMsgObj["message"].(string); ok {
					writeError(w, requestType, resp.StatusCode, msg)
					return
				}
			}
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(resp.StatusCode)
		w.Write(body)
		return
	}

	if requestType == "claude" {
		if strings.Contains(strings.ToLower(resp.Header.Get("Content-Type")), "text/event-stream") {
			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(resp.StatusCode)
			if err := TranslateOpenAIStreamToClaude(resp.Body, w, debug); err != nil {
				log.Printf("failed to translate stream to Claude: %v", err)
			}
		} else {
			translateAndWriteJSON(w, resp, debug, requestType, func(openaiResp map[string]interface{}) (interface{}, error) {
				return TranslateOpenAIResponseToClaude(openaiResp)
			})
		}
	} else if requestType == "responses" {
		if strings.Contains(strings.ToLower(resp.Header.Get("Content-Type")), "text/event-stream") {
			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(resp.StatusCode)
			if err := TranslateOpenAIStreamToResponses(resp.Body, w, debug); err != nil {
				log.Printf("failed to translate stream to Responses: %v", err)
			}
		} else {
			translateAndWriteJSON(w, resp, debug, requestType, func(openaiResp map[string]interface{}) (interface{}, error) {
				return TranslateOpenAIResponseToResponses(openaiResp)
			})
		}
	} else if requestType == "openai" {
		if strings.Contains(strings.ToLower(resp.Header.Get("Content-Type")), "text/event-stream") {
			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(resp.StatusCode)
			if err := TranslateOpenAIStreamToOpenAI(resp.Body, w, debug); err != nil {
				log.Printf("failed to translate openai stream: %v", err)
			}
		} else {
			translateAndWriteJSON(w, resp, debug, requestType, func(openaiResp map[string]interface{}) (interface{}, error) {
				return TranslateOpenAIResponseToOpenAI(openaiResp)
			})
		}
	} else {
		streamResponse(w, resp, debug)
	}
}

// translateAndWriteJSON reads the upstream JSON response, translates it, and writes it to the client.
func translateAndWriteJSON(w http.ResponseWriter, resp *http.Response, debug bool, requestType string, translator func(map[string]interface{}) (interface{}, error)) {
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		writeError(w, requestType, http.StatusInternalServerError, "failed to read upstream response")
		return
	}

	if debug {
		log.Printf("[DEBUG] Upstream Response Body: %s", string(body))
	}

	var openaiResp map[string]interface{}
	if err := json.Unmarshal(body, &openaiResp); err != nil {
		writeError(w, requestType, http.StatusInternalServerError, "failed to parse upstream response JSON")
		return
	}

	translated, err := translator(openaiResp)
	if err != nil {
		writeError(w, requestType, http.StatusInternalServerError, err.Error())
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(resp.StatusCode)

	if debug {
		if translatedBytes, err := marshalNoEscape(translated); err == nil {
			log.Printf("[DEBUG] Translated Response Body: %s", string(translatedBytes))
		}
	}

	writeJSONNoEscape(w, translated)
}
