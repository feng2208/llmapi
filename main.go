package main

import (
	"bufio"
	"bytes"
	_ "embed"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"

	"llmapi/plugins"
)

//go:embed config-template.yaml
var configTemplateContent []byte

var (
	configPath    string
	debug         bool
	printTemplate bool
	showVersion   bool
	Version       = "dev"
)

func init() {
	flag.StringVar(&configPath, "config", "config.yaml", "Path to configuration file")
	flag.BoolVar(&debug, "debug", false, "Enable debug mode (logs raw requests and responses)")
	flag.BoolVar(&printTemplate, "print-template", false, "Print configuration template and exit")
	flag.BoolVar(&showVersion, "v", false, "")
	flag.BoolVar(&showVersion, "version", false, "Print version and exit")
}

func main() {
	flag.Parse()

	if showVersion {
		fmt.Printf("llmapi version: %s\n", Version)
		os.Exit(0)
	}

	if printTemplate {
		fmt.Print(string(configTemplateContent))
		os.Exit(0)
	}

	cfg, err := LoadConfig(configPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: failed to load config from %s: %v\n", configPath, err)
		os.Exit(1)
	}

	stateMgr := NewStateManager()
	router := NewRouter(cfg, stateMgr)
	proxyMgr := NewProxyManager()

	// Define Handlers
	http.HandleFunc("/v1/models", handleModels(cfg))
	http.HandleFunc("/v1/chat/completions", handleChatCompletions(cfg, stateMgr, router, proxyMgr))
	http.HandleFunc("/v1/images/generations", handleImageGenerations(cfg, stateMgr, router, proxyMgr))
	http.HandleFunc("/v1/images/edits", handleImageEdits(cfg, stateMgr, router, proxyMgr))
	http.HandleFunc("/v1/audio/speech", handleAudioSpeech(cfg, stateMgr, router, proxyMgr))
	http.HandleFunc("/v1/audio/transcriptions", handleAudioTranscriptions(cfg, stateMgr, router, proxyMgr))

	fmt.Printf("[INFO] [%s] Server starting, listening on %s\n", time.Now().Format("2006-01-02 15:04:05.000"), cfg.Listen)
	if err := http.ListenAndServe(cfg.Listen, nil); err != nil {
		fmt.Fprintf(os.Stderr, "Error starting server: %v\n", err)
		os.Exit(1)
	}
}

// Handler for GET /v1/models
func handleModels(cfg *Config) http.HandlerFunc {
	type OpenAIModel struct {
		ID      string `json:"id"`
		Object  string `json:"object"`
		Created int64  `json:"created"`
		OwnedBy string `json:"owned_by"`
	}

	type OpenAIModelsList struct {
		Object string        `json:"object"`
		Data   []OpenAIModel `json:"data"`
	}

	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			w.WriteHeader(http.StatusMethodNotAllowed)
			return
		}

		models := make([]OpenAIModel, 0, len(cfg.Models))
		createdTime := time.Now().Unix() // Mock created time
		for _, m := range cfg.Models {
			models = append(models, OpenAIModel{
				ID:      m.Name,
				Object:  "model",
				Created: createdTime,
				OwnedBy: "llmapi",
			})
		}

		resp := OpenAIModelsList{
			Object: "list",
			Data:   models,
		}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}
}

type handlerCtx struct {
	clientKey     string
	matchedClient *ClientAuth
	rawBody       []byte
	reqJSON       map[string]interface{}
	modelName     string
}

func executeUpstreamRequest(
	endpoint plugins.EndpointType,
	w http.ResponseWriter,
	r *http.Request,
	cfg *Config,
	stateMgr *StateManager,
	router *Router,
	proxyMgr *ProxyManager,
	bodyModifier func(rawBody []byte, route *SelectedRoute, r *http.Request) ([]byte, string, error),
) (*SelectedRoute, *http.Response, *handlerCtx, time.Time, bool) {
	startTime := time.Now()

	if r.Method != http.MethodPost {
		http.Error(w, `{"error": {"message": "Method not allowed"}}`, http.StatusMethodNotAllowed)
		return nil, nil, nil, startTime, false
	}

	fmt.Printf("[INFO] [%s] %s request started ...\n", time.Now().Format("2006-01-02 15:04:05.000"), r.URL.Path)

	// 1. Authenticate Client
	authHeader := r.Header.Get("Authorization")
	if len(authHeader) < 8 || authHeader[:7] != "Bearer " {
		fmt.Printf("[WARN] Client authentication failed: Missing or invalid Authorization header\n")
		http.Error(w, `{"error": {"message": "Missing or invalid Authorization header", "type": "invalid_request_error"}}`, http.StatusUnauthorized)
		return nil, nil, nil, startTime, false
	}
	clientKey := authHeader[7:]

	var matchedClient *ClientAuth
	for i := range cfg.Clients.Auth {
		if cfg.Clients.Auth[i].Key == clientKey {
			matchedClient = &cfg.Clients.Auth[i]
			break
		}
	}

	if matchedClient == nil {
		fmt.Printf("[WARN] Client authentication failed: Key not authorized\n")
		http.Error(w, `{"error": {"message": "Invalid client auth key", "type": "invalid_request_error"}}`, http.StatusUnauthorized)
		return nil, nil, nil, startTime, false
	}

	// 2. Client Rate Limiting
	limit := cfg.Clients.RateLimit
	if matchedClient.RateLimit > 0 {
		limit = matchedClient.RateLimit
	}

	if !stateMgr.AllowClient(clientKey, limit) {
		fmt.Printf("[WARN] Client %q rate limit exceeded\n", matchedClient.Name)
		http.Error(w, `{"error": {"message": "Client rate limit exceeded", "type": "rate_limit_error"}}`, http.StatusTooManyRequests)
		return nil, nil, nil, startTime, false
	}

	// 3. Read request body within MaxBodySize limits
	var bodyReader io.Reader = r.Body
	if cfg.MaxBodySize > 0 {
		bodyReader = io.LimitReader(r.Body, cfg.MaxBodySize+1)
	}

	rawBody, err := io.ReadAll(bodyReader)
	if err != nil {
		http.Error(w, `{"error": {"message": "Failed to read request body"}}`, http.StatusInternalServerError)
		return nil, nil, nil, startTime, false
	}

	if cfg.MaxBodySize > 0 && int64(len(rawBody)) > cfg.MaxBodySize {
		http.Error(w, `{"error": {"message": "Request body too large", "type": "invalid_request_error"}}`, http.StatusRequestEntityTooLarge)
		return nil, nil, nil, startTime, false
	}

	// Restore r.Body so that subsequent parses (like ParseMultipartForm) can run if needed.
	r.Body = io.NopCloser(bytes.NewReader(rawBody))

	if debug {
		fmt.Printf("[DEBUG] --- INCOMING REQUEST ---\n")
		fmt.Printf("[DEBUG] URL: %s %s\n", r.Method, r.URL.String())
		for k, vv := range r.Header {
			for _, v := range vv {
				fmt.Printf("[DEBUG] Header: %s: %s\n", k, v)
			}
		}
		bodyStr := string(rawBody)
		if strings.HasPrefix(r.Header.Get("Content-Type"), "application/json") {
			bodyStr = FormatJSON(rawBody)
		}
		fmt.Printf("[DEBUG] Body:\n%s\n", bodyStr)
	}

	// 4. Parse request JSON to get model (or multipart form)
	var reqJSON map[string]interface{}
	var modelName string

	isMultipart := strings.HasPrefix(r.Header.Get("Content-Type"), "multipart/form-data")
	if isMultipart {
		err := r.ParseMultipartForm(32 << 20) // 32MB
		if err != nil {
			http.Error(w, `{"error": {"message": "Failed to parse multipart form"}}`, http.StatusBadRequest)
			return nil, nil, nil, startTime, false
		}
		modelName = r.FormValue("model")
		// Synthesize a reqJSON for backward compatibility / debug logs
		reqJSON = make(map[string]interface{})
		reqJSON["model"] = modelName
		reqJSON["prompt"] = r.FormValue("prompt")
		reqJSON["n"] = r.FormValue("n")
		reqJSON["size"] = r.FormValue("size")
		reqJSON["response_format"] = r.FormValue("response_format")
		reqJSON["stream"] = r.FormValue("stream") == "true" || r.FormValue("stream") == "1"
	} else {
		if err := json.Unmarshal(rawBody, &reqJSON); err != nil {
			http.Error(w, `{"error": {"message": "Invalid JSON in request body"}}`, http.StatusBadRequest)
			return nil, nil, nil, startTime, false
		}
		modelName, _ = reqJSON["model"].(string)
	}

	if modelName == "" {
		http.Error(w, `{"error": {"message": "Missing model parameter", "type": "invalid_request_error"}}`, http.StatusBadRequest)
		return nil, nil, nil, startTime, false
	}

	hCtx := &handlerCtx{
		clientKey:     clientKey,
		matchedClient: matchedClient,
		rawBody:       rawBody,
		reqJSON:       reqJSON,
		modelName:     modelName,
	}

	// 5-9. Select route, build request, execute with retry on 429/500/503
	const maxRetries = 2
	var route *SelectedRoute
	var resp *http.Response

	for attempt := 0; attempt <= maxRetries; attempt++ {
		// 5. Select Route (Model/Provider/AuthKey)
		var err error
		route, err = router.SelectRoute(modelName)
		if err != nil {
			fmt.Printf("[ERROR] Routing failed for model %q: %v\n", modelName, err)
			http.Error(w, fmt.Sprintf(`{"error": {"message": %q, "type": "rate_limit_error"}}`, err.Error()), http.StatusTooManyRequests)
			return nil, nil, hCtx, startTime, false
		}

		// 6. Modify request body
		modifiedBody, contentTypeOut, err := bodyModifier(rawBody, route, r)
		if err != nil {
			http.Error(w, fmt.Sprintf(`{"error": {"message": "Failed to modify request body: %s"}}`, err.Error()), http.StatusInternalServerError)
			return nil, nil, hCtx, startTime, false
		}

		// 7. Build Upstream Request
		p := plugins.Get(route.ModelProvider.ApiType)
		pluginCtx := buildPluginContext(route, r, cfg, proxyMgr)
		pluginCtx.RawRequestBody = rawBody
		targetURL := p.BuildTargetURL(endpoint, r, pluginCtx)

		upstreamReq, err := http.NewRequestWithContext(r.Context(), "POST", targetURL, bytes.NewReader(modifiedBody))
		if err != nil {
			http.Error(w, `{"error": {"message": "Failed to build upstream request"}}`, http.StatusInternalServerError)
			return nil, nil, hCtx, startTime, false
		}

		// Copy request headers
		for k, vv := range r.Header {
			if k == "Host" || k == "Content-Length" || k == "Accept-Encoding" || k == "Content-Type" {
				continue
			}
			for _, v := range vv {
				upstreamReq.Header.Add(k, v)
			}
		}
		upstreamReq.Header.Set("Content-Type", contentTypeOut)
		upstreamReq.Header.Set("Content-Length", strconv.Itoa(len(modifiedBody)))

		// Apply custom request header rules and auth keys from configuration via plugin
		p.ModifyHeaders(upstreamReq, pluginCtx)

		if debug {
			fmt.Printf("[DEBUG] --- OUTGOING REQUEST (attempt %d/%d) ---\n", attempt+1, maxRetries+1)
			fmt.Printf("[DEBUG] Target URL: %s\n", upstreamReq.URL.String())
			for k, vv := range upstreamReq.Header {
				for _, v := range vv {
					fmt.Printf("[DEBUG] Header: %s: %s\n", k, v)
				}
			}
			bodyStr := string(modifiedBody)
			if strings.HasPrefix(upstreamReq.Header.Get("Content-Type"), "application/json") {
				bodyStr = FormatJSON(modifiedBody)
			}
			fmt.Printf("[DEBUG] Body:\n%s\n", bodyStr)
		}

		// 8. Execute Upstream Request
		client, err := proxyMgr.GetClient(route.ModelProvider, cfg.Proxy)
		if err != nil {
			http.Error(w, fmt.Sprintf(`{"error": {"message": "Failed to configure proxy client: %s"}}`, err.Error()), http.StatusInternalServerError)
			return nil, nil, hCtx, startTime, false
		}

		resp, err = client.Do(upstreamReq)
		if err != nil {
			elapsed := time.Since(startTime)
			fmt.Printf("[ERROR] Model=%s Provider=%s Key[%d] 502 %s Error: %v\n",
				modelName, route.ModelProvider.Name, route.KeyIndex, elapsed, err)
			http.Error(w, fmt.Sprintf(`{"error": {"message": "Upstream request failed: %s"}}`, err.Error()), http.StatusBadGateway)
			return nil, nil, hCtx, startTime, false
		}

		// 9. Record state status
		stateMgr.RecordUpstreamResult(route.AuthKey, resp.StatusCode)

		// Check if retry is needed for retryable status codes
		if (resp.StatusCode == 429 || resp.StatusCode == 500 || resp.StatusCode == 503) && attempt < maxRetries {
			resp.Body.Close()
			fmt.Printf("[WARN] Upstream (key[%d]) returned %d, retrying with next key (attempt %d/%d)\n",
				route.KeyIndex, resp.StatusCode, attempt+1, maxRetries)
			startTime = time.Now()
			continue
		}
		break
	}

	if debug {
		fmt.Printf("[DEBUG] --- UPSTREAM RESPONSE ---\n")
		fmt.Printf("[DEBUG] Status: %s\n", resp.Status)
		for k, vv := range resp.Header {
			for _, v := range vv {
				fmt.Printf("[DEBUG] Header: %s: %s\n", k, v)
			}
		}
	}

	return route, resp, hCtx, startTime, true
}

func handleChatCompletions(cfg *Config, stateMgr *StateManager, router *Router, proxyMgr *ProxyManager) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		bodyModifier := func(rawBody []byte, route *SelectedRoute, req *http.Request) ([]byte, string, error) {
			return ModifyRequestBody(rawBody, route, req, cfg, proxyMgr)
		}
		route, resp, hCtx, startTime, ok := executeUpstreamRequest(plugins.EndpointChat, w, r, cfg, stateMgr, router, proxyMgr, bodyModifier)
		if !ok {
			return
		}
		defer resp.Body.Close()

		isSSE := strings.Contains(resp.Header.Get("Content-Type"), "text/event-stream")

		p := plugins.Get(route.ModelProvider.ApiType)
		pluginCtx := buildPluginContext(route, r, cfg, proxyMgr)
		pluginCtx.RawRequestBody = hCtx.rawBody
		pluginCtx.IsStream = isSSE
		pluginCtx.Request = r

		// 10. Copy headers back to client
		p.ModifyResponseHeaders(resp, w.Header(), pluginCtx)

		// 11. Process and stream response body
		var respBody io.Reader = resp.Body
		if debug && isSSE {
			respBody = &loggingReader{r: resp.Body}
		}

		var writer io.Writer = w
		if flusher, ok := w.(http.Flusher); ok {
			writer = flushingWriter{w: w, f: flusher}
		}

		if isSSE {
			w.Header().Set("Content-Type", "text/event-stream; charset=utf-8")
			w.WriteHeader(resp.StatusCode)
			reader := bufio.NewReader(respBody)
			for {
				line, err := reader.ReadBytes('\n')
				if len(line) > 0 {
					modifiedLine, _ := p.TransformStreamChunk(plugins.EndpointChat, line, pluginCtx)
					if modifiedLine != nil {
						_, _ = writer.Write(modifiedLine)
					}
				}
				if err != nil {
					break
				}
			}
		} else {
			rawResp, err := io.ReadAll(respBody)
			if err != nil {
				w.WriteHeader(http.StatusBadGateway)
				return
			}
			if resp.StatusCode == 400 {
				fmt.Printf("[WARN] Upstream returned HTTP 400. Response body:\n%s\n", string(rawResp))
			}

			modified, contentType, err := p.TransformResponse(plugins.EndpointChat, resp, rawResp, pluginCtx)
			if err != nil {
				w.WriteHeader(http.StatusBadGateway)
				return
			}

			if debug {
				upstreamContentType := resp.Header.Get("Content-Type")
				bodyStr := string(rawResp)
				if strings.HasPrefix(upstreamContentType, "application/json") {
					bodyStr = FormatJSON(rawResp)
				}
				fmt.Printf("[DEBUG] --- UPSTREAM RESPONSE BODY (RAW) ---\n%s\n", bodyStr)
			}

			w.Header().Set("Content-Type", contentType)
			w.Header().Set("Content-Length", strconv.Itoa(len(modified)))
			w.WriteHeader(resp.StatusCode)
			_, _ = writer.Write(modified)
		}

		// Access logging
		elapsed := time.Since(startTime)
		fmt.Printf("[INFO] [%s] Model=%s Provider=%s Key[%d] %d %s\n",
			time.Now().Format("2006-01-02 15:04:05.000"), hCtx.modelName, route.ModelProvider.Name, route.KeyIndex, resp.StatusCode, elapsed)
	}
}

func handleImageGenerations(cfg *Config, stateMgr *StateManager, router *Router, proxyMgr *ProxyManager) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		route, resp, hCtx, startTime, ok := executeUpstreamRequest(plugins.EndpointImageGeneration, w, r, cfg, stateMgr, router, proxyMgr, ModifyImageRequestBody)
		if !ok {
			return
		}
		defer resp.Body.Close()

		p := plugins.Get(route.ModelProvider.ApiType)
		pluginCtx := buildPluginContext(route, r, cfg, proxyMgr)
		pluginCtx.RawRequestBody = hCtx.rawBody
		pluginCtx.Request = r

		p.ModifyResponseHeaders(resp, w.Header(), pluginCtx)

		rawResp, err := io.ReadAll(resp.Body)
		if err != nil {
			w.WriteHeader(http.StatusBadGateway)
			return
		}

		if resp.StatusCode == 400 {
			fmt.Printf("[WARN] Upstream returned HTTP 400. Response body:\n%s\n", string(rawResp))
		}

		if debug {
			bodyStr := string(rawResp)
			if strings.HasPrefix(resp.Header.Get("Content-Type"), "application/json") {
				bodyStr = FormatJSON(rawResp)
			}
			fmt.Printf("[DEBUG] --- UPSTREAM RESPONSE BODY (RAW) ---\n%s\n", bodyStr)
		}

		modified, contentType, err := p.TransformResponse(plugins.EndpointImageGeneration, resp, rawResp, pluginCtx)
		if err != nil {
			w.WriteHeader(http.StatusBadGateway)
			return
		}

		w.Header().Set("Content-Type", contentType)
		w.Header().Set("Content-Length", strconv.Itoa(len(modified)))
		w.WriteHeader(resp.StatusCode)
		_, _ = w.Write(modified)

		elapsed := time.Since(startTime)
		fmt.Printf("[INFO] [%s] Model=%s Provider=%s Key[%d] %d %s (Image)\n",
			time.Now().Format("2006-01-02 15:04:05.000"), hCtx.modelName, route.ModelProvider.Name, route.KeyIndex, resp.StatusCode, elapsed)
	}
}

func handleImageEdits(cfg *Config, stateMgr *StateManager, router *Router, proxyMgr *ProxyManager) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		route, resp, hCtx, startTime, ok := executeUpstreamRequest(plugins.EndpointImageEdit, w, r, cfg, stateMgr, router, proxyMgr, ModifyImageEditRequestBody)
		if !ok {
			return
		}
		defer resp.Body.Close()

		p := plugins.Get(route.ModelProvider.ApiType)
		pluginCtx := buildPluginContext(route, r, cfg, proxyMgr)
		pluginCtx.RawRequestBody = hCtx.rawBody
		pluginCtx.Request = r

		p.ModifyResponseHeaders(resp, w.Header(), pluginCtx)

		rawResp, err := io.ReadAll(resp.Body)
		if err != nil {
			w.WriteHeader(http.StatusBadGateway)
			return
		}

		if resp.StatusCode == 400 {
			fmt.Printf("[WARN] Upstream returned HTTP 400. Response body:\n%s\n", string(rawResp))
		}

		if debug {
			bodyStr := string(rawResp)
			if strings.HasPrefix(resp.Header.Get("Content-Type"), "application/json") {
				bodyStr = FormatJSON(rawResp)
			}
			fmt.Printf("[DEBUG] --- UPSTREAM RESPONSE BODY (RAW) ---\n%s\n", bodyStr)
		}

		modified, contentType, err := p.TransformResponse(plugins.EndpointImageEdit, resp, rawResp, pluginCtx)
		if err != nil {
			w.WriteHeader(http.StatusBadGateway)
			return
		}

		w.Header().Set("Content-Type", contentType)
		w.Header().Set("Content-Length", strconv.Itoa(len(modified)))
		w.WriteHeader(resp.StatusCode)
		_, _ = w.Write(modified)

		elapsed := time.Since(startTime)
		fmt.Printf("[INFO] [%s] Model=%s Provider=%s Key[%d] %d %s (Image Edit)\n",
			time.Now().Format("2006-01-02 15:04:05.000"), hCtx.modelName, route.ModelProvider.Name, route.KeyIndex, resp.StatusCode, elapsed)
	}
}

func handleAudioSpeech(cfg *Config, stateMgr *StateManager, router *Router, proxyMgr *ProxyManager) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		route, resp, hCtx, startTime, ok := executeUpstreamRequest(plugins.EndpointAudioSpeech, w, r, cfg, stateMgr, router, proxyMgr, ModifyAudioSpeechRequestBody)
		if !ok {
			return
		}
		defer resp.Body.Close()

		p := plugins.Get(route.ModelProvider.ApiType)
		pluginCtx := buildPluginContext(route, r, cfg, proxyMgr)
		pluginCtx.RawRequestBody = hCtx.rawBody
		pluginCtx.Request = r

		p.ModifyResponseHeaders(resp, w.Header(), pluginCtx)

		rawResp, err := io.ReadAll(resp.Body)
		if err != nil {
			w.WriteHeader(http.StatusBadGateway)
			return
		}

		if resp.StatusCode == 400 {
			fmt.Printf("[WARN] Upstream returned HTTP 400. Response body:\n%s\n", string(rawResp))
		}

		if debug {
			bodyStr := string(rawResp)
			if strings.HasPrefix(resp.Header.Get("Content-Type"), "application/json") {
				bodyStr = FormatJSON(rawResp)
			} else {
				bodyStr = fmt.Sprintf("[Binary audio data, size: %d bytes]", len(rawResp))
			}
			fmt.Printf("[DEBUG] --- UPSTREAM RESPONSE BODY (RAW) ---\n%s\n", bodyStr)
		}

		modified, contentType, err := p.TransformResponse(plugins.EndpointAudioSpeech, resp, rawResp, pluginCtx)
		if err != nil {
			w.WriteHeader(http.StatusBadGateway)
			if json.Valid(rawResp) {
				w.Header().Set("Content-Type", "application/json; charset=utf-8")
				w.Header().Set("Content-Length", strconv.Itoa(len(rawResp)))
				w.WriteHeader(resp.StatusCode)
				_, _ = w.Write(rawResp)
			}
			return
		}

		w.Header().Set("Content-Type", contentType)
		w.Header().Set("Content-Length", strconv.Itoa(len(modified)))
		w.WriteHeader(resp.StatusCode)
		_, _ = w.Write(modified)

		elapsed := time.Since(startTime)
		fmt.Printf("[INFO] [%s] Model=%s Provider=%s Key[%d] %d %s (Speech)\n",
			time.Now().Format("2006-01-02 15:04:05.000"), hCtx.modelName, route.ModelProvider.Name, route.KeyIndex, resp.StatusCode, elapsed)
	}
}

func handleAudioTranscriptions(cfg *Config, stateMgr *StateManager, router *Router, proxyMgr *ProxyManager) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		bodyModifier := func(rawBody []byte, route *SelectedRoute, req *http.Request) ([]byte, string, error) {
			return ModifyAudioTranscriptionRequestBody(rawBody, route, req, cfg, proxyMgr)
		}

		route, resp, hCtx, startTime, ok := executeUpstreamRequest(plugins.EndpointAudioTranscription, w, r, cfg, stateMgr, router, proxyMgr, bodyModifier)
		if !ok {
			return
		}
		defer resp.Body.Close()

		isSSE := strings.Contains(resp.Header.Get("Content-Type"), "text/event-stream")

		p := plugins.Get(route.ModelProvider.ApiType)
		pluginCtx := buildPluginContext(route, r, cfg, proxyMgr)
		pluginCtx.RawRequestBody = hCtx.rawBody
		pluginCtx.IsStream = isSSE
		pluginCtx.Request = r

		var respBody io.Reader = resp.Body
		if debug && isSSE {
			respBody = &loggingReader{r: resp.Body}
		}

		p.ModifyResponseHeaders(resp, w.Header(), pluginCtx)

		var writer io.Writer = w
		if flusher, ok := w.(http.Flusher); ok {
			writer = flushingWriter{w: w, f: flusher}
		}

		if isSSE {
			w.Header().Set("Content-Type", "text/event-stream; charset=utf-8")
			w.WriteHeader(resp.StatusCode)

			reader := bufio.NewReader(respBody)
			for {
				line, err := reader.ReadBytes('\n')
				if len(line) > 0 {
					modifiedLine, _ := p.TransformStreamChunk(plugins.EndpointAudioTranscription, line, pluginCtx)
					if modifiedLine != nil {
						_, _ = writer.Write(modifiedLine)
					}
				}
				if err != nil {
					break
				}
			}
			if route.ModelProvider.ApiType == "gemini" {
				_, _ = writer.Write([]byte("data: [DONE]\n\n"))
			}
		} else {
			rawResp, err := io.ReadAll(resp.Body)
			if err != nil {
				w.WriteHeader(http.StatusBadGateway)
				return
			}

			if resp.StatusCode == 400 {
				fmt.Printf("[WARN] Upstream returned HTTP 400. Response body:\n%s\n", string(rawResp))
			}

			if debug {
				bodyStr := string(rawResp)
				if strings.HasPrefix(resp.Header.Get("Content-Type"), "application/json") {
					bodyStr = FormatJSON(rawResp)
				}
				fmt.Printf("[DEBUG] --- UPSTREAM RESPONSE BODY (RAW) ---\n%s\n", bodyStr)
			}

			modified, contentType, err := p.TransformResponse(plugins.EndpointAudioTranscription, resp, rawResp, pluginCtx)
			if err != nil {
				w.WriteHeader(http.StatusBadGateway)
				return
			}

			w.Header().Set("Content-Type", contentType)
			w.Header().Set("Content-Length", strconv.Itoa(len(modified)))
			w.WriteHeader(resp.StatusCode)
			_, _ = w.Write(modified)
		}

		elapsed := time.Since(startTime)
		fmt.Printf("[INFO] [%s] Model=%s Provider=%s Key[%d] %d %s (Transcription)\n",
			time.Now().Format("2006-01-02 15:04:05.000"), hCtx.modelName, route.ModelProvider.Name, route.KeyIndex, resp.StatusCode, elapsed)
	}
}
