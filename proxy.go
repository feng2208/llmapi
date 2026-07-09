package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"

	"golang.org/x/net/proxy"
)

type ProxyManager struct {
	mu         sync.Mutex
	transports map[string]*http.Transport
}

func NewProxyManager() *ProxyManager {
	return &ProxyManager{
		transports: make(map[string]*http.Transport),
	}
}

func (pm *ProxyManager) GetTransport(proxyURL string) (*http.Transport, error) {
	pm.mu.Lock()
	defer pm.mu.Unlock()

	if t, exists := pm.transports[proxyURL]; exists {
		return t, nil
	}

	transport := &http.Transport{
		MaxIdleConns:        100,
		IdleConnTimeout:     90 * time.Second,
		TLSHandshakeTimeout: 10 * time.Second,
	}

	if proxyURL != "" {
		u, err := url.Parse(proxyURL)
		if err != nil {
			return nil, fmt.Errorf("invalid proxy URL %q: %w", proxyURL, err)
		}

		if u.Scheme == "socks5" || u.Scheme == "socks5h" {
			dialer, err := proxy.FromURL(u, proxy.Direct)
			if err != nil {
				return nil, fmt.Errorf("failed to create socks5 dialer: %w", err)
			}
			transport.DialContext = func(ctx context.Context, network, addr string) (net.Conn, error) {
				if cd, ok := dialer.(proxy.ContextDialer); ok {
					return cd.DialContext(ctx, network, addr)
				}
				return dialer.Dial(network, addr)
			}
		} else if u.Scheme == "http" || u.Scheme == "https" {
			transport.Proxy = http.ProxyURL(u)
		} else {
			return nil, fmt.Errorf("unsupported proxy scheme %q", u.Scheme)
		}
	}

	pm.transports[proxyURL] = transport
	return transport, nil
}

func (pm *ProxyManager) GetClient(mpc *ModelProviderConfig, globalProxy string) (*http.Client, error) {
	proxyURL := mpc.Proxy
	if proxyURL == "" {
		proxyURL = globalProxy
	}

	transport, err := pm.GetTransport(proxyURL)
	if err != nil {
		return nil, err
	}

	return &http.Client{
		Transport: transport,
		Timeout:   mpc.Timeout(),
	}, nil
}

// Deep merges src map into dst map.
func deepMerge(dst, src map[string]interface{}) {
	for k, v := range src {
		if srcMap, ok := v.(map[string]interface{}); ok {
			if dstMap, ok := dst[k].(map[string]interface{}); ok {
				deepMerge(dstMap, srcMap)
				continue
			}
		}
		dst[k] = v
	}
}

// ModifyRequestBody applies the delete, extra, and model replacement rules.
func ModifyRequestBody(rawBody []byte, route *SelectedRoute) ([]byte, error) {
	var body map[string]interface{}
	if err := json.Unmarshal(rawBody, &body); err != nil {
		return nil, fmt.Errorf("failed to parse JSON request body: %w", err)
	}

	// 1. Replace Model Name
	body["model"] = route.ModelProvider.Model

	// 2. Delete Configured Keys
	for _, item := range route.ModelProvider.RequestBody.Delete {
		if str, ok := item.(string); ok {
			delete(body, str)
		} else if m, ok := item.(map[string]interface{}); ok {
			for k := range m {
				delete(body, k)
			}
		} else if m, ok := item.(map[interface{}]interface{}); ok {
			for k := range m {
				if s, ok := k.(string); ok {
					delete(body, s)
				}
			}
		}
	}

	// 3. Add/Replace Extra Fields
	for _, extraMap := range route.ModelProvider.RequestBody.Extra {
		deepMerge(body, extraMap)
	}

	// 4. Inject skip_thought_signature_validator conditionally for gemini provider
	if route.ModelProvider.Name == "gemini" {
		if messages, ok := body["messages"].([]interface{}); ok {
			for _, rawMsg := range messages {
				if msg, ok := rawMsg.(map[string]interface{}); ok {
					role, _ := msg["role"].(string)
					if role == "assistant" || role == "model" {
						if toolCalls, ok := msg["tool_calls"].([]interface{}); ok {
							for _, rawCall := range toolCalls {
								if call, ok := rawCall.(map[string]interface{}); ok {
									// Retrieve or create extra_content
									var extraContent map[string]interface{}
									if rawExtra, exists := call["extra_content"]; exists {
										if m, ok := rawExtra.(map[string]interface{}); ok {
											extraContent = m
										}
									}
									if extraContent == nil {
										extraContent = make(map[string]interface{})
										call["extra_content"] = extraContent
									}

									// Retrieve or create google sub-map
									var google map[string]interface{}
									if rawGoogle, exists := extraContent["google"]; exists {
										if m, ok := rawGoogle.(map[string]interface{}); ok {
											google = m
										}
									}
									if google == nil {
										google = make(map[string]interface{})
										extraContent["google"] = google
									}

									// Inject signature
									google["thought_signature"] = "skip_thought_signature_validator"
								}
							}
						}
					}
				}
			}
		}
	}

	modified, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal modified JSON: %w", err)
	}

	return modified, nil
}

// Custom reader for debug logging of response chunks in real-time.
type loggingReader struct {
	r io.Reader
}

func (lr loggingReader) Read(p []byte) (n int, err error) {
	n, err = lr.r.Read(p)
	if n > 0 {
		fmt.Printf("[DEBUG] Upstream Response Chunk:\n%s\n", string(p[:n]))
	}
	return
}

type flushingWriter struct {
	w io.Writer
	f http.Flusher
}

func (fw flushingWriter) Write(p []byte) (n int, err error) {
	n, err = fw.w.Write(p)
	if fw.f != nil {
		fw.f.Flush()
	}
	return
}

// Suffix helper for streaming tag matching
func getHoldbackPrefix(s, target string) string {
	if s == "" || target == "" {
		return ""
	}
	maxLen := len(s)
	if len(target) < maxLen {
		maxLen = len(target)
	}
	for l := maxLen; l > 0; l-- {
		suffix := s[len(s)-l:]
		prefix := target[:l]
		if suffix == prefix {
			return suffix
		}
	}
	return ""
}

type StreamExtractor struct {
	startTag         string
	endTag           string
	fullBuffer       string
	flushedContent   string
	flushedReasoning string
}

func NewStreamExtractor(startTag, endTag string) *StreamExtractor {
	return &StreamExtractor{
		startTag: startTag,
		endTag:   endTag,
	}
}

func (se *StreamExtractor) ProcessSSELine(line []byte) []byte {
	if se.startTag == "" || se.endTag == "" {
		return line
	}

	trimmed := bytes.TrimSpace(line)
	if !bytes.HasPrefix(trimmed, []byte("data: ")) {
		return line
	}

	payload := bytes.TrimPrefix(trimmed, []byte("data: "))
	if string(payload) == "[DONE]" {
		return line
	}

	var obj map[string]interface{}
	if err := json.Unmarshal(payload, &obj); err != nil {
		return line
	}

	choices, ok := obj["choices"].([]interface{})
	if !ok || len(choices) == 0 {
		return line
	}

	choice, ok := choices[0].(map[string]interface{})
	if !ok {
		return line
	}

	delta, ok := choice["delta"].(map[string]interface{})
	if !ok {
		return line
	}

	content, _ := delta["content"].(string)
	se.fullBuffer += content

	var activeContent string
	var activeReasoning string

	idxStart := strings.Index(se.fullBuffer, se.startTag)
	if idxStart == -1 {
		holdback := getHoldbackPrefix(se.fullBuffer, se.startTag)
		activeContent = se.fullBuffer[:len(se.fullBuffer)-len(holdback)]
		activeReasoning = ""
	} else {
		contentPart := se.fullBuffer[:idxStart]
		idxEnd := strings.Index(se.fullBuffer, se.endTag)
		if idxEnd == -1 {
			reasoningSoFar := se.fullBuffer[idxStart+len(se.startTag):]
			holdback := getHoldbackPrefix(reasoningSoFar, se.endTag)
			activeContent = contentPart
			activeReasoning = reasoningSoFar[:len(reasoningSoFar)-len(holdback)]
		} else {
			reasoningPart := se.fullBuffer[idxStart+len(se.startTag) : idxEnd]
			afterPart := se.fullBuffer[idxEnd+len(se.endTag):]
			activeContent = contentPart + afterPart
			activeReasoning = reasoningPart
		}
	}

	deltaContent := ""
	if len(activeContent) > len(se.flushedContent) {
		deltaContent = activeContent[len(se.flushedContent):]
	}
	deltaReasoning := ""
	if len(activeReasoning) > len(se.flushedReasoning) {
		deltaReasoning = activeReasoning[len(se.flushedReasoning):]
	}

	se.flushedContent = activeContent
	se.flushedReasoning = activeReasoning

	if deltaContent != "" {
		delta["content"] = deltaContent
	} else {
		delete(delta, "content")
	}

	if deltaReasoning != "" {
		delta["reasoning"] = deltaReasoning
		delta["reasoning_content"] = deltaReasoning
	}

	modifiedPayload, err := json.Marshal(obj)
	if err != nil {
		return line
	}

	var result bytes.Buffer
	if bytes.HasSuffix(line, []byte("\r\n")) {
		result.Write([]byte("data: "))
		result.Write(modifiedPayload)
		result.Write([]byte("\r\n"))
	} else {
		result.Write([]byte("data: "))
		result.Write(modifiedPayload)
		result.Write([]byte("\n"))
	}
	return result.Bytes()
}

// ProcessJSONResponse extracts reasoning content from a non-streaming JSON response.
func ProcessJSONResponse(respBody []byte, startTag, endTag string) []byte {
	if startTag == "" || endTag == "" {
		return respBody
	}

	var obj map[string]interface{}
	if err := json.Unmarshal(respBody, &obj); err != nil {
		return respBody
	}

	choices, ok := obj["choices"].([]interface{})
	if !ok || len(choices) == 0 {
		return respBody
	}

	choice, ok := choices[0].(map[string]interface{})
	if !ok {
		return respBody
	}

	message, ok := choice["message"].(map[string]interface{})
	if !ok {
		return respBody
	}

	content, _ := message["content"].(string)
	if content == "" {
		return respBody
	}

	idxStart := strings.Index(content, startTag)
	if idxStart != -1 {
		idxEnd := strings.Index(content, endTag)
		if idxEnd != -1 && idxEnd > idxStart {
			reasoning := content[idxStart+len(startTag) : idxEnd]
			cleanContent := content[:idxStart] + content[idxEnd+len(endTag):]

			message["content"] = cleanContent
			message["reasoning"] = reasoning
			message["reasoning_content"] = reasoning
		}
	}

	modified, err := json.Marshal(obj)
	if err != nil {
		return respBody
	}
	return modified
}
