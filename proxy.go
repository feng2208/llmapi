package main

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math/rand"
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
		srcChildMap, isSrcMap := toMapStringInterface(v)
		if isSrcMap {
			if dstChildMap, isDstMap := toMapStringInterface(dst[k]); isDstMap {
				deepMerge(dstChildMap, srcChildMap)
				dst[k] = dstChildMap
				continue
			} else {
				dst[k] = normalizeMap(srcChildMap)
				continue
			}
		}

		srcSlice, isSrcSlice := toSliceInterface(v)
		if isSrcSlice {
			if dstSlice, isDstSlice := toSliceInterface(dst[k]); isDstSlice {
				merged := append([]interface{}{}, dstSlice...)
				for _, srcItem := range srcSlice {
					exists := false
					for _, dstItem := range merged {
						if areEqualJSON(dstItem, srcItem) {
							exists = true
							break
						}
					}
					if !exists {
						merged = append(merged, normalizeValue(srcItem))
					}
				}
				dst[k] = merged
				continue
			} else {
				dst[k] = normalizeSlice(srcSlice)
				continue
			}
		}

		dst[k] = v
	}
}

func toMapStringInterface(v interface{}) (map[string]interface{}, bool) {
	if v == nil {
		return nil, false
	}
	if m, ok := v.(map[string]interface{}); ok {
		return m, true
	}
	if m, ok := v.(map[interface{}]interface{}); ok {
		res := make(map[string]interface{})
		for k, val := range m {
			strKey := fmt.Sprintf("%v", k)
			res[strKey] = val
		}
		return res, true
	}
	return nil, false
}

func toSliceInterface(v interface{}) ([]interface{}, bool) {
	if v == nil {
		return nil, false
	}
	if s, ok := v.([]interface{}); ok {
		return s, true
	}
	return nil, false
}

func areEqualJSON(a, b interface{}) bool {
	normA := normalizeValue(a)
	normB := normalizeValue(b)
	ja, err1 := json.Marshal(normA)
	jb, err2 := json.Marshal(normB)
	if err1 != nil || err2 != nil {
		return normA == normB
	}
	return string(ja) == string(jb)
}

func normalizeValue(v interface{}) interface{} {
	if v == nil {
		return nil
	}
	if m, ok := toMapStringInterface(v); ok {
		return normalizeMap(m)
	}
	if s, ok := toSliceInterface(v); ok {
		return normalizeSlice(s)
	}
	return v
}

func normalizeSlice(s []interface{}) []interface{} {
	res := make([]interface{}, len(s))
	for i, v := range s {
		res[i] = normalizeValue(v)
	}
	return res
}

func normalizeMap(m map[string]interface{}) map[string]interface{} {
	res := make(map[string]interface{})
	for k, v := range m {
		res[k] = normalizeValue(v)
	}
	return res
}

// GeminiPart represents a single part of content in Gemini API request
type GeminiPart map[string]interface{}

// GeminiContent represents the content of conversation in Gemini API request
type GeminiContent struct {
	Role  string       `json:"role"`
	Parts []GeminiPart `json:"parts"`
}

func generateRandomString(length int) string {
	const charset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
	b := make([]byte, length)
	for i := range b {
		b[i] = charset[rand.Intn(len(charset))]
	}
	return string(b)
}

func parseImageUrlPart(urlStr string) (map[string]interface{}, error) {
	if strings.HasPrefix(urlStr, "data:") {
		// format: data:<mime>;base64,<data>
		parts := strings.SplitN(urlStr, ",", 2)
		if len(parts) != 2 {
			return nil, errors.New("invalid data URI")
		}

		header := parts[0]
		base64Data := parts[1]

		mimeType := "image/jpeg"
		if strings.HasPrefix(header, "data:") && strings.Contains(header, ";") {
			mimePart := strings.TrimPrefix(header, "data:")
			semiIdx := strings.Index(mimePart, ";")
			if semiIdx != -1 {
				mimeType = mimePart[:semiIdx]
			}
		}

		return map[string]interface{}{
			"inline_data": map[string]interface{}{
				"mime_type": mimeType,
				"data":      base64Data,
			},
		}, nil
	} else if strings.HasPrefix(urlStr, "http://") || strings.HasPrefix(urlStr, "https://") {
		client := &http.Client{
			Timeout: 10 * time.Second,
		}
		resp, err := client.Get(urlStr)
		if err != nil {
			return nil, fmt.Errorf("failed to fetch image from URL: %w", err)
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			return nil, fmt.Errorf("failed to fetch image, status code: %d", resp.StatusCode)
		}

		mimeType := resp.Header.Get("Content-Type")
		if mimeType == "" {
			mimeType = "image/jpeg" // Fallback
		}

		data, err := io.ReadAll(resp.Body)
		if err != nil {
			return nil, fmt.Errorf("failed to read image body: %w", err)
		}

		base64Data := base64.StdEncoding.EncodeToString(data)
		return map[string]interface{}{
			"inline_data": map[string]interface{}{
				"mime_type": mimeType,
				"data":      base64Data,
			},
		}, nil
	}

	return nil, fmt.Errorf("unsupported image URL format")
}

func cleanGeminiSchema(schema interface{}) interface{} {
	m, ok := schema.(map[string]interface{})
	if !ok {
		return schema
	}

	cleaned := make(map[string]interface{})
	for k, v := range m {
		// Skip schema version/metadata keys that Gemini's protobuf validator rejects
		if k == "$schema" || k == "$id" || k == "$vocabulary" || k == "$anchor" || k == "additionalProperties" || k == "exclusiveMinimum" {
			continue
		}

		// Recursively clean sub-schemas (e.g. properties, items)
		if k == "properties" {
			if propsMap, ok := v.(map[string]interface{}); ok {
				cleanedProps := make(map[string]interface{})
				for propName, propVal := range propsMap {
					cleanedProps[propName] = cleanGeminiSchema(propVal)
				}
				cleaned[k] = cleanedProps
				continue
			}
		}
		if k == "items" {
			cleaned[k] = cleanGeminiSchema(v)
			continue
		}

		cleaned[k] = v
	}
	return cleaned
}

func TransformRequestToGemini(rawBody []byte, route *SelectedRoute) ([]byte, error) {
	var body map[string]interface{}
	if err := json.Unmarshal(rawBody, &body); err != nil {
		return nil, fmt.Errorf("failed to parse JSON request body: %w", err)
	}

	geminiReq := make(map[string]interface{})

	// 1. Map messages to contents and systemInstruction
	openAIMessages, _ := body["messages"].([]interface{})

	// Collect toolCallID to function name mapping for tool messages
	toolCallIDToName := make(map[string]string)
	for _, msg := range openAIMessages {
		if msgMap, ok := msg.(map[string]interface{}); ok {
			if toolCalls, ok := msgMap["tool_calls"].([]interface{}); ok {
				for _, tc := range toolCalls {
					if tcMap, ok := tc.(map[string]interface{}); ok {
						id, _ := tcMap["id"].(string)
						if fnMap, ok := tcMap["function"].(map[string]interface{}); ok {
							name, _ := fnMap["name"].(string)
							if id != "" && name != "" {
								toolCallIDToName[id] = name
							}
						}
					}
				}
			}
		}
	}

	var geminiContents []GeminiContent
	var systemParts []GeminiPart

	for _, msg := range openAIMessages {
		msgMap, ok := msg.(map[string]interface{})
		if !ok {
			continue
		}

		role, _ := msgMap["role"].(string)
		content := msgMap["content"]

		if role == "system" {
			if txt, ok := content.(string); ok && txt != "" {
				systemParts = append(systemParts, map[string]interface{}{"text": txt})
			} else if arr, ok := content.([]interface{}); ok {
				for _, item := range arr {
					if itemMap, ok := item.(map[string]interface{}); ok {
						if itemMap["type"] == "text" {
							if txt, ok := itemMap["text"].(string); ok && txt != "" {
								systemParts = append(systemParts, map[string]interface{}{"text": txt})
							}
						}
					}
				}
			}
			continue
		}

		var parts []GeminiPart

		// Handle content (text, image)
		if role != "tool" {
			if txt, ok := content.(string); ok && txt != "" {
				parts = append(parts, map[string]interface{}{"text": txt})
			} else if arr, ok := content.([]interface{}); ok {
				for _, item := range arr {
					if itemMap, ok := item.(map[string]interface{}); ok {
						itemType, _ := itemMap["type"].(string)
						if itemType == "text" {
							if txt, ok := itemMap["text"].(string); ok && txt != "" {
								parts = append(parts, map[string]interface{}{"text": txt})
							}
						} else if itemType == "image_url" {
							if imgUrlMap, ok := itemMap["image_url"].(map[string]interface{}); ok {
								if urlStr, ok := imgUrlMap["url"].(string); ok && urlStr != "" {
									part, err := parseImageUrlPart(urlStr)
									if err != nil {
										return nil, fmt.Errorf("failed to process image: %w", err)
									}
									parts = append(parts, part)
								}
							}
						}
					}
				}
			}
		}

		// Handle tool calls in assistant messages
		if toolCalls, ok := msgMap["tool_calls"].([]interface{}); ok {
			for _, tc := range toolCalls {
				if tcMap, ok := tc.(map[string]interface{}); ok {
					if fnMap, ok := tcMap["function"].(map[string]interface{}); ok {
						name, _ := fnMap["name"].(string)
						argsStr, _ := fnMap["arguments"].(string)

						var args map[string]interface{}
						if err := json.Unmarshal([]byte(argsStr), &args); err != nil {
							args = make(map[string]interface{})
						}

						// Retrieve signature if exists in extra_content
						var thoughtSig string
						if extraMap, ok := tcMap["extra_content"].(map[string]interface{}); ok {
							if googleMap, ok := extraMap["google"].(map[string]interface{}); ok {
								if sig, ok := googleMap["thought_signature"].(string); ok {
									thoughtSig = sig
								} else if sig, ok := googleMap["thoughtSignature"].(string); ok {
									thoughtSig = sig
								}
							}
						}

						part := map[string]interface{}{
							"functionCall": map[string]interface{}{
								"name": name,
								"args": args,
							},
						}
						if thoughtSig != "" {
							part["thoughtSignature"] = thoughtSig
						}
						parts = append(parts, part)
					}
				}
			}
		}

		// Handle tool response
		if role == "tool" {
			name, _ := msgMap["name"].(string)
			if name == "" {
				toolCallID, _ := msgMap["tool_call_id"].(string)
				if toolCallID != "" {
					name = toolCallIDToName[toolCallID]
				}
			}

			var respObj interface{}
			var contentStr string
			if s, ok := content.(string); ok {
				contentStr = s
			} else {
				b, _ := json.Marshal(content)
				contentStr = string(b)
			}

			var tempMap map[string]interface{}
			if err := json.Unmarshal([]byte(contentStr), &tempMap); err == nil {
				respObj = tempMap
			} else {
				respObj = map[string]interface{}{
					"result": contentStr,
				}
			}

			parts = append(parts, map[string]interface{}{
				"functionResponse": map[string]interface{}{
					"name":     name,
					"response": respObj,
				},
			})
		}

		var geminiRole string
		if role == "assistant" {
			geminiRole = "model"
		} else {
			geminiRole = "user"
		}

		if len(parts) > 0 {
			if len(geminiContents) > 0 && geminiContents[len(geminiContents)-1].Role == geminiRole {
				geminiContents[len(geminiContents)-1].Parts = append(geminiContents[len(geminiContents)-1].Parts, parts...)
			} else {
				geminiContents = append(geminiContents, GeminiContent{
					Role:  geminiRole,
					Parts: parts,
				})
			}
		}
	}

	if len(geminiContents) > 0 {
		geminiReq["contents"] = geminiContents
	}
	if len(systemParts) > 0 {
		geminiReq["systemInstruction"] = map[string]interface{}{
			"parts": systemParts,
		}
	}

	// 2. Map tools using camelCase "functionDeclarations"
	if openAITools, ok := body["tools"].([]interface{}); ok && len(openAITools) > 0 {
		var functionDeclarations []interface{}
		for _, t := range openAITools {
			if tMap, ok := t.(map[string]interface{}); ok {
				tType, _ := tMap["type"].(string)
				if tType == "function" {
					if fnMap, ok := tMap["function"].(map[string]interface{}); ok {
						decl := map[string]interface{}{
							"name": fnMap["name"],
						}
						if desc, ok := fnMap["description"]; ok {
							decl["description"] = desc
						}
						if params, ok := fnMap["parameters"]; ok {
							decl["parameters"] = cleanGeminiSchema(params)
						}
						functionDeclarations = append(functionDeclarations, decl)
					}
				}
			}
		}
		if len(functionDeclarations) > 0 {
			geminiReq["tools"] = []interface{}{
				map[string]interface{}{
					"functionDeclarations": functionDeclarations,
				},
			}
		}
	}

	// 3. Map tool_choice
	if toolChoice, ok := body["tool_choice"]; ok {
		if tcStr, ok := toolChoice.(string); ok {
			if tcStr == "none" {
				geminiReq["toolConfig"] = map[string]interface{}{
					"functionCallingConfig": map[string]interface{}{
						"mode": "NONE",
					},
				}
			} else if tcStr == "required" {
				geminiReq["toolConfig"] = map[string]interface{}{
					"functionCallingConfig": map[string]interface{}{
						"mode": "ANY",
					},
				}
			} else if tcStr == "auto" {
				geminiReq["toolConfig"] = map[string]interface{}{
					"functionCallingConfig": map[string]interface{}{
						"mode": "AUTO",
					},
				}
			}
		} else if tcMap, ok := toolChoice.(map[string]interface{}); ok {
			if fnMap, ok := tcMap["function"].(map[string]interface{}); ok {
				if name, ok := fnMap["name"].(string); ok && name != "" {
					geminiReq["toolConfig"] = map[string]interface{}{
						"functionCallingConfig": map[string]interface{}{
							"mode":                 "ANY",
							"allowedFunctionNames": []string{name},
						},
					}
				}
			}
		}
	}

	// 4. Map generationConfig
	generationConfig := make(map[string]interface{})

	if temp, ok := body["temperature"].(float64); ok {
		generationConfig["temperature"] = temp
	}
	if topP, ok := body["top_p"].(float64); ok {
		generationConfig["topP"] = topP
	}
	if maxTokens, ok := body["max_tokens"].(float64); ok {
		generationConfig["maxOutputTokens"] = int(maxTokens)
	} else if maxTokens, ok := body["max_tokens"].(int); ok {
		generationConfig["maxOutputTokens"] = maxTokens
	} else if maxTokens, ok := body["max_completion_tokens"].(float64); ok {
		generationConfig["maxOutputTokens"] = int(maxTokens)
	} else if maxTokens, ok := body["max_completion_tokens"].(int); ok {
		generationConfig["maxOutputTokens"] = maxTokens
	}

	if stop, ok := body["stop"]; ok {
		if stopStr, ok := stop.(string); ok {
			generationConfig["stopSequences"] = []string{stopStr}
		} else if stopArr, ok := stop.([]interface{}); ok {
			var stops []string
			for _, s := range stopArr {
				if sStr, ok := s.(string); ok {
					stops = append(stops, sStr)
				}
			}
			generationConfig["stopSequences"] = stops
		}
	}

	if responseFormat, ok := body["response_format"].(map[string]interface{}); ok {
		rfType, _ := responseFormat["type"].(string)
		if rfType == "json_object" {
			generationConfig["responseMimeType"] = "application/json"
		} else if rfType == "json_schema" {
			generationConfig["responseMimeType"] = "application/json"
			if jsonSchema, ok := responseFormat["json_schema"].(map[string]interface{}); ok {
				if schema, ok := jsonSchema["schema"]; ok {
					generationConfig["responseSchema"] = schema
				}
			}
		}
	}

	if len(generationConfig) > 0 {
		geminiReq["generationConfig"] = generationConfig
	}

	// 5. Merge any custom configuration from route.ModelProvider.RequestBody.Extra
	for _, configMap := range route.ModelProvider.RequestBody.Extra {
		deepMerge(geminiReq, configMap)
	}

	return json.Marshal(geminiReq)
}

func ProcessGeminiJSONResponse(rawResp []byte, modelName string) ([]byte, error) {
	var geminiResp map[string]interface{}
	if err := json.Unmarshal(rawResp, &geminiResp); err != nil {
		return nil, err
	}

	choices := []map[string]interface{}{}
	candidates, _ := geminiResp["candidates"].([]interface{})
	for idx, cand := range candidates {
		candMap, ok := cand.(map[string]interface{})
		if !ok {
			continue
		}

		content, _ := candMap["content"].(map[string]interface{})
		parts, _ := content["parts"].([]interface{})

		var contentText string
		var reasoningText string
		var toolCalls []interface{}

		for _, part := range parts {
			partMap, ok := part.(map[string]interface{})
			if !ok {
				continue
			}

			if isThought, _ := partMap["thought"].(bool); isThought {
				if txt, ok := partMap["text"].(string); ok {
					reasoningText += txt
				}
			} else if txt, ok := partMap["text"].(string); ok {
				contentText += txt
			} else if fc, ok := partMap["functionCall"].(map[string]interface{}); ok {
				name, _ := fc["name"].(string)
				args := fc["args"]
				argsStr := "{}"
				if args != nil {
					if b, err := json.Marshal(args); err == nil {
						argsStr = string(b)
					}
				}

				thoughtSig, _ := partMap["thoughtSignature"].(string)
				toolCallID := fmt.Sprintf("call_%d_%s", len(toolCalls), generateRandomString(8))
				tcObj := map[string]interface{}{
					"id":   toolCallID,
					"type": "function",
					"function": map[string]interface{}{
						"name":      name,
						"arguments": argsStr,
					},
				}
				if thoughtSig != "" {
					tcObj["extra_content"] = map[string]interface{}{
						"google": map[string]interface{}{
							"thought_signature": thoughtSig,
							"thoughtSignature":  thoughtSig,
						},
					}
				}
				toolCalls = append(toolCalls, tcObj)
			}
		}

		finishReasonStr := "stop"
		if len(toolCalls) > 0 {
			finishReasonStr = "tool_calls"
		} else if fr, ok := candMap["finishReason"].(string); ok {
			switch fr {
			case "STOP":
				finishReasonStr = "stop"
			case "MAX_TOKENS":
				finishReasonStr = "length"
			case "SAFETY", "RECITATION":
				finishReasonStr = "content_filter"
			case "OTHER":
				finishReasonStr = "stop"
			default:
				finishReasonStr = strings.ToLower(fr)
			}
		}

		message := map[string]interface{}{
			"role":    "assistant",
			"content": contentText,
		}
		if reasoningText != "" {
			message["reasoning_content"] = reasoningText
			message["reasoning"] = reasoningText
		}
		if len(toolCalls) > 0 {
			message["tool_calls"] = toolCalls
			if contentText == "" {
				message["content"] = nil
			}
		}

		choices = append(choices, map[string]interface{}{
			"index":         idx,
			"message":       message,
			"finish_reason": finishReasonStr,
		})
	}

	usage := map[string]interface{}{
		"prompt_tokens":     0,
		"completion_tokens": 0,
		"total_tokens":      0,
	}
	if um, ok := geminiResp["usageMetadata"].(map[string]interface{}); ok {
		if pt, ok := um["promptTokenCount"].(float64); ok {
			usage["prompt_tokens"] = int(pt)
		}
		if ct, ok := um["candidatesTokenCount"].(float64); ok {
			usage["completion_tokens"] = int(ct)
		}
		if tt, ok := um["totalTokenCount"].(float64); ok {
			usage["total_tokens"] = int(tt)
		}
	}

	openAIResp := map[string]interface{}{
		"id":      fmt.Sprintf("chatcmpl-%s", generateRandomString(12)),
		"object":  "chat.completion",
		"created": time.Now().Unix(),
		"model":   modelName,
		"choices": choices,
		"usage":   usage,
	}

	return json.Marshal(openAIResp)
}

func ProcessGeminiSSELine(line []byte, genID string, createdTime int64, modelName string) []byte {
	trimmed := bytes.TrimSpace(line)
	if !bytes.HasPrefix(trimmed, []byte("data: ")) {
		return line
	}

	payload := bytes.TrimPrefix(trimmed, []byte("data: "))
	payloadStr := string(payload)
	if payloadStr == "[DONE]" {
		return line
	}

	var geminiResp map[string]interface{}
	if err := json.Unmarshal(payload, &geminiResp); err != nil {
		return line
	}

	choices := []map[string]interface{}{}
	candidates, _ := geminiResp["candidates"].([]interface{})
	for idx, cand := range candidates {
		candMap, ok := cand.(map[string]interface{})
		if !ok {
			continue
		}

		content, _ := candMap["content"].(map[string]interface{})
		parts, _ := content["parts"].([]interface{})

		var contentText string
		var reasoningText string
		var toolCalls []interface{}

		for _, part := range parts {
			partMap, ok := part.(map[string]interface{})
			if !ok {
				continue
			}

			if isThought, _ := partMap["thought"].(bool); isThought {
				if txt, ok := partMap["text"].(string); ok {
					reasoningText += txt
				}
			} else if txt, ok := partMap["text"].(string); ok {
				contentText += txt
			} else if fc, ok := partMap["functionCall"].(map[string]interface{}); ok {
				name, _ := fc["name"].(string)
				args := fc["args"]
				argsStr := "{}"
				if args != nil {
					if b, err := json.Marshal(args); err == nil {
						argsStr = string(b)
					}
				}

				thoughtSig, _ := partMap["thoughtSignature"].(string)
				toolCallID := fmt.Sprintf("call_%d_%s", len(toolCalls), generateRandomString(8))
				tcObj := map[string]interface{}{
					"index": len(toolCalls),
					"id":    toolCallID,
					"type":  "function",
					"function": map[string]interface{}{
						"name":      name,
						"arguments": argsStr,
					},
				}
				if thoughtSig != "" {
					tcObj["extra_content"] = map[string]interface{}{
						"google": map[string]interface{}{
							"thought_signature": thoughtSig,
							"thoughtSignature":  thoughtSig,
						},
					}
				}
				toolCalls = append(toolCalls, tcObj)
			}
		}

		finishReasonStr := interface{}(nil)
		if len(toolCalls) > 0 {
			finishReasonStr = "tool_calls"
		} else if fr, ok := candMap["finishReason"].(string); ok {
			switch fr {
			case "STOP":
				finishReasonStr = "stop"
			case "MAX_TOKENS":
				finishReasonStr = "length"
			case "SAFETY", "RECITATION":
				finishReasonStr = "content_filter"
			case "OTHER":
				finishReasonStr = "stop"
			default:
				finishReasonStr = strings.ToLower(fr)
			}
		}

		delta := map[string]interface{}{}
		if contentText != "" {
			delta["content"] = contentText
		}
		if reasoningText != "" {
			delta["reasoning_content"] = reasoningText
			delta["reasoning"] = reasoningText
		}
		if len(toolCalls) > 0 {
			delta["tool_calls"] = toolCalls
		}

		choices = append(choices, map[string]interface{}{
			"index":         idx,
			"delta":         delta,
			"finish_reason": finishReasonStr,
		})
	}

	if len(choices) == 0 {
		return nil
	}

	openAIChunk := map[string]interface{}{
		"id":      genID,
		"object":  "chat.completion.chunk",
		"created": createdTime,
		"model":   modelName,
		"choices": choices,
	}

	if um, ok := geminiResp["usageMetadata"].(map[string]interface{}); ok {
		usage := map[string]interface{}{
			"prompt_tokens":     0,
			"completion_tokens": 0,
			"total_tokens":      0,
		}
		if pt, ok := um["promptTokenCount"].(float64); ok {
			usage["prompt_tokens"] = int(pt)
		}
		if ct, ok := um["candidatesTokenCount"].(float64); ok {
			usage["completion_tokens"] = int(ct)
		}
		if tt, ok := um["totalTokenCount"].(float64); ok {
			usage["total_tokens"] = int(tt)
		}
		openAIChunk["usage"] = usage
	}

	modifiedPayload, err := json.Marshal(openAIChunk)
	if err != nil {
		return line
	}

	var result bytes.Buffer
	result.Write([]byte("data: "))
	result.Write(modifiedPayload)
	if bytes.HasSuffix(line, []byte("\r\n")) {
		result.Write([]byte("\r\n"))
	} else {
		result.Write([]byte("\n"))
	}
	return result.Bytes()
}

// ModifyRequestBody applies the delete, extra, and model replacement rules.
func ModifyRequestBody(rawBody []byte, route *SelectedRoute) ([]byte, error) {
	if route.ModelProvider.ApiType == "gemini" {
		return TransformRequestToGemini(rawBody, route)
	}

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
	r             io.Reader
	headerPrinted bool
}

func (lr *loggingReader) Read(p []byte) (n int, err error) {
	n, err = lr.r.Read(p)
	if n > 0 {
		if !lr.headerPrinted {
			fmt.Printf("[DEBUG] --- UPSTREAM RESPONSE BODY (STREAMING) ---\n")
			lr.headerPrinted = true
		}
		fmt.Print(string(p[:n]))
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

// FormatJSON pretty-prints the JSON byte array with 2 spaces indent.
func FormatJSON(data []byte) string {
	var temp interface{}
	if err := json.Unmarshal(data, &temp); err != nil {
		return string(data)
	}
	formatted, err := json.MarshalIndent(temp, "", "  ")
	if err != nil {
		return string(data)
	}
	return string(formatted)
}
