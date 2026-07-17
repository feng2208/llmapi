package main

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"time"
)

// GeminiPart represents a single part of content in Gemini API request
type GeminiPart map[string]interface{}

// GeminiContent represents the content of conversation in Gemini API request
type GeminiContent struct {
	Role  string       `json:"role"`
	Parts []GeminiPart `json:"parts"`
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
						if thoughtSig == "" {
							thoughtSig = "skip_thought_signature_validator"
						}
						part["thoughtSignature"] = thoughtSig
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

	return buildSSELine(modifiedPayload, line)
}

func TransformImageRequestToGemini(rawBody []byte, route *SelectedRoute) ([]byte, error) {
	var body map[string]interface{}
	if err := json.Unmarshal(rawBody, &body); err != nil {
		return nil, fmt.Errorf("failed to parse JSON request body: %w", err)
	}

	prompt, _ := body["prompt"].(string)
	if prompt == "" {
		return nil, errors.New("missing prompt parameter")
	}

	geminiReq := map[string]interface{}{
		"contents": []interface{}{
			map[string]interface{}{
				"role": "user",
				"parts": []interface{}{
					map[string]interface{}{
						"text": prompt,
					},
				},
			},
		},
	}

	generationConfig := map[string]interface{}{
		"responseModalities": []string{"IMAGE"},
	}

	if n, ok := body["n"].(float64); ok {
		generationConfig["candidateCount"] = int(n)
	} else if n, ok := body["n"].(int); ok {
		generationConfig["candidateCount"] = n
	}

	size, _ := body["size"].(string)
	aspectRatio := "1:1"
	imageSize := "1K"
	if size != "" {
		parts := strings.Split(size, "x")
		if len(parts) == 2 {
			width, errW := strconv.Atoi(parts[0])
			height, errH := strconv.Atoi(parts[1])
			if errW == nil && errH == nil {
				if width == height {
					aspectRatio = "1:1"
				} else if width > height {
					if width == 1024 && height == 576 {
						aspectRatio = "16:9"
					} else if width == 1024 && height == 768 {
						aspectRatio = "4:3"
					} else if width == 1152 && height == 896 {
						aspectRatio = "4:3"
					} else if width == 1344 && height == 768 {
						aspectRatio = "16:9"
					} else {
						aspectRatio = "16:9"
					}
				} else {
					if width == 576 && height == 1024 {
						aspectRatio = "9:16"
					} else if width == 768 && height == 1024 {
						aspectRatio = "3:4"
					} else if width == 896 && height == 1152 {
						aspectRatio = "3:4"
					} else if width == 768 && height == 1344 {
						aspectRatio = "9:16"
					} else {
						aspectRatio = "9:16"
					}
				}

				maxDim := width
				if height > width {
					maxDim = height
				}
				if maxDim <= 1024 {
					imageSize = "1K"
				} else if maxDim <= 2048 {
					imageSize = "2K"
				} else {
					imageSize = "4K"
				}
			}
		}
	}

	generationConfig["imageConfig"] = map[string]interface{}{
		"aspectRatio": aspectRatio,
		"imageSize":   imageSize,
	}

	geminiReq["generationConfig"] = generationConfig

	// Merge any custom configuration from route.ModelProvider.RequestBody.Extra
	for _, configMap := range route.ModelProvider.RequestBody.Extra {
		deepMerge(geminiReq, configMap)
	}

	return json.Marshal(geminiReq)
}

func ProcessGeminiImageResponse(rawResp []byte, rawRequest []byte) ([]byte, error) {
	var geminiResp map[string]interface{}
	if err := json.Unmarshal(rawResp, &geminiResp); err != nil {
		return nil, fmt.Errorf("failed to parse Gemini response: %w", err)
	}

	var reqJSON map[string]interface{}
	_ = json.Unmarshal(rawRequest, &reqJSON) // ignore error, might be empty

	type OpenAIImageData struct {
		B64JSON string `json:"b64_json"`
	}

	openAIResp := map[string]interface{}{
		"created": time.Now().Unix(),
		"data":    []OpenAIImageData{},
	}

	// Copy requested extra fields if present
	if background, exists := reqJSON["background"]; exists {
		openAIResp["background"] = background
	}
	if outputFormat, exists := reqJSON["output_format"]; exists {
		openAIResp["output_format"] = outputFormat
	}
	if size, exists := reqJSON["size"]; exists {
		openAIResp["size"] = size
	}
	if quality, exists := reqJSON["quality"]; exists {
		openAIResp["quality"] = quality
	}

	// Include usage block
	inputTokens := 0
	outputTokens := 0
	totalTokens := 0

	if um, ok := geminiResp["usageMetadata"].(map[string]interface{}); ok {
		if pt, ok := um["promptTokenCount"].(float64); ok {
			inputTokens = int(pt)
		} else if pt, ok := um["prompt_token_count"].(float64); ok {
			inputTokens = int(pt)
		}

		if ct, ok := um["candidatesTokenCount"].(float64); ok {
			outputTokens = int(ct)
		} else if ct, ok := um["candidates_token_count"].(float64); ok {
			outputTokens = int(ct)
		}

		if tt, ok := um["totalTokenCount"].(float64); ok {
			totalTokens = int(tt)
		} else if tt, ok := um["total_token_count"].(float64); ok {
			totalTokens = int(tt)
		}
	}

	openAIResp["usage"] = map[string]interface{}{
		"input_tokens":  inputTokens,
		"output_tokens": outputTokens,
		"total_tokens":  totalTokens,
	}

	candidates, _ := geminiResp["candidates"].([]interface{})
	var dataItems []OpenAIImageData

	for _, cand := range candidates {
		candMap, ok := cand.(map[string]interface{})
		if !ok {
			continue
		}

		content, _ := candMap["content"].(map[string]interface{})
		parts, _ := content["parts"].([]interface{})

		for _, part := range parts {
			partMap, ok := part.(map[string]interface{})
			if !ok {
				continue
			}

			var inlineData map[string]interface{}
			if id, ok := partMap["inlineData"].(map[string]interface{}); ok {
				inlineData = id
			} else if id, ok := partMap["inline_data"].(map[string]interface{}); ok {
				inlineData = id
			}

			if inlineData != nil {
				base64Data, _ := inlineData["data"].(string)
				if base64Data != "" {
					item := OpenAIImageData{
						B64JSON: base64Data,
					}
					dataItems = append(dataItems, item)
				}
			}
		}
	}

	openAIResp["data"] = dataItems

	return json.Marshal(openAIResp)
}

func TransformImageEditRequestToGemini(r *http.Request, route *SelectedRoute) ([]byte, error) {
	if r.MultipartForm == nil {
		return nil, errors.New("multipart form not parsed")
	}

	prompt := r.FormValue("prompt")
	if prompt == "" {
		return nil, errors.New("missing prompt parameter")
	}

	var parts []interface{}
	parts = append(parts, map[string]interface{}{
		"text": prompt,
	})

	// Get image
	imageKeys := []string{"image", "image[]"}
	for _, key := range imageKeys {
		for _, fh := range r.MultipartForm.File[key] {
			file, err := fh.Open()
			if err != nil {
				return nil, fmt.Errorf("failed to open image: %w", err)
			}

			data, err := io.ReadAll(file)
			file.Close()
			if err != nil {
				return nil, fmt.Errorf("failed to read image: %w", err)
			}

			mimeType := fh.Header.Get("Content-Type")
			if mimeType == "" || mimeType == "application/octet-stream" {
				fnLower := strings.ToLower(fh.Filename)
				if strings.HasSuffix(fnLower, ".png") {
					mimeType = "image/png"
				} else if strings.HasSuffix(fnLower, ".jpg") || strings.HasSuffix(fnLower, ".jpeg") {
					mimeType = "image/jpeg"
				} else if strings.HasSuffix(fnLower, ".webp") {
					mimeType = "image/webp"
				} else {
					mimeType = "image/png"
				}
			}

			parts = append(parts, map[string]interface{}{
				"inline_data": map[string]interface{}{
					"mime_type": mimeType,
					"data":      base64.StdEncoding.EncodeToString(data),
				},
			})
		}
	}

	// Optionally get mask
	maskHeaders := r.MultipartForm.File["mask"]
	if len(maskHeaders) > 0 {
		fh := maskHeaders[0]
		file, err := fh.Open()
		if err != nil {
			return nil, fmt.Errorf("failed to open mask: %w", err)
		}
		defer file.Close()

		data, err := io.ReadAll(file)
		if err != nil {
			return nil, fmt.Errorf("failed to read mask: %w", err)
		}

		mimeType := fh.Header.Get("Content-Type")
		if mimeType == "" || mimeType == "application/octet-stream" {
			fnLower := strings.ToLower(fh.Filename)
			if strings.HasSuffix(fnLower, ".png") {
				mimeType = "image/png"
			} else if strings.HasSuffix(fnLower, ".jpg") || strings.HasSuffix(fnLower, ".jpeg") {
				mimeType = "image/jpeg"
			} else if strings.HasSuffix(fnLower, ".webp") {
				mimeType = "image/webp"
			} else {
				mimeType = "image/png"
			}
		}

		parts = append(parts, map[string]interface{}{
			"inline_data": map[string]interface{}{
				"mime_type": mimeType,
				"data":      base64.StdEncoding.EncodeToString(data),
			},
		})
	}

	geminiReq := map[string]interface{}{
		"contents": []interface{}{
			map[string]interface{}{
				"role":  "user",
				"parts": parts,
			},
		},
	}

	generationConfig := map[string]interface{}{
		"responseModalities": []string{"IMAGE"},
	}

	nStr := r.FormValue("n")
	if nStr != "" {
		if n, err := strconv.Atoi(nStr); err == nil {
			generationConfig["candidateCount"] = n
		}
	}

	size := r.FormValue("size")
	aspectRatio := "1:1"
	imageSize := "1K"
	if size != "" {
		parts := strings.Split(size, "x")
		if len(parts) == 2 {
			width, errW := strconv.Atoi(parts[0])
			height, errH := strconv.Atoi(parts[1])
			if errW == nil && errH == nil {
				if width == height {
					aspectRatio = "1:1"
				} else if width > height {
					if width == 1024 && height == 576 {
						aspectRatio = "16:9"
					} else if width == 1024 && height == 768 {
						aspectRatio = "4:3"
					} else if width == 1152 && height == 896 {
						aspectRatio = "4:3"
					} else if width == 1344 && height == 768 {
						aspectRatio = "16:9"
					} else {
						aspectRatio = "16:9"
					}
				} else {
					if width == 576 && height == 1024 {
						aspectRatio = "9:16"
					} else if width == 768 && height == 1024 {
						aspectRatio = "3:4"
					} else if width == 896 && height == 1152 {
						aspectRatio = "3:4"
					} else if width == 768 && height == 1344 {
						aspectRatio = "9:16"
					} else {
						aspectRatio = "9:16"
					}
				}

				maxDim := width
				if height > width {
					maxDim = height
				}
				if maxDim <= 1024 {
					imageSize = "1K"
				} else if maxDim <= 2048 {
					imageSize = "2K"
				} else {
					imageSize = "4K"
				}
			}
		}
	}

	generationConfig["imageConfig"] = map[string]interface{}{
		"aspectRatio": aspectRatio,
		"imageSize":   imageSize,
	}

	geminiReq["generationConfig"] = generationConfig

	// Merge any custom configuration from route.ModelProvider.RequestBody.Extra
	for _, configMap := range route.ModelProvider.RequestBody.Extra {
		deepMerge(geminiReq, configMap)
	}

	return json.Marshal(geminiReq)
}

func CreateWavHeader(dataLen int) []byte {
	header := make([]byte, 44)
	// RIFF header
	copy(header[0:4], "RIFF")
	// File size - 8
	fileSize := uint32(dataLen + 36)
	header[4] = byte(fileSize)
	header[5] = byte(fileSize >> 8)
	header[6] = byte(fileSize >> 16)
	header[7] = byte(fileSize >> 24)
	// WAVE
	copy(header[8:12], "WAVE")
	// fmt chunk
	copy(header[12:16], "fmt ")
	// Chunk size (16 for PCM)
	header[16] = 16
	header[17] = 0
	header[18] = 0
	header[19] = 0
	// Audio format (1 for PCM)
	header[20] = 1
	header[21] = 0
	// Number of channels (1)
	header[22] = 1
	header[23] = 0
	// Sample rate (24000)
	sampleRate := uint32(24000)
	header[24] = byte(sampleRate)
	header[25] = byte(sampleRate >> 8)
	header[26] = byte(sampleRate >> 16)
	header[27] = byte(sampleRate >> 24)
	// Byte rate (sampleRate * numChannels * bitsPerSample/8) = 24000 * 1 * 2 = 48000
	byteRate := uint32(48000)
	header[28] = byte(byteRate)
	header[29] = byte(byteRate >> 8)
	header[30] = byte(byteRate >> 16)
	header[31] = byte(byteRate >> 24)
	// Block align (numChannels * bitsPerSample/8) = 2
	header[32] = 2
	header[33] = 0
	// Bits per sample (16)
	header[34] = 16
	header[35] = 0
	// data chunk
	copy(header[36:40], "data")
	// Chunk size
	subchunk2Size := uint32(dataLen)
	header[40] = byte(subchunk2Size)
	header[41] = byte(subchunk2Size >> 8)
	header[42] = byte(subchunk2Size >> 16)
	header[43] = byte(subchunk2Size >> 24)
	return header
}

func TransformAudioSpeechRequestToGemini(rawBody []byte, route *SelectedRoute) ([]byte, error) {
	var body map[string]interface{}
	if err := json.Unmarshal(rawBody, &body); err != nil {
		return nil, fmt.Errorf("failed to parse JSON request body: %w", err)
	}

	inputText, _ := body["input"].(string)
	if inputText == "" {
		return nil, errors.New("missing input parameter")
	}

	voiceName, _ := body["voice"].(string)
	if voiceName == "" {
		voiceName = "Kore" // Default
	}

	// Map OpenAI voice names to Gemini prebuilt voice names
	voiceMapping := map[string]string{
		"alloy":   "Puck",
		"echo":    "Charon",
		"fable":   "Kore",
		"nova":    "Kore",
		"onyx":    "Puck",
		"shimmer": "Puck",
	}

	mappedVoice := voiceName
	if val, ok := voiceMapping[strings.ToLower(voiceName)]; ok {
		mappedVoice = val
	}

	geminiReq := map[string]interface{}{
		"contents": []interface{}{
			map[string]interface{}{
				"role": "user",
				"parts": []interface{}{
					map[string]interface{}{
						"text": inputText,
					},
				},
			},
		},
	}

	generationConfig := map[string]interface{}{
		"responseModalities": []string{"AUDIO"},
		"speechConfig": map[string]interface{}{
			"voiceConfig": map[string]interface{}{
				"prebuiltVoiceConfig": map[string]interface{}{
					"voiceName": mappedVoice,
				},
			},
		},
	}

	if speedVal, ok := body["speed"].(float64); ok && speedVal != 1.0 {
		if speedVal < 0.6 {
			inputText = "[very slow] " + inputText
		} else if speedVal < 0.85 {
			inputText = "[slow] " + inputText
		} else if speedVal > 1.8 {
			inputText = "[very fast] " + inputText
		} else if speedVal > 1.15 {
			inputText = "[fast] " + inputText
		}
		geminiReq["contents"].([]interface{})[0].(map[string]interface{})["parts"].([]interface{})[0].(map[string]interface{})["text"] = inputText
	}

	geminiReq["generationConfig"] = generationConfig

	// Merge any custom configuration from route.ModelProvider.RequestBody.Extra
	for _, configMap := range route.ModelProvider.RequestBody.Extra {
		deepMerge(geminiReq, configMap)
	}

	return json.Marshal(geminiReq)
}

func ProcessGeminiAudioResponse(rawResp []byte, rawRequest []byte) ([]byte, error) {
	var geminiResp map[string]interface{}
	if err := json.Unmarshal(rawResp, &geminiResp); err != nil {
		return nil, fmt.Errorf("failed to parse Gemini response: %w", err)
	}

	candidates, _ := geminiResp["candidates"].([]interface{})
	if len(candidates) == 0 {
		return nil, errors.New("no candidates in Gemini response")
	}

	candMap, ok := candidates[0].(map[string]interface{})
	if !ok {
		return nil, errors.New("invalid candidate format")
	}

	content, _ := candMap["content"].(map[string]interface{})
	parts, _ := content["parts"].([]interface{})
	if len(parts) == 0 {
		return nil, errors.New("no parts in Gemini response content")
	}

	partMap, ok := parts[0].(map[string]interface{})
	if !ok {
		return nil, errors.New("invalid part format")
	}

	var inlineData map[string]interface{}
	if id, ok := partMap["inlineData"].(map[string]interface{}); ok {
		inlineData = id
	} else if id, ok := partMap["inline_data"].(map[string]interface{}); ok {
		inlineData = id
	}

	if inlineData == nil {
		return nil, errors.New("missing inlineData in Gemini response")
	}

	base64Data, _ := inlineData["data"].(string)
	if base64Data == "" {
		return nil, errors.New("missing audio data in Gemini response")
	}

	pcmData, err := base64.StdEncoding.DecodeString(base64Data)
	if err != nil {
		return nil, fmt.Errorf("failed to decode base64 audio data: %w", err)
	}

	var reqJSON map[string]interface{}
	_ = json.Unmarshal(rawRequest, &reqJSON)
	responseFormat, _ := reqJSON["response_format"].(string)

	if strings.ToLower(responseFormat) == "pcm" {
		return pcmData, nil
	}

	// Wrap PCM in a WAV header
	wavHeader := CreateWavHeader(len(pcmData))
	return append(wavHeader, pcmData...), nil
}
