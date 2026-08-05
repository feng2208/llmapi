package plugins

import (
	"bytes"
	"context"
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

// GeminiPlugin handles upstreams with api_type: gemini
type GeminiPlugin struct {
	BasePlugin
}

func (g *GeminiPlugin) Name() string {
	return "gemini"
}

func (g *GeminiPlugin) BuildTargetURL(endpoint EndpointType, req *http.Request, ctx *Context) string {
	targetURL := ctx.UpstreamURL
	u := targetURL
	if !strings.HasSuffix(u, "/") {
		u += "/"
	}
	u += ctx.ModelName
	if endpoint == EndpointImageGeneration || endpoint == EndpointImageEdit || endpoint == EndpointAudioSpeech {
		u += ":generateContent"
	} else {
		var rawBody []byte
		if ctx != nil {
			rawBody = ctx.RawRequestBody
		}
		isStream := IsStreamRequested(req, rawBody, nil)
		if ctx != nil && ctx.IsStream {
			isStream = true
		}
		if isStream {
			u += ":streamGenerateContent"
			if !strings.Contains(u, "?") {
				u += "?alt=sse"
			} else {
				u += "&alt=sse"
			}
		} else {
			u += ":generateContent"
		}
	}
	return u
}

func (g *GeminiPlugin) ModifyHeaders(req *http.Request, ctx *Context) {
	// Standard header deletions & extra maps
	g.BasePlugin.ModifyHeaders(req, ctx)

	// Custom header handling for Gemini upstream
	if ctx.AuthKey == "sk-dummy" {
		req.Header.Del("Authorization")
		req.Header.Del("x-goog-api-key")
	} else if ctx.AuthKey != "" {
		req.Header.Del("Authorization")
		req.Header.Set("x-goog-api-key", ctx.AuthKey)
	}
}

func (g *GeminiPlugin) ModifyResponseHeaders(resp *http.Response, clientHeader http.Header, ctx *Context) {
	for k, vv := range resp.Header {
		if k == "Content-Length" || k == "Content-Type" || strings.HasPrefix(k, "X-Goog-") || strings.HasPrefix(k, "x-goog-") {
			continue
		}
		for _, v := range vv {
			clientHeader.Add(k, v)
		}
	}
}

func (g *GeminiPlugin) TransformRequest(endpoint EndpointType, rawBody []byte, req *http.Request, ctx *Context) ([]byte, string, error) {
	switch endpoint {
	case EndpointImageGeneration:
		modified, err := TransformImageRequestToGemini(rawBody, ctx)
		if err != nil {
			return nil, "", err
		}
		return modified, "application/json; charset=utf-8", nil

	case EndpointImageEdit:
		modified, err := TransformImageEditRequestToGemini(req, ctx)
		if err != nil {
			return nil, "", err
		}
		return modified, "application/json; charset=utf-8", nil

	case EndpointAudioSpeech:
		modified, err := TransformAudioSpeechRequestToGemini(rawBody, ctx)
		if err != nil {
			return nil, "", err
		}
		return modified, "application/json; charset=utf-8", nil

	case EndpointAudioTranscription:
		modified, err := TransformAudioTranscriptionRequestToGemini(req, ctx)
		if err != nil {
			return nil, "", err
		}
		return modified, "application/json; charset=utf-8", nil

	default: // EndpointChat
		modified, err := TransformRequestToGemini(rawBody, req, ctx)
		if err != nil {
			return nil, "", err
		}
		return modified, "application/json; charset=utf-8", nil
	}
}

func (g *GeminiPlugin) TransformResponse(endpoint EndpointType, resp *http.Response, body []byte, ctx *Context) ([]byte, string, error) {
	switch endpoint {
	case EndpointImageGeneration, EndpointImageEdit:
		modified, err := ProcessGeminiImageResponse(body, ctx.RawRequestBody)
		if err != nil {
			return nil, "", err
		}
		return modified, "application/json; charset=utf-8", nil

	case EndpointAudioSpeech:
		modified, err := ProcessGeminiAudioResponse(body, ctx.RawRequestBody)
		if err != nil {
			return nil, "", err
		}
		var reqJSON map[string]interface{}
		_ = json.Unmarshal(ctx.RawRequestBody, &reqJSON)
		responseFormat, _ := reqJSON["response_format"].(string)
		contentType := "audio/wav"
		if strings.ToLower(responseFormat) == "pcm" {
			contentType = "audio/pcm"
		}
		return modified, contentType, nil

	case EndpointAudioTranscription:
		hasTimestamps := false
		responseFormat := "json"
		language := ""
		if ctx.Request != nil {
			responseFormat = ctx.Request.FormValue("response_format")
			language = ctx.Request.FormValue("language")
			rfLower := strings.ToLower(responseFormat)
			if rfLower == "verbose_json" || rfLower == "srt" || rfLower == "vtt" {
				hasTimestamps = true
			}
			if ctx.Request.MultipartForm != nil {
				for _, val := range ctx.Request.MultipartForm.Value["timestamp_granularities[]"] {
					if val == "segment" {
						hasTimestamps = true
					}
				}
				for _, val := range ctx.Request.MultipartForm.Value["timestamp_granularities"] {
					if val == "segment" {
						hasTimestamps = true
					}
				}
			}
		}
		modified, err := ProcessGeminiTranscriptionResponse(body, hasTimestamps, language, responseFormat)
		if err != nil {
			return nil, "", err
		}
		rfLower := strings.ToLower(responseFormat)
		contentType := "application/json; charset=utf-8"
		if rfLower == "text" || rfLower == "srt" || rfLower == "vtt" {
			contentType = "text/plain; charset=utf-8"
		}
		return modified, contentType, nil

	default: // EndpointChat
		modified, err := ProcessGeminiJSONResponse(body, ctx.ModelName)
		if err != nil {
			return nil, "", err
		}
		return modified, "application/json; charset=utf-8", nil
	}
}

func (g *GeminiPlugin) TransformStreamChunk(endpoint EndpointType, chunk []byte, ctx *Context) ([]byte, error) {
	if endpoint == EndpointAudioTranscription {
		delta, _, err := ProcessGeminiSTTStreamLine(chunk)
		if err != nil {
			return nil, err
		}
		if delta != "" {
			deltaChunk := map[string]interface{}{
				"type":     "transcript.text.delta",
				"delta":    delta,
				"logprobs": []interface{}{},
			}
			b, _ := json.Marshal(deltaChunk)
			return []byte(fmt.Sprintf("data: %s\n\n", string(b))), nil
		}
		return nil, nil
	}
	genID := fmt.Sprintf("chatcmpl-%s", GenerateRandomString(12))
	return ProcessGeminiSSELine(chunk, genID, time.Now().Unix(), ctx.ModelName), nil
}

// ----------------------------------------------------------------------------
// Gemini Implementation Details
// ----------------------------------------------------------------------------

type GeminiPart map[string]interface{}

type GeminiContent struct {
	Role  string       `json:"role"`
	Parts []GeminiPart `json:"parts"`
}

func parseImageUrlPart(urlStr string) (map[string]interface{}, error) {
	if strings.HasPrefix(urlStr, "data:") {
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
		client := &http.Client{Timeout: 10 * time.Second}
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
			mimeType = "image/jpeg"
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
		if k == "$schema" || k == "$id" || k == "$vocabulary" || k == "$anchor" || k == "additionalProperties" || k == "exclusiveMinimum" {
			continue
		}

		if k == "type" {
			if typeStr, ok := v.(string); ok {
				cleaned[k] = strings.ToUpper(typeStr)
				continue
			}
		}

		if k == "properties" || k == "$defs" || k == "definitions" {
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
			if itemMap, ok := v.(map[string]interface{}); ok {
				cleaned[k] = cleanGeminiSchema(itemMap)
				continue
			} else if itemArr, ok := v.([]interface{}); ok {
				cleanedArr := make([]interface{}, 0, len(itemArr))
				for _, elem := range itemArr {
					cleanedArr = append(cleanedArr, cleanGeminiSchema(elem))
				}
				cleaned[k] = cleanedArr
				continue
			}
		}

		if k == "allOf" || k == "anyOf" || k == "oneOf" || k == "prefixItems" {
			if schemaArr, ok := v.([]interface{}); ok {
				cleanedArr := make([]interface{}, 0, len(schemaArr))
				for _, elem := range schemaArr {
					cleanedArr = append(cleanedArr, cleanGeminiSchema(elem))
				}
				cleaned[k] = cleanedArr
				continue
			}
		}

		cleaned[k] = v
	}
	return cleaned
}

func DetectMimeTypeFromContent(data []byte) string {
	if len(data) < 4 {
		return "application/octet-stream"
	}
	if bytes.HasPrefix(data, []byte("%PDF")) {
		return "application/pdf"
	}
	if bytes.HasPrefix(data, []byte("\x89PNG\r\n\x1a\n")) {
		return "image/png"
	}
	if bytes.HasPrefix(data, []byte("\xff\xd8\xff")) {
		return "image/jpeg"
	}
	if bytes.HasPrefix(data, []byte("GIF8")) {
		return "image/gif"
	}
	if bytes.HasPrefix(data, []byte("BM")) {
		return "image/bmp"
	}
	if bytes.HasPrefix(data, []byte("fLaC")) {
		return "audio/flac"
	}
	if bytes.HasPrefix(data, []byte("OggS")) {
		return "audio/ogg"
	}
	if bytes.HasPrefix(data, []byte("\x1a\x45\xdf\xa3")) {
		return "video/webm"
	}
	if bytes.HasPrefix(data, []byte("FLV\x01")) {
		return "video/x-flv"
	}
	if bytes.HasPrefix(data, []byte("ID3")) {
		return "audio/mp3"
	}
	if len(data) >= 2 && data[0] == 0xff && (data[1]&0xe0) == 0xe0 {
		return "audio/mp3"
	}
	if bytes.HasPrefix(data, []byte("\x00\x00\x01\xba")) || bytes.HasPrefix(data, []byte("\x00\x00\x01\xb3")) {
		return "video/mpeg"
	}
	if bytes.HasPrefix(data, []byte("\x30\x26\xb2\x75\x8e\x66\xcf\x11")) {
		return "video/wmv"
	}
	if bytes.HasPrefix(data, []byte("RIFF")) && len(data) >= 12 {
		container := string(data[8:12])
		switch container {
		case "WAVE":
			return "audio/wav"
		case "WEBP":
			return "image/webp"
		case "AVI ":
			return "video/avi"
		}
	}
	if bytes.HasPrefix(data, []byte("FORM")) && len(data) >= 12 {
		container := string(data[8:12])
		if container == "AIFF" || container == "AIFC" {
			return "audio/aiff"
		}
	}
	if len(data) >= 12 && string(data[4:8]) == "ftyp" {
		box := string(data[8:12])
		if strings.HasPrefix(box, "mp4") || strings.HasPrefix(box, "isom") || strings.HasPrefix(box, "avc1") {
			return "video/mp4"
		}
		if strings.HasPrefix(box, "3gp") {
			return "video/3gpp"
		}
	}
	detected := http.DetectContentType(data)
	if detected != "application/octet-stream" {
		return detected
	}
	return "application/octet-stream"
}

func detectMimeType(filename string) string {
	idx := strings.LastIndex(filename, ".")
	if idx == -1 || idx == len(filename)-1 {
		return "application/octet-stream"
	}
	ext := strings.ToLower(filename[idx+1:])
	switch ext {
	case "pdf":
		return "application/pdf"
	case "txt":
		return "text/plain"
	case "csv":
		return "text/csv"
	case "html", "htm":
		return "text/html"
	case "json":
		return "application/json"
	case "mp3":
		return "audio/mp3"
	case "wav":
		return "audio/wav"
	case "m4a":
		return "audio/m4a"
	case "mp4":
		return "video/mp4"
	case "mov":
		return "video/quicktime"
	case "png":
		return "image/png"
	case "jpg", "jpeg":
		return "image/jpeg"
	default:
		return "application/octet-stream"
	}
}

func TransformRequestToGemini(rawBody []byte, r *http.Request, ctx *Context) ([]byte, error) {
	var body map[string]interface{}
	if err := json.Unmarshal(rawBody, &body); err != nil {
		return nil, fmt.Errorf("failed to parse JSON request body: %w", err)
	}

	geminiReq := make(map[string]interface{})
	openAIMessages, _ := body["messages"].([]interface{})

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

		if role != "tool" {
			if txt, ok := content.(string); ok && txt != "" {
				parts = append(parts, map[string]interface{}{"text": txt})
			} else if arr, ok := content.([]interface{}); ok {
				var fileParts []GeminiPart
				var textParts []GeminiPart
				for _, item := range arr {
					if itemMap, ok := item.(map[string]interface{}); ok {
						itemType, _ := itemMap["type"].(string)
						if itemType == "text" {
							if txt, ok := itemMap["text"].(string); ok && txt != "" {
								textParts = append(textParts, map[string]interface{}{"text": txt})
							}
						} else if itemType == "image_url" {
							if imgUrlMap, ok := itemMap["image_url"].(map[string]interface{}); ok {
								if urlStr, ok := imgUrlMap["url"].(string); ok && urlStr != "" {
									part, err := parseImageUrlPart(urlStr)
									if err != nil {
										return nil, fmt.Errorf("failed to process image: %w", err)
									}
									fileParts = append(fileParts, part)
								}
							}
						} else if itemType == "input_audio" {
							if inputAudioMap, ok := itemMap["input_audio"].(map[string]interface{}); ok {
								base64Data, _ := inputAudioMap["data"].(string)
								format, _ := inputAudioMap["format"].(string)
								if base64Data != "" {
									dataBytes, err := base64.StdEncoding.DecodeString(base64Data)
									if err == nil {
										mimeType := DetectMimeTypeFromContent(dataBytes)
										if mimeType == "application/octet-stream" {
											if format != "" {
												mimeType = "audio/" + format
											} else {
												mimeType = "audio/wav"
											}
										}
										if len(dataBytes) <= 20*1024*1024 {
											fileParts = append(fileParts, map[string]interface{}{
												"inline_data": map[string]interface{}{
													"mime_type": mimeType,
													"data":      base64Data,
												},
											})
										} else {
											reqCtx := context.Background()
											if r != nil {
												reqCtx = r.Context()
											}
											_, fileURI, err := uploadFileToGemini(reqCtx, ctx.AuthKey, "audio."+format, bytes.NewReader(dataBytes), int64(len(dataBytes)), mimeType, ctx.HTTPClient)
											if err != nil {
												return nil, fmt.Errorf("failed to upload audio to Files API: %w", err)
											}
											fileParts = append(fileParts, map[string]interface{}{
												"file_data": map[string]interface{}{
													"mime_type": mimeType,
													"file_uri":  fileURI,
												},
											})
										}
									}
								}
							}
						} else if itemType == "file" {
							if fileMap, ok := itemMap["file"].(map[string]interface{}); ok {
								fileData, _ := fileMap["file_data"].(string)
								filename, _ := fileMap["filename"].(string)
								if fileData != "" {
									base64Data := ""
									uriMimeType := ""

									if strings.HasPrefix(fileData, "data:") {
										partsData := strings.SplitN(fileData, ",", 2)
										if len(partsData) == 2 {
											header := partsData[0]
											base64Data = partsData[1]
											if strings.Contains(header, ";") {
												mimePart := strings.TrimPrefix(header, "data:")
												semiIdx := strings.Index(mimePart, ";")
												if semiIdx != -1 {
													uriMimeType = mimePart[:semiIdx]
												}
											}
										}
									} else {
										base64Data = fileData
									}

									if base64Data != "" {
										dataBytes, err := base64.StdEncoding.DecodeString(base64Data)
										if err == nil {
											mimeType := DetectMimeTypeFromContent(dataBytes)
											if mimeType == "application/octet-stream" {
												if uriMimeType != "" {
													mimeType = uriMimeType
												} else if filename != "" {
													mimeType = detectMimeType(filename)
												} else {
													mimeType = "application/pdf"
												}
											}

											displayName := filename
											if displayName == "" {
												displayName = "file.pdf"
											}
											if len(dataBytes) <= 20*1024*1024 {
												fileParts = append(fileParts, map[string]interface{}{
													"inline_data": map[string]interface{}{
														"mime_type": mimeType,
														"data":      base64Data,
													},
												})
											} else {
												reqCtx := context.Background()
												if r != nil {
													reqCtx = r.Context()
												}
												_, fileURI, err := uploadFileToGemini(reqCtx, ctx.AuthKey, displayName, bytes.NewReader(dataBytes), int64(len(dataBytes)), mimeType, ctx.HTTPClient)
												if err != nil {
													return nil, fmt.Errorf("failed to upload file to Files API: %w", err)
												}
												fileParts = append(fileParts, map[string]interface{}{
													"file_data": map[string]interface{}{
														"mime_type": mimeType,
														"file_uri":  fileURI,
													},
												})
											}
										}
									}
								}
							}
						}
					}
				}
				parts = append(parts, fileParts...)
				parts = append(parts, textParts...)
			}
		}

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

	toolConfig := make(map[string]interface{})
	if toolChoice, ok := body["tool_choice"]; ok {
		if tcStr, ok := toolChoice.(string); ok {
			if tcStr == "none" {
				toolConfig["functionCallingConfig"] = map[string]interface{}{
					"mode": "NONE",
				}
			} else if tcStr == "required" {
				toolConfig["functionCallingConfig"] = map[string]interface{}{
					"mode": "ANY",
				}
			} else if tcStr == "auto" {
				toolConfig["functionCallingConfig"] = map[string]interface{}{
					"mode": "AUTO",
				}
			}
		} else if tcMap, ok := toolChoice.(map[string]interface{}); ok {
			if fnMap, ok := tcMap["function"].(map[string]interface{}); ok {
				if name, ok := fnMap["name"].(string); ok && name != "" {
					toolConfig["functionCallingConfig"] = map[string]interface{}{
						"mode":                 "ANY",
						"allowedFunctionNames": []string{name},
					}
				}
			}
		}
	}

	toolConfig["includeServerSideToolInvocations"] = true
	geminiReq["toolConfig"] = toolConfig

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

	applyDeleteAndExtra(geminiReq, ctx.RequestBody)

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
				toolCallID := fmt.Sprintf("call_%d_%s", len(toolCalls), GenerateRandomString(8))
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
		"id":      fmt.Sprintf("chatcmpl-%s", GenerateRandomString(12)),
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
				toolCallID := fmt.Sprintf("call_%d_%s", len(toolCalls), GenerateRandomString(8))
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

	return BuildSSELine(modifiedPayload, line)
}

func TransformImageRequestToGemini(rawBody []byte, ctx *Context) ([]byte, error) {
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

	generationConfig := map[string]interface{}{}

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

	applyDeleteAndExtra(geminiReq, ctx.RequestBody)

	return json.Marshal(geminiReq)
}

func ProcessGeminiImageResponse(rawResp []byte, rawRequest []byte) ([]byte, error) {
	var geminiResp map[string]interface{}
	if err := json.Unmarshal(rawResp, &geminiResp); err != nil {
		return nil, fmt.Errorf("failed to parse Gemini response: %w", err)
	}

	var reqJSON map[string]interface{}
	if len(rawRequest) > 0 {
		_ = json.Unmarshal(rawRequest, &reqJSON)
	}

	type OpenAIImageData struct {
		B64JSON string `json:"b64_json"`
	}

	openAIResp := map[string]interface{}{
		"created": time.Now().Unix(),
		"data":    []OpenAIImageData{},
	}

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

func TransformImageEditRequestToGemini(r *http.Request, ctx *Context) ([]byte, error) {
	if r.MultipartForm == nil {
		return nil, errors.New("multipart form not parsed")
	}

	prompt := r.FormValue("prompt")
	if prompt == "" {
		return nil, errors.New("missing prompt parameter")
	}

	var parts []interface{}

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

	parts = append(parts, map[string]interface{}{
		"text": prompt,
	})

	geminiReq := map[string]interface{}{
		"contents": []interface{}{
			map[string]interface{}{
				"role":  "user",
				"parts": parts,
			},
		},
	}

	generationConfig := map[string]interface{}{}

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

	applyDeleteAndExtra(geminiReq, ctx.RequestBody)

	return json.Marshal(geminiReq)
}

func CreateWavHeader(dataLen int) []byte {
	header := make([]byte, 44)
	copy(header[0:4], "RIFF")
	fileSize := uint32(dataLen + 36)
	header[4] = byte(fileSize)
	header[5] = byte(fileSize >> 8)
	header[6] = byte(fileSize >> 16)
	header[7] = byte(fileSize >> 24)
	copy(header[8:12], "WAVE")
	copy(header[12:16], "fmt ")
	header[16] = 16
	header[17] = 0
	header[18] = 0
	header[19] = 0
	header[20] = 1
	header[21] = 0
	header[22] = 1
	header[23] = 0
	sampleRate := uint32(24000)
	header[24] = byte(sampleRate)
	header[25] = byte(sampleRate >> 8)
	header[26] = byte(sampleRate >> 16)
	header[27] = byte(sampleRate >> 24)
	byteRate := uint32(48000)
	header[28] = byte(byteRate)
	header[29] = byte(byteRate >> 8)
	header[30] = byte(byteRate >> 16)
	header[31] = byte(byteRate >> 24)
	header[32] = 2
	header[33] = 0
	header[34] = 16
	header[35] = 0
	copy(header[36:40], "data")
	subchunk2Size := uint32(dataLen)
	header[40] = byte(subchunk2Size)
	header[41] = byte(subchunk2Size >> 8)
	header[42] = byte(subchunk2Size >> 16)
	header[43] = byte(subchunk2Size >> 24)
	return header
}

func TransformAudioSpeechRequestToGemini(rawBody []byte, ctx *Context) ([]byte, error) {
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
		voiceName = "Kore"
	}

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

	instructions, _ := body["instructions"].(string)
	if instructions != "" {
		geminiReq["systemInstruction"] = map[string]interface{}{
			"parts": []interface{}{
				map[string]interface{}{
					"text": instructions,
				},
			},
		}
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

	applyDeleteAndExtra(geminiReq, ctx.RequestBody)

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
	if len(rawRequest) > 0 {
		_ = json.Unmarshal(rawRequest, &reqJSON)
	}
	responseFormat, _ := reqJSON["response_format"].(string)

	if strings.ToLower(responseFormat) == "pcm" {
		return pcmData, nil
	}

	wavHeader := CreateWavHeader(len(pcmData))
	return append(wavHeader, pcmData...), nil
}

func uploadFileToGemini(ctx context.Context, apiKey string, filename string, fileReader io.Reader, fileSize int64, mimeType string, httpClient *http.Client) (string, string, error) {
	client := httpClient
	if client == nil {
		client = &http.Client{Timeout: 300 * time.Second}
	}

	initURL := fmt.Sprintf("https://generativelanguage.googleapis.com/upload/v1beta/files?key=%s", apiKey)
	initBodyObj := map[string]interface{}{
		"file": map[string]interface{}{
			"display_name": filename,
		},
	}
	initBodyBytes, err := json.Marshal(initBodyObj)
	if err != nil {
		return "", "", fmt.Errorf("failed to marshal init request body: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", initURL, bytes.NewReader(initBodyBytes))
	if err != nil {
		return "", "", fmt.Errorf("failed to create init request: %w", err)
	}

	req.Header.Set("X-Goog-Upload-Protocol", "resumable")
	req.Header.Set("X-Goog-Upload-Command", "start")
	req.Header.Set("X-Goog-Upload-Header-Content-Length", fmt.Sprintf("%d", fileSize))
	req.Header.Set("X-Goog-Upload-Header-Content-Type", mimeType)
	req.Header.Set("Content-Type", "application/json")

	resp, err := client.Do(req)
	if err != nil {
		return "", "", fmt.Errorf("initiate upload failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBytes, _ := io.ReadAll(resp.Body)
		return "", "", fmt.Errorf("initiate upload failed with status %d: %s", resp.StatusCode, string(respBytes))
	}

	uploadURL := resp.Header.Get("X-Goog-Upload-URL")
	if uploadURL == "" {
		return "", "", fmt.Errorf("missing X-Goog-Upload-URL header in initiate response")
	}

	uploadReq, err := http.NewRequestWithContext(ctx, "POST", uploadURL, fileReader)
	if err != nil {
		return "", "", fmt.Errorf("failed to create upload request: %w", err)
	}

	uploadReq.Header.Set("Content-Length", fmt.Sprintf("%d", fileSize))
	uploadReq.Header.Set("X-Goog-Upload-Offset", "0")
	uploadReq.Header.Set("X-Goog-Upload-Command", "upload, finalize")

	uploadResp, err := client.Do(uploadReq)
	if err != nil {
		return "", "", fmt.Errorf("upload bytes failed: %w", err)
	}
	defer uploadResp.Body.Close()

	if uploadResp.StatusCode != http.StatusOK && uploadResp.StatusCode != http.StatusCreated {
		respBytes, _ := io.ReadAll(uploadResp.Body)
		return "", "", fmt.Errorf("upload bytes failed with status %d: %s", uploadResp.StatusCode, string(respBytes))
	}

	var uploadResult struct {
		File struct {
			Name  string `json:"name"`
			State string `json:"state"`
			URI   string `json:"uri"`
		} `json:"file"`
	}

	if err := json.NewDecoder(uploadResp.Body).Decode(&uploadResult); err != nil {
		return "", "", fmt.Errorf("failed to decode upload response: %w", err)
	}

	fileName := uploadResult.File.Name
	fileURI := uploadResult.File.URI
	state := uploadResult.File.State

	if state == "PROCESSING" {
		pollURL := fmt.Sprintf("https://generativelanguage.googleapis.com/v1beta/%s?key=%s", fileName, apiKey)
		for {
			select {
			case <-ctx.Done():
				return "", "", ctx.Err()
			default:
			}

			time.Sleep(1 * time.Second)

			pollReq, err := http.NewRequestWithContext(ctx, "GET", pollURL, nil)
			if err != nil {
				return "", "", fmt.Errorf("failed to create poll request: %w", err)
			}

			pollResp, err := client.Do(pollReq)
			if err != nil {
				return "", "", fmt.Errorf("polling failed: %w", err)
			}

			var pollResult struct {
				State string `json:"state"`
			}
			err = json.NewDecoder(pollResp.Body).Decode(&pollResult)
			pollResp.Body.Close()
			if err != nil {
				return "", "", fmt.Errorf("failed to decode poll response: %w", err)
			}

			if pollResult.State == "ACTIVE" {
				break
			}
			if pollResult.State == "FAILED" {
				return "", "", fmt.Errorf("file processing failed on Gemini side")
			}
		}
	}

	return fileName, fileURI, nil
}

func TransformAudioTranscriptionRequestToGemini(r *http.Request, ctx *Context) ([]byte, error) {
	if r.MultipartForm == nil {
		return nil, errors.New("multipart form not parsed")
	}

	fileHeaders := r.MultipartForm.File["file"]
	if len(fileHeaders) == 0 {
		return nil, errors.New("missing file in multipart request")
	}
	fh := fileHeaders[0]

	file, err := fh.Open()
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %w", err)
	}
	defer file.Close()

	fileSize := fh.Size
	mimeType := fh.Header.Get("Content-Type")
	if mimeType == "" || mimeType == "application/octet-stream" {
		fnLower := strings.ToLower(fh.Filename)
		if strings.HasSuffix(fnLower, ".mp3") {
			mimeType = "audio/mp3"
		} else if strings.HasSuffix(fnLower, ".wav") {
			mimeType = "audio/wav"
		} else if strings.HasSuffix(fnLower, ".m4a") {
			mimeType = "audio/m4a"
		} else if strings.HasSuffix(fnLower, ".aac") {
			mimeType = "audio/aac"
		} else if strings.HasSuffix(fnLower, ".ogg") {
			mimeType = "audio/ogg"
		} else if strings.HasSuffix(fnLower, ".flac") {
			mimeType = "audio/flac"
		} else {
			mimeType = "audio/mp3"
		}
	}

	var audioPart map[string]interface{}
	if fileSize <= 20*1024*1024 {
		data, err := io.ReadAll(file)
		if err != nil {
			return nil, fmt.Errorf("failed to read file: %w", err)
		}
		audioPart = map[string]interface{}{
			"inline_data": map[string]interface{}{
				"mime_type": mimeType,
				"data":      base64.StdEncoding.EncodeToString(data),
			},
		}
	} else {
		_, fileURI, err := uploadFileToGemini(r.Context(), ctx.AuthKey, fh.Filename, file, fileSize, mimeType, ctx.HTTPClient)
		if err != nil {
			return nil, fmt.Errorf("failed to upload large file to Gemini Files API: %w", err)
		}
		audioPart = map[string]interface{}{
			"file_data": map[string]interface{}{
				"mime_type": mimeType,
				"file_uri":  fileURI,
			},
		}
	}

	responseFormat := r.FormValue("response_format")
	var hasTimestamps bool
	if strings.ToLower(responseFormat) == "verbose_json" || strings.ToLower(responseFormat) == "srt" || strings.ToLower(responseFormat) == "vtt" {
		hasTimestamps = true
	}
	for _, val := range r.MultipartForm.Value["timestamp_granularities[]"] {
		if val == "segment" {
			hasTimestamps = true
		}
	}
	for _, val := range r.MultipartForm.Value["timestamp_granularities"] {
		if val == "segment" {
			hasTimestamps = true
		}
	}

	userPrompt := r.FormValue("prompt")
	language := r.FormValue("language")

	var instruction string
	if hasTimestamps {
		instruction = "You are a professional audio transcriber. Transcribe the audio precisely. You must provide segment-level timestamps in seconds."
		if language != "" {
			instruction += fmt.Sprintf(" The audio language is %s.", language)
		}
		if userPrompt != "" {
			instruction += fmt.Sprintf(" Style prompt from user: %s", userPrompt)
		}
	} else {
		instruction = "You are a professional audio transcriber. Transcribe the audio precisely. Output ONLY the transcribed text. Do not add any introductory text, explanation, warnings, or markdown formatting (like ```)."
		if language != "" {
			instruction += fmt.Sprintf(" The audio language is %s.", language)
		}
		if userPrompt != "" {
			instruction += fmt.Sprintf(" Style prompt from user: %s", userPrompt)
		}
	}

	parts := []interface{}{
		map[string]interface{}{
			"text": instruction,
		},
		audioPart,
	}

	geminiReq := map[string]interface{}{
		"contents": []interface{}{
			map[string]interface{}{
				"role":  "user",
				"parts": parts,
			},
		},
	}

	generationConfig := map[string]interface{}{}
	if hasTimestamps {
		generationConfig["responseMimeType"] = "application/json"
		generationConfig["responseSchema"] = map[string]interface{}{
			"type": "OBJECT",
			"properties": map[string]interface{}{
				"text": map[string]interface{}{
					"type":        "STRING",
					"description": "The complete transcription of the audio.",
				},
				"segments": map[string]interface{}{
					"type": "ARRAY",
					"items": map[string]interface{}{
						"type": "OBJECT",
						"properties": map[string]interface{}{
							"start": map[string]interface{}{
								"type":        "NUMBER",
								"description": "Start time of the segment in seconds.",
							},
							"end": map[string]interface{}{
								"type":        "NUMBER",
								"description": "End time of the segment in seconds.",
							},
							"text": map[string]interface{}{
								"type":        "STRING",
								"description": "The transcribed text of this segment.",
							},
						},
						"required": []string{"start", "end", "text"},
					},
				},
			},
			"required": []string{"text", "segments"},
		}
	}

	if len(generationConfig) > 0 {
		geminiReq["generationConfig"] = generationConfig
	}

	applyDeleteAndExtra(geminiReq, ctx.RequestBody)

	return json.Marshal(geminiReq)
}

func formatDurationSRT(seconds float64) string {
	h := int(seconds) / 3600
	m := (int(seconds) % 3600) / 60
	s := int(seconds) % 60
	ms := int((seconds - float64(int(seconds))) * 1000)
	return fmt.Sprintf("%02d:%02d:%02d,%03d", h, m, s, ms)
}

func formatDurationVTT(seconds float64) string {
	h := int(seconds) / 3600
	m := (int(seconds) % 3600) / 60
	s := int(seconds) % 60
	ms := int((seconds - float64(int(seconds))) * 1000)
	return fmt.Sprintf("%02d:%02d:%02d.%03d", h, m, s, ms)
}

type GeminiTranscribeStructured struct {
	Text     string `json:"text"`
	Segments []struct {
		Start float64 `json:"start"`
		End   float64 `json:"end"`
		Text  string  `json:"text"`
	} `json:"segments"`
}

func ProcessGeminiTranscriptionResponse(rawResp []byte, hasTimestamps bool, language string, responseFormat string) ([]byte, error) {
	var geminiResp map[string]interface{}
	if err := json.Unmarshal(rawResp, &geminiResp); err != nil {
		return nil, fmt.Errorf("failed to parse Gemini response: %w", err)
	}

	candidates, ok := geminiResp["candidates"].([]interface{})
	if !ok || len(candidates) == 0 {
		return nil, errors.New("no candidates in Gemini response")
	}

	candMap, ok := candidates[0].(map[string]interface{})
	if !ok {
		return nil, errors.New("invalid candidate format")
	}

	content, ok := candMap["content"].(map[string]interface{})
	if !ok {
		return nil, errors.New("missing content in candidate")
	}

	parts, ok := content["parts"].([]interface{})
	if !ok || len(parts) == 0 {
		return nil, errors.New("no parts in candidate content")
	}

	partMap, ok := parts[0].(map[string]interface{})
	if !ok {
		return nil, errors.New("invalid part format")
	}

	textVal, ok := partMap["text"].(string)
	if !ok {
		return nil, errors.New("missing text in candidate part")
	}

	responseFormat = strings.ToLower(responseFormat)

	if hasTimestamps {
		var structuredResp GeminiTranscribeStructured
		if err := json.Unmarshal([]byte(textVal), &structuredResp); err != nil {
			return nil, fmt.Errorf("failed to parse structured transcription JSON from Gemini: %w", err)
		}

		if responseFormat == "srt" {
			var srtBuilder strings.Builder
			for i, seg := range structuredResp.Segments {
				srtBuilder.WriteString(fmt.Sprintf("%d\n", i+1))
				srtBuilder.WriteString(fmt.Sprintf("%s --> %s\n", formatDurationSRT(seg.Start), formatDurationSRT(seg.End)))
				srtBuilder.WriteString(fmt.Sprintf("%s\n\n", strings.TrimSpace(seg.Text)))
			}
			return []byte(srtBuilder.String()), nil
		}

		if responseFormat == "vtt" {
			var vttBuilder strings.Builder
			vttBuilder.WriteString("WEBVTT\n\n")
			for i, seg := range structuredResp.Segments {
				vttBuilder.WriteString(fmt.Sprintf("%d\n", i+1))
				vttBuilder.WriteString(fmt.Sprintf("%s --> %s\n", formatDurationVTT(seg.Start), formatDurationVTT(seg.End)))
				vttBuilder.WriteString(fmt.Sprintf("%s\n\n", strings.TrimSpace(seg.Text)))
			}
			return []byte(vttBuilder.String()), nil
		}

		if responseFormat == "text" {
			return []byte(structuredResp.Text), nil
		}

		if responseFormat == "verbose_json" {
			type OpenAISegment struct {
				ID               int       `json:"id"`
				Seek             int       `json:"seek"`
				Start            float64   `json:"start"`
				End              float64   `json:"end"`
				Text             string    `json:"text"`
				Tokens           []int     `json:"tokens"`
				Temperature      float64   `json:"temperature"`
				AvgLogprob       float64   `json:"avg_logprob"`
				CompressionRatio float64   `json:"compression_ratio"`
				NoSpeechProb     float64   `json:"no_speech_prob"`
			}

			segments := make([]OpenAISegment, 0, len(structuredResp.Segments))
			var maxDuration float64
			for i, seg := range structuredResp.Segments {
				segments = append(segments, OpenAISegment{
					ID:               i,
					Seek:             0,
					Start:            seg.Start,
					End:              seg.End,
					Text:             seg.Text,
					Tokens:           []int{},
					Temperature:      0.0,
					AvgLogprob:       0.0,
					CompressionRatio: 0.0,
					NoSpeechProb:     0.0,
				})
				if seg.End > maxDuration {
					maxDuration = seg.End
				}
			}

			if language == "" {
				language = "english"
			}

			openAIResp := map[string]interface{}{
				"task":     "transcribe",
				"language": language,
				"duration": maxDuration,
				"text":     structuredResp.Text,
				"segments": segments,
				"usage": map[string]interface{}{
					"type":    "duration",
					"seconds": int(maxDuration + 0.5),
				},
			}
			return json.Marshal(openAIResp)
		}

		openAIResp := map[string]interface{}{
			"text": structuredResp.Text,
		}
		return json.Marshal(openAIResp)
	}

	if responseFormat == "text" || responseFormat == "srt" || responseFormat == "vtt" {
		return []byte(textVal), nil
	}

	openAIResp := map[string]interface{}{
		"text": textVal,
	}
	return json.Marshal(openAIResp)
}

type GeminiSTTUsage struct {
	PromptTokens     int
	CandidatesTokens int
	TotalTokens      int
	AudioTokens      int
	TextTokens       int
}

func ProcessGeminiSTTStreamLine(line []byte) (string, *GeminiSTTUsage, error) {
	trimmed := bytes.TrimSpace(line)
	if !bytes.HasPrefix(trimmed, []byte("data: ")) {
		return "", nil, nil
	}

	payload := bytes.TrimPrefix(trimmed, []byte("data: "))
	payloadStr := string(payload)
	if payloadStr == "[DONE]" {
		return "", nil, io.EOF
	}

	var geminiResp map[string]interface{}
	if err := json.Unmarshal(payload, &geminiResp); err != nil {
		return "", nil, err
	}

	var usage *GeminiSTTUsage
	if usageMeta, ok := geminiResp["usageMetadata"].(map[string]interface{}); ok {
		usage = &GeminiSTTUsage{}
		if pt, ok := usageMeta["promptTokenCount"].(float64); ok {
			usage.PromptTokens = int(pt)
		}
		if ct, ok := usageMeta["candidatesTokenCount"].(float64); ok {
			usage.CandidatesTokens = int(ct)
		}
		if tt, ok := usageMeta["totalTokenCount"].(float64); ok {
			usage.TotalTokens = int(tt)
		}
		if details, ok := usageMeta["promptTokensDetails"].([]interface{}); ok {
			for _, det := range details {
				if dMap, ok := det.(map[string]interface{}); ok {
					mod, _ := dMap["modality"].(string)
					tc, _ := dMap["tokenCount"].(float64)
					if mod == "AUDIO" {
						usage.AudioTokens = int(tc)
					} else if mod == "TEXT" {
						usage.TextTokens = int(tc)
					}
				}
			}
		}
	}

	candidates, ok := geminiResp["candidates"].([]interface{})
	if !ok || len(candidates) == 0 {
		return "", usage, nil
	}

	candMap, ok := candidates[0].(map[string]interface{})
	if !ok {
		return "", usage, nil
	}

	content, ok := candMap["content"].(map[string]interface{})
	if !ok {
		return "", usage, nil
	}

	parts, ok := content["parts"].([]interface{})
	if !ok || len(parts) == 0 {
		return "", usage, nil
	}

	partMap, ok := parts[0].(map[string]interface{})
	if !ok {
		return "", usage, nil
	}

	textVal, _ := partMap["text"].(string)
	return textVal, usage, nil
}

func init() {
	Register(&GeminiPlugin{})
}
