package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
)

// Claude API Request & Response Structs
type ClaudeContent struct {
	Type      string             `json:"type"`
	Text      string             `json:"text,omitempty"`
	Source    *ClaudeImageSource `json:"source,omitempty"`
	ID        string             `json:"id,omitempty"`
	Name      string             `json:"name,omitempty"`
	Input     interface{}        `json:"input,omitempty"`
	ToolUseID string             `json:"tool_use_id,omitempty"`
	Content   interface{}        `json:"content,omitempty"`
	IsError   *bool              `json:"is_error,omitempty"`
}

type ClaudeImageSource struct {
	Type      string `json:"type"`
	MediaType string `json:"media_type"`
	Data      string `json:"data"`
}

type ClaudeMessage struct {
	Role    string      `json:"role"`
	Content interface{} `json:"content"` // string or []ClaudeContent
}

type ClaudeTool struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description,omitempty"`
	InputSchema map[string]interface{} `json:"input_schema"`
}

type ClaudeMessagesRequest struct {
	Model         string          `json:"model"`
	Messages      []ClaudeMessage `json:"messages"`
	System        interface{}     `json:"system,omitempty"` // string or []ClaudeContent
	MaxTokens     int             `json:"max_tokens"`
	StopSequences []string        `json:"stop_sequences,omitempty"`
	Stream        bool            `json:"stream,omitempty"`
	Temperature   *float64        `json:"temperature,omitempty"`
	TopP          *float64        `json:"top_p,omitempty"`
	Tools         []ClaudeTool    `json:"tools,omitempty"`
}

type ClaudeMessagesResponse struct {
	ID           string          `json:"id"`
	Type         string          `json:"type"`
	Role         string          `json:"role"`
	Content      []ClaudeContent `json:"content"`
	Model        string          `json:"model"`
	StopReason   *string         `json:"stop_reason"`
	StopSequence *string         `json:"stop_sequence"`
	Usage        ClaudeUsage     `json:"usage"`
}

type ClaudeUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

// OpenAI Responses API Structs
type OpenAIResponsesRequest struct {
	Model        string        `json:"model"`
	Input        interface{}   `json:"input"` // string or []interface{}
	Instructions string        `json:"instructions,omitempty"`
	Tools        []interface{} `json:"tools,omitempty"`
	Stream       bool          `json:"stream,omitempty"`
	Temperature  *float64      `json:"temperature,omitempty"`
	TopP         *float64      `json:"top_p,omitempty"`
	MaxTokens    int           `json:"max_tokens,omitempty"`
}

type OpenAIResponsesResponse struct {
	ID     string                `json:"id"`
	Object string                `json:"object"`
	Model  string                `json:"model"`
	Output []OpenAIOutputMessage `json:"output"`
	Usage  OpenAIUsage           `json:"usage,omitempty"`
}

type OpenAIOutputMessage struct {
	Type    string       `json:"type"`
	Role    string       `json:"role"`
	Content []OpenAIPart `json:"content"`
}

type OpenAIPart struct {
	Type         string              `json:"type"`
	Text         string              `json:"text,omitempty"`
	Function     *OpenAIPartFunction `json:"function_call,omitempty"`
	ExtraContent interface{}         `json:"extra_content,omitempty"`
}

type OpenAIPartFunction struct {
	CallID    string `json:"call_id"`
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type OpenAIUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
}

// marshalNoEscape marshals JSON without escaping HTML characters like < and >
func marshalNoEscape(v interface{}) ([]byte, error) {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	err := enc.Encode(v)
	if err != nil {
		return nil, err
	}
	b := buf.Bytes()
	if len(b) > 0 && b[len(b)-1] == '\n' {
		b = b[:len(b)-1]
	}
	return b, nil
}

// writeJSONNoEscape writes JSON to ResponseWriter without HTML escaping
func writeJSONNoEscape(w http.ResponseWriter, v interface{}) error {
	w.Header().Set("Content-Type", "application/json")
	enc := json.NewEncoder(w)
	enc.SetEscapeHTML(false)
	return enc.Encode(v)
}

// Helper: Marshal interface to JSON string
func marshalJSONString(v interface{}) string {
	if v == nil {
		return "{}"
	}
	data, err := marshalNoEscape(v)
	if err != nil {
		return "{}"
	}
	return string(data)
}

// sanitizeToolName cleans tool names to meet Gemini's strict validation rules
// "Must start with a letter or an underscore. Must be alphameric (a-z, A-Z, 0-9), underscores (_), dots (.), colons (:), or dashes (-), with a maximum length of 128."
// Wait, the error is actually because Claude allows "-" but maybe Gemini is rejecting something else?
// Ah, the user's error says: "Must be alphameric (a-z, A-Z, 0-9), underscores (_), dots (.), colons (:), or dashes (-)"
// Wait, the error explicitly listed dashes (-), so why did it fail?
// Because the user's tool name might have started with a dash, or had characters NOT in that list, or maybe Gemini's actual error message allows hyphens but the tool name had hyphens and the regex rejected it?
// Let's replace any character that is not a-zA-Z0-9_.:- with underscore, and ensure it starts with letter or underscore.
func sanitizeToolName(name string) string {
	if name == "" {
		return name
	}
	var sb strings.Builder
	for i, r := range name {
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || r == '_' {
			sb.WriteRune(r)
		} else if r >= '0' && r <= '9' || r == '.' || r == ':' || r == '-' {
			if i == 0 {
				sb.WriteRune('_') // must start with letter or underscore
			}
			sb.WriteRune(r)
		} else {
			sb.WriteRune('_')
		}
	}
	res := sb.String()
	if len(res) > 128 {
		res = res[:128]
	}
	return res
}

// TranslateClaudeRequestToOpenAI converts Claude /v1/messages request to OpenAI /v1/chat/completions payload map.
func TranslateClaudeRequestToOpenAI(req *ClaudeMessagesRequest) (map[string]interface{}, error) {
	var openAIMessages []interface{}

	// Handle system instructions
	if req.System != nil {
		var systemContent string
		switch v := req.System.(type) {
		case string:
			systemContent = v
		case []interface{}:
			for _, item := range v {
				if m, ok := item.(map[string]interface{}); ok {
					if m["type"] == "text" {
						systemContent += fmt.Sprintf("%v", m["text"])
					}
				}
			}
		case []ClaudeContent:
			for _, item := range v {
				if item.Type == "text" {
					systemContent += item.Text
				}
			}
		}
		if systemContent != "" {
			openAIMessages = append(openAIMessages, map[string]interface{}{
				"role":    "system",
				"content": systemContent,
			})
		}
	}

	// Process messages
	for _, msg := range req.Messages {
		var openAIContent interface{}

		switch contentVal := msg.Content.(type) {
		case string:
			openAIContent = contentVal
		case []interface{}:
			var parts []interface{}
			hasToolUse := false
			hasToolResult := false
			var toolCalls []interface{}
			var toolResults []map[string]interface{}
			var textParts []interface{}

			for _, block := range contentVal {
				blockMap, ok := block.(map[string]interface{})
				if !ok {
					continue
				}
				blockType := fmt.Sprintf("%v", blockMap["type"])

				if blockType == "text" {
					parts = append(parts, map[string]interface{}{
						"type": "text",
						"text": blockMap["text"],
					})
					textParts = append(textParts, blockMap["text"])
				} else if blockType == "image" {
					if source, ok := blockMap["source"].(map[string]interface{}); ok {
						mediaType := source["media_type"]
						data := source["data"]
						parts = append(parts, map[string]interface{}{
							"type": "image_url",
							"image_url": map[string]interface{}{
								"url": fmt.Sprintf("data:%s;base64,%s", mediaType, data),
							},
						})
					}
				} else if blockType == "tool_use" {
					hasToolUse = true
					toolCalls = append(toolCalls, map[string]interface{}{
						"id":   blockMap["id"],
						"type": "function",
						"function": map[string]interface{}{
							"name":      blockMap["name"],
							"arguments": marshalJSONString(blockMap["input"]),
						},
					})
				} else if blockType == "tool_result" {
					hasToolResult = true
					toolContent := blockMap["content"]
					contentStr := ""
					if toolContent != nil {
						if slice, ok := toolContent.([]interface{}); ok {
							for _, item := range slice {
								if m, ok := item.(map[string]interface{}); ok && m["type"] == "text" {
									contentStr += fmt.Sprintf("%v", m["text"])
								}
							}
						} else {
							contentStr = fmt.Sprintf("%v", toolContent)
						}
					}

					if isErr, _ := blockMap["is_error"].(bool); isErr {
						contentStr = "[ERROR] " + contentStr
					}

					toolResults = append(toolResults, map[string]interface{}{
						"role":         "tool",
						"tool_call_id": blockMap["tool_use_id"],
						"content":      contentStr,
					})
				}
			}

			if hasToolResult {
				for _, tr := range toolResults {
					openAIMessages = append(openAIMessages, tr)
				}
				if len(parts) > 0 {
					openAIMessages = append(openAIMessages, map[string]interface{}{
						"role":    msg.Role,
						"content": parts,
					})
				}
				continue
			}

			if hasToolUse {
				msgObj := map[string]interface{}{
					"role": "assistant",
				}
				if len(textParts) > 0 {
					var fullText string
					for _, tp := range textParts {
						fullText += fmt.Sprintf("%v", tp)
					}
					msgObj["content"] = fullText
				}
				msgObj["tool_calls"] = toolCalls
				openAIMessages = append(openAIMessages, msgObj)
				continue
			}

			if len(parts) > 0 {
				openAIContent = parts
			}
		case []ClaudeContent:
			var parts []interface{}
			hasToolUse := false
			hasToolResult := false
			var toolCalls []interface{}
			var toolResults []map[string]interface{}
			var textParts []interface{}

			for _, block := range contentVal {
				blockType := block.Type

				if blockType == "text" {
					parts = append(parts, map[string]interface{}{
						"type": "text",
						"text": block.Text,
					})
					textParts = append(textParts, block.Text)
				} else if blockType == "image" {
					if block.Source != nil {
						mediaType := block.Source.MediaType
						data := block.Source.Data
						parts = append(parts, map[string]interface{}{
							"type": "image_url",
							"image_url": map[string]interface{}{
								"url": fmt.Sprintf("data:%s;base64,%s", mediaType, data),
							},
						})
					}
				} else if blockType == "tool_use" {
					hasToolUse = true
					toolCalls = append(toolCalls, map[string]interface{}{
						"id":   block.ID,
						"type": "function",
						"function": map[string]interface{}{
							"name":      block.Name,
							"arguments": marshalJSONString(block.Input),
						},
					})
				} else if blockType == "tool_result" {
					hasToolResult = true
					toolContent := block.Content
					contentStr := ""
					if toolContent != nil {
						if slice, ok := toolContent.([]interface{}); ok {
							for _, item := range slice {
								if m, ok := item.(map[string]interface{}); ok && m["type"] == "text" {
									contentStr += fmt.Sprintf("%v", m["text"])
								}
							}
						} else if cSlice, ok := toolContent.([]ClaudeContent); ok {
							for _, item := range cSlice {
								if item.Type == "text" {
									contentStr += item.Text
								}
							}
						} else {
							contentStr = fmt.Sprintf("%v", toolContent)
						}
					}

					if block.IsError != nil && *block.IsError {
						contentStr = "[ERROR] " + contentStr
					}

					toolResults = append(toolResults, map[string]interface{}{
						"role":         "tool",
						"tool_call_id": block.ToolUseID,
						"content":      contentStr,
					})
				}
			}

			if hasToolResult {
				for _, tr := range toolResults {
					openAIMessages = append(openAIMessages, tr)
				}
				if len(parts) > 0 {
					openAIMessages = append(openAIMessages, map[string]interface{}{
						"role":    msg.Role,
						"content": parts,
					})
				}
				continue
			}

			if hasToolUse {
				msgObj := map[string]interface{}{
					"role": "assistant",
				}
				if len(textParts) > 0 {
					var fullText string
					for _, tp := range textParts {
						fullText += fmt.Sprintf("%v", tp)
					}
					msgObj["content"] = fullText
				}
				msgObj["tool_calls"] = toolCalls
				openAIMessages = append(openAIMessages, msgObj)
				continue
			}

			if len(parts) > 0 {
				openAIContent = parts
			}
		}

		if openAIContent != nil {
			openAIMessages = append(openAIMessages, map[string]interface{}{
				"role":    msg.Role,
				"content": openAIContent,
			})
		}
	}

	// Process tools
	var openAITools []interface{}
	for _, t := range req.Tools {
		openAITools = append(openAITools, map[string]interface{}{
			"type": "function",
			"function": map[string]interface{}{
				"name":        sanitizeToolName(t.Name),
				"description": t.Description,
				"parameters":  t.InputSchema,
			},
		})
	}

	openaiReq := map[string]interface{}{
		"model":    req.Model,
		"messages": openAIMessages,
	}
	if len(openAITools) > 0 {
		openaiReq["tools"] = openAITools
	}
	if req.MaxTokens > 0 {
		openaiReq["max_tokens"] = req.MaxTokens
	}
	if req.Temperature != nil {
		openaiReq["temperature"] = *req.Temperature
	}
	if req.TopP != nil {
		openaiReq["top_p"] = *req.TopP
	}
	if len(req.StopSequences) > 0 {
		openaiReq["stop"] = req.StopSequences
	}
	if req.Stream {
		openaiReq["stream"] = true
	}

	return openaiReq, nil
}

// TranslateOpenAIResponseToClaude converts OpenAI response to Claude /v1/messages response format.
func TranslateOpenAIResponseToClaude(openaiResp map[string]interface{}) (*ClaudeMessagesResponse, error) {
	choices, ok := openaiResp["choices"].([]interface{})
	if !ok || len(choices) == 0 {
		return nil, errors.New("invalid openai response: missing choices")
	}

	choice, ok := choices[0].(map[string]interface{})
	if !ok {
		return nil, errors.New("invalid openai response: missing choice object")
	}

	message, ok := choice["message"].(map[string]interface{})
	if !ok {
		return nil, errors.New("invalid openai response: missing message object")
	}

	var content []ClaudeContent

	if val, ok := message["content"]; ok && val != nil {
		if contentStr, ok := val.(string); ok && contentStr != "" {
			content = append(content, ClaudeContent{
				Type: "text",
				Text: contentStr,
			})
		}
	}

	if toolCalls, ok := message["tool_calls"].([]interface{}); ok {
		for _, tcVal := range toolCalls {
			tc, ok := tcVal.(map[string]interface{})
			if !ok {
				continue
			}
			tcID := fmt.Sprintf("%v", tc["id"])
			fn, ok := tc["function"].(map[string]interface{})
			if !ok {
				continue
			}
			fnName := fmt.Sprintf("%v", fn["name"])
			fnArgsStr := fmt.Sprintf("%v", fn["arguments"])

			var fnArgs interface{}
			if err := json.Unmarshal([]byte(fnArgsStr), &fnArgs); err != nil {
				fnArgs = fnArgsStr
			}

			content = append(content, ClaudeContent{
				Type:  "tool_use",
				ID:    tcID,
				Name:  fnName,
				Input: fnArgs,
			})
		}
	}

	modelStr := ""
	if m, ok := openaiResp["model"].(string); ok {
		modelStr = m
	}

	respID := ""
	if id, ok := openaiResp["id"].(string); ok {
		respID = "msg_" + id
	}

	stopReason := "end_turn"
	if fr, ok := choice["finish_reason"].(string); ok {
		switch fr {
		case "stop":
			stopReason = "end_turn"
		case "tool_calls":
			stopReason = "tool_use"
		case "length":
			stopReason = "max_tokens"
		default:
			stopReason = fr
		}
	}

	var inputTokens, outputTokens int
	if usage, ok := openaiResp["usage"].(map[string]interface{}); ok {
		if it, ok := usage["prompt_tokens"].(float64); ok {
			inputTokens = int(it)
		}
		if ot, ok := usage["completion_tokens"].(float64); ok {
			outputTokens = int(ot)
		}
	}

	return &ClaudeMessagesResponse{
		ID:           respID,
		Type:         "message",
		Role:         "assistant",
		Content:      content,
		Model:        modelStr,
		StopReason:   &stopReason,
		StopSequence: nil,
		Usage: ClaudeUsage{
			InputTokens:  inputTokens,
			OutputTokens: outputTokens,
		},
	}, nil
}

// TranslateResponsesRequestToOpenAI converts OpenAI Responses API payload to traditional Chat Completions payload.
func TranslateResponsesRequestToOpenAI(req *OpenAIResponsesRequest) (map[string]interface{}, error) {
	var openAIMessages []interface{}

	if req.Instructions != "" {
		openAIMessages = append(openAIMessages, map[string]interface{}{
			"role":    "system",
			"content": req.Instructions,
		})
	}

	if req.Input != nil {
		switch inputVal := req.Input.(type) {
		case string:
			openAIMessages = append(openAIMessages, map[string]interface{}{
				"role":    "user",
				"content": inputVal,
			})
		case []interface{}:
			var parts []interface{}
			isMessageHistory := false

			for _, el := range inputVal {
				elMap, ok := el.(map[string]interface{})
				if !ok {
					continue
				}
				if _, hasRole := elMap["role"]; hasRole {
					isMessageHistory = true
					break
				}
				elType := fmt.Sprintf("%v", elMap["type"])
				if elType == "function_call" || elType == "function_call_output" {
					isMessageHistory = true
					break
				}
			}

			if isMessageHistory {
				for _, el := range inputVal {
					elMap, ok := el.(map[string]interface{})
					if !ok {
						continue
					}

					elType := fmt.Sprintf("%v", elMap["type"])
					if elType == "function_call" {
						var argsStr string
						if argsVal, ok := elMap["arguments"]; ok {
							switch av := argsVal.(type) {
							case string:
								argsStr = av
							default:
								argsStr = marshalJSONString(av)
							}
						}
						tcObj := map[string]interface{}{
							"id":   elMap["call_id"],
							"type": "function",
							"function": map[string]interface{}{
								"name":      elMap["name"],
								"arguments": argsStr,
							},
						}
						if ec, exists := elMap["extra_content"]; exists && ec != nil {
							tcObj["extra_content"] = ec
						}
						openAIMessages = append(openAIMessages, map[string]interface{}{
							"role": "assistant",
							"tool_calls": []interface{}{
								tcObj,
							},
						})
						continue
					}

					if elType == "function_call_output" {
						openAIMessages = append(openAIMessages, map[string]interface{}{
							"role":         "tool",
							"tool_call_id": elMap["call_id"],
							"content":      elMap["output"],
						})
						continue
					}

					var role string
					if rVal, ok := elMap["role"]; ok && rVal != nil {
						role = fmt.Sprintf("%v", rVal)
					}
					content := elMap["content"]

					if role == "developer" {
						role = "system"
					}

					if contentSlice, ok := content.([]interface{}); ok {
						var processedContent []interface{}
						for _, part := range contentSlice {
							partMap, ok := part.(map[string]interface{})
							if !ok {
								processedContent = append(processedContent, part)
								continue
							}
							partType := fmt.Sprintf("%v", partMap["type"])
							if partType == "input_text" || partType == "text" || partType == "output_text" {
								processedContent = append(processedContent, map[string]interface{}{
									"type": "text",
									"text": partMap["text"],
								})
							} else if partType == "input_image" {
								processedContent = append(processedContent, map[string]interface{}{
									"type": "image_url",
									"image_url": map[string]interface{}{
										"url": partMap["image_url"],
									},
								})
							} else {
								processedContent = append(processedContent, part)
							}
						}
						content = processedContent
					}

					msgObj := map[string]interface{}{
						"role":    role,
						"content": content,
					}
					if tc, hasTC := elMap["tool_calls"]; hasTC {
						msgObj["tool_calls"] = tc
					}
					openAIMessages = append(openAIMessages, msgObj)
				}
			} else {
				for _, el := range inputVal {
					elMap, ok := el.(map[string]interface{})
					if !ok {
						continue
					}
					elType := fmt.Sprintf("%v", elMap["type"])
					if elType == "text" || elType == "input_text" || elType == "output_text" {
						parts = append(parts, map[string]interface{}{
							"type": "text",
							"text": elMap["text"],
						})
					} else if elType == "input_image" {
						parts = append(parts, map[string]interface{}{
							"type": "image_url",
							"image_url": map[string]interface{}{
								"url": elMap["image_url"],
							},
						})
					}
				}
				if len(parts) > 0 {
					openAIMessages = append(openAIMessages, map[string]interface{}{
						"role":    "user",
						"content": parts,
					})
				}
			}
		}
	}

	openaiReq := map[string]interface{}{
		"model":    req.Model,
		"messages": openAIMessages,
	}

	if len(req.Tools) > 0 {
		var chatTools []interface{}
		for _, tool := range req.Tools {
			toolMap, ok := tool.(map[string]interface{})
			if !ok {
				chatTools = append(chatTools, tool)
				continue
			}
			toolType := fmt.Sprintf("%v", toolMap["type"])
			if toolType == "function" {
				// Responses API flat format -> Chat Completions nested format
				fnDef := map[string]interface{}{}
				if name, ok := toolMap["name"]; ok {
					fnDef["name"] = sanitizeToolName(fmt.Sprintf("%v", name))
				}
				if desc, ok := toolMap["description"]; ok {
					fnDef["description"] = desc
				}
				if params, ok := toolMap["parameters"]; ok {
					fnDef["parameters"] = params
				}
				if strict, ok := toolMap["strict"]; ok {
					fnDef["strict"] = strict
				}
				chatTools = append(chatTools, map[string]interface{}{
					"type":     "function",
					"function": fnDef,
				})
			}
		}
		openaiReq["tools"] = chatTools
	}
	if req.Temperature != nil {
		openaiReq["temperature"] = *req.Temperature
	}
	if req.TopP != nil {
		openaiReq["top_p"] = *req.TopP
	}
	if req.MaxTokens > 0 {
		openaiReq["max_tokens"] = req.MaxTokens
	}
	if req.Stream {
		openaiReq["stream"] = true
	}

	return openaiReq, nil
}

// TranslateOpenAIResponseToResponses converts OpenAI Chat Completions Response to Responses API Response.
func TranslateOpenAIResponseToResponses(openaiResp map[string]interface{}) (*OpenAIResponsesResponse, error) {
	choices, ok := openaiResp["choices"].([]interface{})
	if !ok || len(choices) == 0 {
		return nil, errors.New("invalid openai response: missing choices")
	}

	choice, ok := choices[0].(map[string]interface{})
	if !ok {
		return nil, errors.New("invalid openai response: missing choice object")
	}

	message, ok := choice["message"].(map[string]interface{})
	if !ok {
		return nil, errors.New("invalid openai response: missing message object")
	}

	var outputParts []OpenAIPart

	if val, ok := message["content"]; ok && val != nil {
		if contentStr, ok := val.(string); ok && contentStr != "" {
			outputParts = append(outputParts, OpenAIPart{
				Type: "text",
				Text: contentStr,
			})
		}
	}

	if toolCalls, ok := message["tool_calls"].([]interface{}); ok {
		for _, tcVal := range toolCalls {
			tc, ok := tcVal.(map[string]interface{})
			if !ok {
				continue
			}
			tcID := fmt.Sprintf("%v", tc["id"])
			fn, ok := tc["function"].(map[string]interface{})
			if !ok {
				continue
			}
			fnName := fmt.Sprintf("%v", fn["name"])
			fnArgsStr := fmt.Sprintf("%v", fn["arguments"])

			part := OpenAIPart{
				Type: "function_call",
				Function: &OpenAIPartFunction{
					CallID:    tcID,
					Name:      fnName,
					Arguments: fnArgsStr,
				},
			}
			if ec, exists := tc["extra_content"]; exists {
				part.ExtraContent = ec
			}
			outputParts = append(outputParts, part)
		}
	}

	var finalOutput []OpenAIOutputMessage
	if len(outputParts) > 0 {
		finalOutput = append(finalOutput, OpenAIOutputMessage{
			Type:    "message",
			Role:    "assistant",
			Content: outputParts,
		})
	}

	modelStr := ""
	if m, ok := openaiResp["model"].(string); ok {
		modelStr = m
	}

	respID := ""
	if id, ok := openaiResp["id"].(string); ok {
		respID = id
	}

	var pt, ct int
	if usage, ok := openaiResp["usage"].(map[string]interface{}); ok {
		if p, ok := usage["prompt_tokens"].(float64); ok {
			pt = int(p)
		}
		if c, ok := usage["completion_tokens"].(float64); ok {
			ct = int(c)
		}
	}

	return &OpenAIResponsesResponse{
		ID:     respID,
		Object: "response",
		Model:  modelStr,
		Output: finalOutput,
		Usage: OpenAIUsage{
			PromptTokens:     pt,
			CompletionTokens: ct,
		},
	}, nil
}

// OpenAI SSE payload structure for streaming parser
type OpenAIMessageChunk struct {
	ID      string `json:"id"`
	Model   string `json:"model"`
	Choices []struct {
		Index int `json:"index"`
		Delta struct {
			Content   string `json:"content"`
			ToolCalls []struct {
				Index    int    `json:"index"`
				ID       string `json:"id"`
				Type     string `json:"type"`
				Function struct {
					Name      string `json:"name"`
					Arguments string `json:"arguments"`
				} `json:"function"`
				ExtraContent interface{} `json:"extra_content,omitempty"`
			} `json:"tool_calls"`
		} `json:"delta"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage *struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
	} `json:"usage,omitempty"`
}

type debugWriter struct {
	w        io.Writer
	debugBuf bytes.Buffer
}

func (dw *debugWriter) Write(p []byte) (n int, err error) {
	n, err = dw.w.Write(p)
	if n > 0 {
		dw.debugBuf.Write(p[:n])
	}
	return
}

// TranslateOpenAIStreamToClaude parses OpenAI SSE stream and writes Claude SSE stream.
func TranslateOpenAIStreamToClaude(r io.Reader, w io.Writer, debug bool) error {
	var targetWriter io.Writer = w
	var dw *debugWriter
	if debug {
		dw = &debugWriter{w: w}
		targetWriter = dw
	}

	scanner := bufio.NewScanner(r)
	// Use a 10MB buffer to handle extremely large lines (e.g. nested images or large thought blocks)
	buf := make([]byte, 0, 1024*1024)
	scanner.Buffer(buf, 10*1024*1024)
	flusher, hasFlush := w.(http.Flusher)

	messageStarted := false
	textBlockStarted := false
	nextBlockIndex := 0
	activeToolCalls := make(map[int]int) // OpenAI tc.Index -> Claude block index
	stopReasonSent := false

	var lastChunkID string
	var lastChunkModel string

	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			// Proxy non-data lines (like spacing) directly
			if line == "" {
				fmt.Fprint(targetWriter, "\n")
				if hasFlush {
					flusher.Flush()
				}
			}
			continue
		}

		dataPayload := strings.TrimPrefix(line, "data: ")
		if dataPayload == "[DONE]" {
			break
		}

		var chunk OpenAIMessageChunk
		if err := json.Unmarshal([]byte(dataPayload), &chunk); err != nil {
			log.Printf("failed to parse OpenAI chunk: %v", err)
			continue
		}

		if chunk.ID != "" {
			lastChunkID = chunk.ID
		}
		if chunk.Model != "" {
			lastChunkModel = chunk.Model
		}

		if len(chunk.Choices) == 0 {
			// Might be a usage-only or empty chunk
			if chunk.Usage != nil && messageStarted {
				// Send message usage delta
				usageEvent := map[string]interface{}{
					"type": "message_delta",
					"usage": map[string]interface{}{
						"output_tokens": chunk.Usage.CompletionTokens,
					},
				}
				writeSSEEvent(targetWriter, "message_delta", usageEvent)
			}
			continue
		}

		choice := chunk.Choices[0]

		// 1. Send message_start if not sent yet
		if !messageStarted {
			msgID := chunk.ID
			if !strings.HasPrefix(msgID, "msg_") {
				msgID = "msg_" + msgID
			}
			startEvent := map[string]interface{}{
				"type": "message_start",
				"message": map[string]interface{}{
					"id":            msgID,
					"type":          "message",
					"role":          "assistant",
					"content":       []interface{}{},
					"model":         chunk.Model,
					"stop_reason":   nil,
					"stop_sequence": nil,
					"usage": map[string]interface{}{
						"input_tokens":  0,
						"output_tokens": 0,
					},
				},
			}
			if err := writeSSEEvent(targetWriter, "message_start", startEvent); err != nil {
				return err
			}
			messageStarted = true
		}

		// 2. Handle Text Delta
		if choice.Delta.Content != "" {
			if !textBlockStarted {
				startBlock := map[string]interface{}{
					"type":  "content_block_start",
					"index": nextBlockIndex,
					"content_block": map[string]interface{}{
						"type": "text",
						"text": "",
					},
				}
				if err := writeSSEEvent(targetWriter, "content_block_start", startBlock); err != nil {
					return err
				}
				textBlockStarted = true
				nextBlockIndex++
			}

			textBlockIdx := nextBlockIndex - 1 // text block was the last one opened
			deltaEvent := map[string]interface{}{
				"type":  "content_block_delta",
				"index": textBlockIdx,
				"delta": map[string]interface{}{
					"type": "text_delta",
					"text": choice.Delta.Content,
				},
			}
			if err := writeSSEEvent(targetWriter, "content_block_delta", deltaEvent); err != nil {
				return err
			}
		}

		// 3. Handle Tool Calls Delta
		if len(choice.Delta.ToolCalls) > 0 {
			for _, tc := range choice.Delta.ToolCalls {
				blockIdx, active := activeToolCalls[tc.Index]
				if !active && tc.ID != "" {
					blockIdx = nextBlockIndex
					activeToolCalls[tc.Index] = blockIdx
					nextBlockIndex++
					// Start tool use block
					startBlock := map[string]interface{}{
						"type":  "content_block_start",
						"index": blockIdx,
						"content_block": map[string]interface{}{
							"type":  "tool_use",
							"id":    tc.ID,
							"name":  tc.Function.Name,
							"input": map[string]interface{}{},
						},
					}
					if err := writeSSEEvent(targetWriter, "content_block_start", startBlock); err != nil {
						return err
					}
				}

				if tc.Function.Arguments != "" {
					deltaEvent := map[string]interface{}{
						"type":  "content_block_delta",
						"index": blockIdx,
						"delta": map[string]interface{}{
							"type":         "input_json_delta",
							"partial_json": tc.Function.Arguments,
						},
					}
					if err := writeSSEEvent(targetWriter, "content_block_delta", deltaEvent); err != nil {
						return err
					}
				}
			}
		}

		// 4. Handle finish reason
		if choice.FinishReason != "" {
			if textBlockStarted {
				textBlockIdx := 0 // text block is always index 0 when it exists
				for i := 0; i < nextBlockIndex; i++ {
					isToolBlock := false
					for _, bi := range activeToolCalls {
						if bi == i {
							isToolBlock = true
							break
						}
					}
					if !isToolBlock {
						textBlockIdx = i
						break
					}
				}
				stopBlock := map[string]interface{}{
					"type":  "content_block_stop",
					"index": textBlockIdx,
				}
				if err := writeSSEEvent(targetWriter, "content_block_stop", stopBlock); err != nil {
					return err
				}
				textBlockStarted = false
			}

			for _, blockIdx := range activeToolCalls {
				stopBlock := map[string]interface{}{
					"type":  "content_block_stop",
					"index": blockIdx,
				}
				if err := writeSSEEvent(targetWriter, "content_block_stop", stopBlock); err != nil {
					return err
				}
			}
			activeToolCalls = make(map[int]int)

			stopReason := "end_turn"
			switch choice.FinishReason {
			case "stop":
				stopReason = "end_turn"
			case "tool_calls":
				stopReason = "tool_use"
			case "length":
				stopReason = "max_tokens"
			default:
				stopReason = choice.FinishReason
			}

			msgDelta := map[string]interface{}{
				"type": "message_delta",
				"delta": map[string]interface{}{
					"stop_reason":   stopReason,
					"stop_sequence": nil,
				},
			}
			if chunk.Usage != nil {
				msgDelta["usage"] = map[string]interface{}{
					"output_tokens": chunk.Usage.CompletionTokens,
				}
			}
			if err := writeSSEEvent(targetWriter, "message_delta", msgDelta); err != nil {
				return err
			}
			stopReasonSent = true
		}

		if hasFlush {
			flusher.Flush()
		}
	}

	// Close the message cleanly
	if messageStarted {
		if !stopReasonSent {
			msgDelta := map[string]interface{}{
				"type": "message_delta",
				"delta": map[string]interface{}{
					"stop_reason":   "end_turn",
					"stop_sequence": nil,
				},
			}
			writeSSEEvent(targetWriter, "message_delta", msgDelta)
		}
		writeSSEEvent(targetWriter, "message_stop", map[string]interface{}{"type": "message_stop"})
	} else {
		// Fallback start + stop if stream was completely empty
		startEvent := map[string]interface{}{
			"type": "message_start",
			"message": map[string]interface{}{
				"id":          "msg_" + lastChunkID,
				"type":        "message",
				"role":        "assistant",
				"content":     []interface{}{},
				"model":       lastChunkModel,
				"stop_reason": "end_turn",
			},
		}
		writeSSEEvent(targetWriter, "message_start", startEvent)
		writeSSEEvent(targetWriter, "message_stop", map[string]interface{}{"type": "message_stop"})
	}

	if debug && dw != nil {
		log.Printf("[DEBUG] Claude Response Body (Stream): %s", dw.debugBuf.String())
	}

	if hasFlush {
		flusher.Flush()
	}
	return nil
}

type responsesToolCallState struct {
	id           string
	name         string
	arguments    strings.Builder
	outputIndex  int
	itemAdded    bool
	done         bool
	extraContent interface{}
}

type responsesStreamState struct {
	seq              int
	responseCreated  bool
	messageItemAdded bool
	contentPartAdded bool
	messageItemID    string
	accumulatedText  strings.Builder
	activeToolCalls  map[int]*responsesToolCallState
	lastChunkID      string
	lastChunkModel   string
	messageItemDone  bool
}

func (s *responsesStreamState) nextSeq() int {
	n := s.seq
	s.seq++
	return n
}

func (s *responsesStreamState) emit(w io.Writer, eventType string, data map[string]interface{}) error {
	data["type"] = eventType
	data["sequence_number"] = s.nextSeq()
	return writeSSEEvent(w, eventType, data)
}

func (s *responsesStreamState) buildResponse(status string, output []interface{}) map[string]interface{} {
	resp := map[string]interface{}{
		"id":     s.lastChunkID,
		"object": "response",
		"model":  s.lastChunkModel,
		"status": status,
		"output": output,
	}
	return resp
}

func (s *responsesStreamState) ensureCreated(w io.Writer) error {
	if s.responseCreated {
		return nil
	}
	if err := s.emit(w, "response.created", map[string]interface{}{
		"response": s.buildResponse("in_progress", []interface{}{}),
	}); err != nil {
		return err
	}
	s.responseCreated = true
	return s.emit(w, "response.in_progress", map[string]interface{}{
		"response": s.buildResponse("in_progress", []interface{}{}),
	})
}

func (s *responsesStreamState) messageOutputIndex() int {
	return 0
}

func (s *responsesStreamState) toolOutputIndex(tcIndex int) int {
	if s.messageItemAdded {
		return tcIndex + 1
	}
	return tcIndex
}

func (s *responsesStreamState) ensureMessageItem(w io.Writer) error {
	if s.messageItemAdded {
		return nil
	}
	s.messageItemID = "msg_" + s.lastChunkID
	if err := s.emit(w, "response.output_item.added", map[string]interface{}{
		"output_index": s.messageOutputIndex(),
		"item": map[string]interface{}{
			"type":    "message",
			"id":      s.messageItemID,
			"role":    "assistant",
			"status":  "in_progress",
			"content": []interface{}{},
		},
	}); err != nil {
		return err
	}
	s.messageItemAdded = true
	return nil
}

func (s *responsesStreamState) ensureContentPart(w io.Writer) error {
	if s.contentPartAdded {
		return nil
	}
	if err := s.ensureMessageItem(w); err != nil {
		return err
	}
	if err := s.emit(w, "response.content_part.added", map[string]interface{}{
		"item_id":       s.messageItemID,
		"output_index":  s.messageOutputIndex(),
		"content_index": 0,
		"part": map[string]interface{}{
			"type": "output_text",
			"text": "",
		},
	}); err != nil {
		return err
	}
	s.contentPartAdded = true
	return nil
}

func (s *responsesStreamState) ensureToolCallItem(w io.Writer, tcIndex int, tcID, tcName string, extraContent interface{}) (*responsesToolCallState, error) {
	if s.activeToolCalls == nil {
		s.activeToolCalls = make(map[int]*responsesToolCallState)
	}
	state, ok := s.activeToolCalls[tcIndex]
	if !ok {
		state = &responsesToolCallState{
			id:           tcID,
			outputIndex:  s.toolOutputIndex(tcIndex),
			extraContent: extraContent,
		}
		s.activeToolCalls[tcIndex] = state
	}
	if tcID != "" {
		state.id = tcID
	}
	if tcName != "" {
		state.name = tcName
	}
	if extraContent != nil {
		state.extraContent = extraContent
	}
	if state.itemAdded || state.id == "" {
		return state, nil
	}
	item := map[string]interface{}{
		"type":      "function_call",
		"id":        state.id,
		"call_id":   state.id,
		"name":      state.name,
		"arguments": "",
		"status":    "in_progress",
	}
	if state.extraContent != nil {
		item["extra_content"] = state.extraContent
	}
	if err := s.emit(w, "response.output_item.added", map[string]interface{}{
		"output_index": state.outputIndex,
		"item":         item,
	}); err != nil {
		return nil, err
	}
	state.itemAdded = true
	return state, nil
}

func (s *responsesStreamState) finalizeMessageItem(w io.Writer) error {
	if !s.contentPartAdded || s.messageItemDone {
		return nil
	}
	text := s.accumulatedText.String()
	if err := s.emit(w, "response.output_text.done", map[string]interface{}{
		"item_id":       s.messageItemID,
		"output_index":  s.messageOutputIndex(),
		"content_index": 0,
		"text":          text,
	}); err != nil {
		return err
	}
	if err := s.emit(w, "response.output_item.done", map[string]interface{}{
		"output_index": s.messageOutputIndex(),
		"item": map[string]interface{}{
			"type":   "message",
			"id":     s.messageItemID,
			"role":   "assistant",
			"status": "completed",
			"content": []interface{}{
				map[string]interface{}{
					"type": "output_text",
					"text": text,
				},
			},
		},
	}); err != nil {
		return err
	}
	s.messageItemDone = true
	return nil
}

func (s *responsesStreamState) finalizeToolCalls(w io.Writer) error {
	for _, state := range s.activeToolCalls {
		if !state.itemAdded || state.done {
			continue
		}
		args := state.arguments.String()
		if err := s.emit(w, "response.function_call_arguments.done", map[string]interface{}{
			"item_id":      state.id,
			"output_index": state.outputIndex,
			"name":         state.name,
			"arguments":    args,
		}); err != nil {
			return err
		}
		item := map[string]interface{}{
			"type":      "function_call",
			"id":        state.id,
			"call_id":   state.id,
			"name":      state.name,
			"arguments": args,
			"status":    "completed",
		}
		if state.extraContent != nil {
			item["extra_content"] = state.extraContent
		}
		if err := s.emit(w, "response.output_item.done", map[string]interface{}{
			"output_index": state.outputIndex,
			"item":         item,
		}); err != nil {
			return err
		}
		state.done = true
	}
	return nil
}

func (s *responsesStreamState) buildCompletedOutput() []interface{} {
	var output []interface{}
	if s.messageItemAdded {
		text := s.accumulatedText.String()
		output = append(output, map[string]interface{}{
			"type":   "message",
			"id":     s.messageItemID,
			"role":   "assistant",
			"status": "completed",
			"content": []interface{}{
				map[string]interface{}{
					"type": "output_text",
					"text": text,
				},
			},
		})
	}
	for _, state := range s.activeToolCalls {
		if !state.itemAdded {
			continue
		}
		item := map[string]interface{}{
			"type":      "function_call",
			"id":        state.id,
			"call_id":   state.id,
			"name":      state.name,
			"arguments": state.arguments.String(),
			"status":    "completed",
		}
		if state.extraContent != nil {
			item["extra_content"] = state.extraContent
		}
		output = append(output, item)
	}
	return output
}

func (s *responsesStreamState) finalize(w io.Writer) error {
	if !s.responseCreated {
		return nil
	}
	if err := s.finalizeMessageItem(w); err != nil {
		return err
	}
	if err := s.finalizeToolCalls(w); err != nil {
		return err
	}
	return s.emit(w, "response.completed", map[string]interface{}{
		"response": s.buildResponse("completed", s.buildCompletedOutput()),
	})
}

// TranslateOpenAIStreamToResponses parses OpenAI Chat Completions stream and writes OpenAI Responses API SSE stream.
func TranslateOpenAIStreamToResponses(r io.Reader, w io.Writer, debug bool) error {
	var targetWriter io.Writer = w
	var dw *debugWriter
	if debug {
		dw = &debugWriter{w: w}
		targetWriter = dw
	}

	scanner := bufio.NewScanner(r)
	// Use a 10MB buffer to handle extremely large lines (e.g. nested images or large thought blocks)
	buf := make([]byte, 0, 1024*1024)
	scanner.Buffer(buf, 10*1024*1024)
	flusher, hasFlush := w.(http.Flusher)

	state := &responsesStreamState{}

	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			if line == "" {
				fmt.Fprint(targetWriter, "\n")
				if hasFlush {
					flusher.Flush()
				}
			}
			continue
		}

		dataPayload := strings.TrimPrefix(line, "data: ")
		if dataPayload == "[DONE]" {
			break
		}

		var chunk OpenAIMessageChunk
		if err := json.Unmarshal([]byte(dataPayload), &chunk); err != nil {
			log.Printf("failed to parse OpenAI chunk: %v", err)
			continue
		}

		state.lastChunkID = chunk.ID
		state.lastChunkModel = chunk.Model

		if len(chunk.Choices) == 0 {
			continue
		}

		if err := state.ensureCreated(targetWriter); err != nil {
			return err
		}

		choice := chunk.Choices[0]

		if choice.Delta.Content != "" {
			if err := state.ensureContentPart(targetWriter); err != nil {
				return err
			}
			state.accumulatedText.WriteString(choice.Delta.Content)
			if err := state.emit(targetWriter, "response.output_text.delta", map[string]interface{}{
				"item_id":       state.messageItemID,
				"output_index":  state.messageOutputIndex(),
				"content_index": 0,
				"delta":         choice.Delta.Content,
			}); err != nil {
				return err
			}
		}

		for _, tc := range choice.Delta.ToolCalls {
			if err := state.ensureCreated(targetWriter); err != nil {
				return err
			}
			toolState, err := state.ensureToolCallItem(targetWriter, tc.Index, tc.ID, tc.Function.Name, tc.ExtraContent)
			if err != nil {
				return err
			}
			if tc.Function.Arguments != "" {
				toolState.arguments.WriteString(tc.Function.Arguments)
				if err := state.emit(targetWriter, "response.function_call_arguments.delta", map[string]interface{}{
					"item_id":      toolState.id,
					"output_index": toolState.outputIndex,
					"delta":        tc.Function.Arguments,
				}); err != nil {
					return err
				}
			}
		}

		if hasFlush {
			flusher.Flush()
		}
	}

	if err := scanner.Err(); err != nil {
		log.Printf("scanner error during OpenAI to Responses stream: %v", err)
		return fmt.Errorf("scanner error: %w", err)
	}

	if err := state.finalize(targetWriter); err != nil {
		return err
	}

	if debug && dw != nil {
		log.Printf("[DEBUG] Responses Response Body (Stream): %s", dw.debugBuf.String())
	}

	if hasFlush {
		flusher.Flush()
	}
	return nil
}

func writeSSEEvent(w io.Writer, event string, data interface{}) error {
	payload, err := marshalNoEscape(data)
	if err != nil {
		return err
	}
	if _, err := fmt.Fprintf(w, "event: %s\ndata: %s\n\n", event, string(payload)); err != nil {
		return err
	}
	return nil
}
