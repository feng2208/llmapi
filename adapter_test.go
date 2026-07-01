package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

func TestTranslateClaudeRequestToOpenAI(t *testing.T) {
	req := &ClaudeMessagesRequest{
		Model:  "gemini-flash-lite",
		System: "You are a helpful assistant.",
		Messages: []ClaudeMessage{
			{
				Role: "user",
				Content: []interface{}{
					map[string]interface{}{
						"type": "text",
						"text": "Analyze this image:",
					},
					map[string]interface{}{
						"type": "image",
						"source": map[string]interface{}{
							"type":       "base64",
							"media_type": "image/jpeg",
							"data":       "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=",
						},
					},
				},
			},
		},
		Tools: []ClaudeTool{
			{
				Name:        "get_weather",
				Description: "Get the current weather",
				InputSchema: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"location": map[string]interface{}{
							"type": "string",
						},
					},
					"required": []interface{}{"location"},
				},
			},
		},
		MaxTokens:   100,
		Temperature: float64Ptr(0.7),
	}

	openaiReq, err := TranslateClaudeRequestToOpenAI(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Verify model
	if openaiReq["model"] != "gemini-flash-lite" {
		t.Errorf("expected model gemini-flash-lite, got %v", openaiReq["model"])
	}

	// Verify system message
	msgs, ok := openaiReq["messages"].([]interface{})
	if !ok || len(msgs) != 2 {
		t.Fatalf("expected 2 messages, got %v", msgs)
	}

	sysMsg := msgs[0].(map[string]interface{})
	if sysMsg["role"] != "system" || sysMsg["content"] != "You are a helpful assistant." {
		t.Errorf("invalid system message: %v", sysMsg)
	}

	// Verify user message with text and image_url
	userMsg := msgs[1].(map[string]interface{})
	if userMsg["role"] != "user" {
		t.Errorf("expected role user, got %v", userMsg["role"])
	}
	contentParts, ok := userMsg["content"].([]interface{})
	if !ok || len(contentParts) != 2 {
		t.Fatalf("expected 2 content parts, got %v", userMsg["content"])
	}

	p1 := contentParts[0].(map[string]interface{})
	if p1["type"] != "text" || p1["text"] != "Analyze this image:" {
		t.Errorf("invalid text part: %v", p1)
	}

	p2 := contentParts[1].(map[string]interface{})
	imgURL, ok := p2["image_url"].(map[string]interface{})
	if !ok || p2["type"] != "image_url" {
		t.Errorf("invalid image part: %v", p2)
	}
	if !strings.HasPrefix(imgURL["url"].(string), "data:image/jpeg;base64,") {
		t.Errorf("invalid image_url format: %v", imgURL["url"])
	}

	// Verify tools
	tools, ok := openaiReq["tools"].([]interface{})
	if !ok || len(tools) != 1 {
		t.Fatalf("expected 1 tool, got %v", openaiReq["tools"])
	}
	toolFn := tools[0].(map[string]interface{})["function"].(map[string]interface{})
	if toolFn["name"] != "get_weather" {
		t.Errorf("expected get_weather tool, got %v", toolFn["name"])
	}

	// Verify max_tokens, temperature
	if openaiReq["max_tokens"] != 100 {
		t.Errorf("expected max_tokens 100, got %v", openaiReq["max_tokens"])
	}
	if openaiReq["temperature"] != 0.7 {
		t.Errorf("expected temperature 0.7, got %v", openaiReq["temperature"])
	}
}

func TestTranslateOpenAIResponseToClaude(t *testing.T) {
	openaiResp := map[string]interface{}{
		"id":    "chatcmpl-123",
		"model": "gemini-2.5-flash-lite",
		"choices": []interface{}{
			map[string]interface{}{
				"index": 0,
				"message": map[string]interface{}{
					"role":    "assistant",
					"content": "Hello! I called the tool.",
					"tool_calls": []interface{}{
						map[string]interface{}{
							"id":   "call_abc",
							"type": "function",
							"function": map[string]interface{}{
								"name":      "get_weather",
								"arguments": `{"location":"Beijing"}`,
							},
						},
					},
				},
				"finish_reason": "tool_calls",
			},
		},
		"usage": map[string]interface{}{
			"prompt_tokens":     15.0,
			"completion_tokens": 20.0,
		},
	}

	claudeResp, err := TranslateOpenAIResponseToClaude(openaiResp)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if claudeResp.ID != "msg_chatcmpl-123" {
		t.Errorf("expected ID msg_chatcmpl-123, got %v", claudeResp.ID)
	}
	if claudeResp.Model != "gemini-2.5-flash-lite" {
		t.Errorf("expected model, got %v", claudeResp.Model)
	}
	if claudeResp.StopReason == nil || *claudeResp.StopReason != "tool_use" {
		t.Errorf("expected stop_reason tool_use, got %v", claudeResp.StopReason)
	}
	if claudeResp.StopSequence != nil {
		t.Errorf("expected stop_sequence nil, got %v", claudeResp.StopSequence)
	}
	if len(claudeResp.Content) != 2 {
		t.Fatalf("expected 2 content blocks, got %d", len(claudeResp.Content))
	}

	c1 := claudeResp.Content[0]
	if c1.Type != "text" || c1.Text != "Hello! I called the tool." {
		t.Errorf("invalid text block: %v", c1)
	}

	c2 := claudeResp.Content[1]
	if c2.Type != "tool_use" || c2.ID != "call_abc" || c2.Name != "get_weather" {
		t.Errorf("invalid tool block: %v", c2)
	}
	argsMap, ok := c2.Input.(map[string]interface{})
	if !ok || argsMap["location"] != "Beijing" {
		t.Errorf("invalid tool input: %v", c2.Input)
	}

	if claudeResp.Usage.InputTokens != 15 || claudeResp.Usage.OutputTokens != 20 {
		t.Errorf("invalid usage: %v", claudeResp.Usage)
	}

	// Verify JSON serialization includes stop_sequence as null
	jsonBytes, err := json.Marshal(claudeResp)
	if err != nil {
		t.Fatalf("failed to marshal response: %v", err)
	}
	jsonStr := string(jsonBytes)
	if !strings.Contains(jsonStr, `"stop_sequence":null`) {
		t.Errorf("expected stop_sequence:null in JSON, got: %s", jsonStr)
	}
	if !strings.Contains(jsonStr, `"stop_reason":"tool_use"`) {
		t.Errorf("expected stop_reason in JSON, got: %s", jsonStr)
	}
}

func TestTranslateOpenAIResponseToResponses_WithThoughts(t *testing.T) {
	openaiResp := map[string]interface{}{
		"id":    "resp-123",
		"model": "gemini-2.5-flash-lite",
		"choices": []interface{}{
			map[string]interface{}{
				"index": 0,
				"message": map[string]interface{}{
					"role":    "assistant",
					"content": "<thought>\n**Calculating volume of gold for Pluto layer**\n\nStarting with the approximation...\n</thought>\nPluto has a lot of gold.",
				},
				"finish_reason": "stop",
			},
		},
		"usage": map[string]interface{}{
			"prompt_tokens":     12.0,
			"completion_tokens": 18.0,
		},
	}

	responsesResp, err := TranslateOpenAIResponseToResponses(openaiResp)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if responsesResp.ID != "resp-123" {
		t.Errorf("expected ID resp-123, got %s", responsesResp.ID)
	}
	if len(responsesResp.Output) != 2 {
		t.Fatalf("expected 2 output items, got %d", len(responsesResp.Output))
	}

	// 1. Verify reasoning item
	reasoningItem, ok := responsesResp.Output[0].(map[string]interface{})
	if !ok || reasoningItem["type"] != "reasoning" {
		t.Errorf("expected first item to be reasoning, got %v", responsesResp.Output[0])
	}
	if reasoningItem["id"] != "rs_resp-123" {
		t.Errorf("expected reasoning ID rs_resp-123, got %v", reasoningItem["id"])
	}
	content, ok := reasoningItem["content"].([]interface{})
	if !ok || len(content) != 1 {
		t.Fatalf("expected 1 content part, got %v", reasoningItem["content"])
	}
	contentPart, ok := content[0].(map[string]interface{})
	if !ok || contentPart["type"] != "reasoning_text" || contentPart["text"] != "**Calculating volume of gold for Pluto layer**\n\nStarting with the approximation..." {
		t.Errorf("invalid reasoning text: %v", contentPart)
	}

	// 2. Verify message item
	msg, ok := responsesResp.Output[1].(OpenAIOutputMessage)
	if !ok || msg.Role != "assistant" || msg.Type != "message" {
		t.Errorf("invalid message output: %v", responsesResp.Output[1])
	}
	if msg.ID != "msg_resp-123" {
		t.Errorf("expected message ID msg_resp-123, got %s", msg.ID)
	}
	if len(msg.Content) != 1 {
		t.Fatalf("expected 1 output part, got %d", len(msg.Content))
	}
	part := msg.Content[0]
	if part.Type != "output_text" || part.Text != "Pluto has a lot of gold." {
		t.Errorf("invalid text part: %v", part)
	}

	// 3. Verify usage
	if responsesResp.Usage.InputTokens != 12 || responsesResp.Usage.OutputTokens != 18 || responsesResp.Usage.TotalTokens != 30 {
		t.Errorf("invalid usage: %+v", responsesResp.Usage)
	}
}

func TestTranslateOpenAIStreamToClaude(t *testing.T) {
	streamInput := `data: {"id":"chatcmpl-123","model":"gemini-2.5-flash-lite","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"},"finish_reason":null}]}
data: {"id":"chatcmpl-123","model":"gemini-2.5-flash-lite","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":null}]}
data: {"id":"chatcmpl-123","model":"gemini-2.5-flash-lite","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_abc","type":"function","function":{"name":"get_weather","arguments":""}}]},"finish_reason":null}]}
data: {"id":"chatcmpl-123","model":"gemini-2.5-flash-lite","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"location\":\"Beijing\"}"}}]},"finish_reason":null}]}
data: {"id":"chatcmpl-123","model":"gemini-2.5-flash-lite","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":10,"completion_tokens":25}}
data: [DONE]
`
	var outputBuf bytes.Buffer
	err := TranslateOpenAIStreamToClaude(strings.NewReader(streamInput), &outputBuf, false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	outputStr := outputBuf.String()

	// Verify event presence
	events := []string{
		"event: message_start",
		"event: content_block_start",
		"event: content_block_delta",
		"event: content_block_stop",
		"event: message_delta",
		"event: message_stop",
	}

	for _, ev := range events {
		if !strings.Contains(outputStr, ev) {
			t.Errorf("missing expected event: %s in output:\n%s", ev, outputStr)
		}
	}

	if !strings.Contains(outputStr, "Hello") || !strings.Contains(outputStr, " world") {
		t.Errorf("missing content tokens in output:\n%s", outputStr)
	}
	if !strings.Contains(outputStr, "call_abc") || !strings.Contains(outputStr, "get_weather") {
		t.Errorf("missing tool call details in output:\n%s", outputStr)
	}
	if !strings.Contains(outputStr, `"stop_sequence":null`) {
		t.Errorf("missing stop_sequence:null in message_start event:\n%s", outputStr)
	}
}

func TestTranslateOpenAIStreamToClaude_ToolCallOnly(t *testing.T) {
	// Tool call without any preceding text content
	streamInput := `data: {"id":"chatcmpl-456","model":"gpt-4","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_xyz","type":"function","function":{"name":"search","arguments":""}}]},"finish_reason":null}]}
data: {"id":"chatcmpl-456","model":"gpt-4","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"q\":\"hello\"}"}}]},"finish_reason":null}]}
data: {"id":"chatcmpl-456","model":"gpt-4","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}
data: [DONE]
`
	var outputBuf bytes.Buffer
	err := TranslateOpenAIStreamToClaude(strings.NewReader(streamInput), &outputBuf, false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	outputStr := outputBuf.String()

	// Tool call should be at index 0 (no text block preceding it)
	if !strings.Contains(outputStr, `"index":0`) {
		t.Errorf("tool call block should be at index 0:\n%s", outputStr)
	}
	if !strings.Contains(outputStr, `"call_xyz"`) {
		t.Errorf("missing tool call ID:\n%s", outputStr)
	}
	if !strings.Contains(outputStr, `"search"`) {
		t.Errorf("missing tool function name:\n%s", outputStr)
	}
	if !strings.Contains(outputStr, `"stop_reason":"tool_use"`) {
		t.Errorf("missing stop_reason tool_use:\n%s", outputStr)
	}
}

func TestTranslateResponsesRequestToOpenAI(t *testing.T) {
	req := &OpenAIResponsesRequest{
		Model:        "gemini-flash-lite",
		Instructions: "You are a translator.",
		Input: []interface{}{
			map[string]interface{}{
				"role":    "user",
				"content": "Hi",
			},
		},
		Stream: true,
	}

	openaiReq, err := TranslateResponsesRequestToOpenAI(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if openaiReq["model"] != "gemini-flash-lite" {
		t.Errorf("expected model, got %v", openaiReq["model"])
	}

	msgs := openaiReq["messages"].([]interface{})
	if len(msgs) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(msgs))
	}

	m0 := msgs[0].(map[string]interface{})
	if m0["role"] != "system" || m0["content"] != "You are a translator." {
		t.Errorf("invalid system message: %v", m0)
	}

	m1 := msgs[1].(map[string]interface{})
	if m1["role"] != "user" || m1["content"] != "Hi" {
		t.Errorf("invalid user message: %v", m1)
	}
}

func TestTranslateResponsesRequestToOpenAIFunctionCallOutput(t *testing.T) {
	req := &OpenAIResponsesRequest{
		Model: "gemini-flash-lite",
		Input: []interface{}{
			map[string]interface{}{
				"type":    "function_call_output",
				"call_id": "call_abc",
				"output":  "Beijing is sunny",
			},
		},
	}
	openaiReq, err := TranslateResponsesRequestToOpenAI(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	msgs := openaiReq["messages"].([]interface{})
	if len(msgs) != 1 {
		t.Fatalf("expected 1 message, got %d", len(msgs))
	}
	m := msgs[0].(map[string]interface{})
	if m["role"] != "tool" || m["tool_call_id"] != "call_abc" || m["content"] != "Beijing is sunny" {
		t.Errorf("invalid tool message: %v", m)
	}
}

func TestTranslateClaudeToolResult(t *testing.T) {
	req := &ClaudeMessagesRequest{
		Model: "gemini-flash-lite",
		Messages: []ClaudeMessage{
			{
				Role: "user",
				Content: []interface{}{
					map[string]interface{}{
						"type":        "tool_result",
						"tool_use_id": "call_abc",
						"is_error":    true,
						"content":     "Failed to get weather",
					},
				},
			},
		},
	}
	openaiReq, err := TranslateClaudeRequestToOpenAI(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	msgs := openaiReq["messages"].([]interface{})
	if len(msgs) != 1 {
		t.Fatalf("expected 1 message, got %d", len(msgs))
	}
	m := msgs[0].(map[string]interface{})
	if m["role"] != "tool" || m["tool_call_id"] != "call_abc" || m["content"] != "[ERROR] Failed to get weather" {
		t.Errorf("invalid tool message: %v", m)
	}
}

func TestTranslateOpenAIResponseToResponses(t *testing.T) {
	openaiResp := map[string]interface{}{
		"id":    "resp-123",
		"model": "gemini-2.5-flash-lite",
		"choices": []interface{}{
			map[string]interface{}{
				"index": 0,
				"message": map[string]interface{}{
					"role":    "assistant",
					"content": "Responses API response content",
				},
				"finish_reason": "stop",
			},
		},
		"usage": map[string]interface{}{
			"prompt_tokens":     12.0,
			"completion_tokens": 18.0,
		},
	}

	responsesResp, err := TranslateOpenAIResponseToResponses(openaiResp)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if responsesResp.ID != "resp-123" {
		t.Errorf("expected ID resp-123, got %s", responsesResp.ID)
	}
	if len(responsesResp.Output) != 1 {
		t.Fatalf("expected 1 output message, got %d", len(responsesResp.Output))
	}
	msg, ok := responsesResp.Output[0].(OpenAIOutputMessage)
	if !ok || msg.Role != "assistant" || msg.Type != "message" {
		t.Errorf("invalid message output: %v", responsesResp.Output[0])
	}
	if msg.ID != "msg_resp-123" {
		t.Errorf("expected message ID msg_resp-123, got %s", msg.ID)
	}
	if len(msg.Content) != 1 {
		t.Fatalf("expected 1 output part, got %d", len(msg.Content))
	}
	part := msg.Content[0]
	if part.Type != "output_text" || part.Text != "Responses API response content" {
		t.Errorf("invalid text part: %v", part)
	}
	if responsesResp.Usage.InputTokens != 12 || responsesResp.Usage.OutputTokens != 18 || responsesResp.Usage.TotalTokens != 30 {
		t.Errorf("invalid usage: %+v", responsesResp.Usage)
	}
}

func TestTranslateOpenAIStreamToResponses(t *testing.T) {
	streamInput := `data: {"id":"resp-123","model":"gemini-2.5-flash-lite","choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}]}
data: {"id":"resp-123","model":"gemini-2.5-flash-lite","choices":[{"index":0,"delta":{"content":" there"},"finish_reason":null}]}
data: [DONE]
`
	var outputBuf bytes.Buffer
	err := TranslateOpenAIStreamToResponses(strings.NewReader(streamInput), &outputBuf, false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	outputStr := outputBuf.String()

	expectedEvents := []string{
		"event: response.created",
		"event: response.in_progress",
		"event: response.output_item.added",
		"event: response.content_part.added",
		"event: response.output_text.delta",
		"event: response.output_text.done",
		"event: response.output_item.done",
		"event: response.completed",
	}
	for _, event := range expectedEvents {
		if !strings.Contains(outputStr, event) {
			t.Errorf("missing %s event: %s", event, outputStr)
		}
	}

	if strings.Contains(outputStr, "response.output_item.part.delta") {
		t.Errorf("should not contain deprecated event type: %s", outputStr)
	}
	if strings.Contains(outputStr, "response.done") {
		t.Errorf("should not contain deprecated event type: %s", outputStr)
	}
	if !strings.Contains(outputStr, `"delta":"Hi"`) || !strings.Contains(outputStr, `"delta":" there"`) {
		t.Errorf("missing expected text deltas: %s", outputStr)
	}
	if !strings.Contains(outputStr, `"text":"Hi there"`) {
		t.Errorf("missing finalized text: %s", outputStr)
	}
}

func TestTranslateOpenAIStreamToResponses_ToolCalls(t *testing.T) {
	streamInput := `data: {"id":"resp-456","model":"gpt-4","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_abc","type":"function","function":{"name":"get_weather","arguments":""}}]},"finish_reason":null}]}
data: {"id":"resp-456","model":"gpt-4","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"loc"}}]},"finish_reason":null}]}
data: {"id":"resp-456","model":"gpt-4","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\":\"NYC\"}"}}]},"finish_reason":"tool_calls"}]}
data: [DONE]
`
	var outputBuf bytes.Buffer
	err := TranslateOpenAIStreamToResponses(strings.NewReader(streamInput), &outputBuf, false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	outputStr := outputBuf.String()

	expectedEvents := []string{
		"event: response.created",
		"event: response.output_item.added",
		"event: response.function_call_arguments.delta",
		"event: response.function_call_arguments.done",
		"event: response.output_item.done",
		"event: response.completed",
	}
	for _, event := range expectedEvents {
		if !strings.Contains(outputStr, event) {
			t.Errorf("missing %s event: %s", event, outputStr)
		}
	}
	if !strings.Contains(outputStr, `"name":"get_weather"`) {
		t.Errorf("missing function name: %s", outputStr)
	}
	if !strings.Contains(outputStr, `"arguments":"{\"loc\":\"NYC\"}"`) {
		t.Errorf("missing finalized arguments: %s", outputStr)
	}
}

func TestTranslateResponsesRequestToOpenAI_InputText(t *testing.T) {
	req := &OpenAIResponsesRequest{
		Model: "gemini-flash-lite",
		Input: []interface{}{
			map[string]interface{}{
				"role": "user",
				"content": []interface{}{
					map[string]interface{}{
						"type": "input_text",
						"text": "Hello there",
					},
				},
			},
		},
	}
	openaiReq, err := TranslateResponsesRequestToOpenAI(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	msgs := openaiReq["messages"].([]interface{})
	if len(msgs) != 1 {
		t.Fatalf("expected 1 message, got %d", len(msgs))
	}
	m0 := msgs[0].(map[string]interface{})
	contentParts := m0["content"].([]interface{})
	if len(contentParts) != 1 {
		t.Fatalf("expected 1 content part, got %d", len(contentParts))
	}
	p0 := contentParts[0].(map[string]interface{})
	if p0["type"] != "text" || p0["text"] != "Hello there" {
		t.Errorf("expected text part, got %v", p0)
	}

	reqNoRole := &OpenAIResponsesRequest{
		Model: "gemini-flash-lite",
		Input: []interface{}{
			map[string]interface{}{
				"type": "input_text",
				"text": "Hello there without role",
			},
		},
	}
	openaiReq2, err := TranslateResponsesRequestToOpenAI(reqNoRole)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	msgs2 := openaiReq2["messages"].([]interface{})
	m0_2 := msgs2[0].(map[string]interface{})
	contentParts2 := m0_2["content"].([]interface{})
	p0_2 := contentParts2[0].(map[string]interface{})
	if p0_2["type"] != "text" || p0_2["text"] != "Hello there without role" {
		t.Errorf("expected text part, got %v", p0_2)
	}
}

func TestTranslateResponsesRequestToOpenAI_OutputText(t *testing.T) {
	req := &OpenAIResponsesRequest{
		Model: "gemini-flash-lite",
		Input: []interface{}{
			map[string]interface{}{
				"role": "assistant",
				"content": []interface{}{
					map[string]interface{}{
						"type": "output_text",
						"text": "This is an assistant response",
					},
				},
			},
		},
	}
	openaiReq, err := TranslateResponsesRequestToOpenAI(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	msgs := openaiReq["messages"].([]interface{})
	if len(msgs) != 1 {
		t.Fatalf("expected 1 message, got %d", len(msgs))
	}
	m0 := msgs[0].(map[string]interface{})
	contentParts := m0["content"].([]interface{})
	if len(contentParts) != 1 {
		t.Fatalf("expected 1 content part, got %d", len(contentParts))
	}
	p0 := contentParts[0].(map[string]interface{})
	if p0["type"] != "text" || p0["text"] != "This is an assistant response" {
		t.Errorf("expected text part mapped from output_text, got %v", p0)
	}

	reqNoRole := &OpenAIResponsesRequest{
		Model: "gemini-flash-lite",
		Input: []interface{}{
			map[string]interface{}{
				"type": "output_text",
				"text": "Hello there without role as output_text",
			},
		},
	}
	openaiReq2, err := TranslateResponsesRequestToOpenAI(reqNoRole)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	msgs2 := openaiReq2["messages"].([]interface{})
	m0_2 := msgs2[0].(map[string]interface{})
	contentParts2 := m0_2["content"].([]interface{})
	p0_2 := contentParts2[0].(map[string]interface{})
	if p0_2["type"] != "text" || p0_2["text"] != "Hello there without role as output_text" {
		t.Errorf("expected text part mapped from output_text, got %v", p0_2)
	}
}

func TestTranslateResponsesRequestToOpenAIFunctionCall(t *testing.T) {
	req := &OpenAIResponsesRequest{
		Model: "gemini-flash-lite",
		Input: []interface{}{
			map[string]interface{}{
				"type":      "function_call",
				"call_id":   "call_abc",
				"name":      "get_weather",
				"arguments": `{"location":"Beijing"}`,
			},
		},
	}
	openaiReq, err := TranslateResponsesRequestToOpenAI(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	msgs := openaiReq["messages"].([]interface{})
	if len(msgs) != 1 {
		t.Fatalf("expected 1 message, got %d", len(msgs))
	}
	m := msgs[0].(map[string]interface{})
	if m["role"] != "assistant" {
		t.Errorf("expected role assistant, got %v", m["role"])
	}
	toolCalls, ok := m["tool_calls"].([]interface{})
	if !ok || len(toolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %v", m["tool_calls"])
	}
	tc := toolCalls[0].(map[string]interface{})
	if tc["id"] != "call_abc" || tc["type"] != "function" {
		t.Errorf("invalid tool call format: %v", tc)
	}
	fn := tc["function"].(map[string]interface{})
	if fn["name"] != "get_weather" || fn["arguments"] != `{"location":"Beijing"}` {
		t.Errorf("invalid function call details: %v", fn)
	}
}

func TestTranslateResponsesRequestToOpenAIFunctionCallWithExtraContent(t *testing.T) {
	req := &OpenAIResponsesRequest{
		Model: "gemini-flash-lite",
		Input: []interface{}{
			map[string]interface{}{
				"type":      "function_call",
				"call_id":   "call_abc",
				"name":      "get_weather",
				"arguments": `{"location":"Beijing"}`,
				"extra_content": map[string]interface{}{
					"google": map[string]interface{}{
						"thought_signature": "sig_123",
					},
				},
			},
		},
	}
	openaiReq, err := TranslateResponsesRequestToOpenAI(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	msgs := openaiReq["messages"].([]interface{})
	if len(msgs) != 1 {
		t.Fatalf("expected 1 message, got %d", len(msgs))
	}
	m := msgs[0].(map[string]interface{})
	toolCalls, ok := m["tool_calls"].([]interface{})
	if !ok || len(toolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %v", m["tool_calls"])
	}
	tc := toolCalls[0].(map[string]interface{})
	ec, ok := tc["extra_content"].(map[string]interface{})
	if !ok {
		t.Fatalf("missing or invalid extra_content in tool call: %v", tc)
	}
	google := ec["google"].(map[string]interface{})
	if google["thought_signature"] != "sig_123" {
		t.Errorf("expected thought_signature sig_123, got %v", google["thought_signature"])
	}
}

func TestTranslateOpenAIResponseToResponsesWithExtraContent(t *testing.T) {
	openaiResp := map[string]interface{}{
		"id": "chatcmpl-123",
		"choices": []interface{}{
			map[string]interface{}{
				"message": map[string]interface{}{
					"role": "assistant",
					"tool_calls": []interface{}{
						map[string]interface{}{
							"id":   "call_abc",
							"type": "function",
							"function": map[string]interface{}{
								"name":      "get_weather",
								"arguments": `{"location":"Beijing"}`,
							},
							"extra_content": map[string]interface{}{
								"google": map[string]interface{}{
									"thought_signature": "sig_123",
								},
							},
						},
					},
				},
			},
		},
	}
	resp, err := TranslateOpenAIResponseToResponses(openaiResp)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(resp.Output) != 1 {
		t.Fatalf("expected 1 output message, got %d", len(resp.Output))
	}
	m, ok := resp.Output[0].(OpenAIOutputMessage)
	if !ok || len(m.Content) != 1 {
		t.Fatalf("expected 1 part, got %d", len(m.Content))
	}
	part := m.Content[0]
	if part.Type != "function_call" {
		t.Errorf("expected type function_call, got %s", part.Type)
	}
	ec, ok := part.ExtraContent.(map[string]interface{})
	if !ok {
		t.Fatalf("missing or invalid extra_content in part: %v", part)
	}
	google := ec["google"].(map[string]interface{})
	if google["thought_signature"] != "sig_123" {
		t.Errorf("expected thought_signature sig_123, got %v", google["thought_signature"])
	}
}

func TestTranslateOpenAIStreamToResponsesWithExtraContent(t *testing.T) {
	streamInput := `data: {"id":"resp-456","model":"gpt-4","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_abc","type":"function","function":{"name":"get_weather","arguments":""},"extra_content":{"google":{"thought_signature":"sig_123"}}}]},"finish_reason":null}]}
data: {"id":"resp-456","model":"gpt-4","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{}"}}]},"finish_reason":"tool_calls"}]}
data: [DONE]
`
	var outputBuf bytes.Buffer
	err := TranslateOpenAIStreamToResponses(strings.NewReader(streamInput), &outputBuf, false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	outputStr := outputBuf.String()
	if !strings.Contains(outputStr, `"thought_signature":"sig_123"`) {
		t.Errorf("missing thought_signature in SSE events: %s", outputStr)
	}
}

func TestRewriteBodyGeminiThoughtSignatureInjection(t *testing.T) {
	pr := &ProxyRouter{}
	body := map[string]interface{}{
		"messages": []interface{}{
			map[string]interface{}{
				"role":    "user",
				"content": "Hello",
			},
			map[string]interface{}{
				"role": "assistant",
				"tool_calls": []interface{}{
					map[string]interface{}{
						"id":   "call_123",
						"type": "function",
						"function": map[string]interface{}{
							"name":      "test_tool",
							"arguments": "{}",
						},
					},
				},
			},
		},
	}
	pConfig := ModelProviderConfig{
		Name:     "gemini",
		Model:    "gemini-2.5-flash",
		Upstream: "https://generativelanguage.googleapis.com",
	}

	newBytes, err := pr.rewriteBody(body, pConfig)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var newBody map[string]interface{}
	if err := json.Unmarshal(newBytes, &newBody); err != nil {
		t.Fatalf("failed to unmarshal rewritten body: %v", err)
	}

	msgs := newBody["messages"].([]interface{})
	assistantMsg := msgs[1].(map[string]interface{})
	toolCalls := assistantMsg["tool_calls"].([]interface{})
	tc := toolCalls[0].(map[string]interface{})

	ec, ok := tc["extra_content"].(map[string]interface{})
	if !ok {
		t.Fatalf("extra_content missing or not a map: %v", tc)
	}
	google, ok := ec["google"].(map[string]interface{})
	if !ok {
		t.Fatalf("google missing or not a map: %v", ec)
	}
	if google["thought_signature"] != "c2tpcF90aG91Z2h0X3NpZ25hdHVyZV92YWxpZGF0b3I=" {
		t.Errorf("expected thought_signature skip sentinel, got %v", google["thought_signature"])
	}
	if google["thoughtSignature"] != "c2tpcF90aG91Z2h0X3NpZ25hdHVyZV92YWxpZGF0b3I=" {
		t.Errorf("expected thoughtSignature skip sentinel, got %v", google["thoughtSignature"])
	}
}

func TestTranslateClaudeRequestToOpenAI_SystemClaudeContent(t *testing.T) {
	req := &ClaudeMessagesRequest{
		Model: "gemini-flash-lite",
		System: []ClaudeContent{
			{
				Type: "text",
				Text: "System Instruction Part 1.",
			},
			{
				Type: "text",
				Text: " System Instruction Part 2.",
			},
		},
		Messages: []ClaudeMessage{
			{
				Role:    "user",
				Content: "Hello",
			},
		},
	}
	openaiReq, err := TranslateClaudeRequestToOpenAI(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	msgs := openaiReq["messages"].([]interface{})
	if len(msgs) != 2 {
		t.Fatalf("expected 2 messages, got %v", msgs)
	}

	sysMsg := msgs[0].(map[string]interface{})
	if sysMsg["role"] != "system" || sysMsg["content"] != "System Instruction Part 1. System Instruction Part 2." {
		t.Errorf("invalid system message: %v", sysMsg)
	}
}

func TestTranslateClaudeRequestToOpenAI_MessagesClaudeContent(t *testing.T) {
	req := &ClaudeMessagesRequest{
		Model: "gemini-flash-lite",
		Messages: []ClaudeMessage{
			{
				Role: "user",
				Content: []ClaudeContent{
					{
						Type: "text",
						Text: "Hello there",
					},
				},
			},
		},
	}
	openaiReq, err := TranslateClaudeRequestToOpenAI(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	msgs := openaiReq["messages"].([]interface{})
	if len(msgs) != 1 {
		t.Fatalf("expected 1 message, got %v", msgs)
	}

	userMsg := msgs[0].(map[string]interface{})
	contentParts := userMsg["content"].([]interface{})
	if len(contentParts) != 1 {
		t.Fatalf("expected 1 content part, got %v", contentParts)
	}
	part := contentParts[0].(map[string]interface{})
	if part["type"] != "text" || part["text"] != "Hello there" {
		t.Errorf("invalid part content: %v", part)
	}
}

func TestTranslateClaudeRequestToOpenAI_ToolResultMixedWithText(t *testing.T) {
	req := &ClaudeMessagesRequest{
		Model: "gemini-flash-lite",
		Messages: []ClaudeMessage{
			{
				Role: "user",
				Content: []interface{}{
					map[string]interface{}{
						"type": "text",
						"text": "And here is the answer: ",
					},
					map[string]interface{}{
						"type":        "tool_result",
						"tool_use_id": "call_abc",
						"content":     "The result is 42",
					},
				},
			},
		},
	}
	openaiReq, err := TranslateClaudeRequestToOpenAI(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	msgs := openaiReq["messages"].([]interface{})
	if len(msgs) != 2 {
		t.Fatalf("expected 2 messages, got %v", msgs)
	}

	// Sibling 0 should be the tool message
	m0 := msgs[0].(map[string]interface{})
	if m0["role"] != "tool" || m0["tool_call_id"] != "call_abc" || m0["content"] != "The result is 42" {
		t.Errorf("invalid tool message: %v", m0)
	}

	// Sibling 1 should be the user text part message (appended after tool message)
	m1 := msgs[1].(map[string]interface{})
	if m1["role"] != "user" {
		t.Errorf("expected role user for second message, got %v", m1["role"])
	}
	contentParts, ok := m1["content"].([]interface{})
	if !ok || len(contentParts) != 1 {
		t.Fatalf("expected 1 content part in user message, got %v", m1["content"])
	}
	part := contentParts[0].(map[string]interface{})
	if part["type"] != "text" || part["text"] != "And here is the answer: " {
		t.Errorf("invalid part content: %v", part)
	}
}

func TestTranslateClaudeRequestToOpenAI_NilToolContent(t *testing.T) {
	req := &ClaudeMessagesRequest{
		Model: "gemini-flash-lite",
		Messages: []ClaudeMessage{
			{
				Role: "user",
				Content: []interface{}{
					map[string]interface{}{
						"type":        "tool_result",
						"tool_use_id": "call_abc",
						"content":     nil,
					},
				},
			},
		},
	}
	openaiReq, err := TranslateClaudeRequestToOpenAI(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	msgs := openaiReq["messages"].([]interface{})
	if len(msgs) != 1 {
		t.Fatalf("expected 1 message, got %v", msgs)
	}

	m0 := msgs[0].(map[string]interface{})
	if m0["role"] != "tool" || m0["content"] != "" {
		t.Errorf("invalid tool message content: expected empty string, got %v", m0["content"])
	}
}

func float64Ptr(v float64) *float64 {
	return &v
}

func TestSanitizeToolName(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{"valid_name_1", "valid_name_1"},
		{"1_invalid_start", "_1_invalid_start"},
		{"-invalid-start", "_-invalid-start"},
		{"with-dashes-and:colons", "with-dashes-and:colons"},
		{"with spaces", "with_spaces"},
		{"with!@#symbols", "with___symbols"},
		{"with.dots", "with.dots"},
		{"", ""},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			if got := sanitizeToolName(tt.input); got != tt.expected {
				t.Errorf("sanitizeToolName(%q) = %q; want %q", tt.input, got, tt.expected)
			}
		})
	}
}

func TestTranslateOpenAIStreamToResponses_WithThoughts(t *testing.T) {
	// Simulate split tags across chunks
	streamInput := `data: {"id":"resp-123","model":"gpt-4","choices":[{"index":0,"delta":{"content":"Prefix <tho"},"finish_reason":null}]}
data: {"id":"resp-123","model":"gpt-4","choices":[{"index":0,"delta":{"content":"ught>Thinking</thou"},"finish_reason":null}]}
data: {"id":"resp-123","model":"gpt-4","choices":[{"index":0,"delta":{"content":"ght>Suffix"},"finish_reason":null}]}
data: [DONE]
`
	var outputBuf bytes.Buffer
	err := TranslateOpenAIStreamToResponses(strings.NewReader(streamInput), &outputBuf, false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	outputStr := outputBuf.String()
	// t.Logf("Output: %s", outputStr)

	// Verify Reasoning Events
	if !strings.Contains(outputStr, "response.output_item.added") || !strings.Contains(outputStr, `"type":"reasoning"`) {
		t.Errorf("missing reasoning item added event")
	}
	if !strings.Contains(outputStr, "response.reasoning_text.delta") || !strings.Contains(outputStr, `"delta":"Thinking"`) {
		t.Errorf("missing reasoning text delta")
	}
	if !strings.Contains(outputStr, "response.reasoning_text.done") {
		t.Errorf("missing reasoning text done")
	}

	// Verify Message Events
	if !strings.Contains(outputStr, `"type":"output_text"`) {
		t.Errorf("missing content part added for message")
	}
	if !strings.Contains(outputStr, `"delta":"Prefix "`) {
		t.Errorf("missing message text delta 'Prefix '")
	}
	if !strings.Contains(outputStr, `"delta":"Suffix"`) {
		t.Errorf("missing message text delta 'Suffix'")
	}

	// Verify final response object in response.completed
	if !strings.Contains(outputStr, `"status":"completed"`) {
		t.Errorf("missing response.completed event")
	}
	// Verify total output contains both reasoning and message
	if strings.Count(outputStr, `"type":"reasoning"`) < 2 { // once in added, once in done/completed
		t.Errorf("reasoning item not fully accounted for in output")
	}
	if strings.Count(outputStr, `"type":"message"`) < 2 {
		t.Errorf("message item not fully accounted for in output")
	}
}

func TestTranslateOpenAIResponseToOpenAI_WithThoughts(t *testing.T) {
	openaiResp := map[string]interface{}{
		"choices": []interface{}{
			map[string]interface{}{
				"message": map[string]interface{}{
					"role":    "assistant",
					"content": "<thought>I am thinking.</thought>Hello world!",
				},
			},
		},
	}

	translated, err := TranslateOpenAIResponseToOpenAI(openaiResp)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	resp := translated.(map[string]interface{})
	choices := resp["choices"].([]interface{})
	msg := choices[0].(map[string]interface{})["message"].(map[string]interface{})

	if msg["reasoning"] != "I am thinking." {
		t.Errorf("expected reasoning 'I am thinking.', got %v", msg["reasoning"])
	}
	if msg["content"] != "Hello world!" {
		t.Errorf("expected content 'Hello world!', got %v", msg["content"])
	}
}

func TestTranslateOpenAIStreamToOpenAI_WithThoughts(t *testing.T) {
	streamInput := `data: {"id":"chatcmpl-123","choices":[{"index":0,"delta":{"content":"Prefix <tho"},"finish_reason":null}]}
data: {"id":"chatcmpl-123","choices":[{"index":0,"delta":{"content":"ught>Thinking</thou"},"finish_reason":null}]}
data: {"id":"chatcmpl-123","choices":[{"index":0,"delta":{"content":"ght>Suffix"},"finish_reason":null}]}
data: [DONE]
`
	var outputBuf bytes.Buffer
	err := TranslateOpenAIStreamToOpenAI(strings.NewReader(streamInput), &outputBuf, false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	outputStr := outputBuf.String()

	if !strings.Contains(outputStr, `"content":"Prefix "`) {
		t.Errorf("missing content 'Prefix '")
	}
	if !strings.Contains(outputStr, `"reasoning":"Thinking"`) {
		t.Errorf("missing reasoning 'Thinking'")
	}
	if !strings.Contains(outputStr, `"content":"Suffix"`) {
		t.Errorf("missing content 'Suffix'")
	}
	if !strings.Contains(outputStr, "data: [DONE]") {
		t.Errorf("missing DONE data")
	}
}

