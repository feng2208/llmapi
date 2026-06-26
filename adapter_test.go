package main

import (
	"bytes"
	"strings"
	"testing"
)

func TestTranslateClaudeRequestToOpenAI(t *testing.T) {
	req := &ClaudeMessagesRequest{
		Model: "gemini-flash-lite",
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

	if claudeResp.ID != "chatcmpl-123" {
		t.Errorf("expected ID chatcmpl-123, got %v", claudeResp.ID)
	}
	if claudeResp.Model != "gemini-2.5-flash-lite" {
		t.Errorf("expected model, got %v", claudeResp.Model)
	}
	if claudeResp.StopReason != "tool_use" {
		t.Errorf("expected stop_reason tool_use, got %v", claudeResp.StopReason)
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
				"type": "function_call_output",
				"call_id": "call_abc",
				"output": "Beijing is sunny",
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
	msg := responsesResp.Output[0]
	if msg.Role != "assistant" || msg.Type != "message" {
		t.Errorf("invalid message output: %v", msg)
	}
	if len(msg.Content) != 1 {
		t.Fatalf("expected 1 output part, got %d", len(msg.Content))
	}
	part := msg.Content[0]
	if part.Type != "text" || part.Text != "Responses API response content" {
		t.Errorf("invalid text part: %v", part)
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
				"type": "function_call",
				"call_id": "call_abc",
				"name": "get_weather",
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

func float64Ptr(v float64) *float64 {
	return &v
}
