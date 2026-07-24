package main

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"strings"
	"testing"
	"time"
)

// 1. Test Config Loading
func TestLoadConfig(t *testing.T) {
	// Use the config-template.yaml in workspace to test
	cfg, err := LoadConfig("config-template.yaml")
	if err != nil {
		t.Fatalf("Failed to load config template: %v", err)
	}

	if cfg.Listen != "0.0.0.0:3000" {
		t.Errorf("Expected listen '0.0.0.0:3000', got %q", cfg.Listen)
	}

	if cfg.MaxBodySize != 10485760 {
		t.Errorf("Expected MaxBodySize 10485760, got %d", cfg.MaxBodySize)
	}

	if len(cfg.Models) == 0 {
		t.Fatal("Expected at least one model, got 0")
	}

	if cfg.Models[0].Name != "gemini-flash-lite" {
		t.Errorf("Expected first model to be gemini-flash-lite, got %q", cfg.Models[0].Name)
	}

	// Verify request_body configurations
	modelProv := cfg.Models[0].Providers[0]
	if len(modelProv.RequestHeaders.Delete) != 1 || modelProv.RequestHeaders.Delete[0] != "bad_header" {
		t.Errorf("Expected request_headers.delete to contain ['bad_header'], got %v", modelProv.RequestHeaders.Delete)
	}
	if len(modelProv.RequestHeaders.Extra) != 1 || modelProv.RequestHeaders.Extra[0]["Cookie"] != "mycookie=hello" {
		t.Errorf("Expected request_headers.extra to contain Cookie, got %v", modelProv.RequestHeaders.Extra)
	}

	if len(modelProv.RequestBody.Delete) == 0 {
		t.Error("Expected delete config to have items")
	}
	if len(modelProv.RequestBody.Extra) == 0 {
		t.Error("Expected extra config to have items")
	}

	if modelProv.ReasoningStart != "<thought>" || modelProv.ReasoningEnd != "</thought>" {
		t.Errorf("Expected reasoning tags '<thought>' and '</thought>', got '%s' and '%s'", modelProv.ReasoningStart, modelProv.ReasoningEnd)
	}
}

// 2. Test StateManager (Rate Limiting and 429 Lock)
func TestStateManager(t *testing.T) {
	sm := NewStateManager()

	// Test Client Rate Limiting
	// Limiter with rate of 60 per minute = 1 per second
	// Burst is 60.
	if !sm.AllowClient("client1", 60.0) {
		t.Error("Client first request should be allowed")
	}

	// Test Upstream 429 locking
	key := "test-key-1"
	if sm.IsUpstreamLocked(key) {
		t.Error("Key should not be locked initially")
	}

	// First 429: lock for 1 minute
	sm.RecordUpstreamResult(key, 429)
	if !sm.IsUpstreamLocked(key) {
		t.Error("Key should be locked after first 429")
	}

	// Simulate lock expiration
	sm.mu.Lock()
	sm.backoffStates[key].lockedUntil = time.Now().Add(-1 * time.Second) // Force expired lock
	sm.mu.Unlock()

	if sm.IsUpstreamLocked(key) {
		t.Error("Key should be unlocked after lock expiration")
	}

	sm.RecordUpstreamResult(key, 200)
	sm.mu.RLock()
	consec := sm.backoffStates[key].consecutive429s
	sm.mu.RUnlock()
	if consec != 0 {
		t.Errorf("Expected consecutive 429s to reset to 0, got %d", consec)
	}

	// 3 consecutive 429s -> lock until next UTC 00:00
	sm.RecordUpstreamResult(key, 429)
	sm.RecordUpstreamResult(key, 429)
	sm.RecordUpstreamResult(key, 429)

	if !sm.IsUpstreamLocked(key) {
		t.Error("Key should be locked after consecutive 429s")
	}

	sm.mu.RLock()
	lockedUntil := sm.backoffStates[key].lockedUntil
	sm.mu.RUnlock()

	now := time.Now().UTC()
	expectedMidnight := time.Date(now.Year(), now.Month(), now.Day()+1, 0, 0, 0, 0, time.UTC)
	// They should be equal within 2 second range to avoid tiny timing variations
	if lockedUntil.Sub(expectedMidnight).Abs() > 2*time.Second {
		t.Errorf("Expected lockedUntil to be around %v, got %v", expectedMidnight, lockedUntil)
	}
}

// 3. Test Routing and Load Balancing (Sequential providers, Round Robin keys)
func TestRouter(t *testing.T) {
	// Create a custom config for testing
	cfg := &Config{
		Models: []ModelConfig{
			{
				Name: "test-model",
				Providers: []ModelProviderConfig{
					{
						Name:  "prov-1",
						Model: "upstream-model-1",
					},
					{
						Name:  "prov-2",
						Model: "upstream-model-2",
					},
				},
			},
		},
		Providers: []ProviderConf{
			{
				Name:      "prov-1",
				RateLimit: 60.0,
				AuthKeys:  []string{"key-1-a", "key-1-b"},
			},
			{
				Name:      "prov-2",
				RateLimit: 60.0,
				AuthKeys:  []string{"key-2-a"},
			},
		},
	}

	sm := NewStateManager()
	router := NewRouter(cfg, sm)

	// Step 3a: Select route first time. It should pick a valid key of prov-1.
	route, err := router.SelectRoute("test-model")
	if err != nil {
		t.Fatalf("Failed to select route: %v", err)
	}
	if route.ModelProvider.Name != "prov-1" {
		t.Errorf("Expected prov-1, got %s", route.ModelProvider.Name)
	}
	if route.AuthKey != "key-1-a" && route.AuthKey != "key-1-b" {
		t.Errorf("Expected route.AuthKey to be either key-1-a or key-1-b, got %s", route.AuthKey)
	}

	// Step 3b: Select route second time. Since it is random, it should still pick a valid key of prov-1.
	route2, err := router.SelectRoute("test-model")
	if err != nil {
		t.Fatalf("Failed to select route second time: %v", err)
	}
	if route2.ModelProvider.Name != "prov-1" {
		t.Errorf("Expected prov-1, got %s", route2.ModelProvider.Name)
	}
	if route2.AuthKey != "key-1-a" && route2.AuthKey != "key-1-b" {
		t.Errorf("Expected route2.AuthKey to be either key-1-a or key-1-b, got %s", route2.AuthKey)
	}

	// Step 3c: Disable all keys of prov-1 (simulate lock)
	sm.RecordUpstreamResult("key-1-a", 429)
	sm.RecordUpstreamResult("key-1-b", 429)

	// Selecting route now should fallback to prov-2 because all keys of prov-1 are locked
	route3, err := router.SelectRoute("test-model")
	if err != nil {
		t.Fatalf("Failed to fallback to prov-2: %v", err)
	}
	if route3.ModelProvider.Name != "prov-2" {
		t.Errorf("Expected fallback to prov-2, got %s", route3.ModelProvider.Name)
	}
	if route3.AuthKey != "key-2-a" {
		t.Errorf("Expected key-2-a, got %s", route3.AuthKey)
	}
}

// 4. Test Request Body Modification
func TestModifyRequestBody(t *testing.T) {
	rawJSON := []byte(`{
		"model": "client-model-name",
		"reasoning_effort": "high",
		"temperature": 0.7,
		"messages": [
			{"role": "user", "content": "hello"},
			{
				"role": "assistant",
				"tool_calls": [
					{
						"id": "call-1",
						"type": "function",
						"function": {"name": "test"}
					}
				]
			}
		]
	}`)

	routeGemini := &SelectedRoute{
		ModelProvider: &ModelProviderConfig{
			Name:  "gemini",
			Model: "upstream-gemini",
			RequestBody: RequestBodyConfig{
				Delete: []interface{}{
					"reasoning_effort",
				},
				Extra: []map[string]interface{}{
					{
						"extra_field_1": "value1",
						"extra_body": map[string]interface{}{
							"google": map[string]interface{}{
								"thinking_config": map[string]interface{}{
									"thinking_level":   "high",
									"include_thoughts": true,
								},
							},
						},
					},
				},
			},
		},
	}

	// 1. Test with Gemini provider (bypass should be injected)
	modifiedGemini, _, err := ModifyRequestBody(rawJSON, routeGemini, nil, nil, nil)
	if err != nil {
		t.Fatalf("Failed to modify body for gemini: %v", err)
	}

	var parsedGemini map[string]interface{}
	if err := json.Unmarshal(modifiedGemini, &parsedGemini); err != nil {
		t.Fatalf("Modified gemini body is invalid JSON: %v", err)
	}

	// Verify model replace
	if parsedGemini["model"] != "upstream-gemini" {
		t.Errorf("Expected model upstream-gemini, got %v", parsedGemini["model"])
	}

	// Verify delete
	if _, exists := parsedGemini["reasoning_effort"]; exists {
		t.Error("Expected reasoning_effort to be deleted, but it exists")
	}

	// Verify existing untouched fields
	if parsedGemini["temperature"] != 0.7 {
		t.Errorf("Expected temperature to remain 0.7, got %v", parsedGemini["temperature"])
	}

	// Verify extra fields
	if parsedGemini["extra_field_1"] != "value1" {
		t.Errorf("Expected extra_field_1 to be value1, got %v", parsedGemini["extra_field_1"])
	}

	// Verify deep merged fields
	extraBody, ok := parsedGemini["extra_body"].(map[string]interface{})
	if !ok {
		t.Fatal("Expected extra_body to be a map")
	}
	google, ok := extraBody["google"].(map[string]interface{})
	if !ok {
		t.Fatal("Expected extra_body.google to be a map")
	}
	thinkingConfig, ok := google["thinking_config"].(map[string]interface{})
	if !ok {
		t.Fatal("Expected google.thinking_config to be a map")
	}
	if thinkingConfig["thinking_level"] != "high" || thinkingConfig["include_thoughts"] != true {
		t.Errorf("Deep merge values incorrect: %v", thinkingConfig)
	}

	// Verify tool_call signature bypass injection (gemini channel only)
	messagesGemini, ok := parsedGemini["messages"].([]interface{})
	if !ok || len(messagesGemini) < 2 {
		t.Fatal("Expected messages list with at least 2 items")
	}
	assistantMsgGemini, ok := messagesGemini[1].(map[string]interface{})
	if !ok {
		t.Fatal("Expected second message to be a map")
	}
	toolCallsGemini, ok := assistantMsgGemini["tool_calls"].([]interface{})
	if !ok || len(toolCallsGemini) == 0 {
		t.Fatal("Expected tool_calls list")
	}
	callGemini, ok := toolCallsGemini[0].(map[string]interface{})
	if !ok {
		t.Fatal("Expected tool call to be a map")
	}
	extraContent, ok := callGemini["extra_content"].(map[string]interface{})
	if !ok {
		t.Fatal("Expected extra_content map")
	}
	googleContent, ok := extraContent["google"].(map[string]interface{})
	if !ok {
		t.Fatal("Expected google sub-map")
	}
	if googleContent["thought_signature"] != "skip_thought_signature_validator" {
		t.Errorf("Expected thought_signature skip_thought_signature_validator, got %v", googleContent["thought_signature"])
	}

	// 2. Test with Non-Gemini provider (bypass should NOT be injected)
	routeOther := &SelectedRoute{
		ModelProvider: &ModelProviderConfig{
			Name:  "openai",
			Model: "gpt-4o",
		},
	}

	modifiedOther, _, err := ModifyRequestBody(rawJSON, routeOther, nil, nil, nil)
	if err != nil {
		t.Fatalf("Failed to modify body for other: %v", err)
	}

	var parsedOther map[string]interface{}
	if err := json.Unmarshal(modifiedOther, &parsedOther); err != nil {
		t.Fatalf("Modified other body is invalid JSON: %v", err)
	}

	messagesOther, ok := parsedOther["messages"].([]interface{})
	if !ok || len(messagesOther) < 2 {
		t.Fatal("Expected messages list with at least 2 items")
	}
	assistantMsgOther, ok := messagesOther[1].(map[string]interface{})
	if !ok {
		t.Fatal("Expected second message to be a map")
	}
	toolCallsOther, ok := assistantMsgOther["tool_calls"].([]interface{})
	if !ok || len(toolCallsOther) == 0 {
		t.Fatal("Expected tool_calls list")
	}
	callOther, ok := toolCallsOther[0].(map[string]interface{})
	if !ok {
		t.Fatal("Expected tool call to be a map")
	}
	if _, exists := callOther["extra_content"]; exists {
		t.Error("Expected extra_content to NOT exist for non-gemini provider")
	}
}

// 5. Test JSON non-streaming reasoning extraction
func TestProcessJSONResponse(t *testing.T) {
	rawJSON := []byte(`{
		"choices": [{
			"message": {
				"role": "assistant",
				"content": "Before solving this, <thought>I am reasoning step-by-step</thought>the answer is 42."
			}
		}]
	}`)

	modified := ProcessJSONResponse(rawJSON, "<thought>", "</thought>")
	var obj map[string]interface{}
	if err := json.Unmarshal(modified, &obj); err != nil {
		t.Fatalf("Invalid JSON returned: %v", err)
	}

	choices := obj["choices"].([]interface{})
	choice := choices[0].(map[string]interface{})
	message := choice["message"].(map[string]interface{})

	content := message["content"].(string)
	reasoning := message["reasoning"].(string)
	reasoningContent := message["reasoning_content"].(string)

	if content != "Before solving this, the answer is 42." {
		t.Errorf("Expected stripped content, got %q", content)
	}
	if reasoning != "I am reasoning step-by-step" {
		t.Errorf("Expected reasoning field to contain the steps, got %q", reasoning)
	}
	if reasoningContent != "I am reasoning step-by-step" {
		t.Errorf("Expected reasoning_content to contain the steps, got %q", reasoningContent)
	}
}

// 6. Test Streaming reasoning extraction (SSE state machine)
func TestProcessSSELine(t *testing.T) {
	extractor := NewStreamExtractor("<thought>", "</thought>")

	// Simulation of consecutive incoming stream lines
	lines := [][]byte{
		[]byte("data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n"),
		[]byte("data: {\"choices\":[{\"delta\":{\"content\":\"Hello, <th\"}}]}\n"),
		[]byte("data: {\"choices\":[{\"delta\":{\"content\":\"ought>I think therefore \"}}]}\n"),
		[]byte("data: {\"choices\":[{\"delta\":{\"content\":\"I am.</th\"}}]}\n"),
		[]byte("data: {\"choices\":[{\"delta\":{\"content\":\"ought> Nice.\"}}]}\n"),
	}

	var results []string
	for _, l := range lines {
		out := extractor.ProcessSSELine(l)
		results = append(results, string(out))
	}

	// Verify delta content is correctly split in downstream output
	// Expected parts:
	// Part 1: "Hello, " as content
	// Part 2: "I think therefore I am." as reasoning/reasoning_content
	// Part 3: " Nice." as content

	var combinedContent string
	var combinedReasoning string

	for _, r := range results {
		if !strings.HasPrefix(r, "data: ") {
			continue
		}
		trimmed := strings.TrimPrefix(r, "data: ")
		trimmed = strings.TrimSuffix(trimmed, "\n")
		trimmed = strings.TrimSuffix(trimmed, "\r")

		var obj map[string]interface{}
		if err := json.Unmarshal([]byte(trimmed), &obj); err != nil {
			continue
		}
		choices := obj["choices"].([]interface{})
		choice := choices[0].(map[string]interface{})
		delta := choice["delta"].(map[string]interface{})

		if c, exists := delta["content"]; exists {
			combinedContent += c.(string)
		}
		if rContent, exists := delta["reasoning"]; exists {
			combinedReasoning += rContent.(string)
		}
	}

	if combinedContent != "Hello,  Nice." {
		t.Errorf("Expected combinedContent 'Hello,  Nice.', got %q", combinedContent)
	}

	if combinedReasoning != "I think therefore I am." {
		t.Errorf("Expected combinedReasoning 'I think therefore I am.', got %q", combinedReasoning)
	}
}

// 7. Test Gemini Request and Response Translation
func TestGeminiTranslation(t *testing.T) {
	// A helper route representing gemini api_type
	route := &SelectedRoute{
		ModelProvider: &ModelProviderConfig{
			Name:    "gemini-provider",
			Model:   "gemini-2.5-flash",
			ApiType: "gemini",
			RequestBody: RequestBodyConfig{
				Extra: []map[string]interface{}{
					{
						"generationConfig": map[string]interface{}{
							"thinkingConfig": map[string]interface{}{
								"thinkingBudget": 1024,
							},
						},
					},
				},
			},
		},
	}

	// 7.1 Test Request Translation
	openAIReq := map[string]interface{}{
		"model": "gpt-4o",
		"messages": []interface{}{
			map[string]interface{}{
				"role":    "system",
				"content": "You are a helpful assistant.",
			},
			map[string]interface{}{
				"role": "user",
				"content": []interface{}{
					map[string]interface{}{
						"type": "text",
						"text": "What is in this image and what is the temperature in London?",
					},
					map[string]interface{}{
						"type": "image_url",
						"image_url": map[string]interface{}{
							"url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
						},
					},
				},
			},
			map[string]interface{}{
				"role": "assistant",
				"tool_calls": []interface{}{
					map[string]interface{}{
						"id":   "call_xyz123",
						"type": "function",
						"function": map[string]interface{}{
							"name":      "get_current_temperature",
							"arguments": `{"location":"London"}`,
						},
						"extra_content": map[string]interface{}{
							"google": map[string]interface{}{
								"thought_signature": "CuoCAY89a19w6WWC6i",
							},
						},
					},
				},
			},
			map[string]interface{}{
				"role":         "tool",
				"tool_call_id": "call_xyz123",
				"content":      `{"temperature":"15C"}`,
			},
		},
		"tools": []interface{}{
			map[string]interface{}{
				"type": "function",
				"function": map[string]interface{}{
					"name":        "get_current_temperature",
					"description": "Gets the current temperature.",
					"parameters": map[string]interface{}{
						"$schema": "http://json-schema.org/draft-07/schema#",
						"type":    "object",
						"properties": map[string]interface{}{
							"location": map[string]interface{}{
								"type":                 "string",
								"additionalProperties": false,
							},
						},
						"required":             []interface{}{"location"},
						"additionalProperties": false,
					},
				},
			},
		},
		"tool_choice": "required",
		"temperature": 0.7,
		"max_tokens":  2048,
		"stop":        []interface{}{"\n", "stop_word"},
	}

	reqBytes, err := json.Marshal(openAIReq)
	if err != nil {
		t.Fatalf("Failed to marshal OpenAI request: %v", err)
	}

	translatedBytes, err := TransformRequestToGemini(reqBytes, route, nil, nil, nil)
	if err != nil {
		t.Fatalf("TransformRequestToGemini failed: %v", err)
	}

	var geminiReq map[string]interface{}
	if err := json.Unmarshal(translatedBytes, &geminiReq); err != nil {
		t.Fatalf("Failed to unmarshal translated Gemini request: %v", err)
	}

	// Verify System Instruction
	sysInst, ok := geminiReq["systemInstruction"].(map[string]interface{})
	if !ok {
		t.Errorf("Expected systemInstruction to be set")
	} else {
		parts, _ := sysInst["parts"].([]interface{})
		if len(parts) != 1 || parts[0].(map[string]interface{})["text"] != "You are a helpful assistant." {
			t.Errorf("System instruction text not mapped correctly: %v", parts)
		}
	}

	// Verify Contents & Roles
	contents, ok := geminiReq["contents"].([]interface{})
	if !ok || len(contents) != 3 {
		t.Fatalf("Expected 3 items in contents, got %v", len(contents))
	}

	// Turn 1: user text & inline image data
	turn1 := contents[0].(map[string]interface{})
	if turn1["role"] != "user" {
		t.Errorf("Expected first role to be user, got %v", turn1["role"])
	}
	parts1 := turn1["parts"].([]interface{})
	if len(parts1) != 2 {
		t.Fatalf("Expected 2 parts in user content, got %d", len(parts1))
	}
	if parts1[1].(map[string]interface{})["text"] != "What is in this image and what is the temperature in London?" {
		t.Errorf("Expected user text part, got %v", parts1[1])
	}
	inlineData := parts1[0].(map[string]interface{})["inline_data"].(map[string]interface{})
	if inlineData["mime_type"] != "image/png" || inlineData["data"] != "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==" {
		t.Errorf("Expected user inline image data, got %v", inlineData)
	}

	// Turn 2: assistant tool call with thoughtSignature
	turn2 := contents[1].(map[string]interface{})
	if turn2["role"] != "model" {
		t.Errorf("Expected second role to be model, got %v", turn2["role"])
	}
	parts2 := turn2["parts"].([]interface{})
	if len(parts2) != 1 {
		t.Fatalf("Expected 1 part in assistant message, got %d", len(parts2))
	}
	fcPart := parts2[0].(map[string]interface{})
	fc := fcPart["functionCall"].(map[string]interface{})
	if fc["name"] != "get_current_temperature" {
		t.Errorf("Expected functionCall name 'get_current_temperature', got %v", fc["name"])
	}
	if fcPart["thoughtSignature"] != "CuoCAY89a19w6WWC6i" {
		t.Errorf("Expected thoughtSignature 'CuoCAY89a19w6WWC6i', got %v", fcPart["thoughtSignature"])
	}

	// Turn 3: tool output
	turn3 := contents[2].(map[string]interface{})
	if turn3["role"] != "user" {
		t.Errorf("Expected third role to be user, got %v", turn3["role"])
	}
	parts3 := turn3["parts"].([]interface{})
	if len(parts3) != 1 {
		t.Fatalf("Expected 1 part in tool message, got %d", len(parts3))
	}
	fr := parts3[0].(map[string]interface{})["functionResponse"].(map[string]interface{})
	if fr["name"] != "get_current_temperature" {
		t.Errorf("Expected functionResponse name 'get_current_temperature', got %v", fr["name"])
	}
	respMap := fr["response"].(map[string]interface{})
	if respMap["temperature"] != "15C" {
		t.Errorf("Expected response temperature '15C', got %v", respMap)
	}

	// Verify Tools (functionDeclarations)
	tools, ok := geminiReq["tools"].([]interface{})
	if !ok || len(tools) != 1 {
		t.Fatalf("Expected 1 tools entry, got %v", tools)
	}
	fd := tools[0].(map[string]interface{})["functionDeclarations"].([]interface{})
	if len(fd) != 1 || fd[0].(map[string]interface{})["name"] != "get_current_temperature" {
		t.Errorf("Expected get_current_temperature function declaration, got %v", fd)
	}
	params := fd[0].(map[string]interface{})["parameters"].(map[string]interface{})
	if params["type"] != "OBJECT" {
		t.Errorf("Expected root parameters type 'OBJECT', got %v", params["type"])
	}
	if _, ok := params["$schema"]; ok {
		t.Errorf("Expected '$schema' to be cleaned from parameters")
	}
	if _, ok := params["additionalProperties"]; ok {
		t.Errorf("Expected 'additionalProperties' to be cleaned from parameters")
	}
	locationProp := params["properties"].(map[string]interface{})["location"].(map[string]interface{})
	if locationProp["type"] != "STRING" {
		t.Errorf("Expected property type 'STRING', got %v", locationProp["type"])
	}
	if _, ok := locationProp["additionalProperties"]; ok {
		t.Errorf("Expected nested 'additionalProperties' to be cleaned from location properties")
	}

	// Verify Tool Choice
	toolConfig, ok := geminiReq["toolConfig"].(map[string]interface{})
	if !ok {
		t.Fatalf("Expected toolConfig to be set")
	}
	if toolConfig["includeServerSideToolInvocations"] != true {
		t.Errorf("Expected includeServerSideToolInvocations to be true, got %v", toolConfig["includeServerSideToolInvocations"])
	}
	fcConfig := toolConfig["functionCallingConfig"].(map[string]interface{})
	if fcConfig["mode"] != "ANY" {
		t.Errorf("Expected mode 'ANY', got %v", fcConfig["mode"])
	}

	// Verify generationConfig & Merge configs
	genConfig, ok := geminiReq["generationConfig"].(map[string]interface{})
	if !ok {
		t.Fatalf("Expected generationConfig to be set")
	}
	if genConfig["temperature"].(float64) != 0.7 {
		t.Errorf("Expected temperature 0.7, got %v", genConfig["temperature"])
	}
	if genConfig["maxOutputTokens"].(float64) != 2048 {
		t.Errorf("Expected maxOutputTokens 2048, got %v", genConfig["maxOutputTokens"])
	}
	stops := genConfig["stopSequences"].([]interface{})
	if len(stops) != 2 || stops[0].(string) != "\n" || stops[1].(string) != "stop_word" {
		t.Errorf("Expected stopSequences [\"\\n\", \"stop_word\"], got %v", stops)
	}
	// Verify custom config merge
	thinkingConfig := genConfig["thinkingConfig"].(map[string]interface{})
	if thinkingConfig["thinkingBudget"].(float64) != 1024 {
		t.Errorf("Expected merged thinkingBudget 1024, got %v", thinkingConfig["thinkingBudget"])
	}

	// 7.2 Test Response Translation
	geminiRespJSON := `{
		"candidates": [
			{
				"content": {
					"role": "model",
					"parts": [
						{
							"text": "Let me think...",
							"thought": true
						},
						{
							"text": "Here is the temperature."
						},
						{
							"functionCall": {
								"name": "get_current_temperature",
								"args": {
									"location": "London"
								}
							},
							"thoughtSignature": "CuoCAY89a19w6WWC6i"
						}
					]
				},
				"finishReason": "STOP"
			}
		],
		"usageMetadata": {
			"promptTokenCount": 36,
			"candidatesTokenCount": 7,
			"totalTokenCount": 117
		}
	}`

	translatedResp, err := ProcessGeminiJSONResponse([]byte(geminiRespJSON), "gemini-2.5-flash")
	if err != nil {
		t.Fatalf("ProcessGeminiJSONResponse failed: %v", err)
	}

	var openAIResp map[string]interface{}
	if err := json.Unmarshal(translatedResp, &openAIResp); err != nil {
		t.Fatalf("Failed to unmarshal translated OpenAI response: %v", err)
	}

	choices := openAIResp["choices"].([]interface{})
	if len(choices) != 1 {
		t.Fatalf("Expected 1 choice, got %d", len(choices))
	}
	choice := choices[0].(map[string]interface{})
	message := choice["message"].(map[string]interface{})

	if message["role"] != "assistant" {
		t.Errorf("Expected role 'assistant', got %v", message["role"])
	}
	if message["content"] != "Here is the temperature." {
		t.Errorf("Expected content 'Here is the temperature.', got %q", message["content"])
	}
	if message["reasoning_content"] != "Let me think..." {
		t.Errorf("Expected reasoning_content 'Let me think...', got %q", message["reasoning_content"])
	}
	if choice["finish_reason"] != "tool_calls" {
		t.Errorf("Expected finish_reason 'tool_calls', got %v", choice["finish_reason"])
	}

	toolCalls := message["tool_calls"].([]interface{})
	if len(toolCalls) != 1 {
		t.Fatalf("Expected 1 tool call, got %d", len(toolCalls))
	}
	tc := toolCalls[0].(map[string]interface{})
	if tc["type"] != "function" {
		t.Errorf("Expected type 'function', got %v", tc["type"])
	}
	fn := tc["function"].(map[string]interface{})
	if fn["name"] != "get_current_temperature" {
		t.Errorf("Expected function name 'get_current_temperature', got %v", fn["name"])
	}
	if fn["arguments"] != `{"location":"London"}` {
		t.Errorf("Expected arguments '{\\\"location\\\":\\\"London\\\"}', got %q", fn["arguments"])
	}
	extra := tc["extra_content"].(map[string]interface{})
	googleSig := extra["google"].(map[string]interface{})["thought_signature"].(string)
	if googleSig != "CuoCAY89a19w6WWC6i" {
		t.Errorf("Expected thought_signature 'CuoCAY89a19w6WWC6i', got %v", googleSig)
	}

	// 7.3 Test SSE Response Translation
	sseLines := [][]byte{
		[]byte(`data: {"candidates": [{"content": {"role": "model","parts": [{"text": "Thinking process...","thought": true}]}}]}` + "\n"),
		[]byte(`data: {"candidates": [{"content": {"role": "model","parts": [{"text": "Hello, world!"}]}}]}` + "\n"),
		[]byte(`data: {"candidates": [{"content": {"role": "model","parts": [{"functionCall": {"name": "test_func","args": {}},"thoughtSignature": "xyz"}]},"finishReason": "STOP"}]}` + "\n"),
	}

	var outputLines []string
	for _, l := range sseLines {
		out := ProcessGeminiSSELine(l, "chatcmpl-test", 12345678, "gemini-2.5-flash")
		if out != nil {
			outputLines = append(outputLines, string(out))
		}
	}

	if len(outputLines) != 3 {
		t.Fatalf("Expected 3 SSE output lines, got %d", len(outputLines))
	}

	// Verify Chunk 1 (Thinking)
	var chunk1 map[string]interface{}
	cleaned1 := strings.TrimPrefix(strings.TrimSpace(outputLines[0]), "data: ")
	if err := json.Unmarshal([]byte(cleaned1), &chunk1); err != nil {
		t.Fatalf("Failed to parse chunk 1: %v", err)
	}
	c1Delta := chunk1["choices"].([]interface{})[0].(map[string]interface{})["delta"].(map[string]interface{})
	if c1Delta["reasoning_content"] != "Thinking process..." {
		t.Errorf("Expected reasoning_content 'Thinking process...', got %v", c1Delta["reasoning_content"])
	}

	// Verify Chunk 2 (Content)
	var chunk2 map[string]interface{}
	cleaned2 := strings.TrimPrefix(strings.TrimSpace(outputLines[1]), "data: ")
	if err := json.Unmarshal([]byte(cleaned2), &chunk2); err != nil {
		t.Fatalf("Failed to parse chunk 2: %v", err)
	}
	c2Delta := chunk2["choices"].([]interface{})[0].(map[string]interface{})["delta"].(map[string]interface{})
	if c2Delta["content"] != "Hello, world!" {
		t.Errorf("Expected content 'Hello, world!', got %v", c2Delta["content"])
	}

	// Verify Chunk 3 (Tool Call)
	var chunk3 map[string]interface{}
	cleaned3 := strings.TrimPrefix(strings.TrimSpace(outputLines[2]), "data: ")
	if err := json.Unmarshal([]byte(cleaned3), &chunk3); err != nil {
		t.Fatalf("Failed to parse chunk 3: %v", err)
	}
	c3Choice := chunk3["choices"].([]interface{})[0].(map[string]interface{})
	c3Delta := c3Choice["delta"].(map[string]interface{})
	c3TC := c3Delta["tool_calls"].([]interface{})[0].(map[string]interface{})
	if c3TC["function"].(map[string]interface{})["name"] != "test_func" {
		t.Errorf("Expected tool_call function name 'test_func', got %v", c3TC)
	}
	if c3Choice["finish_reason"] != "tool_calls" {
		t.Errorf("Expected finish_reason 'tool_calls', got %v", c3Choice["finish_reason"])
	}
}

// 8. Test deepMerge recursive merging and map normalization
func TestDeepMergeTypes(t *testing.T) {
	dst := map[string]interface{}{
		"generationConfig": map[string]interface{}{
			"maxOutputTokens": 32000,
			"topP":            0.95,
		},
	}

	src := map[string]interface{}{
		"generationConfig": map[interface{}]interface{}{
			"thinkingConfig": map[interface{}]interface{}{
				"includeThoughts": true,
				"thinkingLevel":   "HIGH",
			},
		},
	}

	deepMerge(dst, src)

	genConfig, ok := dst["generationConfig"].(map[string]interface{})
	if !ok {
		t.Fatalf("Expected generationConfig to be map[string]interface{}")
	}

	// Verify original keys are preserved
	if genConfig["maxOutputTokens"] != 32000 {
		t.Errorf("Expected maxOutputTokens 32000, got %v", genConfig["maxOutputTokens"])
	}
	if genConfig["topP"] != 0.95 {
		t.Errorf("Expected topP 0.95, got %v", genConfig["topP"])
	}

	// Verify merged nested keys are present and converted
	thinkingConfig, ok := genConfig["thinkingConfig"].(map[string]interface{})
	if !ok {
		t.Fatalf("Expected thinkingConfig to be map[string]interface{}, got: %v", genConfig["thinkingConfig"])
	}

	if thinkingConfig["includeThoughts"] != true {
		t.Errorf("Expected includeThoughts true, got %v", thinkingConfig["includeThoughts"])
	}
	if thinkingConfig["thinkingLevel"] != "HIGH" {
		t.Errorf("Expected thinkingLevel 'HIGH', got %v", thinkingConfig["thinkingLevel"])
	}
}

// 10. Test deepMerge slice/array merging
func TestDeepMergeSlices(t *testing.T) {
	dst := map[string]interface{}{
		"tools": []interface{}{
			map[string]interface{}{
				"function": map[string]interface{}{
					"name":        "get_weather",
					"description": "get the weather",
				},
			},
		},
	}

	src := map[string]interface{}{
		"tools": []interface{}{
			map[interface{}]interface{}{
				"google_search": map[interface{}]interface{}{},
			},
			// Duplicate function to test deduplication
			map[interface{}]interface{}{
				"function": map[interface{}]interface{}{
					"name":        "get_weather",
					"description": "get the weather",
				},
			},
		},
	}

	deepMerge(dst, src)

	tools, ok := dst["tools"].([]interface{})
	if !ok {
		t.Fatalf("Expected tools to be a slice")
	}

	// Should have exactly 2 tools (the original get_weather, the new google_search, and the duplicate is deduplicated)
	if len(tools) != 2 {
		t.Fatalf("Expected tools slice length to be 2, got %d. Tools: %v", len(tools), tools)
	}

	tool1, ok := tools[0].(map[string]interface{})
	if !ok {
		t.Fatalf("Expected tool 1 to be a map")
	}
	if fn, ok := tool1["function"].(map[string]interface{}); !ok || fn["name"] != "get_weather" {
		t.Errorf("Expected tool 1 to be get_weather function, got %v", tool1)
	}

	tool2, ok := tools[1].(map[string]interface{})
	if !ok {
		t.Fatalf("Expected tool 2 to be a map")
	}
	if _, exists := tool2["google_search"]; !exists {
		t.Errorf("Expected tool 2 to be google_search map, got %v", tool2)
	}
}

func TestTransformImageRequestToGemini(t *testing.T) {
	route := &SelectedRoute{
		ModelProvider: &ModelProviderConfig{
			Model: "gemini-3.1-flash-image",
			RequestBody: RequestBodyConfig{
				Extra: []map[string]interface{}{
					{
						"generationConfig": map[string]interface{}{
							"thinkingConfig": map[string]interface{}{
								"thinkingLevel": "HIGH",
							},
						},
					},
				},
			},
		},
	}

	reqBody := []byte(`{
		"prompt": "A cute orange cat",
		"n": 2,
		"size": "1024x768"
	}`)

	modified, err := TransformImageRequestToGemini(reqBody, route)
	if err != nil {
		t.Fatalf("Failed to transform: %v", err)
	}

	var m map[string]interface{}
	if err := json.Unmarshal(modified, &m); err != nil {
		t.Fatalf("Modified is not valid JSON: %v", err)
	}

	// Verify prompt
	contents := m["contents"].([]interface{})
	parts := contents[0].(map[string]interface{})["parts"].([]interface{})
	if parts[0].(map[string]interface{})["text"] != "A cute orange cat" {
		t.Errorf("Expected prompt 'A cute orange cat', got %v", parts[0])
	}

	// Verify generationConfig
	genConfig := m["generationConfig"].(map[string]interface{})
	if genConfig["candidateCount"].(float64) != 2 {
		t.Errorf("Expected candidateCount 2, got %v", genConfig["candidateCount"])
	}

	// Verify imageConfig size parsing (1024x768 is 4:3 aspect ratio, 1024 max dim is 1K size)
	imgConfig := genConfig["imageConfig"].(map[string]interface{})
	if imgConfig["aspectRatio"] != "4:3" {
		t.Errorf("Expected aspectRatio '4:3', got %q", imgConfig["aspectRatio"])
	}
	if imgConfig["imageSize"] != "1K" {
		t.Errorf("Expected imageSize '1K', got %q", imgConfig["imageSize"])
	}

	// Verify merged extra
	thinking := genConfig["thinkingConfig"].(map[string]interface{})
	if thinking["thinkingLevel"] != "HIGH" {
		t.Errorf("Expected merged thinkingLevel 'HIGH', got %q", thinking["thinkingLevel"])
	}
}

func TestProcessGeminiImageResponse(t *testing.T) {
	geminiResp := []byte(`{
		"candidates": [
			{
				"content": {
					"parts": [
						{
							"inlineData": {
								"mimeType": "image/png",
								"data": "iVBORw0KGgoAAAANSUhEUgAA"
							}
						}
					]
				}
			}
		],
		"usageMetadata": {
			"promptTokenCount": 3,
			"candidatesTokenCount": 1120,
			"totalTokenCount": 1123
		}
	}`)

	clientReq := []byte(`{
		"prompt": "A cute orange cat",
		"size": "1024x1024",
		"quality": "medium",
		"output_format": "png",
		"background": "opaque"
	}`)

	// 1. Test DEFAULT response format (omitted), which should return 'b64_json'
	respDefault, err := ProcessGeminiImageResponse(geminiResp, clientReq)
	if err != nil {
		t.Fatalf("Failed to process default: %v", err)
	}

	var mDefault map[string]interface{}
	if err := json.Unmarshal(respDefault, &mDefault); err != nil {
		t.Fatalf("Processed default resp is not valid JSON: %v", err)
	}

	if mDefault["background"] != "opaque" {
		t.Errorf("Expected background 'opaque', got %v", mDefault["background"])
	}
	if mDefault["output_format"] != "png" {
		t.Errorf("Expected output_format 'png', got %v", mDefault["output_format"])
	}
	if mDefault["size"] != "1024x1024" {
		t.Errorf("Expected size '1024x1024', got %v", mDefault["size"])
	}
	if mDefault["quality"] != "medium" {
		t.Errorf("Expected quality 'medium', got %v", mDefault["quality"])
	}

	usage := mDefault["usage"].(map[string]interface{})
	if usage["total_tokens"].(float64) != 1123 {
		t.Errorf("Expected usage total_tokens 1123, got %v", usage["total_tokens"])
	}
	if usage["input_tokens"].(float64) != 3 {
		t.Errorf("Expected usage input_tokens 3, got %v", usage["input_tokens"])
	}
	if usage["output_tokens"].(float64) != 1120 {
		t.Errorf("Expected usage output_tokens 1120, got %v", usage["output_tokens"])
	}

	dataDefault := mDefault["data"].([]interface{})
	if len(dataDefault) != 1 {
		t.Fatalf("Expected 1 data item, got %d", len(dataDefault))
	}

	itemDefault := dataDefault[0].(map[string]interface{})
	if itemDefault["b64_json"] != "iVBORw0KGgoAAAANSUhEUgAA" {
		t.Errorf("Expected b64_json 'iVBORw0KGgoAAAANSUhEUgAA', got %q", itemDefault["b64_json"])
	}

	// 2. Test that explicit 'url' client format is IGNORED and returns 'b64_json' anyway
	clientReqURL := []byte(`{
		"response_format": "url"
	}`)
	respURL, err := ProcessGeminiImageResponse(geminiResp, clientReqURL)
	if err != nil {
		t.Fatalf("Failed to process: %v", err)
	}

	var mURL map[string]interface{}
	_ = json.Unmarshal(respURL, &mURL)
	dataURL := mURL["data"].([]interface{})
	itemURL := dataURL[0].(map[string]interface{})
	if itemURL["b64_json"] != "iVBORw0KGgoAAAANSUhEUgAA" {
		t.Errorf("Expected b64_json 'iVBORw0KGgoAAAANSUhEUgAA', got %q", itemURL["b64_json"])
	}
}

func TestModifyImageRequestBody(t *testing.T) {
	// 1. OpenAI-compatible route
	routeOpenAI := &SelectedRoute{
		ModelProvider: &ModelProviderConfig{
			Model: "dall-e-3",
			RequestBody: RequestBodyConfig{
				Delete: []interface{}{"moderation"},
				Extra: []map[string]interface{}{
					{"style": "vivid"},
				},
			},
		},
	}

	clientReq := []byte(`{
		"prompt": "A sunset",
		"moderation": "auto",
		"size": "512x512"
	}`)

	modOpenAI, _, err := ModifyImageRequestBody(clientReq, routeOpenAI, nil)
	if err != nil {
		t.Fatalf("Failed: %v", err)
	}

	var mOpenAI map[string]interface{}
	_ = json.Unmarshal(modOpenAI, &mOpenAI)

	if mOpenAI["model"] != "dall-e-3" {
		t.Errorf("Expected model 'dall-e-3', got %v", mOpenAI["model"])
	}
	if _, exists := mOpenAI["moderation"]; exists {
		t.Error("Expected moderation to be deleted")
	}
	if mOpenAI["style"] != "vivid" {
		t.Errorf("Expected style 'vivid', got %v", mOpenAI["style"])
	}

	// 2. Gemini route
	routeGemini := &SelectedRoute{
		ModelProvider: &ModelProviderConfig{
			Model:   "gemini-3.1-flash-image",
			ApiType: "gemini",
		},
	}

	modGemini, _, err := ModifyImageRequestBody(clientReq, routeGemini, nil)
	if err != nil {
		t.Fatalf("Failed: %v", err)
	}

	var mGemini map[string]interface{}
	_ = json.Unmarshal(modGemini, &mGemini)
	if _, exists := mGemini["contents"]; !exists {
		t.Error("Expected transformed Gemini request with 'contents' field")
	}
}

func TestGeminiTranslationBypass(t *testing.T) {
	route := &SelectedRoute{
		ModelProvider: &ModelProviderConfig{
			Name:    "gemini-provider",
			Model:   "gemini-2.5-flash",
			ApiType: "gemini",
		},
	}

	openAIReq := map[string]interface{}{
		"model": "gpt-4o",
		"messages": []interface{}{
			map[string]interface{}{
				"role": "assistant",
				"tool_calls": []interface{}{
					map[string]interface{}{
						"id":   "call_xyz123",
						"type": "function",
						"function": map[string]interface{}{
							"name":      "get_current_temperature",
							"arguments": `{"location":"London"}`,
						},
					},
				},
			},
		},
	}

	reqBytes, err := json.Marshal(openAIReq)
	if err != nil {
		t.Fatalf("Failed to marshal OpenAI request: %v", err)
	}

	translatedBytes, err := TransformRequestToGemini(reqBytes, route, nil, nil, nil)
	if err != nil {
		t.Fatalf("TransformRequestToGemini failed: %v", err)
	}

	var geminiReq map[string]interface{}
	if err := json.Unmarshal(translatedBytes, &geminiReq); err != nil {
		t.Fatalf("Failed to unmarshal translated Gemini request: %v", err)
	}

	contents, ok := geminiReq["contents"].([]interface{})
	if !ok || len(contents) != 1 {
		t.Fatalf("Expected 1 item in contents, got %v", len(contents))
	}

	turn := contents[0].(map[string]interface{})
	parts := turn["parts"].([]interface{})
	if len(parts) != 1 {
		t.Fatalf("Expected 1 part, got %d", len(parts))
	}

	fcPart := parts[0].(map[string]interface{})
	if fcPart["thoughtSignature"] != "skip_thought_signature_validator" {
		t.Errorf("Expected thoughtSignature 'skip_thought_signature_validator', got %v", fcPart["thoughtSignature"])
	}
}

func TestModifyImageEditRequestBody(t *testing.T) {
	// Construct route for Gemini
	routeGemini := &SelectedRoute{
		ModelProvider: &ModelProviderConfig{
			Name:    "gemini",
			Model:   "gemini-2.5-flash",
			ApiType: "gemini",
			RequestBody: RequestBodyConfig{
				Extra: []map[string]interface{}{
					{"generationConfig": map[string]interface{}{"temperature": 0.5}},
				},
			},
		},
		AuthKey:  "test-api-key",
		KeyIndex: 0,
	}

	// Construct route for OpenAI (non-gemini)
	routeOpenAI := &SelectedRoute{
		ModelProvider: &ModelProviderConfig{
			Name:    "openai",
			Model:   "dall-e-2",
			ApiType: "openai",
			RequestBody: RequestBodyConfig{
				Delete: []interface{}{"n"},
				Extra: []map[string]interface{}{
					{"size": "512x512"},
				},
			},
		},
		AuthKey:  "test-api-key",
		KeyIndex: 0,
	}

	// Create a multipart request body helper
	createMultipartRequest := func(t *testing.T) (*http.Request, []byte) {
		var buf bytes.Buffer
		writer := multipart.NewWriter(&buf)

		_ = writer.WriteField("model", "original-model")
		_ = writer.WriteField("prompt", "a cute dog wearing a hat")
		_ = writer.WriteField("n", "1")
		_ = writer.WriteField("size", "1024x1024")

		part, err := writer.CreateFormFile("image", "dog.png")
		if err != nil {
			t.Fatalf("failed to create image field: %v", err)
		}
		_, _ = part.Write([]byte("fake-png-data"))

		_ = writer.Close()

		req, err := http.NewRequest("POST", "/v1/images/edits", &buf)
		if err != nil {
			t.Fatalf("failed to create request: %v", err)
		}
		req.Header.Set("Content-Type", writer.FormDataContentType())

		return req, buf.Bytes()
	}

	// Test 1: Gemini Translation
	reqGemini, rawBodyGemini := createMultipartRequest(t)
	// We need to parse multipart form first as executeUpstreamRequest does
	err := reqGemini.ParseMultipartForm(32 << 20)
	if err != nil {
		t.Fatalf("failed to parse form: %v", err)
	}

	modifiedGemini, contentTypeGemini, err := ModifyImageEditRequestBody(rawBodyGemini, routeGemini, reqGemini)
	if err != nil {
		t.Fatalf("Gemini ModifyImageEditRequestBody failed: %v", err)
	}

	if contentTypeGemini != "application/json; charset=utf-8" {
		t.Errorf("Expected Content-Type 'application/json; charset=utf-8', got %q", contentTypeGemini)
	}

	var geminiReq map[string]interface{}
	if err := json.Unmarshal(modifiedGemini, &geminiReq); err != nil {
		t.Fatalf("Failed to parse gemini request JSON: %v", err)
	}

	contents, ok := geminiReq["contents"].([]interface{})
	if !ok || len(contents) == 0 {
		t.Fatalf("Invalid contents in Gemini request: %v", geminiReq)
	}
	parts := contents[0].(map[string]interface{})["parts"].([]interface{})
	if len(parts) != 2 {
		t.Fatalf("Expected 2 parts (prompt text and inline image), got %d", len(parts))
	}
	promptText := parts[1].(map[string]interface{})["text"].(string)
	if promptText != "a cute dog wearing a hat" {
		t.Errorf("Expected prompt text 'a cute dog wearing a hat', got %q", promptText)
	}
	inlineData := parts[0].(map[string]interface{})["inline_data"].(map[string]interface{})
	if inlineData["mime_type"] != "image/png" {
		t.Errorf("Expected mime_type 'image/png', got %v", inlineData["mime_type"])
	}

	// Check generationConfig
	genConfig := geminiReq["generationConfig"].(map[string]interface{})
	if genConfig["candidateCount"].(float64) != 1 {
		t.Errorf("Expected candidateCount 1, got %v", genConfig["candidateCount"])
	}
	if genConfig["temperature"].(float64) != 0.5 {
		t.Errorf("Expected temperature 0.5, got %v", genConfig["temperature"])
	}
	imageConfig := genConfig["imageConfig"].(map[string]interface{})
	if imageConfig["aspectRatio"] != "1:1" {
		t.Errorf("Expected aspectRatio 1:1, got %v", imageConfig["aspectRatio"])
	}

	// Test 2: OpenAI (non-gemini) Modification
	reqOpenAI, rawBodyOpenAI := createMultipartRequest(t)
	err = reqOpenAI.ParseMultipartForm(32 << 20)
	if err != nil {
		t.Fatalf("failed to parse form: %v", err)
	}

	modifiedOpenAI, contentTypeOpenAI, err := ModifyImageEditRequestBody(rawBodyOpenAI, routeOpenAI, reqOpenAI)
	if err != nil {
		t.Fatalf("OpenAI ModifyImageEditRequestBody failed: %v", err)
	}

	if !strings.HasPrefix(contentTypeOpenAI, "multipart/form-data; boundary=") {
		t.Errorf("Expected Content-Type starting with 'multipart/form-data; boundary=', got %q", contentTypeOpenAI)
	}

	// Parse modified body back as multipart request
	parsedReq, err := http.NewRequest("POST", "/v1/images/edits", bytes.NewReader(modifiedOpenAI))
	if err != nil {
		t.Fatalf("failed to construct request from modified: %v", err)
	}
	parsedReq.Header.Set("Content-Type", contentTypeOpenAI)
	err = parsedReq.ParseMultipartForm(32 << 20)
	if err != nil {
		t.Fatalf("failed to parse modified multipart form: %v", err)
	}

	if parsedReq.FormValue("model") != "dall-e-2" {
		t.Errorf("Expected model replaced with 'dall-e-2', got %q", parsedReq.FormValue("model"))
	}
	if parsedReq.FormValue("size") != "512x512" {
		t.Errorf("Expected size replaced with extra field '512x512', got %q", parsedReq.FormValue("size"))
	}
	if parsedReq.FormValue("n") != "" {
		t.Errorf("Expected key 'n' deleted, but got %q", parsedReq.FormValue("n"))
	}

	// Verify image file is still present
	imageHeaders := parsedReq.MultipartForm.File["image"]
	if len(imageHeaders) == 0 {
		t.Error("Expected image file header in modified body, got none")
	} else {
		fh := imageHeaders[0]
		file, err := fh.Open()
		if err != nil {
			t.Fatalf("failed to open image: %v", err)
		}
		data, _ := io.ReadAll(file)
		if string(data) != "fake-png-data" {
			t.Errorf("Expected file content 'fake-png-data', got %q", string(data))
		}
	}
}

func TestModifyImageEditRequestBody_MultipleImages(t *testing.T) {
	routeGemini := &SelectedRoute{
		ModelProvider: &ModelProviderConfig{
			Model:   "gemini-1.5-pro",
			ApiType: "gemini",
			RequestBody: RequestBodyConfig{
				Extra: []map[string]interface{}{
					{"temperature": 0.5},
				},
			},
		},
	}

	routeOpenAI := &SelectedRoute{
		ModelProvider: &ModelProviderConfig{
			Model:   "dall-e-2",
			ApiType: "openai",
			RequestBody: RequestBodyConfig{
				Delete: []interface{}{"n"},
				Extra: []map[string]interface{}{
					{"size": "512x512"},
				},
			},
		},
	}

	createMultipartRequestMultiple := func(t *testing.T) (*http.Request, []byte) {
		var buf bytes.Buffer
		writer := multipart.NewWriter(&buf)

		_ = writer.WriteField("model", "original-model")
		_ = writer.WriteField("prompt", "compare these two images")
		_ = writer.WriteField("n", "1")
		_ = writer.WriteField("size", "1024x1024")

		part1, err := writer.CreateFormFile("image[]", "img1.png")
		if err != nil {
			t.Fatalf("failed to create image[] part 1: %v", err)
		}
		_, _ = part1.Write([]byte("fake-png-data-1"))

		part2, err := writer.CreateFormFile("image[]", "img2.png")
		if err != nil {
			t.Fatalf("failed to create image[] part 2: %v", err)
		}
		_, _ = part2.Write([]byte("fake-png-data-2"))

		_ = writer.Close()

		req, err := http.NewRequest("POST", "/v1/images/edits", &buf)
		if err != nil {
			t.Fatalf("failed to create request: %v", err)
		}
		req.Header.Set("Content-Type", writer.FormDataContentType())

		return req, buf.Bytes()
	}

	// Test 1: Gemini Translation with Multiple Images
	reqGemini, rawBodyGemini := createMultipartRequestMultiple(t)
	err := reqGemini.ParseMultipartForm(32 << 20)
	if err != nil {
		t.Fatalf("failed to parse form: %v", err)
	}

	modifiedGemini, contentTypeGemini, err := ModifyImageEditRequestBody(rawBodyGemini, routeGemini, reqGemini)
	if err != nil {
		t.Fatalf("Gemini ModifyImageEditRequestBody failed: %v", err)
	}

	if contentTypeGemini != "application/json; charset=utf-8" {
		t.Errorf("Expected Content-Type 'application/json; charset=utf-8', got %q", contentTypeGemini)
	}

	var geminiReq map[string]interface{}
	if err := json.Unmarshal(modifiedGemini, &geminiReq); err != nil {
		t.Fatalf("Failed to parse gemini request JSON: %v", err)
	}

	contents, ok := geminiReq["contents"].([]interface{})
	if !ok || len(contents) == 0 {
		t.Fatalf("Invalid contents in Gemini request: %v", geminiReq)
	}
	parts := contents[0].(map[string]interface{})["parts"].([]interface{})
	// Expected: 1 text part (prompt) + 2 image parts
	if len(parts) != 3 {
		t.Fatalf("Expected 3 parts (prompt text and 2 inline images), got %d", len(parts))
	}
	promptText := parts[2].(map[string]interface{})["text"].(string)
	if promptText != "compare these two images" {
		t.Errorf("Expected prompt text 'compare these two images', got %q", promptText)
	}

	// Verify image 1
	img1Data := parts[0].(map[string]interface{})["inline_data"].(map[string]interface{})
	if img1Data["mime_type"] != "image/png" {
		t.Errorf("Expected img1 mime_type 'image/png', got %v", img1Data["mime_type"])
	}
	// Verify image 2
	img2Data := parts[1].(map[string]interface{})["inline_data"].(map[string]interface{})
	if img2Data["mime_type"] != "image/png" {
		t.Errorf("Expected img2 mime_type 'image/png', got %v", img2Data["mime_type"])
	}

	// Test 2: OpenAI Modification with Multiple Images
	reqOpenAI, rawBodyOpenAI := createMultipartRequestMultiple(t)
	err = reqOpenAI.ParseMultipartForm(32 << 20)
	if err != nil {
		t.Fatalf("failed to parse form: %v", err)
	}

	modifiedOpenAI, contentTypeOpenAI, err := ModifyImageEditRequestBody(rawBodyOpenAI, routeOpenAI, reqOpenAI)
	if err != nil {
		t.Fatalf("OpenAI ModifyImageEditRequestBody failed: %v", err)
	}

	// Parse modified body back as multipart request
	parsedReq, err := http.NewRequest("POST", "/v1/images/edits", bytes.NewReader(modifiedOpenAI))
	if err != nil {
		t.Fatalf("failed to construct request from modified: %v", err)
	}
	parsedReq.Header.Set("Content-Type", contentTypeOpenAI)
	err = parsedReq.ParseMultipartForm(32 << 20)
	if err != nil {
		t.Fatalf("failed to parse modified multipart form: %v", err)
	}

	// Verify image[] files are still present and both of them exist
	imageHeaders := parsedReq.MultipartForm.File["image[]"]
	if len(imageHeaders) != 2 {
		t.Errorf("Expected 2 image[] file headers in modified body, got %d", len(imageHeaders))
	} else {
		fh1 := imageHeaders[0]
		file1, _ := fh1.Open()
		data1, _ := io.ReadAll(file1)
		file1.Close()
		if string(data1) != "fake-png-data-1" {
			t.Errorf("Expected file 1 content 'fake-png-data-1', got %q", string(data1))
		}

		fh2 := imageHeaders[1]
		file2, _ := fh2.Open()
		data2, _ := io.ReadAll(file2)
		file2.Close()
		if string(data2) != "fake-png-data-2" {
			t.Errorf("Expected file 2 content 'fake-png-data-2', got %q", string(data2))
		}
	}
}

func TestAudioSpeechTranslation(t *testing.T) {
	// Test request translation from OpenAI to Gemini
	openAIReq := []byte(`{
		"model": "tts-1",
		"input": "Hello world from the speech generator!",
		"voice": "alloy",
		"speed": 1.5,
		"response_format": "mp3",
		"instructions": "Speak in a cheerful and positive tone."
	}`)

	route := &SelectedRoute{
		ModelProvider: &ModelProviderConfig{
			Name:    "gemini-provider",
			Model:   "gemini-3.1-flash-tts-preview",
			ApiType: "gemini",
		},
	}

	translated, err := TransformAudioSpeechRequestToGemini(openAIReq, route)
	if err != nil {
		t.Fatalf("TransformAudioSpeechRequestToGemini failed: %v", err)
	}

	var parsed map[string]interface{}
	if err := json.Unmarshal(translated, &parsed); err != nil {
		t.Fatalf("Failed to parse translated request JSON: %v", err)
	}

	// Verify systemInstruction
	sysInst, ok := parsed["systemInstruction"].(map[string]interface{})
	if !ok {
		t.Errorf("Expected systemInstruction to be set")
	} else {
		sysParts := sysInst["parts"].([]interface{})
		if len(sysParts) != 1 || sysParts[0].(map[string]interface{})["text"] != "Speak in a cheerful and positive tone." {
			t.Errorf("Expected systemInstruction 'Speak in a cheerful and positive tone.', got %v", sysParts)
		}
	}

	// Verify input text and speed tag prefix
	contents := parsed["contents"].([]interface{})
	parts := contents[0].(map[string]interface{})["parts"].([]interface{})
	text := parts[0].(map[string]interface{})["text"].(string)
	if !strings.Contains(text, "[fast]") || !strings.Contains(text, "Hello world") {
		t.Errorf("Expected [fast] prefix, got %q", text)
	}

	// Verify voice config mapping (alloy -> Puck)
	genConfig := parsed["generationConfig"].(map[string]interface{})
	modalities := genConfig["responseModalities"].([]interface{})
	if modalities[0].(string) != "AUDIO" {
		t.Errorf("Expected responseModalities [\"AUDIO\"], got %v", modalities)
	}
	speechConfig := genConfig["speechConfig"].(map[string]interface{})
	voiceName := speechConfig["voiceConfig"].(map[string]interface{})["prebuiltVoiceConfig"].(map[string]interface{})["voiceName"].(string)
	if voiceName != "Puck" {
		t.Errorf("Expected mapped voice name 'Puck', got %q", voiceName)
	}

	// Test response translation
	geminiResp := []byte(`{
		"candidates": [
			{
				"content": {
					"parts": [
						{
							"inlineData": {
								"mimeType": "audio/pcm",
								"data": "SGVsbG8="
							}
						}
					]
				}
			}
		]
	}`)

	// SGVsbG8= is base64 for "Hello"
	outWav, err := ProcessGeminiAudioResponse(geminiResp, openAIReq)
	if err != nil {
		t.Fatalf("ProcessGeminiAudioResponse failed: %v", err)
	}

	// It should have WAV header by default since response_format is "mp3"
	if len(outWav) <= 44 {
		t.Errorf("Expected WAV output to be longer than 44 bytes, got %d", len(outWav))
	}
	if string(outWav[0:4]) != "RIFF" {
		t.Errorf("Expected RIFF prefix, got %q", string(outWav[0:4]))
	}
	if string(outWav[44:]) != "Hello" {
		t.Errorf("Expected decrypted content 'Hello', got %q", string(outWav[44:]))
	}

	// If response_format is "pcm", it should return raw PCM data
	pcmReq := []byte(`{
		"model": "tts-1",
		"input": "Hello world",
		"voice": "alloy",
		"response_format": "pcm"
	}`)
	outPCM, err := ProcessGeminiAudioResponse(geminiResp, pcmReq)
	if err != nil {
		t.Fatalf("ProcessGeminiAudioResponse failed with pcm format: %v", err)
	}
	if string(outPCM) != "Hello" {
		t.Errorf("Expected raw PCM data 'Hello', got %q", string(outPCM))
	}
}

func createTestMultipartRequest(t *testing.T, fields map[string]string, fileFieldName, fileName, fileContent string) *http.Request {
	var b bytes.Buffer
	w := multipart.NewWriter(&b)

	for k, v := range fields {
		err := w.WriteField(k, v)
		if err != nil {
			t.Fatalf("Failed to write field %s: %v", k, err)
		}
	}

	if fileFieldName != "" {
		part, err := w.CreateFormFile(fileFieldName, fileName)
		if err != nil {
			t.Fatalf("Failed to create form file %s: %v", fileFieldName, err)
		}
		_, err = part.Write([]byte(fileContent))
		if err != nil {
			t.Fatalf("Failed to write file content: %v", err)
		}
	}

	err := w.Close()
	if err != nil {
		t.Fatalf("Failed to close writer: %v", err)
	}

	req, err := http.NewRequest("POST", "/v1/audio/transcriptions", &b)
	if err != nil {
		t.Fatalf("Failed to create request: %v", err)
	}
	req.Header.Set("Content-Type", w.FormDataContentType())

	return req
}

func TestModifyAudioTranscriptionMultipartBody(t *testing.T) {
	fields := map[string]string{
		"model":           "whisper-1",
		"response_format": "json",
	}
	req := createTestMultipartRequest(t, fields, "file", "test.mp3", "audiobytes")

	err := req.ParseMultipartForm(32 << 20)
	if err != nil {
		t.Fatalf("ParseMultipartForm failed: %v", err)
	}

	route := &SelectedRoute{
		ModelProvider: &ModelProviderConfig{
			Model: "upstream-whisper",
			RequestBody: RequestBodyConfig{
				Delete: []interface{}{"response_format"},
				Extra: []map[string]interface{}{
					{"temperature": 0.5},
				},
			},
		},
	}

	body, contentType, err := ModifyAudioTranscriptionMultipartBody(req, route)
	if err != nil {
		t.Fatalf("ModifyAudioTranscriptionMultipartBody failed: %v", err)
	}

	if !strings.Contains(contentType, "multipart/form-data") {
		t.Errorf("Expected multipart content type, got %s", contentType)
	}

	// Build a new request to parse and verify the modified fields
	parsedReq, err := http.NewRequest("POST", "/v1/audio/transcriptions", bytes.NewReader(body))
	if err != nil {
		t.Fatalf("Failed to create parsing request: %v", err)
	}
	parsedReq.Header.Set("Content-Type", contentType)

	err = parsedReq.ParseMultipartForm(32 << 20)
	if err != nil {
		t.Fatalf("ParseMultipartForm on modified body failed: %v", err)
	}

	if parsedReq.FormValue("model") != "upstream-whisper" {
		t.Errorf("Expected model to be replaced with 'upstream-whisper', got %q", parsedReq.FormValue("model"))
	}

	if parsedReq.FormValue("response_format") != "" {
		t.Errorf("Expected response_format to be deleted, but got %q", parsedReq.FormValue("response_format"))
	}

	if parsedReq.FormValue("temperature") != "0.5" {
		t.Errorf("Expected temperature to be extra injected with value '0.5', got %q", parsedReq.FormValue("temperature"))
	}
}

func TestTransformAudioTranscriptionRequestToGeminiInline(t *testing.T) {
	fields := map[string]string{
		"model":           "whisper-1",
		"response_format": "verbose_json",
		"prompt":          "transcribe this clearly",
		"language":        "en",
	}
	// Content-Type of file will be auto-detected or default to audio/mp3 based on filename
	req := createTestMultipartRequest(t, fields, "file", "audio.mp3", "dummy-mp3-bytes")

	err := req.ParseMultipartForm(32 << 20)
	if err != nil {
		t.Fatalf("ParseMultipartForm failed: %v", err)
	}

	route := &SelectedRoute{
		ModelProvider: &ModelProviderConfig{
			Model: "gemini-flash-1.5-transcribe",
		},
		AuthKey: "api-key",
	}

	cfg := &Config{
		Proxy: "http://localhost:8080",
	}
	proxyMgr := NewProxyManager()

	modified, err := TransformAudioTranscriptionRequestToGemini(req, route, cfg, proxyMgr)
	if err != nil {
		t.Fatalf("TransformAudioTranscriptionRequestToGemini failed: %v", err)
	}

	var geminiReq map[string]interface{}
	if err := json.Unmarshal(modified, &geminiReq); err != nil {
		t.Fatalf("Transformed body is not valid JSON: %v", err)
	}

	contents, ok := geminiReq["contents"].([]interface{})
	if !ok || len(contents) == 0 {
		t.Fatal("Missing or invalid contents in Gemini request")
	}

	parts, ok := contents[0].(map[string]interface{})["parts"].([]interface{})
	if !ok || len(parts) != 2 {
		t.Fatalf("Expected 2 parts (instruction text + audio part), got %d parts", len(parts))
	}

	instruction, ok := parts[0].(map[string]interface{})["text"].(string)
	if !ok || !strings.Contains(instruction, "Transcribe") {
		t.Errorf("Expected transcription instructions, got %v", parts[0])
	}
	if !strings.Contains(instruction, "transcribe this clearly") {
		t.Errorf("Expected user style prompt in instructions, got %q", instruction)
	}
	if !strings.Contains(instruction, "language is en") {
		t.Errorf("Expected language instruction, got %q", instruction)
	}

	audioPart := parts[1].(map[string]interface{})
	inlineData, ok := audioPart["inline_data"].(map[string]interface{})
	if !ok {
		t.Fatalf("Expected inline_data part, got %v", audioPart)
	}

	if inlineData["mime_type"] != "audio/mp3" {
		t.Errorf("Expected audio/mp3 mime_type, got %v", inlineData["mime_type"])
	}

	expectedBase64 := base64.StdEncoding.EncodeToString([]byte("dummy-mp3-bytes"))
	if inlineData["data"] != expectedBase64 {
		t.Errorf("Expected base64 audio data %q, got %q", expectedBase64, inlineData["data"])
	}

	genConfig, ok := geminiReq["generationConfig"].(map[string]interface{})
	if !ok {
		t.Fatal("Missing generationConfig")
	}

	if genConfig["responseMimeType"] != "application/json" {
		t.Errorf("Expected responseMimeType application/json, got %v", genConfig["responseMimeType"])
	}

	responseSchema, ok := genConfig["responseSchema"].(map[string]interface{})
	if !ok || responseSchema["type"] != "OBJECT" {
		t.Errorf("Expected Object responseSchema, got %v", responseSchema)
	}
}

func TestProcessGeminiTranscriptionResponse_Simple(t *testing.T) {
	geminiResp := []byte(`{
		"candidates": [
			{
				"content": {
					"parts": [
						{
							"text": "This is a simple transcription."
						}
					]
				}
			}
		]
	}`)

	// Test mapping to JSON
	outJSON, err := ProcessGeminiTranscriptionResponse(geminiResp, false, "", "json")
	if err != nil {
		t.Fatalf("ProcessGeminiTranscriptionResponse failed: %v", err)
	}

	var parsed map[string]interface{}
	if err := json.Unmarshal(outJSON, &parsed); err != nil {
		t.Fatalf("Output is not valid JSON: %v", err)
	}

	if parsed["text"] != "This is a simple transcription." {
		t.Errorf("Expected text 'This is a simple transcription.', got %q", parsed["text"])
	}

	// Test mapping to raw text
	outText, err := ProcessGeminiTranscriptionResponse(geminiResp, false, "", "text")
	if err != nil {
		t.Fatalf("ProcessGeminiTranscriptionResponse failed: %v", err)
	}

	if string(outText) != "This is a simple transcription." {
		t.Errorf("Expected raw text 'This is a simple transcription.', got %q", string(outText))
	}
}

func TestProcessGeminiTranscriptionResponse_VerboseTimestamps(t *testing.T) {
	geminiStructuredJSON := `{
		"text": "Hello world. Testing SRT conversion.",
		"segments": [
			{"start": 0.0, "end": 1.5, "text": "Hello world."},
			{"start": 1.5, "end": 4.25, "text": "Testing SRT conversion."}
		]
	}`

	geminiResp := []byte(fmt.Sprintf(`{
		"candidates": [
			{
				"content": {
					"parts": [
						{
							"text": %q
						}
					]
				}
			}
		]
	}`, geminiStructuredJSON))

	// 1. Test mapping to verbose_json
	outVerbose, err := ProcessGeminiTranscriptionResponse(geminiResp, true, "french", "verbose_json")
	if err != nil {
		t.Fatalf("ProcessGeminiTranscriptionResponse verbose_json failed: %v", err)
	}

	var parsed map[string]interface{}
	if err := json.Unmarshal(outVerbose, &parsed); err != nil {
		t.Fatalf("Output verbose_json is not valid JSON: %v", err)
	}

	if parsed["text"] != "Hello world. Testing SRT conversion." {
		t.Errorf("Expected text 'Hello world. Testing SRT conversion.', got %q", parsed["text"])
	}

	if parsed["language"] != "french" {
		t.Errorf("Expected language 'french', got %q", parsed["language"])
	}

	if parsed["duration"].(float64) != 4.25 {
		t.Errorf("Expected duration 4.25, got %v", parsed["duration"])
	}

	segments, ok := parsed["segments"].([]interface{})
	if !ok || len(segments) != 2 {
		t.Fatalf("Expected 2 segments, got %v", parsed["segments"])
	}

	seg1 := segments[0].(map[string]interface{})
	if seg1["id"].(float64) != 0 || seg1["start"].(float64) != 0.0 || seg1["end"].(float64) != 1.5 || seg1["text"] != "Hello world." {
		t.Errorf("Segment 1 mismatch: %v", seg1)
	}

	seg2 := segments[1].(map[string]interface{})
	if seg2["id"].(float64) != 1 || seg2["start"].(float64) != 1.5 || seg2["end"].(float64) != 4.25 || seg2["text"] != "Testing SRT conversion." {
		t.Errorf("Segment 2 mismatch: %v", seg2)
	}

	// 2. Test mapping to srt format
	outSRT, err := ProcessGeminiTranscriptionResponse(geminiResp, true, "", "srt")
	if err != nil {
		t.Fatalf("ProcessGeminiTranscriptionResponse srt failed: %v", err)
	}

	expectedSRT := "1\n00:00:00,000 --> 00:00:01,500\nHello world.\n\n2\n00:00:01,500 --> 00:00:04,250\nTesting SRT conversion.\n\n"
	// Replace CRLF if any for OS independence
	normalizedSRT := strings.ReplaceAll(string(outSRT), "\r\n", "\n")
	if normalizedSRT != expectedSRT {
		t.Errorf("SRT content mismatch.\nExpected:\n%q\nGot:\n%q", expectedSRT, normalizedSRT)
	}

	// 3. Test mapping to vtt format
	outVTT, err := ProcessGeminiTranscriptionResponse(geminiResp, true, "", "vtt")
	if err != nil {
		t.Fatalf("ProcessGeminiTranscriptionResponse vtt failed: %v", err)
	}

	expectedVTT := "WEBVTT\n\n1\n00:00:00.000 --> 00:00:01.500\nHello world.\n\n2\n00:00:01.500 --> 00:00:04.250\nTesting SRT conversion.\n\n"
	normalizedVTT := strings.ReplaceAll(string(outVTT), "\r\n", "\n")
	if normalizedVTT != expectedVTT {
		t.Errorf("VTT content mismatch.\nExpected:\n%q\nGot:\n%q", expectedVTT, normalizedVTT)
	}
}

func TestProcessGeminiSTTStreamLine(t *testing.T) {
	line := []byte(`data: {"candidates":[{"content":{"parts":[{"text":"Skies of blue"}]}}], "usageMetadata": {"promptTokenCount": 140, "candidatesTokenCount": 10, "totalTokenCount": 150, "promptTokensDetails": [{"modality": "AUDIO", "tokenCount": 100}, {"modality": "TEXT", "tokenCount": 40}]}}`)
	delta, usage, err := ProcessGeminiSTTStreamLine(line)
	if err != nil {
		t.Fatalf("ProcessGeminiSTTStreamLine failed: %v", err)
	}

	if delta != "Skies of blue" {
		t.Errorf("Expected delta 'Skies of blue', got %q", delta)
	}

	if usage == nil {
		t.Fatal("Expected parsed usageMetadata, got nil")
	}

	if usage.PromptTokens != 140 || usage.CandidatesTokens != 10 || usage.TotalTokens != 150 {
		t.Errorf("Usage mismatch: %+v", usage)
	}

	if usage.AudioTokens != 100 || usage.TextTokens != 40 {
		t.Errorf("Usage promptTokensDetails mismatch: %+v", usage)
	}

	// Test EOF line
	doneLine := []byte(`data: [DONE]`)
	_, _, err = ProcessGeminiSTTStreamLine(doneLine)
	if err != io.EOF {
		t.Errorf("Expected io.EOF on [DONE], got %v", err)
	}
}

func TestTransformRequestToGemini_InputAudio(t *testing.T) {
	route := &SelectedRoute{
		ModelProvider: &ModelProviderConfig{
			Name:    "gemini-provider",
			Model:   "gemini-2.5-flash",
			ApiType: "gemini",
		},
	}

	openAIReq := map[string]interface{}{
		"model": "gpt-4o",
		"messages": []interface{}{
			map[string]interface{}{
				"role": "user",
				"content": []interface{}{
					map[string]interface{}{
						"type": "text",
						"text": "What is in this recording?",
					},
					map[string]interface{}{
						"type": "input_audio",
						"input_audio": map[string]interface{}{
							"data":   "UklGRgAAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA=", // valid tiny wav base64
							"format": "wav",
						},
					},
				},
			},
		},
	}

	reqBytes, err := json.Marshal(openAIReq)
	if err != nil {
		t.Fatalf("Failed to marshal OpenAI request: %v", err)
	}

	translatedBytes, err := TransformRequestToGemini(reqBytes, route, nil, nil, nil)
	if err != nil {
		t.Fatalf("TransformRequestToGemini failed: %v", err)
	}

	var geminiReq map[string]interface{}
	if err := json.Unmarshal(translatedBytes, &geminiReq); err != nil {
		t.Fatalf("Failed to unmarshal translated Gemini request: %v", err)
	}

	contents, ok := geminiReq["contents"].([]interface{})
	if !ok || len(contents) != 1 {
		t.Fatalf("Expected 1 item in contents, got %v", len(contents))
	}

	parts := contents[0].(map[string]interface{})["parts"].([]interface{})
	if len(parts) != 2 {
		t.Fatalf("Expected 2 parts, got %d", len(parts))
	}

	inlineData, ok := parts[0].(map[string]interface{})["inline_data"].(map[string]interface{})
	if !ok {
		t.Fatalf("Expected inline_data part, got %v", parts[0])
	}

	if inlineData["mime_type"] != "audio/wav" {
		t.Errorf("Expected audio/wav mime_type, got %v", inlineData["mime_type"])
	}
}

func TestTransformRequestToGemini_File(t *testing.T) {
	route := &SelectedRoute{
		ModelProvider: &ModelProviderConfig{
			Name:    "gemini-provider",
			Model:   "gemini-2.5-flash",
			ApiType: "gemini",
		},
	}

	openAIReq := map[string]interface{}{
		"model": "gpt-4o",
		"messages": []interface{}{
			map[string]interface{}{
				"role": "user",
				"content": []interface{}{
					map[string]interface{}{
						"type": "text",
						"text": "What is in this file?",
					},
					map[string]interface{}{
						"type": "file",
						"file": map[string]interface{}{
							"file_data": "JVBERi0xLjQK",
							"filename":  "report.pdf",
						},
					},
				},
			},
		},
	}

	reqBytes, err := json.Marshal(openAIReq)
	if err != nil {
		t.Fatalf("Failed to marshal OpenAI request: %v", err)
	}

	translatedBytes, err := TransformRequestToGemini(reqBytes, route, nil, nil, nil)
	if err != nil {
		t.Fatalf("TransformRequestToGemini failed: %v", err)
	}

	var geminiReq map[string]interface{}
	if err := json.Unmarshal(translatedBytes, &geminiReq); err != nil {
		t.Fatalf("Failed to unmarshal translated Gemini request: %v", err)
	}

	contents, ok := geminiReq["contents"].([]interface{})
	if !ok || len(contents) != 1 {
		t.Fatalf("Expected 1 item in contents, got %v", len(contents))
	}

	parts := contents[0].(map[string]interface{})["parts"].([]interface{})
	if len(parts) != 2 {
		t.Fatalf("Expected 2 parts, got %d", len(parts))
	}

	inlineData, ok := parts[0].(map[string]interface{})["inline_data"].(map[string]interface{})
	if !ok {
		t.Fatalf("Expected inline_data part, got %v", parts[0])
	}

	if inlineData["mime_type"] != "application/pdf" {
		t.Errorf("Expected application/pdf mime_type, got %v", inlineData["mime_type"])
	}
}

func TestDetectMimeTypeFromContent(t *testing.T) {
	tests := []struct {
		name     string
		data     []byte
		expected string
	}{
		{"PDF", []byte("%PDF-1.4\n..."), "application/pdf"},
		{"PNG", []byte("\x89PNG\r\n\x1a\nblah"), "image/png"},
		{"JPEG", []byte("\xff\xd8\xff\xe0\x00\x10JFIF"), "image/jpeg"},
		{"GIF", []byte("GIF89a..."), "image/gif"},
		{"BMP", []byte("BM..."), "image/bmp"},
		{"FLAC", []byte("fLaC..."), "audio/flac"},
		{"OGG", []byte("OggS..."), "audio/ogg"},
		{"MP3_ID3", []byte("ID3..."), "audio/mp3"},
		{"MP3_FrameSync", []byte{0xff, 0xfb, 0x00, 0x00}, "audio/mp3"},
		{"WAV", []byte("RIFF\x00\x00\x00\x00WAVE..."), "audio/wav"},
		{"AVI", []byte("RIFF\x00\x00\x00\x00AVI ..."), "video/avi"},
		{"MP4", append([]byte("\x00\x00\x00\x18ftypmp42"), []byte("isommp42")...), "video/mp4"},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := detectMimeTypeFromContent(tc.data)
			if got != tc.expected {
				t.Errorf("Expected %q, got %q", tc.expected, got)
			}
		})
	}
}

func TestModifyRequestHeaders(t *testing.T) {
	route := &SelectedRoute{
		ModelProvider: &ModelProviderConfig{
			RequestHeaders: RequestHeadersConfig{
				Delete: []string{"X-Bad-Header", "Content-Type"},
				Extra: []map[string]interface{}{
					{"Cookie": "session=abc"},
					{"X-Custom-Header": "hello"},
					{"X-Merged-Header": "extra1"},
				},
			},
		},
	}

	req, err := http.NewRequest("POST", "http://localhost", nil)
	if err != nil {
		t.Fatalf("Failed to create request: %v", err)
	}

	req.Header.Set("X-Bad-Header", "should-be-deleted")
	req.Header.Set("Content-Type", "should-be-deleted")
	req.Header.Set("Cookie", "init=123")
	req.Header.Set("X-Merged-Header", "init1")

	ModifyRequestHeaders(req, route)

	// Check deleted headers
	if v := req.Header.Get("X-Bad-Header"); v != "" {
		t.Errorf("Expected X-Bad-Header to be deleted, got %q", v)
	}
	if v := req.Header.Get("Content-Type"); v != "" {
		t.Errorf("Expected Content-Type to be deleted, got %q", v)
	}

	// Check custom new header
	if v := req.Header.Get("X-Custom-Header"); v != "hello" {
		t.Errorf("Expected X-Custom-Header to be 'hello', got %q", v)
	}

	// Check merged Cookie with "; "
	if v := req.Header.Get("Cookie"); v != "init=123; session=abc" {
		t.Errorf("Expected Cookie to be 'init=123; session=abc', got %q", v)
	}

	// Check merged standard header with ", "
	if v := req.Header.Get("X-Merged-Header"); v != "init1, extra1" {
		t.Errorf("Expected X-Merged-Header to be 'init1, extra1', got %q", v)
	}
}

func TestDeleteNestedPath(t *testing.T) {
	bodyJSON := `{
		"model": "gpt-4o",
		"messages": [
			{
				"role": "system",
				"content": "You must output ..."
			},
			{
				"role": "user",
				"content": "今天是星期几"
			},
			{
				"role": "assistant",
				"content": "今天是星期五。",
				"reasoning_content": "The user is asking ..."
			}
		],
		"settings": {
			"options": {
				"temperature": 0.5,
				"max_tokens": 100
			}
		}
	}`

	var body map[string]interface{}
	if err := json.Unmarshal([]byte(bodyJSON), &body); err != nil {
		t.Fatalf("Failed to parse body JSON: %v", err)
	}

	// 1. Delete a nested key inside a slice with bracket syntax: messages[].reasoning_content
	deleteNestedPath(body, "messages[].reasoning_content")

	messages, ok := body["messages"].([]interface{})
	if !ok {
		t.Fatalf("Expected messages to be an array")
	}

	for _, m := range messages {
		msgMap := m.(map[string]interface{})
		if _, ok := msgMap["reasoning_content"]; ok {
			t.Errorf("Expected reasoning_content to be deleted from assistant message")
		}
	}

	// 2. Delete a nested key inside a slice with auto-fallback dot syntax: messages.content
	deleteNestedPath(body, "messages.content")

	for _, m := range messages {
		msgMap := m.(map[string]interface{})
		if _, ok := msgMap["content"]; ok {
			t.Errorf("Expected content to be deleted from message")
		}
		// role should still be there
		if _, ok := msgMap["role"]; !ok {
			t.Errorf("Expected role to remain intact")
		}
	}

	// 3. Delete deep nested key: settings.options.max_tokens
	deleteNestedPath(body, "settings.options.max_tokens")

	settings := body["settings"].(map[string]interface{})
	options := settings["options"].(map[string]interface{})
	if _, ok := options["max_tokens"]; ok {
		t.Errorf("Expected settings.options.max_tokens to be deleted")
	}
	if options["temperature"].(float64) != 0.5 {
		t.Errorf("Expected settings.options.temperature to remain 0.5, got %v", options["temperature"])
	}
}


