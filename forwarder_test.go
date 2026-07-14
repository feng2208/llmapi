package main

import (
	"encoding/json"
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
	modifiedGemini, err := ModifyRequestBody(rawJSON, routeGemini)
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

	modifiedOther, err := ModifyRequestBody(rawJSON, routeOther)
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

	translatedBytes, err := TransformRequestToGemini(reqBytes, route)
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
	if parts1[0].(map[string]interface{})["text"] != "What is in this image and what is the temperature in London?" {
		t.Errorf("Expected user text part, got %v", parts1[0])
	}
	inlineData := parts1[1].(map[string]interface{})["inline_data"].(map[string]interface{})
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
	if _, ok := params["$schema"]; ok {
		t.Errorf("Expected '$schema' to be cleaned from parameters")
	}
	if _, ok := params["additionalProperties"]; ok {
		t.Errorf("Expected 'additionalProperties' to be cleaned from parameters")
	}
	locationProp := params["properties"].(map[string]interface{})["location"].(map[string]interface{})
	if _, ok := locationProp["additionalProperties"]; ok {
		t.Errorf("Expected nested 'additionalProperties' to be cleaned from location properties")
	}

	// Verify Tool Choice
	toolConfig, ok := geminiReq["toolConfig"].(map[string]interface{})
	if !ok {
		t.Fatalf("Expected toolConfig to be set")
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
