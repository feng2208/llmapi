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

	// Step 3a: Select route first time.
	route, err := router.SelectRoute("test-model")
	if err != nil {
		t.Fatalf("Failed to select route: %v", err)
	}
	if route.ModelProvider.Name != "prov-1" {
		t.Errorf("Expected prov-1, got %s", route.ModelProvider.Name)
	}
	firstKey := route.AuthKey

	// Step 3b: Select route second time. It should round-robin to the other key of prov-1.
	route2, err := router.SelectRoute("test-model")
	if err != nil {
		t.Fatalf("Failed to select route second time: %v", err)
	}
	if route2.ModelProvider.Name != "prov-1" {
		t.Errorf("Expected prov-1, got %s", route2.ModelProvider.Name)
	}
	if route2.AuthKey == firstKey {
		t.Errorf("Expected different key for round-robin, both were %s", firstKey)
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
