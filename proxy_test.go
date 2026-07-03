package main

import (
	"bytes"
	"compress/gzip"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func TestProxyGzipUpstreamDecompression(t *testing.T) {
	// 1. Create mock upstream server that returns gzip-compressed response
	mockUpstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Verify that we stripped Accept-Encoding if requestType != "openai"
		// If Accept-Encoding was stripped, Go's transport might have added "gzip" back
		// Let's check if the Accept-Encoding contains gzip or if it is empty/something else.
		// Go's transport automatically decompresses if we don't pass Accept-Encoding ourselves.
		// For our mock server, we will return gzip-compressed data anyway, setting Content-Encoding: gzip.
		
		responseJSON := `{
			"id": "chatcmpl-123",
			"object": "chat.completion",
			"created": 1677652288,
			"model": "gpt-4o",
			"choices": [
				{
					"index": 0,
					"message": {
						"role": "assistant",
						"content": "Hello from mock compressed upstream!"
					},
					"finish_reason": "stop"
				}
			],
			"usage": {
				"prompt_tokens": 9,
				"completion_tokens": 12,
				"total_tokens": 21
			}
		}`

		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Content-Encoding", "gzip")
		w.WriteHeader(http.StatusOK)

		gw := gzip.NewWriter(w)
		_, err := gw.Write([]byte(responseJSON))
		if err != nil {
			t.Errorf("failed to write gzip response: %v", err)
		}
		gw.Close()
	}))
	defer mockUpstream.Close()

	// 2. Build configuration with Mock Upstream
	cfg := &Config{
		Listen:      "127.0.0.1:3000",
		MaxBodySize: 10 * 1024 * 1024,
		Clients: ClientsConfig{
			RateLimit: 100,
			Auth: []ClientAuthKey{
				{
					Name:      "test-client",
					Key:       "sk-test-client-key",
					RateLimit: 100,
				},
			},
		},
		Models: []ModelConfig{
			{
				Name: "claude-3-5-sonnet",
				Providers: []ModelProviderConfig{
					{
						Name:     "mock-provider",
						Upstream: mockUpstream.URL,
						Model:    "gpt-4o",
						Timeout:  5 * time.Second,
					},
				},
			},
		},
		Providers: []ProviderAuthConfig{
			{
				Name:      "mock-provider",
				RateLimit: 100,
				AuthKeys:  []string{"sk-upstream-key"},
			},
		},
	}

	keyManagers := map[string]*KeyManager{
		"mock-provider": NewKeyManager("mock-provider", []string{"sk-upstream-key"}, 100),
	}

	// 3. Initialize ProxyRouter
	router := NewProxyRouter(cfg, keyManagers, true)

	// 4. Send Claude API request to the proxy
	claudeReq := `{
		"model": "claude-3-5-sonnet",
		"messages": [
			{
				"role": "user",
				"content": "Hi"
			}
		],
		"max_tokens": 100
	}`

	req := httptest.NewRequest("POST", "/v1/messages", bytes.NewBufferString(claudeReq))
	req.Header.Set("Authorization", "Bearer sk-test-client-key")
	req.Header.Set("Content-Type", "application/json")
	// Simulate user client requesting compressed response
	req.Header.Set("Accept-Encoding", "gzip, deflate, br")

	recorder := httptest.NewRecorder()
	router.ServeHTTP(recorder, req)

	// 5. Verify Response
	if recorder.Code != http.StatusOK {
		t.Fatalf("expected HTTP status 200, got %d. Response: %s", recorder.Code, recorder.Body.String())
	}

	var claudeResp ClaudeMessagesResponse
	if err := json.Unmarshal(recorder.Body.Bytes(), &claudeResp); err != nil {
		t.Fatalf("failed to parse Claude response: %v. Body: %s", err, recorder.Body.String())
	}

	if len(claudeResp.Content) == 0 {
		t.Fatal("expected non-empty content in response")
	}

	expectedText := "Hello from mock compressed upstream!"
	if claudeResp.Content[0].Text != expectedText {
		t.Errorf("expected text %q, got %q", expectedText, claudeResp.Content[0].Text)
	}
}

func TestProxyGzipUpstreamDecompression_OpenAI(t *testing.T) {
	// 1. Create mock upstream server that returns gzip-compressed response
	mockUpstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		responseJSON := `{
			"id": "chatcmpl-123",
			"object": "chat.completion",
			"created": 1677652288,
			"model": "gpt-4o",
			"choices": [
				{
					"index": 0,
					"message": {
						"role": "assistant",
						"content": "Hello from mock compressed upstream for OpenAI!"
					},
					"finish_reason": "stop"
				}
			],
			"usage": {
				"prompt_tokens": 9,
				"completion_tokens": 12,
				"total_tokens": 21
			}
		}`

		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Content-Encoding", "gzip")
		w.WriteHeader(http.StatusOK)

		gw := gzip.NewWriter(w)
		_, err := gw.Write([]byte(responseJSON))
		if err != nil {
			t.Errorf("failed to write gzip response: %v", err)
		}
		gw.Close()
	}))
	defer mockUpstream.Close()

	// 2. Build configuration with Mock Upstream
	cfg := &Config{
		Listen:      "127.0.0.1:3000",
		MaxBodySize: 10 * 1024 * 1024,
		Clients: ClientsConfig{
			RateLimit: 100,
			Auth: []ClientAuthKey{
				{
					Name:      "test-client",
					Key:       "sk-test-client-key",
					RateLimit: 100,
				},
			},
		},
		Models: []ModelConfig{
			{
				Name: "gpt-4o",
				Providers: []ModelProviderConfig{
					{
						Name:     "mock-provider",
						Upstream: mockUpstream.URL,
						Model:    "gpt-4o",
						Timeout:  5 * time.Second,
					},
				},
			},
		},
		Providers: []ProviderAuthConfig{
			{
				Name:      "mock-provider",
				RateLimit: 100,
				AuthKeys:  []string{"sk-upstream-key"},
			},
		},
	}

	keyManagers := map[string]*KeyManager{
		"mock-provider": NewKeyManager("mock-provider", []string{"sk-upstream-key"}, 100),
	}

	// 3. Initialize ProxyRouter
	router := NewProxyRouter(cfg, keyManagers, true)

	// 4. Send OpenAI API request to the proxy
	openaiReq := `{
		"model": "gpt-4o",
		"messages": [
			{
				"role": "user",
				"content": "Hi"
			}
		]
	}`

	req := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewBufferString(openaiReq))
	req.Header.Set("Authorization", "Bearer sk-test-client-key")
	req.Header.Set("Content-Type", "application/json")
	// Simulate user client requesting compressed response
	req.Header.Set("Accept-Encoding", "gzip, deflate, br")

	recorder := httptest.NewRecorder()
	router.ServeHTTP(recorder, req)

	// 5. Verify Response
	if recorder.Code != http.StatusOK {
		t.Fatalf("expected HTTP status 200, got %d. Response: %s", recorder.Code, recorder.Body.String())
	}

	var openaiResp map[string]interface{}
	if err := json.Unmarshal(recorder.Body.Bytes(), &openaiResp); err != nil {
		t.Fatalf("failed to parse OpenAI response: %v. Body: %s", err, recorder.Body.String())
	}

	choices, ok := openaiResp["choices"].([]interface{})
	if !ok || len(choices) == 0 {
		t.Fatal("expected non-empty choices in response")
	}

	choice := choices[0].(map[string]interface{})
	message := choice["message"].(map[string]interface{})
	content := message["content"].(string)

	expectedText := "Hello from mock compressed upstream for OpenAI!"
	if content != expectedText {
		t.Errorf("expected text %q, got %q", expectedText, content)
	}
}
