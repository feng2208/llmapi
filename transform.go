package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"strings"
)

// applyDeleteAndExtra deletes the configured delete keys and merges extra fields.
func applyDeleteAndExtra(body map[string]interface{}, route *SelectedRoute) {
	// Delete Configured Keys
	for _, item := range route.ModelProvider.RequestBody.Delete {
		if str, ok := item.(string); ok {
			deleteNestedPath(body, str)
		} else if m, ok := item.(map[string]interface{}); ok {
			for k := range m {
				deleteNestedPath(body, k)
			}
		} else if m, ok := item.(map[interface{}]interface{}); ok {
			for k := range m {
				if s, ok := k.(string); ok {
					deleteNestedPath(body, s)
				}
			}
		}
	}

	// Add/Replace Extra Fields
	for _, extraMap := range route.ModelProvider.RequestBody.Extra {
		deepMerge(body, extraMap)
	}
}

// deleteNestedPath recursively traverses and deletes a key from maps and slices using dot/brackets notation (e.g. "messages[].reasoning_content").
func deleteNestedPath(data interface{}, path string) {
	if data == nil || path == "" {
		return
	}

	idx := strings.Index(path, ".")
	if idx != -1 {
		currKey := path[:idx]
		restPath := path[idx+1:]

		// Check if current key represents an array format (e.g. "messages[]")
		isArray := false
		if strings.HasSuffix(currKey, "[]") {
			currKey = strings.TrimSuffix(currKey, "[]")
			isArray = true
		}

		if m, ok := data.(map[string]interface{}); ok {
			val := m[currKey]
			if isArray {
				if slice, ok := val.([]interface{}); ok {
					for _, item := range slice {
						deleteNestedPath(item, restPath)
					}
				}
			} else {
				// Fallback: if user didn't write "[]" but the actual value is a slice, automatically loop
				if slice, ok := val.([]interface{}); ok {
					for _, item := range slice {
						deleteNestedPath(item, restPath)
					}
				} else {
					deleteNestedPath(val, restPath)
				}
			}
		}
		return
	}

	// End of path (deepest key, e.g. "reasoning_content")
	if m, ok := data.(map[string]interface{}); ok {
		delete(m, path)
	} else if slice, ok := data.([]interface{}); ok {
		for _, item := range slice {
			if itemMap, ok := item.(map[string]interface{}); ok {
				delete(itemMap, path)
			}
		}
	}
}

// ModifyRequestBody applies the delete, extra, and model replacement rules.
func ModifyRequestBody(rawBody []byte, route *SelectedRoute, r *http.Request, cfg *Config, proxyMgr *ProxyManager) ([]byte, string, error) {
	if route.ModelProvider.ApiType == "gemini" {
		modified, err := TransformRequestToGemini(rawBody, route, r, cfg, proxyMgr)
		if err != nil {
			return nil, "", err
		}
		return modified, "application/json; charset=utf-8", nil
	}

	var body map[string]interface{}
	if err := json.Unmarshal(rawBody, &body); err != nil {
		return nil, "", fmt.Errorf("failed to parse JSON request body: %w", err)
	}

	// 1. Replace Model Name
	body["model"] = route.ModelProvider.Model

	// 2. Apply delete & extra config
	applyDeleteAndExtra(body, route)

	// 3. Inject skip_thought_signature_validator conditionally for gemini provider
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
		return nil, "", fmt.Errorf("failed to marshal modified JSON: %w", err)
	}

	return modified, "application/json; charset=utf-8", nil
}

func ModifyImageRequestBody(rawBody []byte, route *SelectedRoute, r *http.Request) ([]byte, string, error) {
	if route.ModelProvider.ApiType == "gemini" {
		modified, err := TransformImageRequestToGemini(rawBody, route)
		if err != nil {
			return nil, "", err
		}
		return modified, "application/json; charset=utf-8", nil
	}

	var body map[string]interface{}
	if err := json.Unmarshal(rawBody, &body); err != nil {
		return nil, "", fmt.Errorf("failed to parse JSON request body: %w", err)
	}

	// 1. Replace Model Name
	body["model"] = route.ModelProvider.Model

	// 2. Apply delete & extra config
	applyDeleteAndExtra(body, route)

	modified, err := json.Marshal(body)
	if err != nil {
		return nil, "", fmt.Errorf("failed to marshal modified JSON: %w", err)
	}

	return modified, "application/json; charset=utf-8", nil
}

type StreamExtractor struct {
	startTag         string
	endTag           string
	reasoningField   string
	fullBuffer       string
	flushedContent   string
	flushedReasoning string
}

func NewStreamExtractor(startTag, endTag, reasoningField string) *StreamExtractor {
	return &StreamExtractor{
		startTag:       startTag,
		endTag:         endTag,
		reasoningField: reasoningField,
	}
}

func extractReasoningValue(val interface{}) string {
	if val == nil {
		return ""
	}
	if s, ok := val.(string); ok {
		return s
	}
	var sb strings.Builder
	if slice, ok := val.([]interface{}); ok {
		for _, item := range slice {
			if itemStr, ok := item.(string); ok {
				sb.WriteString(itemStr)
			} else if itemMap, ok := item.(map[string]interface{}); ok {
				if txt, ok := itemMap["text"].(string); ok {
					sb.WriteString(txt)
				} else if content, ok := itemMap["content"].(string); ok {
					sb.WriteString(content)
				}
			}
		}
		return sb.String()
	}
	if m, ok := val.(map[string]interface{}); ok {
		if txt, ok := m["text"].(string); ok {
			return txt
		}
		if content, ok := m["content"].(string); ok {
			return content
		}
	}
	return ""
}

func processReasoningField(container map[string]interface{}, reasoningField string) {
	if container == nil || reasoningField == "" {
		return
	}

	var extractedReasoning strings.Builder
	var normalTextParts []string
	hasContentField := false
	contentIsArrayOrMap := false

	// Case 1: Direct key on container, e.g. container["thinking"]
	if val, ok := container[reasoningField]; ok {
		extractedReasoning.WriteString(extractReasoningValue(val))
		delete(container, reasoningField)
	}

	// Case 2: Inside container["content"]
	if contentVal, ok := container["content"]; ok && contentVal != nil {
		hasContentField = true
		if slice, ok := contentVal.([]interface{}); ok {
			contentIsArrayOrMap = true
			for _, item := range slice {
				if itemMap, ok := item.(map[string]interface{}); ok {
					if thinkVal, ok := itemMap[reasoningField]; ok {
						extractedReasoning.WriteString(extractReasoningValue(thinkVal))
					} else if txt, ok := itemMap["text"].(string); ok {
						normalTextParts = append(normalTextParts, txt)
					} else if itemType, ok := itemMap["type"].(string); ok && itemType == "text" {
						if txtVal, ok := itemMap["text"].(string); ok {
							normalTextParts = append(normalTextParts, txtVal)
						}
					}
				} else if itemStr, ok := item.(string); ok {
					normalTextParts = append(normalTextParts, itemStr)
				}
			}
		} else if itemMap, ok := contentVal.(map[string]interface{}); ok {
			contentIsArrayOrMap = true
			if thinkVal, ok := itemMap[reasoningField]; ok {
				extractedReasoning.WriteString(extractReasoningValue(thinkVal))
			} else if txt, ok := itemMap["text"].(string); ok {
				normalTextParts = append(normalTextParts, txt)
			}
		}
	}

	// If we extracted reasoning text:
	if extractedReasoning.Len() > 0 {
		rText := extractedReasoning.String()
		container["reasoning"] = rText
		container["reasoning_content"] = rText

		if contentIsArrayOrMap {
			if len(normalTextParts) > 0 {
				container["content"] = strings.Join(normalTextParts, "")
			} else {
				delete(container, "content")
			}
		}
	} else if contentIsArrayOrMap && hasContentField {
		if len(normalTextParts) > 0 {
			container["content"] = strings.Join(normalTextParts, "")
		}
	}
}

func (se *StreamExtractor) ProcessSSELine(line []byte) []byte {
	if (se.startTag == "" || se.endTag == "") && se.reasoningField == "" {
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

	// Process reasoning_field if configured
	if se.reasoningField != "" {
		processReasoningField(delta, se.reasoningField)
	}

	// Process tag-based extraction if startTag/endTag configured
	if se.startTag != "" && se.endTag != "" {
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
	}

	modifiedPayload, err := json.Marshal(obj)
	if err != nil {
		return line
	}

	return buildSSELine(modifiedPayload, line)
}

// ProcessJSONResponse extracts reasoning content from a non-streaming JSON response.
func ProcessJSONResponse(respBody []byte, startTag, endTag, reasoningField string) []byte {
	if (startTag == "" || endTag == "") && reasoningField == "" {
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

	if reasoningField != "" {
		processReasoningField(message, reasoningField)
	}

	if startTag != "" && endTag != "" {
		content, _ := message["content"].(string)
		if content != "" {
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
		}
	}

	modified, err := json.Marshal(obj)
	if err != nil {
		return respBody
	}
	return modified
}

func ModifyImageEditMultipartBody(r *http.Request, route *SelectedRoute) ([]byte, string, error) {
	if r.MultipartForm == nil {
		return nil, "", errors.New("multipart form not parsed")
	}

	var buf bytes.Buffer
	writer := multipart.NewWriter(&buf)

	// Collect text fields
	fields := make(map[string]string)
	for k, values := range r.MultipartForm.Value {
		if len(values) > 0 {
			fields[k] = values[0]
		}
	}

	// 1. Replace Model Name
	fields["model"] = route.ModelProvider.Model

	// 2. Apply delete & extra config
	for _, item := range route.ModelProvider.RequestBody.Delete {
		if str, ok := item.(string); ok {
			delete(fields, str)
		} else if m, ok := item.(map[string]interface{}); ok {
			for k := range m {
				delete(fields, k)
			}
		} else if m, ok := item.(map[interface{}]interface{}); ok {
			for k := range m {
				if s, ok := k.(string); ok {
					delete(fields, s)
				}
			}
		}
	}
	for _, extraMap := range route.ModelProvider.RequestBody.Extra {
		for k, v := range extraMap {
			fields[k] = fmt.Sprintf("%v", v)
		}
	}

	// Write text fields
	for k, v := range fields {
		err := writer.WriteField(k, v)
		if err != nil {
			return nil, "", err
		}
	}

	// Write files
	for fieldName, fileHeaders := range r.MultipartForm.File {
		for _, fh := range fileHeaders {
			file, err := fh.Open()
			if err != nil {
				return nil, "", err
			}

			part, err := writer.CreateFormFile(fieldName, fh.Filename)
			if err != nil {
				file.Close()
				return nil, "", err
			}
			_, err = io.Copy(part, file)
			file.Close()
			if err != nil {
				return nil, "", err
			}
		}
	}

	err := writer.Close()
	if err != nil {
		return nil, "", err
	}

	return buf.Bytes(), writer.FormDataContentType(), nil
}

func ModifyImageEditRequestBody(rawBody []byte, route *SelectedRoute, r *http.Request) ([]byte, string, error) {
	if route.ModelProvider.ApiType == "gemini" {
		modified, err := TransformImageEditRequestToGemini(r, route)
		if err != nil {
			return nil, "", err
		}
		return modified, "application/json; charset=utf-8", nil
	}

	// For standard API providers, rebuild the multipart form with modified fields
	modifiedBody, contentType, err := ModifyImageEditMultipartBody(r, route)
	if err != nil {
		return nil, "", err
	}
	return modifiedBody, contentType, nil
}

func ModifyAudioSpeechRequestBody(rawBody []byte, route *SelectedRoute, r *http.Request) ([]byte, string, error) {
	if route.ModelProvider.ApiType == "gemini" {
		modified, err := TransformAudioSpeechRequestToGemini(rawBody, route)
		if err != nil {
			return nil, "", err
		}
		return modified, "application/json; charset=utf-8", nil
	}

	var body map[string]interface{}
	if err := json.Unmarshal(rawBody, &body); err != nil {
		return nil, "", fmt.Errorf("failed to parse JSON request body: %w", err)
	}

	// 1. Replace Model Name
	body["model"] = route.ModelProvider.Model

	// 2. Apply delete & extra config
	applyDeleteAndExtra(body, route)

	modified, err := json.Marshal(body)
	if err != nil {
		return nil, "", fmt.Errorf("failed to marshal modified JSON: %w", err)
	}

	return modified, "application/json; charset=utf-8", nil
}

func ModifyAudioTranscriptionMultipartBody(r *http.Request, route *SelectedRoute) ([]byte, string, error) {
	if r.MultipartForm == nil {
		return nil, "", errors.New("multipart form not parsed")
	}

	var buf bytes.Buffer
	writer := multipart.NewWriter(&buf)

	// Collect text fields
	fields := make(map[string][]string)
	for k, values := range r.MultipartForm.Value {
		fields[k] = values
	}

	// 1. Replace Model Name
	fields["model"] = []string{route.ModelProvider.Model}

	// 2. Apply delete & extra config
	for _, item := range route.ModelProvider.RequestBody.Delete {
		if str, ok := item.(string); ok {
			delete(fields, str)
		} else if m, ok := item.(map[string]interface{}); ok {
			for k := range m {
				delete(fields, k)
			}
		} else if m, ok := item.(map[interface{}]interface{}); ok {
			for k := range m {
				if s, ok := k.(string); ok {
					delete(fields, s)
				}
			}
		}
	}
	for _, extraMap := range route.ModelProvider.RequestBody.Extra {
		for k, v := range extraMap {
			fields[k] = []string{fmt.Sprintf("%v", v)}
		}
	}

	// Write text fields
	for k, vv := range fields {
		for _, v := range vv {
			err := writer.WriteField(k, v)
			if err != nil {
				return nil, "", err
			}
		}
	}

	// Write files
	for fieldName, fileHeaders := range r.MultipartForm.File {
		for _, fh := range fileHeaders {
			file, err := fh.Open()
			if err != nil {
				return nil, "", err
			}

			part, err := writer.CreateFormFile(fieldName, fh.Filename)
			if err != nil {
				file.Close()
				return nil, "", err
			}
			_, err = io.Copy(part, file)
			file.Close()
			if err != nil {
				return nil, "", err
			}
		}
	}

	err := writer.Close()
	if err != nil {
		return nil, "", err
	}

	return buf.Bytes(), writer.FormDataContentType(), nil
}

func ModifyAudioTranscriptionRequestBody(rawBody []byte, route *SelectedRoute, r *http.Request, cfg *Config, proxyMgr *ProxyManager) ([]byte, string, error) {
	if route.ModelProvider.ApiType == "gemini" {
		modified, err := TransformAudioTranscriptionRequestToGemini(r, route, cfg, proxyMgr)
		if err != nil {
			return nil, "", err
		}
		return modified, "application/json; charset=utf-8", nil
	}

	modifiedBody, contentType, err := ModifyAudioTranscriptionMultipartBody(r, route)
	if err != nil {
		return nil, "", err
	}
	return modifiedBody, contentType, nil
}

// ModifyRequestHeaders applies configured request header deletions and additions/merges.
func ModifyRequestHeaders(req *http.Request, route *SelectedRoute) {
	for _, hName := range route.ModelProvider.RequestHeaders.Delete {
		req.Header.Del(hName)
	}
	for _, extraMap := range route.ModelProvider.RequestHeaders.Extra {
		for k, v := range extraMap {
			valStr := fmt.Sprintf("%v", v)
			canonicalKey := http.CanonicalHeaderKey(k)
			existing := req.Header[canonicalKey]
			if len(existing) == 0 {
				req.Header.Set(canonicalKey, valStr)
			} else {
				if canonicalKey == "Cookie" {
					combined := strings.Join(existing, "; ")
					req.Header.Set(canonicalKey, combined+"; "+valStr)
				} else {
					combined := strings.Join(existing, ", ")
					req.Header.Set(canonicalKey, combined+", "+valStr)
				}
			}
		}
	}
}
