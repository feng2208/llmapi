package plugins

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

// OpenAIPlugin is the default plugin handling standard OpenAI-compatible upstreams.
type OpenAIPlugin struct {
	BasePlugin
}

func (p *OpenAIPlugin) Name() string {
	return "openai"
}

func (p *OpenAIPlugin) TransformRequest(endpoint EndpointType, rawBody []byte, req *http.Request, ctx *Context) ([]byte, string, error) {
	switch endpoint {
	case EndpointImageEdit:
		return transformOpenAIImageEditRequest(req, ctx)
	case EndpointAudioTranscription:
		return transformOpenAIAudioTranscriptionRequest(req, ctx)
	default:
		return transformOpenAIJsonRequest(rawBody, ctx)
	}
}

func transformOpenAIJsonRequest(rawBody []byte, ctx *Context) ([]byte, string, error) {
	var body map[string]interface{}
	if err := json.Unmarshal(rawBody, &body); err != nil {
		return nil, "", fmt.Errorf("failed to parse JSON request body: %w", err)
	}

	// 1. Replace Model Name
	if ctx.ModelName != "" {
		body["model"] = ctx.ModelName
	}

	// 2. Apply delete & extra config
	applyDeleteAndExtra(body, ctx.RequestBody)

	// 3. Inject skip_thought_signature_validator conditionally for gemini provider name
	if ctx.ProviderName == "gemini" {
		injectGeminiSignatureValidator(body)
	}

	modified, err := json.Marshal(body)
	if err != nil {
		return nil, "", fmt.Errorf("failed to marshal modified JSON: %w", err)
	}

	return modified, "application/json; charset=utf-8", nil
}

func transformOpenAIImageEditRequest(r *http.Request, ctx *Context) ([]byte, string, error) {
	if r.MultipartForm == nil {
		return nil, "", errors.New("multipart form not parsed")
	}

	var buf bytes.Buffer
	writer := multipart.NewWriter(&buf)

	fields := make(map[string]string)
	for k, values := range r.MultipartForm.Value {
		if len(values) > 0 {
			fields[k] = values[0]
		}
	}

	if ctx.ModelName != "" {
		fields["model"] = ctx.ModelName
	}

	for _, item := range ctx.RequestBody.Delete {
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
	for _, extraMap := range ctx.RequestBody.Extra {
		for k, v := range extraMap {
			fields[k] = fmt.Sprintf("%v", v)
		}
	}

	for k, v := range fields {
		if err := writer.WriteField(k, v); err != nil {
			return nil, "", err
		}
	}

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

	if err := writer.Close(); err != nil {
		return nil, "", err
	}

	return buf.Bytes(), writer.FormDataContentType(), nil
}

func transformOpenAIAudioTranscriptionRequest(r *http.Request, ctx *Context) ([]byte, string, error) {
	if r.MultipartForm == nil {
		return nil, "", errors.New("multipart form not parsed")
	}

	var buf bytes.Buffer
	writer := multipart.NewWriter(&buf)

	fields := make(map[string][]string)
	for k, values := range r.MultipartForm.Value {
		fields[k] = values
	}

	if ctx.ModelName != "" {
		fields["model"] = []string{ctx.ModelName}
	}

	for _, item := range ctx.RequestBody.Delete {
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
	for _, extraMap := range ctx.RequestBody.Extra {
		for k, v := range extraMap {
			fields[k] = []string{fmt.Sprintf("%v", v)}
		}
	}

	for k, vv := range fields {
		for _, v := range vv {
			if err := writer.WriteField(k, v); err != nil {
				return nil, "", err
			}
		}
	}

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

	if err := writer.Close(); err != nil {
		return nil, "", err
	}

	return buf.Bytes(), writer.FormDataContentType(), nil
}

func applyDeleteAndExtra(body map[string]interface{}, reqBodyConfig RequestBodyConfig) {
	for _, item := range reqBodyConfig.Delete {
		if str, ok := item.(string); ok {
			DeleteNestedPath(body, str)
		} else if m, ok := item.(map[string]interface{}); ok {
			for k := range m {
				DeleteNestedPath(body, k)
			}
		} else if m, ok := item.(map[interface{}]interface{}); ok {
			for k := range m {
				if s, ok := k.(string); ok {
					DeleteNestedPath(body, s)
				}
			}
		}
	}

	for _, extraMap := range reqBodyConfig.Extra {
		DeepMerge(body, extraMap)
	}
}

func DeleteNestedPath(data interface{}, path string) {
	if data == nil || path == "" {
		return
	}

	idx := strings.Index(path, ".")
	if idx != -1 {
		currKey := path[:idx]
		restPath := path[idx+1:]

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
						DeleteNestedPath(item, restPath)
					}
				}
			} else {
				if slice, ok := val.([]interface{}); ok {
					for _, item := range slice {
						DeleteNestedPath(item, restPath)
					}
				} else {
					DeleteNestedPath(val, restPath)
				}
			}
		}
		return
	}

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

func injectGeminiSignatureValidator(body map[string]interface{}) {
	if messages, ok := body["messages"].([]interface{}); ok {
		for _, rawMsg := range messages {
			if msg, ok := rawMsg.(map[string]interface{}); ok {
				role, _ := msg["role"].(string)
				if role == "assistant" || role == "model" {
					if toolCalls, ok := msg["tool_calls"].([]interface{}); ok {
						for _, rawCall := range toolCalls {
							if call, ok := rawCall.(map[string]interface{}); ok {
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

								google["thought_signature"] = "skip_thought_signature_validator"
							}
						}
					}
				}
			}
		}
	}
}

func (p *OpenAIPlugin) TransformResponse(endpoint EndpointType, resp *http.Response, body []byte, ctx *Context) ([]byte, string, error) {
	contentType := ""
	if resp != nil {
		contentType = resp.Header.Get("Content-Type")
	}
	if contentType == "" {
		contentType = "application/json; charset=utf-8"
	}

	if endpoint == EndpointChat {
		modified := ProcessJSONResponse(body, ctx.ReasoningStart, ctx.ReasoningEnd, ctx.ReasoningField)
		return modified, contentType, nil
	}

	return body, contentType, nil
}

func (p *OpenAIPlugin) TransformStreamChunk(endpoint EndpointType, chunk []byte, ctx *Context) ([]byte, error) {
	if endpoint == EndpointChat && (ctx.ReasoningStart != "" || ctx.ReasoningEnd != "" || ctx.ReasoningField != "") {
		extractor, ok := ctx.StreamState.(*StreamExtractor)
		if !ok || extractor == nil {
			extractor = NewStreamExtractor(ctx.ReasoningStart, ctx.ReasoningEnd, ctx.ReasoningField)
			ctx.StreamState = extractor
		}
		return extractor.ProcessSSELine(chunk), nil
	}
	return chunk, nil
}

// ----------------------------------------------------------------------------
// Reasoning Extraction Logic for OpenAI-compatible Upstreams
// ----------------------------------------------------------------------------

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

	if val, ok := container[reasoningField]; ok {
		extractedReasoning.WriteString(extractReasoningValue(val))
		delete(container, reasoningField)
	}

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

	if se.reasoningField != "" {
		processReasoningField(delta, se.reasoningField)
	}

	if se.startTag != "" && se.endTag != "" {
		content, _ := delta["content"].(string)
		se.fullBuffer += content

		var activeContent string
		var activeReasoning string

		idxStart := strings.Index(se.fullBuffer, se.startTag)
		if idxStart == -1 {
			holdback := GetHoldbackPrefix(se.fullBuffer, se.startTag)
			activeContent = se.fullBuffer[:len(se.fullBuffer)-len(holdback)]
			activeReasoning = ""
		} else {
			contentPart := se.fullBuffer[:idxStart]
			idxEnd := strings.Index(se.fullBuffer, se.endTag)
			if idxEnd == -1 {
				reasoningSoFar := se.fullBuffer[idxStart+len(se.startTag):]
				holdback := GetHoldbackPrefix(reasoningSoFar, se.endTag)
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

	return BuildSSELine(modifiedPayload, line)
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

func init() {
	Register(&OpenAIPlugin{})
}
