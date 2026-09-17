package plugins

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math/rand"
	"net/http"
	"strings"
	"time"
)

// IsStreamRequested checks if a request specifies streaming via JSON body, form values, or query parameters.
func IsStreamRequested(r *http.Request, rawBody []byte, reqJSON map[string]interface{}) bool {
	if reqJSON != nil {
		if v, ok := reqJSON["stream"]; ok {
			switch val := v.(type) {
			case bool:
				if val {
					return true
				}
			case string:
				if val == "true" || val == "1" {
					return true
				}
			case float64:
				if val != 0 {
					return true
				}
			}
		}
	} else if len(rawBody) > 0 {
		var body map[string]interface{}
		if err := json.Unmarshal(rawBody, &body); err == nil {
			if v, ok := body["stream"]; ok {
				switch val := v.(type) {
				case bool:
					if val {
						return true
					}
				case string:
					if val == "true" || val == "1" {
						return true
					}
				case float64:
					if val != 0 {
						return true
					}
				}
			}
		}
	}

	if r != nil {
		if r.FormValue("stream") == "true" || r.FormValue("stream") == "1" {
			return true
		}
		if query := r.URL.Query(); query.Get("stream") == "true" || query.Get("stream") == "1" {
			return true
		}
		if strings.Contains(r.URL.RawQuery, "stream=true") || strings.Contains(r.URL.RawQuery, "stream=1") {
			return true
		}
	}

	return false
}

func GenerateRandomString(length int) string {
	const charset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
	b := make([]byte, length)
	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	for i := range b {
		b[i] = charset[r.Intn(len(charset))]
	}
	return string(b)
}

func BuildSSELine(payload []byte, originalLine []byte) []byte {
	var result bytes.Buffer
	result.Write([]byte("data: "))
	result.Write(payload)
	if bytes.HasSuffix(originalLine, []byte("\r\n")) {
		result.Write([]byte("\r\n"))
	} else {
		result.Write([]byte("\n"))
	}
	return result.Bytes()
}

// BuildOpenAIErrorBody normalizes any upstream error (Gemini, OpenAI-compatible,
// or plain text) into OpenAI {"error": {...}} shape while preserving
// rate-limit / quota details (Retry-After header, RetryInfo.retryDelay,
// QuotaFailure details).
func BuildOpenAIErrorBody(statusCode int, respHeader http.Header, rawBody []byte) []byte {
	retryAfter := ""
	if respHeader != nil {
		retryAfter = respHeader.Get("Retry-After")
	}

	var parsed map[string]interface{}
	var errObj map[string]interface{}
	var topMsg string
	if len(rawBody) > 0 && json.Valid(rawBody) {
		if err := json.Unmarshal(rawBody, &parsed); err == nil {
			if e, ok := parsed["error"].(map[string]interface{}); ok {
				errObj = e
			} else if s, ok := parsed["error"].(string); ok {
				topMsg = s
			}
			if m, ok := parsed["message"].(string); ok && m != "" {
				topMsg = m
			}
		}
	}

	message := ""
	upstreamStatus := ""
	upstreamType := ""
	var upstreamDetails interface{}
	if errObj != nil {
		if m, ok := errObj["message"].(string); ok && m != "" {
			message = m
		}
		if s, ok := errObj["status"].(string); ok {
			upstreamStatus = s
		}
		if t, ok := errObj["type"].(string); ok {
			upstreamType = t
		}
		if d, ok := errObj["details"]; ok {
			upstreamDetails = d
		}
	}
	if message == "" {
		message = topMsg
	}
	if message == "" {
		s := strings.TrimSpace(string(rawBody))
		if len(s) > 2000 {
			s = s[:2000] + "..."
		}
		message = s
	}
	if message == "" {
		if statusCode != 0 {
			message = fmt.Sprintf("Upstream request failed with status %d", statusCode)
		} else {
			message = "Upstream request failed"
		}
	}

	// Extract RetryInfo.retryDelay from Gemini-style details.
	retryDelay := ""
	if arr, ok := upstreamDetails.([]interface{}); ok {
		for _, item := range arr {
			m, ok := item.(map[string]interface{})
			if !ok {
				continue
			}
			if t, _ := m["@type"].(string); strings.Contains(t, "RetryInfo") {
				if d, _ := m["retryDelay"].(string); d != "" {
					retryDelay = d
					break
				}
			}
		}
	}

	effectiveRetry := retryAfter
	if effectiveRetry == "" {
		effectiveRetry = retryDelay
	}
	// Keep retry/quota hints visible to clients that only surface message.
	if effectiveRetry != "" && !strings.Contains(message, effectiveRetry) {
		message = fmt.Sprintf("%s (retry after %s)", message, effectiveRetry)
	}

	errType := "api_error"
	switch {
	case statusCode == 429:
		errType = "rate_limit_error"
	case statusCode == 401:
		errType = "authentication_error"
	case statusCode == 403:
		errType = "permission_error"
	case statusCode == 400:
		errType = "invalid_request_error"
	}
	if upstreamType != "" && (statusCode < 400 || errType == "api_error") {
		errType = upstreamType
	}

	out := map[string]interface{}{
		"message": message,
		"type":    errType,
		"code":    statusCode,
	}
	if effectiveRetry != "" {
		out["retry_after"] = effectiveRetry
	}
	if upstreamStatus != "" {
		out["upstream_status"] = upstreamStatus
	}
	if upstreamDetails != nil {
		out["upstream_details"] = upstreamDetails
	}

	b, err := json.Marshal(map[string]interface{}{"error": out})
	if err != nil {
		return []byte(fmt.Sprintf(`{"error":{"message":%q,"type":"api_error","code":%d}}`, message, statusCode))
	}
	return b
}

// DeepMerge merges src map into dst map with slice deduplication and map normalization.
func DeepMerge(dst, src map[string]interface{}) {
	for k, v := range src {
		srcChildMap, isSrcMap := toMapStringInterface(v)
		if isSrcMap {
			if dstChildMap, isDstMap := toMapStringInterface(dst[k]); isDstMap {
				DeepMerge(dstChildMap, srcChildMap)
				dst[k] = dstChildMap
			} else {
				dst[k] = normalizeMap(srcChildMap)
			}
			continue
		}

		srcSlice, isSrcSlice := toSliceInterface(v)
		if isSrcSlice {
			if dstSlice, isDstSlice := toSliceInterface(dst[k]); isDstSlice {
				merged := append([]interface{}{}, dstSlice...)
				for _, srcItem := range srcSlice {
					exists := false
					for _, dstItem := range merged {
						if areEqualJSON(dstItem, srcItem) {
							exists = true
							break
						}
					}
					if !exists {
						merged = append(merged, normalizeValue(srcItem))
					}
				}
				dst[k] = merged
			} else {
				dst[k] = normalizeSlice(srcSlice)
			}
			continue
		}

		dst[k] = v
	}
}

func toMapStringInterface(v interface{}) (map[string]interface{}, bool) {
	if v == nil {
		return nil, false
	}
	if m, ok := v.(map[string]interface{}); ok {
		return m, true
	}
	if m, ok := v.(map[interface{}]interface{}); ok {
		res := make(map[string]interface{})
		for k, val := range m {
			strKey := fmt.Sprintf("%v", k)
			res[strKey] = val
		}
		return res, true
	}
	return nil, false
}

func toSliceInterface(v interface{}) ([]interface{}, bool) {
	if v == nil {
		return nil, false
	}
	if s, ok := v.([]interface{}); ok {
		return s, true
	}
	return nil, false
}

func areEqualJSON(a, b interface{}) bool {
	normA := normalizeValue(a)
	normB := normalizeValue(b)
	ja, err1 := json.Marshal(normA)
	jb, err2 := json.Marshal(normB)
	if err1 != nil || err2 != nil {
		return normA == normB
	}
	return string(ja) == string(jb)
}

func normalizeValue(v interface{}) interface{} {
	if v == nil {
		return nil
	}
	if m, ok := toMapStringInterface(v); ok {
		return normalizeMap(m)
	}
	if s, ok := toSliceInterface(v); ok {
		return normalizeSlice(s)
	}
	return v
}

func normalizeSlice(s []interface{}) []interface{} {
	res := make([]interface{}, len(s))
	for i, v := range s {
		res[i] = normalizeValue(v)
	}
	return res
}

func normalizeMap(m map[string]interface{}) map[string]interface{} {
	res := make(map[string]interface{})
	for k, v := range m {
		res[k] = normalizeValue(v)
	}
	return res
}

func GetHoldbackPrefix(s, target string) string {
	if s == "" || target == "" {
		return ""
	}
	maxLen := len(s)
	if len(target) < maxLen {
		maxLen = len(target)
	}
	for l := maxLen; l > 0; l-- {
		suffix := s[len(s)-l:]
		prefix := target[:l]
		if suffix == prefix {
			return suffix
		}
	}
	return ""
}
