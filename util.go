package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"math/rand"
	"net/http"
)

func generateRandomString(length int) string {
	const charset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
	b := make([]byte, length)
	for i := range b {
		b[i] = charset[rand.Intn(len(charset))]
	}
	return string(b)
}

// FormatJSON pretty-prints the JSON byte array with 2 spaces indent.
func FormatJSON(data []byte) string {
	var temp interface{}
	if err := json.Unmarshal(data, &temp); err != nil {
		return string(data)
	}
	formatted, err := json.MarshalIndent(temp, "", "  ")
	if err != nil {
		return string(data)
	}
	return string(formatted)
}

type loggingReader struct {
	r             io.Reader
	headerPrinted bool
}

func (lr *loggingReader) Read(p []byte) (n int, err error) {
	n, err = lr.r.Read(p)
	if n > 0 {
		if !lr.headerPrinted {
			fmt.Printf("[DEBUG] --- UPSTREAM RESPONSE BODY (STREAMING) ---\n")
			lr.headerPrinted = true
		}
		fmt.Print(string(p[:n]))
	}
	return
}

type flushingWriter struct {
	w io.Writer
	f http.Flusher
}

func (fw flushingWriter) Write(p []byte) (n int, err error) {
	n, err = fw.w.Write(p)
	if fw.f != nil {
		fw.f.Flush()
	}
	return
}

func getHoldbackPrefix(s, target string) string {
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

// Deep merges src map into dst map.
func deepMerge(dst, src map[string]interface{}) {
	for k, v := range src {
		srcChildMap, isSrcMap := toMapStringInterface(v)
		if isSrcMap {
			if dstChildMap, isDstMap := toMapStringInterface(dst[k]); isDstMap {
				deepMerge(dstChildMap, srcChildMap)
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

func buildSSELine(payload []byte, originalLine []byte) []byte {
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
