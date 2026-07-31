package plugins

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math/rand"
	"time"
)

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
