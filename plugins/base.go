package plugins

import (
	"fmt"
	"net/http"
	"strings"
	"sync"
)

// ProviderPlugin is the interface that all upstream provider plugins must implement.
type ProviderPlugin interface {
	// Name returns the identifier of the plugin, e.g. "openai", "gemini", "claude".
	Name() string

	// ModifyHeaders applies header modifications (auth key, delete/extra rules, etc.).
	ModifyHeaders(req *http.Request, ctx *Context)

	// ModifyResponseHeaders copies and filters upstream response headers back to the client.
	ModifyResponseHeaders(resp *http.Response, clientHeader http.Header, ctx *Context)

	// TransformRequest transforms the incoming OpenAI request body to the provider's upstream format.
	TransformRequest(endpoint EndpointType, rawBody []byte, req *http.Request, ctx *Context) ([]byte, string, error)

	// TransformResponse transforms the provider's non-streaming response body back to OpenAI format.
	TransformResponse(endpoint EndpointType, resp *http.Response, body []byte, ctx *Context) ([]byte, string, error)

	// TransformStreamChunk transforms a single SSE line chunk from upstream into OpenAI SSE format.
	TransformStreamChunk(endpoint EndpointType, chunk []byte, ctx *Context) ([]byte, error)

	// BuildTargetURL constructs the destination upstream URL for a request.
	BuildTargetURL(endpoint EndpointType, req *http.Request, ctx *Context) string
}

var (
	mu       sync.RWMutex
	registry = make(map[string]ProviderPlugin)
)

// Register registers a plugin under its Name().
func Register(plugin ProviderPlugin) {
	mu.Lock()
	defer mu.Unlock()
	if plugin == nil {
		panic("plugins: Register plugin is nil")
	}
	name := strings.ToLower(plugin.Name())
	registry[name] = plugin
}

// Get retrieves a registered plugin by apiType.
// If apiType is empty or not found, it returns the default "openai" plugin.
func Get(apiType string) ProviderPlugin {
	mu.RLock()
	defer mu.RUnlock()

	key := strings.ToLower(strings.TrimSpace(apiType))
	if key != "" {
		if p, ok := registry[key]; ok {
			return p
		}
	}
	if p, ok := registry["openai"]; ok {
		return p
	}
	return &BasePlugin{}
}

// BasePlugin provides standard default implementation (pass-through for OpenAI format).
type BasePlugin struct{}

func (b *BasePlugin) Name() string {
	return "base"
}

func (b *BasePlugin) ModifyHeaders(req *http.Request, ctx *Context) {
	if ctx.AuthKey == "sk-dummy" {
		req.Header.Del("Authorization")
		req.Header.Del("x-goog-api-key")
	} else if ctx.AuthKey != "" {
		req.Header.Set("Authorization", "Bearer "+ctx.AuthKey)
		req.Header.Del("x-goog-api-key")
	}

	// Standard header deletion
	for _, hName := range ctx.RequestHeaders.Delete {
		req.Header.Del(hName)
	}

	// Standard header additions/merges
	for _, extraMap := range ctx.RequestHeaders.Extra {
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

func (b *BasePlugin) ModifyResponseHeaders(resp *http.Response, clientHeader http.Header, ctx *Context) {
	isSSE := strings.Contains(resp.Header.Get("Content-Type"), "text/event-stream")
	for k, vv := range resp.Header {
		if !isSSE && k == "Content-Length" {
			continue
		}
		for _, v := range vv {
			clientHeader.Add(k, v)
		}
	}
}

func (b *BasePlugin) TransformRequest(endpoint EndpointType, rawBody []byte, req *http.Request, ctx *Context) ([]byte, string, error) {
	return rawBody, "application/json; charset=utf-8", nil
}

func (b *BasePlugin) TransformResponse(endpoint EndpointType, resp *http.Response, body []byte, ctx *Context) ([]byte, string, error) {
	contentType := resp.Header.Get("Content-Type")
	if contentType == "" {
		contentType = "application/json; charset=utf-8"
	}
	return body, contentType, nil
}

func (b *BasePlugin) TransformStreamChunk(endpoint EndpointType, chunk []byte, ctx *Context) ([]byte, error) {
	return chunk, nil
}

func (b *BasePlugin) BuildTargetURL(endpoint EndpointType, req *http.Request, ctx *Context) string {
	return ctx.UpstreamURL
}
