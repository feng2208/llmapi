package main

import (
	"net/http"

	"llmapi/plugins"
)

func buildPluginContext(route *SelectedRoute, r *http.Request, cfg *Config, proxyMgr *ProxyManager) *plugins.Context {
	var httpClient *http.Client
	if proxyMgr != nil && route != nil && cfg != nil {
		if client, err := proxyMgr.GetClient(route.ModelProvider, cfg.Proxy); err == nil {
			httpClient = client
		}
	}

	modelName := ""
	providerName := ""
	upstreamURL := ""
	authKey := ""
	reasoningStart := ""
	reasoningEnd := ""
	reasoningField := ""
	var reqHeadersConfig plugins.RequestHeadersConfig
	var reqBodyConfig plugins.RequestBodyConfig

	if route != nil {
		modelName = route.ModelProvider.Model
		providerName = route.ModelProvider.Name
		upstreamURL = route.ModelProvider.Upstream
		authKey = route.AuthKey
		reasoningStart = route.ModelProvider.ReasoningStart
		reasoningEnd = route.ModelProvider.ReasoningEnd
		reasoningField = route.ModelProvider.ReasoningField
		reqHeadersConfig = plugins.RequestHeadersConfig{
			Delete: route.ModelProvider.RequestHeaders.Delete,
			Extra:  route.ModelProvider.RequestHeaders.Extra,
		}
		reqBodyConfig = plugins.RequestBodyConfig{
			Delete: route.ModelProvider.RequestBody.Delete,
			Extra:  route.ModelProvider.RequestBody.Extra,
		}
	}

	isStream := false
	if r != nil {
		isStream = plugins.IsStreamRequested(r, nil, nil)
	}

	return &plugins.Context{
		UpstreamURL:    upstreamURL,
		ModelName:      modelName,
		ProviderName:   providerName,
		AuthKey:        authKey,
		ReasoningStart: reasoningStart,
		ReasoningEnd:   reasoningEnd,
		ReasoningField: reasoningField,
		IsStream:       isStream,
		Request:        r,
		HTTPClient:     httpClient,
		RequestHeaders: reqHeadersConfig,
		RequestBody:    reqBodyConfig,
	}
}

// ModifyRequestBody applies the delete, extra, and model replacement rules using plugins.
func ModifyRequestBody(rawBody []byte, route *SelectedRoute, r *http.Request, cfg *Config, proxyMgr *ProxyManager) ([]byte, string, error) {
	p := plugins.Get(route.ModelProvider.ApiType)
	ctx := buildPluginContext(route, r, cfg, proxyMgr)
	return p.TransformRequest(plugins.EndpointChat, rawBody, r, ctx)
}

func ModifyImageRequestBody(rawBody []byte, route *SelectedRoute, r *http.Request) ([]byte, string, error) {
	p := plugins.Get(route.ModelProvider.ApiType)
	ctx := buildPluginContext(route, r, nil, nil)
	return p.TransformRequest(plugins.EndpointImageGeneration, rawBody, r, ctx)
}

type StreamExtractor = plugins.StreamExtractor

func NewStreamExtractor(startTag, endTag, reasoningField string) *StreamExtractor {
	return plugins.NewStreamExtractor(startTag, endTag, reasoningField)
}

func ProcessJSONResponse(respBody []byte, startTag, endTag, reasoningField string) []byte {
	return plugins.ProcessJSONResponse(respBody, startTag, endTag, reasoningField)
}

func ModifyImageEditRequestBody(rawBody []byte, route *SelectedRoute, r *http.Request) ([]byte, string, error) {
	p := plugins.Get(route.ModelProvider.ApiType)
	ctx := buildPluginContext(route, r, nil, nil)
	return p.TransformRequest(plugins.EndpointImageEdit, rawBody, r, ctx)
}

func ModifyAudioSpeechRequestBody(rawBody []byte, route *SelectedRoute, r *http.Request) ([]byte, string, error) {
	p := plugins.Get(route.ModelProvider.ApiType)
	ctx := buildPluginContext(route, r, nil, nil)
	return p.TransformRequest(plugins.EndpointAudioSpeech, rawBody, r, ctx)
}

func ModifyAudioTranscriptionRequestBody(rawBody []byte, route *SelectedRoute, r *http.Request, cfg *Config, proxyMgr *ProxyManager) ([]byte, string, error) {
	p := plugins.Get(route.ModelProvider.ApiType)
	ctx := buildPluginContext(route, r, cfg, proxyMgr)
	return p.TransformRequest(plugins.EndpointAudioTranscription, rawBody, r, ctx)
}

// ModifyRequestHeaders applies configured request header deletions and additions/merges using plugins.
func ModifyRequestHeaders(req *http.Request, route *SelectedRoute) {
	p := plugins.Get(route.ModelProvider.ApiType)
	ctx := buildPluginContext(route, req, nil, nil)
	p.ModifyHeaders(req, ctx)
}
