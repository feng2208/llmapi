package plugins

import (
	"net/http"
)

type EndpointType string

const (
	EndpointChat               EndpointType = "chat"
	EndpointImageGeneration   EndpointType = "image_generation"
	EndpointImageEdit         EndpointType = "image_edit"
	EndpointAudioSpeech        EndpointType = "audio_speech"
	EndpointAudioTranscription EndpointType = "audio_transcription"
)

// RequestBodyConfig holds delete and extra rules for request body.
type RequestBodyConfig struct {
	Delete []interface{}
	Extra  []map[string]interface{}
}

// RequestHeadersConfig holds delete and extra rules for request headers.
type RequestHeadersConfig struct {
	Delete []string
	Extra  []map[string]interface{}
}

// Context contains all routing and configuration parameters needed by plugins.
type Context struct {
	UpstreamURL    string
	ModelName      string
	ProviderName   string
	AuthKey        string
	ReasoningStart string
	ReasoningEnd   string
	ReasoningField string
	IsStream       bool
	RawRequestBody []byte
	Request        *http.Request
	HTTPClient     *http.Client
	RequestHeaders RequestHeadersConfig
	RequestBody    RequestBodyConfig
	StreamState    interface{}
}
