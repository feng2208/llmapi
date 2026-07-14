package main

import (
	"fmt"
	"io"
	"os"
	"time"

	"gopkg.in/yaml.v3"
)

type Config struct {
	Listen      string         `yaml:"listen"`
	MaxBodySize int64          `yaml:"max_body_size"`
	Proxy       string         `yaml:"proxy"`
	Clients     ClientsConfig  `yaml:"clients"`
	Models      []ModelConfig  `yaml:"models"`
	Providers   []ProviderConf `yaml:"providers"`
}

type ClientsConfig struct {
	RateLimit float64      `yaml:"rate_limit"` // Requests per minute
	Auth      []ClientAuth `yaml:"auth"`
}

type ClientAuth struct {
	Name      string  `yaml:"name"`
	Key       string  `yaml:"key"`
	RateLimit float64 `yaml:"rate_limit,omitempty"` // Requests per minute (optional, overrides global client limit)
}

type ModelConfig struct {
	Name      string                `yaml:"name"`
	Providers []ModelProviderConfig `yaml:"providers"`
}

type ModelProviderConfig struct {
	Name           string            `yaml:"name"`
	Upstream       string            `yaml:"upstream"`
	Model          string            `yaml:"model"`
	RateLimit      *float64          `yaml:"rate_limit,omitempty"` // Requests per minute (optional, independent)
	TimeoutStr     string            `yaml:"timeout"`              // e.g. "120s"
	Proxy          string            `yaml:"proxy"`
	ReasoningStart string            `yaml:"reasoning_start"`
	ReasoningEnd   string            `yaml:"reasoning_end"`
	RequestBody    RequestBodyConfig `yaml:"request_body"`
	ApiType        string            `yaml:"api_type"`
}

type RequestBodyConfig struct {
	Delete []interface{}            `yaml:"delete"`
	Extra  []map[string]interface{} `yaml:"extra"`
}

type ProviderConf struct {
	Name      string   `yaml:"name"`
	RateLimit float64  `yaml:"rate_limit"` // Requests per minute for single provider
	AuthKeys  []string `yaml:"auth_keys"`
}

// Timeout parses Duration from ModelProviderConfig.TimeoutStr.
func (mpc *ModelProviderConfig) Timeout() time.Duration {
	if mpc.TimeoutStr == "" {
		return 300 * time.Second // Default timeout 5 min
	}
	d, err := time.ParseDuration(mpc.TimeoutStr)
	if err != nil {
		return 300 * time.Second
	}
	return d
}

// LoadConfig reads and parses YAML configuration.
func LoadConfig(path string) (*Config, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open config: %w", err)
	}
	defer file.Close()

	data, err := io.ReadAll(file)
	if err != nil {
		return nil, fmt.Errorf("failed to read config: %w", err)
	}

	var cfg Config
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("failed to parse yaml: %w", err)
	}

	return &cfg, nil
}
