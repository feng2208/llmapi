package main

import (
	"errors"
	"fmt"
	"os"
	"time"

	"gopkg.in/yaml.v3"
)

type Config struct {
	Listen      string               `yaml:"listen"`
	MaxBodySize int64                `yaml:"max_body_size"`
	Proxy       string               `yaml:"proxy"`
	Clients     ClientsConfig        `yaml:"clients"`
	Models      []ModelConfig        `yaml:"models"`
	Providers   []ProviderAuthConfig `yaml:"providers"`
}

type ClientsConfig struct {
	RateLimit int             `yaml:"rate_limit"`
	Auth      []ClientAuthKey `yaml:"auth"`
}

type ClientAuthKey struct {
	Name      string `yaml:"name"`
	Key       string `yaml:"key"`
	RateLimit int    `yaml:"rate_limit"`
}

type ModelConfig struct {
	Name      string                `yaml:"name"`
	Providers []ModelProviderConfig `yaml:"providers"`
}

type ModelProviderConfig struct {
	Name            string        `yaml:"name"`
	Upstream        string        `yaml:"upstream"`
	Model           string        `yaml:"model"`
	RateLimit       int           `yaml:"rate_limit"`
	Timeout         time.Duration `yaml:"timeout"`
	Proxy           string        `yaml:"proxy"`
	ReasoningEffort string        `yaml:"reasoning_effort"`
	IncludeThoughts bool          `yaml:"include_thoughts"`
}

type ProviderAuthConfig struct {
	Name      string   `yaml:"name"`
	RateLimit int      `yaml:"rate_limit"`
	AuthKeys  []string `yaml:"auth_keys"`
}

func loadConfig(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read config file: %w", err)
	}

	var cfg Config
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("failed to parse config file: %w", err)
	}

	// Set defaults
	if cfg.Listen == "" {
		cfg.Listen = "0.0.0.0:3000"
	}
	if cfg.MaxBodySize == 0 {
		cfg.MaxBodySize = 10485760 // 10MB
	}
	if cfg.Clients.RateLimit == 0 {
		cfg.Clients.RateLimit = 10
	}

	for i := range cfg.Models {
		for j := range cfg.Models[i].Providers {
			if cfg.Models[i].Providers[j].Timeout == 0 {
				cfg.Models[i].Providers[j].Timeout = 120 * time.Second
			}
		}
	}

	// Validate config
	if len(cfg.Models) == 0 {
		return nil, errors.New("at least one model must be configured")
	}

	providerMap := make(map[string]bool)
	for _, p := range cfg.Providers {
		if len(p.AuthKeys) == 0 {
			return nil, fmt.Errorf("provider %s must have at least one auth key", p.Name)
		}
		providerMap[p.Name] = true
	}

	modelMap := make(map[string]bool)
	for _, m := range cfg.Models {
		if modelMap[m.Name] {
			return nil, fmt.Errorf("duplicate model name: %s", m.Name)
		}
		modelMap[m.Name] = true

		if len(m.Providers) == 0 {
			return nil, fmt.Errorf("model %s must have at least one provider", m.Name)
		}

		for _, p := range m.Providers {
			if !providerMap[p.Name] {
				return nil, fmt.Errorf("model %s references undefined provider: %s", m.Name, p.Name)
			}
		}
	}

	return &cfg, nil
}
