package main

import (
	"errors"
	"math/rand"
)

type SelectedRoute struct {
	ModelProvider *ModelProviderConfig
	AuthKey       string
	KeyIndex      int
}

type Router struct {
	cfg   *Config
	state *StateManager
}

func NewRouter(cfg *Config, state *StateManager) *Router {
	return &Router{
		cfg:   cfg,
		state: state,
	}
}

// Find matching ModelConfig
func (r *Router) findModelConfig(modelName string) *ModelConfig {
	for i := range r.cfg.Models {
		if r.cfg.Models[i].Name == modelName {
			return &r.cfg.Models[i]
		}
	}
	return nil
}

// Find matching ProviderConf
func (r *Router) findProviderConf(providerName string) *ProviderConf {
	for i := range r.cfg.Providers {
		if r.cfg.Providers[i].Name == providerName {
			return &r.cfg.Providers[i]
		}
	}
	return nil
}

// SelectRoute chooses the best provider and auth key for the requested model.
// Returns SelectedRoute or an error if no upstreams are available.
func (r *Router) SelectRoute(modelName string) (*SelectedRoute, error) {
	modelCfg := r.findModelConfig(modelName)
	if modelCfg == nil {
		return nil, errors.New("model not found")
	}

	if len(modelCfg.Providers) == 0 {
		return nil, errors.New("no providers configured for this model")
	}

	// Iterate through the model's providers in order
	for i := range modelCfg.Providers {
		modelProvider := &modelCfg.Providers[i]
		providerConf := r.findProviderConf(modelProvider.Name)
		if providerConf == nil || len(providerConf.AuthKeys) == 0 {
			continue // Skip if provider config or keys are missing
		}

		numKeys := len(providerConf.AuthKeys)

		// Generate a random permutation of indices (shuffled array from 0 to numKeys-1)
		indices := rand.Perm(numKeys)
		found := false
		var selectedKey string
		var selectedIndex int

		for _, idx := range indices {
			key := providerConf.AuthKeys[idx]

			// Check 429 lock status
			if r.state.IsUpstreamLocked(key) {
				continue
			}

			// Determine which rate limit to apply and check token bucket
			var limitKey string
			var limitAmt float64

			if modelProvider.RateLimit != nil {
				// Use Model-specific limit (independent bucket keyed by model name)
				limitKey = modelName
				limitAmt = *modelProvider.RateLimit
			} else {
				// Use Provider global limit (independent bucket keyed by provider name)
				limitKey = providerConf.Name
				limitAmt = providerConf.RateLimit
			}

			// If allowed, it consumes a token and we lock down this key
			if r.state.AllowUpstream(key, limitKey, limitAmt) {
				selectedKey = key
				selectedIndex = idx
				found = true
				break
			}
		}

		if found {
			return &SelectedRoute{
				ModelProvider: modelProvider,
				AuthKey:       selectedKey,
				KeyIndex:      selectedIndex,
			}, nil
		}
	}

	return nil, errors.New("all providers/keys are currently rate-limited or locked")
}
