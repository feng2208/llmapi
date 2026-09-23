package main

import (
	"errors"
	"sync"
)

type SelectedRoute struct {
	ModelProvider *ModelProviderConfig
	AuthKey       string
	KeyIndex      int
}

type Router struct {
	cfg            *Config
	state          *StateManager
	mu             sync.Mutex
	activeKeyIndex map[string]int
}

func NewRouter(cfg *Config, state *StateManager) *Router {
	return &Router{
		cfg:            cfg,
		state:          state,
		activeKeyIndex: make(map[string]int),
	}
}

// AdvanceKey advances the active key index for a model-provider pair to the next key.
func (r *Router) AdvanceKey(modelName, providerName string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	mapKey := modelName + ":" + providerName
	r.activeKeyIndex[mapKey]++
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

		mapKey := modelName + ":" + modelProvider.Name
		r.mu.Lock()
		currIdx := r.activeKeyIndex[mapKey] % numKeys
		if currIdx < 0 {
			currIdx = 0
		}

		found := false
		var selectedKey string
		var selectedIndex int

		for k := 0; k < numKeys; k++ {
			idx := (currIdx + k) % numKeys
			key := providerConf.AuthKeys[idx]

			// Check 429 lock status
			if r.state.IsUpstreamLocked(key) {
				continue
			}

			// If model-level rate limit is set, check token bucket
			if modelProvider.RateLimit != nil {
				if !r.state.AllowUpstream(key, modelName, *modelProvider.RateLimit) {
					continue
				}
			}

			selectedKey = key
			selectedIndex = idx
			r.activeKeyIndex[mapKey] = idx
			found = true
			break
		}
		r.mu.Unlock()

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
