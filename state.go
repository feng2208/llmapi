package main

import (
	"sync"
	"time"

	"golang.org/x/time/rate"
)

type backoffState struct {
	consecutive429s int
	lockedUntil     time.Time
}

type StateManager struct {
	mu               sync.RWMutex
	clientLimiters   map[string]*rate.Limiter
	upstreamLimiters map[string]*rate.Limiter
	backoffStates    map[string]*backoffState
}

func NewStateManager() *StateManager {
	return &StateManager{
		clientLimiters:   make(map[string]*rate.Limiter),
		upstreamLimiters: make(map[string]*rate.Limiter),
		backoffStates:    make(map[string]*backoffState),
	}
}

// AllowClient checks if the client is allowed to make a request.
func (sm *StateManager) AllowClient(clientKey string, limit float64) bool {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	limiter, exists := sm.clientLimiters[clientKey]
	if !exists {
		// Limit is requests per minute
		burst := int(limit)
		if burst < 1 {
			burst = 1
		}
		limiter = rate.NewLimiter(rate.Limit(limit/60.0), burst)
		sm.clientLimiters[clientKey] = limiter
	}

	return limiter.Allow()
}

// AllowUpstream checks if the upstream key is allowed under its rate limit and backoff status.
func (sm *StateManager) AllowUpstream(authKey string, limitKey string, limit float64) bool {
	sm.mu.RLock()
	// First check backoff lock
	state, locked := sm.backoffStates[authKey]
	if locked && time.Now().Before(state.lockedUntil) {
		sm.mu.RUnlock()
		return false
	}
	sm.mu.RUnlock()

	// Now check rate limit
	sm.mu.Lock()
	defer sm.mu.Unlock()

	// Recheck lock inside write lock to avoid race
	state, locked = sm.backoffStates[authKey]
	if locked && time.Now().Before(state.lockedUntil) {
		return false
	}

	// Dynamic token bucket key: "authKey:limitKey"
	// limitKey is either the model name or the provider name depending on which limit is applied.
	mapKey := authKey + ":" + limitKey
	limiter, exists := sm.upstreamLimiters[mapKey]
	if !exists {
		burst := int(limit)
		if burst < 1 {
			burst = 1
		}
		limiter = rate.NewLimiter(rate.Limit(limit/60.0), burst)
		sm.upstreamLimiters[mapKey] = limiter
	}

	return limiter.Allow()
}

// RecordUpstreamResult records HTTP status code for upstream key to update backoff status.
func (sm *StateManager) RecordUpstreamResult(authKey string, statusCode int) {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	state, exists := sm.backoffStates[authKey]
	if !exists {
		state = &backoffState{}
		sm.backoffStates[authKey] = state
	}

	if statusCode == 429 {
		state.consecutive429s++
		if state.consecutive429s >= 3 {
			// Lock until next UTC 00:00
			now := time.Now().UTC()
			nextMidnight := time.Date(now.Year(), now.Month(), now.Day()+1, 0, 0, 0, 0, time.UTC)
			state.lockedUntil = nextMidnight
		} else {
			// Lock for 1 minute
			state.lockedUntil = time.Now().Add(1 * time.Minute)
		}
	} else if statusCode == 200 {
		state.consecutive429s = 0
	}
}

// IsUpstreamLocked checks if the key is locked due to 429 backoff.
func (sm *StateManager) IsUpstreamLocked(authKey string) bool {
	sm.mu.RLock()
	defer sm.mu.RUnlock()

	state, exists := sm.backoffStates[authKey]
	if !exists {
		return false
	}
	return time.Now().Before(state.lockedUntil)
}
