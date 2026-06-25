package main

import (
	"sync"
	"time"
)

// ClientRateLimiter provides a simple fixed-window rate limiter per API key.
// The window size is 1 minute.
type ClientRateLimiter struct {
	mu            sync.Mutex
	limits        map[string]*clientLimit
	defaultLimit  int
	clientConfigs map[string]int // key string -> specific limit
}

type clientLimit struct {
	count      int
	windowTime time.Time
}

func NewClientRateLimiter(defaultLimit int, authConfigs []ClientAuthKey) *ClientRateLimiter {
	clientConfigs := make(map[string]int)
	for _, auth := range authConfigs {
		if auth.RateLimit > 0 {
			clientConfigs[auth.Key] = auth.RateLimit
		}
	}

	return &ClientRateLimiter{
		limits:        make(map[string]*clientLimit),
		defaultLimit:  defaultLimit,
		clientConfigs: clientConfigs,
	}
}

// Allow checks if the given key is allowed to make a request.
// It returns true if allowed, false if rate limited.
func (rl *ClientRateLimiter) Allow(key string) bool {
	rl.mu.Lock()
	defer rl.mu.Unlock()

	now := time.Now()
	// Truncate to the current minute to create a fixed window
	currentWindow := now.Truncate(time.Minute)

	limit, exists := rl.limits[key]
	if !exists || limit.windowTime.Before(currentWindow) {
		// New window or first time
		rl.limits[key] = &clientLimit{
			count:      1,
			windowTime: currentWindow,
		}
		return true
	}

	// Determine the limit for this key
	maxAllowed := rl.defaultLimit
	if specificLimit, hasSpecific := rl.clientConfigs[key]; hasSpecific {
		maxAllowed = specificLimit
	}

	if limit.count >= maxAllowed {
		return false
	}

	limit.count++
	return true
}
