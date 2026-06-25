package main

import (
	"errors"
	"fmt"
	"log"
	"sync"
	"time"
	_ "time/tzdata"
)

var ptLocation *time.Location

func init() {
	var err error
	ptLocation, err = time.LoadLocation("America/Los_Angeles")
	if err != nil {
		log.Printf("WARN: failed to load America/Los_Angeles timezone: %v", err)
		ptLocation = time.UTC
	}
}

func nextDayPTCooldown() time.Time {
	nowPT := time.Now().In(ptLocation)
	nextDay := nowPT.AddDate(0, 0, 1)
	return time.Date(nextDay.Year(), nextDay.Month(), nextDay.Day(), 0, 0, 0, 0, ptLocation)
}

type keyRateLimit struct {
	count      int
	windowTime time.Time
}

// KeyManager manages key selection, rate limiting, and cooldown for a single provider.
type KeyManager struct {
	mu             sync.Mutex
	providerName   string
	keys           []string
	currentIndex   int
	globalLimit    int
	cooldownMap    map[int]time.Time
	rateLimits     map[string]*keyRateLimit
	consecutive429 map[int]int
}

func NewKeyManager(providerName string, keys []string, globalLimit int) *KeyManager {
	return &KeyManager{
		providerName:   providerName,
		keys:           keys,
		globalLimit:    globalLimit,
		cooldownMap:    make(map[int]time.Time),
		rateLimits:     make(map[string]*keyRateLimit),
		consecutive429: make(map[int]int),
	}
}

// GetKey returns the next available key and its index using round-robin.
// It applies the modelRateLimit if > 0, otherwise falls back to globalRateLimit.
func (km *KeyManager) GetKey(modelID string, modelRateLimit int) (string, int, error) {
	km.mu.Lock()
	defer km.mu.Unlock()

	now := time.Now()
	n := len(km.keys)

	for i := 0; i < n; i++ {
		idx := (km.currentIndex + i) % n
		
		if km.isKeyInCooldown(idx, now) {
			continue
		}

		if km.tryConsumeToken(idx, modelID, modelRateLimit, now) {
			km.currentIndex = (idx + 1) % n
			return km.keys[idx], idx, nil
		}
	}

	return "", -1, errors.New("all keys exhausted or rate limited")
}

func (km *KeyManager) isKeyInCooldown(index int, now time.Time) bool {
	expiry, exists := km.cooldownMap[index]
	if !exists {
		return false
	}
	if now.After(expiry) {
		delete(km.cooldownMap, index)
		return false
	}
	return true
}

func (km *KeyManager) tryConsumeToken(index int, modelID string, modelRateLimit int, now time.Time) bool {
	var bucketKey string
	var limitValue int

	if modelRateLimit > 0 {
		bucketKey = fmt.Sprintf("model:%s:%d", modelID, index)
		limitValue = modelRateLimit
	} else {
		bucketKey = fmt.Sprintf("global:%d", index)
		limitValue = km.globalLimit
	}

	if limitValue <= 0 {
		return true // No limit
	}

	currentWindow := now.Truncate(time.Minute)
	limit, exists := km.rateLimits[bucketKey]

	if !exists || limit.windowTime.Before(currentWindow) {
		km.rateLimits[bucketKey] = &keyRateLimit{
			count:      1,
			windowTime: currentWindow,
		}
		return true
	}

	if limit.count >= limitValue {
		return false
	}

	limit.count++
	return true
}

// ResetFailures resets the consecutive 429 counter on a successful request.
func (km *KeyManager) ResetFailures(index int) {
	km.mu.Lock()
	defer km.mu.Unlock()
	km.consecutive429[index] = 0
}

// Mark429 handles a 429 response with graduated cooldown.
func (km *KeyManager) Mark429(index int) {
	km.mu.Lock()
	defer km.mu.Unlock()

	km.consecutive429[index]++
	count := km.consecutive429[index]

	if count >= 3 {
		km.cooldownMap[index] = nextDayPTCooldown()
		log.Printf("[WARN] provider=%s key[%d]: 3 consecutive 429s, cooldown until next day PT", km.providerName, index)
	} else {
		km.cooldownMap[index] = time.Now().Add(1 * time.Minute)
		log.Printf("[WARN] provider=%s key[%d]: 429 #%d, cooldown for 1 minute", km.providerName, index, count)
	}
}

// MarkFailed handles 401/403 responses by cooling down until next day PT.
func (km *KeyManager) MarkFailed(index int) {
	km.mu.Lock()
	defer km.mu.Unlock()
	km.cooldownMap[index] = nextDayPTCooldown()
	km.consecutive429[index] = 0 // reset just in case
	log.Printf("[WARN] provider=%s key[%d]: hard failure (e.g. 401/403), cooldown until next day PT", km.providerName, index)
}
