package main

import (
	"context"
	"fmt"
	"net"
	"net/http"
	"net/url"
	"sync"
	"time"

	"golang.org/x/net/proxy"
)

type ProxyManager struct {
	mu         sync.Mutex
	transports map[string]*http.Transport
}

func NewProxyManager() *ProxyManager {
	return &ProxyManager{
		transports: make(map[string]*http.Transport),
	}
}

func (pm *ProxyManager) GetTransport(proxyURL string) (*http.Transport, error) {
	pm.mu.Lock()
	defer pm.mu.Unlock()

	if t, exists := pm.transports[proxyURL]; exists {
		return t, nil
	}

	transport := &http.Transport{
		MaxIdleConns:        100,
		IdleConnTimeout:     90 * time.Second,
		TLSHandshakeTimeout: 10 * time.Second,
	}

	if proxyURL != "" {
		u, err := url.Parse(proxyURL)
		if err != nil {
			return nil, fmt.Errorf("invalid proxy URL %q: %w", proxyURL, err)
		}

		if u.Scheme == "socks5" || u.Scheme == "socks5h" {
			dialer, err := proxy.FromURL(u, proxy.Direct)
			if err != nil {
				return nil, fmt.Errorf("failed to create socks5 dialer: %w", err)
			}
			transport.DialContext = func(ctx context.Context, network, addr string) (net.Conn, error) {
				if cd, ok := dialer.(proxy.ContextDialer); ok {
					return cd.DialContext(ctx, network, addr)
				}
				return dialer.Dial(network, addr)
			}
		} else if u.Scheme == "http" || u.Scheme == "https" {
			transport.Proxy = http.ProxyURL(u)
		} else {
			return nil, fmt.Errorf("unsupported proxy scheme %q", u.Scheme)
		}
	}

	pm.transports[proxyURL] = transport
	return transport, nil
}

func (pm *ProxyManager) GetClient(mpc *ModelProviderConfig, globalProxy string) (*http.Client, error) {
	proxyURL := mpc.Proxy
	if proxyURL == "" {
		proxyURL = globalProxy
	}

	transport, err := pm.GetTransport(proxyURL)
	if err != nil {
		return nil, err
	}

	return &http.Client{
		Transport: transport,
		Timeout:   mpc.Timeout(),
	}, nil
}
