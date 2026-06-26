package main

import (
	"fmt"
	"strings"
)

func printBanner(cfg *Config) {
	sep := strings.Repeat("─", 60)
	fmt.Println(sep)
	fmt.Printf("  API Proxy started on %s\n", cfg.Listen)
	fmt.Printf("  Global proxy: %s\n", cfg.Proxy)
	fmt.Printf("  Client Rate Limit: %d/min\n", cfg.Clients.RateLimit)
	fmt.Println(sep)

	fmt.Println("  Configured Models:")
	for _, m := range cfg.Models {
		fmt.Printf("  - %s\n", m.Name)
		for _, p := range m.Providers {
			proxyInfo := "direct"
			if p.Proxy != "" {
				proxyInfo = p.Proxy
			} else if cfg.Proxy != "" {
				proxyInfo = cfg.Proxy + " (global)"
			}

			if p.RateLimit > 0 {
				fmt.Printf("      -> provider: %s, upstream_model: %s (timeout: %s, rate_limit: %d/min, proxy: %s)\n",
					p.Name, p.Model, p.Timeout, p.RateLimit, proxyInfo)
			} else {
				fmt.Printf("      -> provider: %s, upstream_model: %s (timeout: %s, proxy: %s)\n",
					p.Name, p.Model, p.Timeout, proxyInfo)
			}
		}
	}

	fmt.Println(sep)
	fmt.Println("  Configured Auth Providers:")
	for _, p := range cfg.Providers {
		fmt.Printf("  - %s: %d keys, Rate Limit: %d/min\n", p.Name, len(p.AuthKeys), p.RateLimit)
	}
	fmt.Println(sep)
}

