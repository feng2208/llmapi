package main

import (
	"context"
	_ "embed"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"
)

var Version = "dev"

//go:embed config-template.yaml
var ConfigTemplate string

func main() {
	var configPath string
	var printTemplate bool
	var debug bool
	var showVersion bool

	flag.StringVar(&configPath, "config", "config.yaml", "Path to configuration file")
	flag.BoolVar(&printTemplate, "print-template", false, "Print the config template and exit")
	flag.BoolVar(&debug, "debug", false, "Enable debug mode to log detailed requests and responses")
	flag.BoolVar(&showVersion, "version", false, "Print the version and exit")
	flag.Parse()

	if showVersion {
		fmt.Println(Version)
		os.Exit(0)
	}

	if printTemplate {
		fmt.Println(ConfigTemplate)
		os.Exit(0)
	}

	cfg, err := loadConfig(configPath)
	if err != nil {
		log.Fatalf("Fatal error loading config: %v", err)
	}

	printBanner(cfg)

	keyManagers := make(map[string]*KeyManager)
	for _, p := range cfg.Providers {
		keyManagers[p.Name] = NewKeyManager(p.Name, p.AuthKeys, p.RateLimit)
	}

	router := NewProxyRouter(cfg, keyManagers, debug)

	server := &http.Server{
		Addr:    cfg.Listen,
		Handler: router,
	}

	go func() {
		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("Server failed: %v", err)
		}
	}()

	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	log.Println("Shutting down server...")

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	if err := server.Shutdown(ctx); err != nil {
		log.Fatalf("Server forced to shutdown: %v", err)
	}

	log.Println("Server exiting")
}
