package main

import (
	"context"
	"io"
	"log"
	"net"
	"net/http"
	"net/url"

	"golang.org/x/net/proxy"
)

var hopByHopHeaders = map[string]bool{
	"Connection":          true,
	"Keep-Alive":          true,
	"Proxy-Authenticate":  true,
	"Proxy-Authorization": true,
	"Te":                  true,
	"Trailer":             true,
	"Transfer-Encoding":   true,
	"Upgrade":             true,
}

func buildTransport(proxyURLStr string) *http.Transport {
	if proxyURLStr == "" {
		return http.DefaultTransport.(*http.Transport).Clone()
	}

	proxyURL, err := url.Parse(proxyURLStr)
	if err != nil {
		log.Printf("WARN: invalid proxy URL %s: %v, falling back to default transport", proxyURLStr, err)
		return http.DefaultTransport.(*http.Transport).Clone()
	}

	transport := http.DefaultTransport.(*http.Transport).Clone()

	if proxyURL.Scheme == "socks5" {
		// Use golang.org/x/net/proxy for SOCKS5
		// proxy.Direct causes DNS resolution to happen on the proxy server (remote DNS) if an address name is used instead of IP.
		dialer, err := proxy.SOCKS5("tcp", proxyURL.Host, nil, proxy.Direct)
		if err != nil {
			log.Printf("WARN: failed to create socks5 dialer: %v", err)
			return transport
		}

		transport.DialContext = func(ctx context.Context, network, addr string) (net.Conn, error) {
			return dialer.Dial(network, addr)
		}
		// Clear HTTP proxy if any was inherited
		transport.Proxy = nil
	} else if proxyURL.Scheme == "http" || proxyURL.Scheme == "https" {
		transport.Proxy = http.ProxyURL(proxyURL)
	}

	return transport
}

func copyHeaders(dst, src http.Header) {
	for k, vv := range src {
		if hopByHopHeaders[k] {
			continue
		}
		for _, v := range vv {
			dst.Add(k, v)
		}
	}
}

func streamResponse(w http.ResponseWriter, resp *http.Response, debug bool) {
	copyHeaders(w.Header(), resp.Header)
	w.WriteHeader(resp.StatusCode)

	flusher, ok := w.(http.Flusher)
	buf := make([]byte, 4096)

	var debugBody []byte

	for {
		n, err := resp.Body.Read(buf)
		if n > 0 {
			w.Write(buf[:n])
			if ok {
				flusher.Flush()
			}

			if debug && len(debugBody) < 16384 {
				debugBody = append(debugBody, buf[:n]...)
			}
		}
		if err != nil {
			if err != io.EOF {
				log.Printf("error reading upstream response: %v", err)
			}
			break
		}
	}

	if debug {
		log.Printf("[DEBUG] Upstream Response Body: %s", string(debugBody))
	}
}
