package main

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

// FormatJSON pretty-prints the JSON byte array with 2 spaces indent.
func FormatJSON(data []byte) string {
	var temp interface{}
	if err := json.Unmarshal(data, &temp); err != nil {
		return string(data)
	}
	formatted, err := json.MarshalIndent(temp, "", "  ")
	if err != nil {
		return string(data)
	}
	return string(formatted)
}

type loggingReader struct {
	r             io.Reader
	headerPrinted bool
}

func (lr *loggingReader) Read(p []byte) (n int, err error) {
	n, err = lr.r.Read(p)
	if n > 0 {
		if !lr.headerPrinted {
			fmt.Printf("[DEBUG] --- UPSTREAM RESPONSE BODY (STREAMING) ---\n")
			lr.headerPrinted = true
		}
		fmt.Print(string(p[:n]))
	}
	return
}

type flushingWriter struct {
	w io.Writer
	f http.Flusher
}

func (fw flushingWriter) Write(p []byte) (n int, err error) {
	n, err = fw.w.Write(p)
	if fw.f != nil {
		fw.f.Flush()
	}
	return
}


