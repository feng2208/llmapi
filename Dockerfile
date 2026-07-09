# Build stage
FROM golang:alpine AS builder

WORKDIR /app
COPY . .

# Version argument
ARG VERSION=dev

# Download dependencies
RUN go mod download

# Build the binary
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -ldflags="-s -w -X 'main.Version=${VERSION}'" -o llmapi .

# Final stage
FROM alpine:latest

# Add ca-certificates in case the proxy needs to make TLS connections
RUN apk --no-cache add ca-certificates

WORKDIR /app
COPY --from=builder /app/llmapi /app/llmapi

# Use a non-root user for security
RUN adduser -D proxyuser
USER proxyuser

ENTRYPOINT ["/app/llmapi"]
