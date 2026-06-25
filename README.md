# LLM API Proxy

一个轻量级、高性能的 LLM API（如 Google Gemini）反向代理服务器。它提供完全**兼容 OpenAI**的接口。

## 功能特性

- **OpenAI 兼容**：接受 `/v1/chat/completions` 请求。
- **429 熔断降级**：自动禁用无用 key。
- **Reasoning 与思考过程支持**：可设置思考深度。
- **代理支持**：支持配置 HTTP 或 SOCKS5 代理。


## 编译

```bash
go mod tidy
go build -o llmapi .
```

## 配置

配置文件基于 YAML 格式，内置了一个基本模板，你可以使用以下命令查看：

```bash
./llmapi -print-template
```

然后，你可以将配置保存为 `config.yaml` 并在运行时代入。

## 运行

默认加载当前目录下的 `config.yaml`：

```bash
./llmapi
```

或者指定配置文件：

```bash
./llmapi -config /path/to/my-config.yaml
```

## 测试

启动后，你可以使用 `curl` 或者标准的 OpenAI SDK 进行测试：

```bash
curl http://127.0.0.1:3000/v1/models \
  -H "Authorization: Bearer sk-wxfoeeeeeeeeeeeeeeeeeeee"
```

发送对话请求（支持 SSE 流式返回）：
```bash
curl -X POST http://127.0.0.1:3000/v1/chat/completions \
  -H "Authorization: Bearer sk-wxfoeeeeeeeeeeeeeeeeeeee" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-flash-lite",
    "stream": true,
    "messages": [
      {
        "role": "user",
        "content": "请给我介绍一下人工智能的历史"
      }
    ]
  }'
```
