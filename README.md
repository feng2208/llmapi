# LLM API Forwarder (llmapi)

`llmapi` 是一款高性能、支持高并发的 Go 语言编写的 OpenAI Chat Completions API 转发网关。


## 🚀 核心特性

*   **智能模型路由与降级 (Fallback)**
    *   根据客户端请求的 `model` 参数自动选择上游渠道。
    *   支持为同一个模型配置多个上游 Provider（按顺序尝试），当前 Provider 节点全部 Auth Key 失效时，自动平滑 fallback 降级到下一个 Provider。
*   **多 Key 轮询与限流管理**
    *   单个 Provider 支持配置多个 Auth Key 轮询（Round-Robin）使用。
    *   **独立限流桶**：支持配置 Model 级限流和 Provider 全局级限流。每个 Key 使用其中一种限流设置，在内存中维护完全独立的 Token Bucket 计数桶，互不干扰。
*   **精细化 429 惩罚退避策略**
    *   若某个 Auth Key 请求上游遭遇 `429 Too Many Requests`，自动暂停使用该 Key **1分钟**。
    *   若连续 **3 次** 遭遇 429，则该 Key 将被锁定并暂停使用，直至**次日 UTC 时间 0:00** 自动解锁。
*   **动态请求体修改**
    *   **删除 (Delete)**：支持从请求体中按键无条件删除特定字段（如 `reasoning_effort`），解决部分模型对不兼容参数报错的问题。
    *   **额外注入 (Extra & Deep Merge)**：支持深度合并（Deep Merge）并向请求体注入额外的参数（如配置 Gemini 的 `thinking_config` 以强制开启深度思考）。
*   **深度思考流式还原 (Reasoning Extraction)**
    *   自动提取上游混在 `content` 正文中的思考块内容（根据配置的 `reasoning_start` 和 `reasoning_end`，如 `<thought>` 与 `</thought>`）。
    *   将思考块从正文内容中剥离，并实时转换输出为 DeepSeek 标准的 `reasoning` 和 `reasoning_content` 双字段（多客户端高兼容推荐），使标准的第三方 OpenAI 客户端也能获得精美的原生思考流式渲染。
*   **支持代理**
    *   支持配置全局或 Provider 特定的代理（支持 `http`、`https` 或 `socks5` 协议）。



## 📦 快速启动

### 方式一：本地运行

1.  从 release 下载可执行文件。
2.  生成本地配置文件：
    ```bash
    llmapi -print-template > config.yaml
    ```
3.  根据需要修改 `config.yaml`（参照下方配置指南）。
4.  启动服务：
    ```bash
    llmapi -config config.yaml
    ```

### 方式二：Docker 容器化部署

挂载配置文件运行：
```bash
docker run -d \
  -p 3000:3000 \
  -v $(pwd)/config.yaml:/app/config.yaml \
  --name llmapi \
  llmapi:latest -c /app/config.yaml
```



## ⚙️ 配置文件说明 (`config.yaml`)

下面是一个完整的配置示例：

```yaml
listen: "0.0.0.0:3000"     # 监听端口
max_body_size: 10485760   # 单次请求体大小限制 (10MB)，0 代表无限制

# 全局代理 (可选，支持 socks5:// 或 http://)
proxy: socks5://127.0.0.1:1080

# 客户端认证及限流
clients:
  rate_limit: 10          # 全局默认客户端限流 (每分钟请求数)
  auth:
    - name: mykey
      key: sk-wxfoeeeeeeeeeeeeeeeeeeee
      rate_limit: 20      # 覆盖全局，该客户端限流为 20次/分钟
    - name: mykey2
      key: sk-wxfouuuuuuuuuuuuuuuu2

# 模型路由分配
models:
  - name: gemini-flash-lite  # 客户端请求的模型名称
    providers:
      - name: gemini
        upstream: https://generativelanguage.googleapis.com/v1beta/openai/chat/completions
        model: gemini-2.5-flash-lite  # 映射到上游的真实模型名称
        rate_limit: 10                # 针对该 model-key 的独立限流 (每分钟)，缺省则使用 Provider 级限流
        timeout: 120s                 # 请求超时时间
        proxy: socks5://127.0.0.1:1080 # 覆盖全局代理
        reasoning_start: "<thought>"  # 思考块开始标记
        reasoning_end: "</thought>"    # 思考块结束标记
        request_body:
          delete:
            - reasoning_effort        # 转发上游时剔除此参数
          extra:
            - extra_body:             # 向 JSON 根注入额外参数强制开启思维流
                google:
                  thinking_config:
                    thinking_level: high
                    include_thoughts: true
            
# 上游提供商凭证
providers:
  - name: gemini
    rate_limit: 10           # 全局 Provider 级默认限流 (每分钟)
    auth_keys:
      - AIaaaaaaaaaaaaaaaaaaaa # 支持轮询的多个 API Key
      - AIbbbbbbbbbbbbbbbbbbbb
```

