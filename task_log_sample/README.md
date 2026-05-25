# Task Log 说明

## Log 文件要求

每次提交需包含 `task1_logs.log` 与 `task2_logs.log` 两个文件。

每个 log 文件的**每一行**必须是一条合法的 JSON 数据，记录 Agent 每次调用 LLM 的完整 response，且必须包含以下字段：

| 字段 | 说明 |
|---|---|
| `timestamp` | 本次 LLM 调用完成时的 ISO 8601 时间戳（含时区），如 `2026-05-06T09:02:54.524886+00:00` |
| `elapsed_seconds` | 本次 LLM 调用耗时（秒） |
| `response` 或 `tool_calls` | LLM 本次输出的文本内容或工具调用记录，**至少存在其中一个字段** |

`response` 与 `tool_calls` 字段是评审的核心依据：系统会通过分析这两个字段的内容，验证提交的 `code/` 目录中的代码是否完全由 Agent 生成，而非人工编写或修改。**请确保 log 中完整保留 LLM 的每次输出。**

---

## Task1 与 Task2 的执行方式

**Task1 与 Task2 可以分别独立执行，也可以由 Agent 在同一个 session 中连续完成。**

### 分别执行

启动两个独立的 Agent session，分别完成 Task1 和 Task2，各自产生对应的 log 文件。两个 log 文件的时间线相互独立。

### 一次执行（同一 session）

由同一个 Agent session 依次完成 Task1 和 Task2。此时 `task1_logs.log` 与 `task2_logs.log` **应保持一致**，即两个文件内容相同，均为该 session 的完整 LLM 调用记录。

---

## 时间限制

**单个 log 文件中，最后一条记录的 `timestamp` 与第一条记录的 `timestamp` 之差不得超过 12 小时。**

超过 12 小时将被判定为违规，对应 task 得分记为 0 分。

---

## Log 记录工具

本目录下提供了 `openai-log/proxy.py`，这是一个本地转发代理，可自动拦截并记录所有经过的 LLM API 请求，生成符合上述格式要求的 log 文件。

### 使用方式

安装依赖：

```bash
pip install -r openai-log/requirements.txt
```

启动代理：

```bash
python openai-log/proxy.py --port 8080 --target https://api.openai.com --log-dir ./logs
```

将 Agent 的 API base URL 指向 `http://localhost:8080`，代理即会自动转发请求并将每次 LLM 调用写入 log。

### 参数说明

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--port` | `8080` | 本地监听端口 |
| `--target` | `https://api.openai.com` | 转发目标 API 地址 |
| `--log-dir` | `./logs` | log 文件输出目录 |

### Anthropic 接口

`proxy.py` 目前仅针对 **OpenAI 兼容格式**实现了 response 解析与 log 记录逻辑。如果 Agent 使用 Anthropic 接口（`/v1/messages`），需要自行修改 `proxy.py` 中的响应解析部分：

- 非流式响应：参考 `_extract_assistant_message` 函数，适配 Anthropic 的响应结构（`content[].text`）
- 流式响应：参考 `parse_sse_chunks` 函数，适配 Anthropic 的 SSE 事件格式（`content_block_delta` 等事件类型）
