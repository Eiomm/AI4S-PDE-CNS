# Agent Workspace

这个目录用于官方要求的 Agent 独立代码生成流程。

```text
agent_workspace/
├── code/             # GPT-5.5 Agent 生成或修改的提交代码
├── prompts/          # 给 Agent 的最小 prompt 与 memory packet
└── logs/             # Agent runner 自己的执行摘要；官方 API 原始 log 在 logs/
```

重要规则：

1. 最终 `submissions/<name>/code/` 默认只从 `agent_workspace/code` 复制。
2. Agent 生成代码时不应读取当前仓库的 `src/`、旧仓库代码或预存提交代码。
3. Agent 可以读取：
   - `memory/contract/task1_rules.yaml`
   - `memory/episodic/runs.jsonl` 的检索摘要
   - `memory/findings/metric_leaderboard.csv`
   - `memory/wisdom/strategy_summary.md`
   - 官方数据 shape / checkpoint 路径 / 允许的命令说明
4. Agent 生成代码的 API 请求必须经过 `scripts/start_openai_proxy.sh`。
5. 当前 `src/` 是 harness，不是最终可提交代码来源。
