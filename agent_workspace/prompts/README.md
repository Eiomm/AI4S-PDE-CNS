# Agent Prompts

当前 Task1 正式 prompt：

```text
agent_workspace/prompts/task1_planner.md
```

Runner 解析 Agent 输出用的 JSON schema：

```text
agent_workspace/prompts/action_schema.json
```

prompt 应只包含：

- 赛题硬规则；
- memory_query 输出的小 packet；
- 当前实验目标；
- 允许生成的文件列表；
- 允许执行的验证命令。
- 允许参考的公开 baseline / 工具箱边界。
- Skill / Workflow 模块边界：规则读取、数据检查、baseline replay、checkpoint 微调、
  预测校验、合规日志整理、submission 打包。

prompt 不应包含：

- 当前 `src/` 代码全文；
- 旧仓库代码全文；
- 历史完整日志；
- 大段 memory 全量内容。

旧仓库参考 prompt 在：

```text
/autodl-fs/data/AI4S-PDE-CNS/agent/tasks/task1/prompt.md
```

它只作为历史参考，不作为当前 runner 的默认 prompt。
