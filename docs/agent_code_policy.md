# Agent Code Policy

赛题要求：

- 日志必须是 Agent 运行过程中的真实记录，格式符合赛题要求。
- 代码必须由 Agent 独立生成，不得读取预存代码。
- 预测结果必须由神经网络模型产生，不得调用数值求解器。

## 本仓库执行方式

当前 `src/`、`scripts/` 是 Task1 harness，用来：

- 管理数据、checkpoint、run、memory；
- 验证 shape / finite / 前 10 帧；
- 启动官方 proxy；
- 组织 Agent 工作区。

最终提交代码不默认来自 `src/`。正式打包时：

```bash
bash scripts/run_in_env.sh scripts/task1_make_submission.py \
  --run-dir runs/task1/<UTC timestamp> \
  --submission-name <name>
```

脚本会要求存在：

```text
agent_workspace/code/
```

该目录应由 GPT-5.5 Agent 通过官方 proxy 生成或修改。只有调试时才允许：

```bash
--allow-harness-code
```

带这个参数生成的包不能作为最终提交。

## Agent 允许读取什么

为了避免“读取预存代码”，Agent 生成最终 `code/` 时只应读取：

- `memory/contract/task1_rules.yaml`
- `memory_query.py` 的 retrieval packet
- 数据和 checkpoint 的路径、shape 摘要
- 允许命令和输出格式说明
- 允许的本地 baseline 知识库路径、API 说明和高层思路

Agent 不应读取：

- 当前 `src/` 目录代码
- 旧仓库代码
- 历史提交目录里的 `code/`
- 任何预写好的模型/推理源码全文

允许作为知识库参考的本地 baseline：

- FNO / NeuralOperator: `/autodl-fs/data/AI4Sv2/Task1/data/baselinecode/neuraloperator`
- DeepONet: `/autodl-fs/data/AI4Sv2/Task1/data/baselinecode/deeponet`
- PI-DeepONet: `/autodl-fs/data/AI4Sv2/Task1/data/baselinecode/Physics-informed-DeepONets`

边界是：可以参考 baseline 的 README、配置、API、模型结构和高层设计思路；
不能把大段 baseline 源码逐字复制成最终 `code/`。如果需要使用第三方库，
优先通过依赖和 API 调用表达，而不是内嵌其源码。

## 为什么这样设计

这样可以把两件事分开：

1. harness 负责实验管理和校验；
2. final `code/` 由 Agent 在官方 log 追溯下生成。

这比直接把当前工程代码复制进 submission 更符合赛题要求。
