# AI4S-PDE-CNS Task1：PDE 神经算子科研 Agent

本目录是 AI4S PDE 神经算子赛道 Task 1 的实际代码根目录，面向 PDEBench 1D Burgers 固定物理环境预测。项目目标不是手工提交一个固定模型，而是构建一个能自动完成“理解规则 -> 生成实验代码 -> 调用工具运行实验 -> 验证结果 -> 写入记忆 -> 整理提交”的科研工作流 Agent。

Agent 可以使用官方 Task 1 checkpoint 进行微调，但不能调用数值求解器，不能使用额外数据。最终提交必须同时满足预测文件、计时文件、LLM JSONL 日志和 `code/` 可追溯要求。

## 核心约束

- `task1_pred.hdf5` 必须包含数据集 `tensor`，shape 为 `(N, 200, 256)`。
- 前 10 个时间步必须与测试输入完全一致，容差为 `1e-3`。
- 官方缩放设置必须被显式处理：`reduced_resolution_t=5`、`reduced_resolution=4`。模型尺度上的 1 个时间步对应原始 PDEBench 的 5 个时间步，1024 空间点下采样为 256 空间点。
- Task 1 总分上限 150 分：预测精度最高 75 分，训练耗时最高 35 分，推理耗时最高 40 分。
- 训练耗时满分线为 60 分钟；推理耗时必须小于 2 分钟，否则该任务为 0 分。
- `task1_logs.log` 必须是 JSONL，每行包含 `timestamp`、`elapsed_seconds`，并至少包含 `response` 或 `tool_calls`。
- `code/` 中的代码必须能从 Agent 日志中追溯，不能只提交说明文档或 manifest。

## 设计原则

本项目把强约束放在规则、输出格式、日志、时间预算、数据尺度、代码可追溯性和工具白名单上；模型结构、损失函数、微调范围、后处理、集成策略、消融顺序和回滚策略保留给 Agent 自主探索。

多轮实验由 `--max-rounds` 控制。Agent 不会因为 `submit_ready` 或中间分数达标提前停止；每一轮都必须根据 memory、上一轮执行日志和当前 leaderboard 继续做科研决策。

## 目录结构

```text
.
├── agent_workspace/
│   ├── prompts/                 # Agent 主 prompt 和 action schema
│   ├── logs/                    # 每轮 Agent 决策记录，运行时生成，不入库
│   ├── executor_logs/           # 工具执行记录，运行时生成，不入库
│   ├── tool_outputs/            # 工具产物索引，运行时生成，不入库
│   └── code/                    # 每轮 Agent 生成的实验代码，运行时生成，不入库
├── configs/                     # baseline、ensemble、Agent、微调配置
├── docs/                        # 工作流、日志、代码策略和 toolbox 设计文档
├── memory/                      # 长期记忆、实验结论、失败库、规则契约
├── scripts/                     # Agent loop、工具执行器、验证、预测、打包脚本
├── src/ai4sv2_task1/            # Task1 通用库：IO、metrics、predict、submission
├── task_log_sample/openai-log/  # 官方兼容 LLM 调用日志代理
└── tests/                       # 轻量单元测试和核心逻辑校验
```

## 数据与权重

数据和 checkpoint 不随 Git 提交。请在本地按需放置：

```text
data/task1_test.hdf5
data/task1_val.hdf5
data/pdebench_burgers/raw/1D_Burgers_Sols_Nu0.001.hdf5
checkpoints/official/nu0.001_fno.pt
checkpoints/official/nu0.001_unet_pf20.pt
```

`.gitignore` 会忽略 HDF5、checkpoint、压缩包、实验日志、submission、`.env` 和运行目录。

## 环境准备

安装基础依赖：

```bash
pip install -r requirements.txt
pip install -r task_log_sample/openai-log/requirements.txt
```

复制环境变量模板并填写密钥：

```bash
cp .env.example .env
```

默认脚本使用 `/root/miniconda3/envs/ai4s-pde-cns/bin/python`。如果你的环境不同，请修改 `scripts/run_in_env.sh` 和 `scripts/task1_agent_loop.sh` 中的 Python 路径。

## 启动 Agent 闭环

推荐使用 loop 脚本，它会读取 `.env`，启动本地 LLM 日志代理，并执行多轮 Agent 决策与工具调用：

```bash
bash scripts/task1_agent_loop.sh --config configs/agent_gpt55.yaml --max-rounds 10
```

单轮调试：

```bash
bash scripts/task1_agent_loop.sh --config configs/agent_gpt55.yaml --max-rounds 1
```

运行时产物按统一时间戳写入：

```text
agent_workspace/logs/agent_<timestamp>/
agent_workspace/executor_logs/agent_<timestamp>.json
runs/task1/<timestamp>/
```

## 常用工具链

运行本地 checkpoint 微调工具：

```bash
bash scripts/run_in_env.sh scripts/task1_finetune_local.py --config configs/finetune_base.yaml
```

验证预测文件：

```bash
bash scripts/run_in_env.sh scripts/task1_validate.py --pred runs/task1/<run_id>/task1_pred.hdf5
```

整理提交包：

```bash
bash scripts/run_in_env.sh scripts/task1_make_submission.py --run-dir runs/task1/<run_id>
```

实际推荐让 Agent 通过 action schema 选择工具，而不是人工直接串命令。人工命令主要用于调试和复核。

## 可迭代模块

新增科研动作时，优先按以下路径接入：

1. 在 `scripts/task1_tool_executor.py` 增加白名单工具实现。
2. 在 `agent_workspace/prompts/action_schema.json` 增加对应 tool request schema。
3. 在 `agent_workspace/prompts/task1_planner.md` 说明工具输入输出和合规边界。
4. 在 `docs/toolbox_design.md` 记录设计意图。
5. 在 `memory/procedures/` 或 `memory/failures/` 沉淀可复用经验。
6. 增加轻量测试或 smoke check，确保工具不会破坏日志、命名和代码追溯规则。

## 代码审查 Skill

本项目建议把“代码审查”作为固定 Skill 使用。后续只要说“启动代码审查”，Codex 会按固定流程复查：

- 当前变更范围和 Git 状态；
- prompt/schema/runner/executor/tool/memory/submission 的契约一致性；
- 数据、权重、密钥和实验产物是否被误提交；
- 必要的 `py_compile`、JSON/YAML 解析和定向 smoke test；
- 以问题优先的格式输出审查结论，并在需要时修复阻塞性 bug。

## 提交规范

- 不提交 `.env`、API key、数据集、checkpoint、运行日志、预测文件和 submission 压缩包。
- 分支按任务维护，本分支为 `Task1`。
- 每次较大改动应能说明对应的 Agent 工作流收益：规则更清晰、工具更稳定、代码更可追溯、实验更可复现或提交更合规。
