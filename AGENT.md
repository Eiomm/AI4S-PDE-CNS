# AI4S-PDE-CNS Agent Guide

本文件是给后续 Codex、Claude Code、Cursor、OpenCode 或其他代码智能体使用的项目入口说明。进入本仓库后，先读本文件，再执行任何训练、生成代码或提交打包动作。

## 项目目标

本仓库用于参加第四届世界科学智能大赛 AI4S 智能体 CNS 挑战赛的任务 4：神经算子 PDE 智能体。

第一阶段目标不是冲榜，而是构建可审计的 Agent 科研闭环：

- 通过 API LLM 做规划、代码生成、实验分析。
- 使用工具执行本地文件读写、训练、推理和提交校验。
- 记录每次 LLM 调用和实验动作，保证代码与日志可追溯。
- 后续接入 FNO / `neuraloperator` baseline，再做 Task 1 微调和 Task 2 从头训练。

官方页面：

```text
https://competition.ai4s.com.cn/race/7/description
```

## 关键比赛约束

- Task 1 可以使用 PDEBench 官方 checkpoint 微调。
- Task 2 必须从头训练，不能使用 Task 1 的数据或 checkpoint。
- 不允许调用数值求解器生成额外训练数据。
- 每个提交任务必须同时包含：
  - `task{N}_pred.hdf5`
  - `task{N}_time.csv`
  - `task{N}_logs.log`
- 预测文件 shape 必须是 `(N, 200, 256)`。
- 前 10 个时间步必须与测试输入初始条件一致，容差按官方要求处理。
- `task{N}_logs.log` 每一行必须是合法 JSON。
- 每条 LLM 调用记录必须至少包含：
  - `timestamp`
  - `elapsed_seconds`
- Task 2 的 Agent 思考与训练总时间必须控制在 12 小时内。
- `code/` 中的最终提交代码必须能从日志中追溯生成过程。

## 仓库结构

```text
agent/        Agent 主循环、LLM 客户端、日志、工具、提交校验
code/         官方提交用源码目录；不要把它做成 Python package
configs/      task 配置和 LLM provider 示例配置
data/         官方数据包和解压数据；默认不入 git
docs/         赛题摘要、methodology 草稿、项目说明
runs/         每次 Agent 实验输出、日志、manifest、临时提交
submission/   最终提交目录模板
tests/        单元测试和 smoke test
```

重要：`code/` 目录不要添加 `__init__.py`。Python 标准库也叫 `code`，把 `code/` 变成包会遮蔽标准库，导致 `pytest` 等工具异常。

## 当前数据位置

官方数据已放在：

```text
D:\Study\AI4S-PDE-CNS\data\data_and_sample_submission\data_and_sample_submission\train_val_test_init
```

已确认的主要数据：

```text
task1_test.hdf5       tensor: (1000, 10, 256)
task1_val.hdf5        tensor: (100, 200, 256)
task2_part0_train.h5  tensor: (1000, 320, 256), nu: (1000,)
task2_part1_train.h5  tensor: (1000, 320, 256), nu: (1000,)
task2_part2_train.h5  tensor: (1000, 320, 256), nu: (1000,)
task2_test.h5         tensor: (1000, 10, 256)
task2_val.h5          tensor: (100, 210, 256), nu: (100,)
```

Task 1 坐标 key 使用短横线：

```text
t-coordinate
x-coordinate
```

Task 2 坐标 key 使用下划线：

```text
t_coordinate
x_coordinate
```

## 环境建议

当前系统 Python 可能是 3.13。正式训练请创建 Python 3.11 或 3.12 环境。

```powershell
cd D:\Study\AI4S-PDE-CNS
conda create -n ai4s-pde-cns python=3.12 -y
conda activate ai4s-pde-cns
pip install -e ".[dev]"
```

安装 PyTorch 时按本机 CUDA 驱动选择官方命令。本机规划环境为 RTX 5070 12GB，训练时优先使用小 batch 和短 epoch 做 smoke test。

## 常用命令

运行测试：

```powershell
python -m pytest -q
```

检查环境：

```powershell
python -m agent.env_check
```

运行 mock Agent：

```powershell
python -m agent.run --task task1 --config configs\task1.yaml
```

校验提交目录：

```powershell
python -m agent.validate_submission --path runs\local-smoke-submission
```

打包通过校验的 run：

```powershell
python -m agent.pack_submission --run runs\<run_id>
```

运行占位 baseline 时使用脚本路径，不要使用 `python -m code.baseline_stub`：

```powershell
python code\baseline_stub.py --input data\...\task1_test.hdf5 --output runs\...\task1_pred.hdf5 --dataset-key tensor
```

## LLM API 接入

默认配置使用 `provider: mock`，适合本地 smoke test。

真实 Agent 运行前，配置对应 API key：

```powershell
$env:DEEPSEEK_API_KEY="..."
$env:KIMI_CODE_API_KEY="..."
$env:KIMI_API_KEY="..."
$env:MOONSHOT_API_KEY="..."
$env:SILICONFLOW_API_KEY="..."
```

Provider 配置集中在：

```text
configs\llm_providers.yaml
```

可用 profile：

```text
deepseek             DeepSeek 官方 API，默认 deepseek-v4-pro
deepseek_flash       DeepSeek 官方 API，默认 deepseek-v4-flash
kimi                 Kimi Code API，固定模型 kimi-for-coding
kimi_open_platform   Moonshot/Kimi 开放平台，默认 kimi-k2.6
siliconflow_glm      硅基流动 GLM profile
siliconflow_deepseek 硅基流动 DeepSeek profile
```

任务配置通过 `llm_profile` 切换模型供应商：

```yaml
llm_profile: kimi
llm_profiles_path: llm_providers.yaml
```

先做 API 连通性检查：

```powershell
python -m agent.check_llm --config configs\deepseek.example.yaml
python -m agent.check_llm --config configs\kimi.example.yaml
python -m agent.check_llm --config configs\siliconflow.example.yaml
```

运行 Agent 示例：

```powershell
python -m agent.run --task task1 --config configs\deepseek.example.yaml
python -m agent.run --task task1 --config configs\kimi.example.yaml
python -m agent.run --task task1 --config configs\siliconflow.example.yaml
```

不要让真实 API Agent 一开始就训练大模型。推荐顺序：

1. 让 Agent 只读取赛题摘要并生成实验计划。
2. 检查 `task1_logs.log` 是否是合法 JSONL。
3. 让 Agent 修改小文件或生成脚本草稿。
4. 再接入 FNO baseline 的训练/推理。

## Agent 工作流

Agent 第一版必须遵循 observe-plan-act-record：

1. `observe`：读取赛题摘要、项目目录、实验输出和错误日志。
2. `plan`：调用 LLM 生成下一步动作。
3. `act`：只执行白名单工具。
4. `record`：记录 LLM response、工具动作、文件写入和 shell 输出。
5. `stop`：达到任务目标、时间预算或失败阈值后停止。

所有 LLM 调用必须通过 `LLMCallLogger`，不要直接绕过 logger 调 API。

## 工具安全规则

- 默认只允许在项目目录和当前 run 目录内读写。
- 大数据、模型权重、日志和 zip 文件不提交到 git。
- 不要删除 `data/` 下用户补充的数据。
- 不要私下复用 Task 1 checkpoint 到 Task 2。
- 不要用数值求解器生成额外训练数据。
- 不要手工编辑最终提交代码后伪装成 Agent 生成。

## 下一步开发路线

### Step 1: Task 1 zero-train baseline

目标是先打通链路，不追求高分：

```text
读取 task1_test.hdf5
生成 shape 正确的 task1_pred.hdf5
生成 task1_time.csv
生成 task1_logs.log
validate_submission 通过
pack_submission 通过
```

### Step 2: FNO baseline

接入 `neuraloperator` 或官方 PDEBench FNO checkpoint：

```text
code/train_task1_fno.py
code/infer_task1_fno.py
code/model_task1_fno.py
```

先在 `task1_val.hdf5` 上做短时验证，再对 `task1_test.hdf5` 推理。

### Step 3: Task 1 light fine-tuning

在训练耗时分档内做轻量微调。必须比较：

```text
不训练 checkpoint 推理分数
短时微调后分数
训练时间扣分
推理时间扣分
```

### Step 4: Task 2 isolated training

Task 2 必须独立：

```text
不加载 Task 1 checkpoint
不使用 Task 1 数据
使用官方 Task 2 train/val/test
推理时不能依赖 test Nu
```

## 完成标准

每次重要修改后至少运行：

```powershell
python -m pytest -q
```

涉及提交文件时还要运行：

```powershell
python -m agent.validate_submission --path <submission_or_run_dir>
```

只有在命令输出确认通过后，才能声称“完成”或“可提交”。
