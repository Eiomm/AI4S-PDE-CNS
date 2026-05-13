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

## 赛题详细规则

### 赛题概述

赛道：AI4S 智能体 CNS 挑战赛 — 神经算子 PDE 智能体（Race 7）

题目：1D Burgers 方程预测。给定前 10 个时间步的初始条件，预测后续 190 个时间步（共 200 步，256 个空间点的均匀网格）。

### Task 1：固定物理环境

- 物理参数固定（Nu ≈ 0.001），与 PDEBench 标准 1D Burgers 一致。
- **允许**使用 PDEBench 官方预训练 FNO checkpoint 进行微调。
- **允许**使用 PDEBench 1D Burgers 公开训练数据。
- 输入: `(N, 10, 256)` → 输出: `(N, 200, 256)`

### Task 2：多物理环境

- 不同样本对应不同 Nu 值（粘性系数），Agent 需要在多物理环境下泛化。
- **必须从头训练**，严禁使用 Task 1 的数据或 checkpoint。
- **严禁**在推理阶段依赖测试集的 Nu 值（Nu 是预测目标的一部分，不可见）。
- 使用官方提供的 Task 2 train/val/test 数据。
- Agent 思考 + 训练总时间必须控制在 **12 小时**以内。

### 通用禁止事项

- **禁止**调用数值求解器（如 FDM、FVM、FEM、谱方法等）生成额外训练数据。
- **禁止**手工编辑最终提交代码后伪装成 Agent 生成。
- **禁止**私下复用 Task 1 checkpoint 到 Task 2。

---

### 提交文件格式

每个提交的 task 必须包含以下三个文件（缺一不可）：

| 文件 | 格式要求 |
|------|---------|
| `task{N}_pred.hdf5` | 预测张量，shape 必须为 `(N, 200, 256)`，前 10 帧必须与官方初始条件一致 |
| `task{N}_time.csv` | CSV，必须包含 `train_time` 和 `inference_time` 两列，各一行数值（秒） |
| `task{N}_logs.log` | JSONL 格式，每行一条合法 JSON 记录 |

此外还需包含：
- `submission.json`：包含 `problem_id: "PDE_Burgers"`、`code_path: "code"`
- `code/`：完整的提交代码目录

### Log 文件详细规范

`task{N}_logs.log` 每一行是合法 JSON，必须包含以下字段：

| 字段 | 说明 |
|------|------|
| `timestamp` | ISO 8601 时间戳（含时区），如 `2026-05-06T09:02:54.524886+00:00` |
| `elapsed_seconds` | 本次 LLM 调用耗时（秒） |
| `response` 或 `tool_calls` | LLM 输出内容或工具调用记录，**至少存在一个** |

关键约束：
- `response` 和 `tool_calls` 是评审核心依据：系统通过分析这两个字段验证 `code/` 中的代码是否完全由 Agent 生成。
- **单个 log 文件中，最后一条与第一条记录的 `timestamp` 之差不得超过 12 小时。** 超时该 task 得 0 分。
- 如果 Task 1 和 Task 2 由同一个 Agent session 连续完成，`task1_logs.log` 与 `task2_logs.log` 应保持一致（内容相同）。

---

### 评分规则（本地代理）

官方评分分三段计算（去除前 10 帧初始条件后，对剩余 190 帧）：

| 段 | 时间步范围 | 评估方式 |
|----|-----------|---------|
| Segment 1 | 步 0–47（相对前 50 帧） | Relative MSE → `100 * exp(-20 * rel_mse)` |
| Segment 2 | 步 47–95（相对前 50 帧） | Relative MSE → `100 * exp(-10 * rel_mse)` |
| Segment 3 | 步 95–190（长时预测） | RMSE → Lorentzian `100 / (1 + 10 * rmse)` + 可能的 Frechet 分量 |

> 注意：官方页面未提供 Segment 3 Frechet 分量的完整实现细节。本地使用 Lorentzian 分量作为透明代理，同时保留 raw MSE、forecast MSE 和逐段指标。

### 训练时间折算

官方使用 A100 GPU 作为基准，根据提交的 `train_time` 和 `inference_time` 进行时间扣分。本机使用 RTX 5070，在官方公布 RTX 5070 → A100 折算系数前，以本地实测时间记录。

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

默认输出：

```text
runs\<run_id>\pred.zip
```

运行占位 baseline 时使用脚本路径，不要使用 `python -m code.baseline_stub`：

```powershell
python code\baseline_stub.py --input data\...\task1_test.hdf5 --output runs\...\task1_pred.hdf5
```

`baseline_stub.py` 默认 `--dataset-key auto`，会优先识别官方 `tensor` 数据集；仅在特殊文件结构下手动指定 `--dataset-key`。

校验预测前 10 帧是否匹配官方初始条件：

```powershell
python -c "from agent.submission import validate_initial_condition; validate_initial_condition(r'runs\...\task1_pred.hdf5', r'data\...\task1_test.hdf5'); print('ok')"
```

生成完整 Task 1 zero-train 提交目录：

```powershell
python -m agent.zero_submission --input data\data_and_sample_submission\data_and_sample_submission\train_val_test_init\task1_test.hdf5 --output-dir runs\task1-zero-submission --code-dir code
python -m agent.validate_submission --path runs\task1-zero-submission
```


评估 Task 1 validation prediction：

```powershell
python code\evaluate_task1.py --prediction runs\task1-val-zero\task1_val_pred.hdf5 --target data\data_and_sample_submission\data_and_sample_submission\train_val_test_init\task1_val.hdf5 --output runs\task1-val-zero\metrics.json
```

当前 zero-train validation 对照指标（2026-05-10）：

```text
mse: 0.06385390007387708
forecast_mse: 0.06721463165671283
long_horizon_mse: 0.06452582283150855
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

## 自主实验工作流

Task 1 现在支持 AIDE/ML-Master 风格的自主实验层：

```powershell
python -m agent.run_task1_autonomous_experiment --config configs\kimi.example.yaml --study-name task1-autonomous --max-iterations 3
```

真实 Task 1 轻量 bootstrap 可先不依赖 LLM，直接生成一个保守的 weight-search 节点：

```powershell
D:\Junao\ProgramData\anaconda3\envs\Hwpytorch\python.exe -m agent.run_task1_autonomous_experiment --config configs\task1_mock.yaml --study-name task1-autonomous-bootstrap --max-iterations 1 --bootstrap-weight-search --bootstrap-grid-step 0.01 --bootstrap-grid-radius 2 --metric competition_score_proxy --maximize --checkpoint-override nu0.1=runs\task1-finetune-nu0.1-lr3e-6-short-proxy\best.pt
```

该 bootstrap 会围绕当前最佳 `nu0.01/nu0.1` 权重做 line search；每个候选的 validation 指标写入 journal artifact 的 `candidate_results` 并摊平成 `candidate_comparison.csv`，节点 best 继续进入 `experiment_comparison.csv` 和全局 registry。

输出目录：

```text
runs\<study_name>\journal.json
runs\<study_name>\planner_logs.log
runs\<study_name>\autonomous_summary.json
runs\<study_name>\journal_report.md
runs\<study_name>\experiment_results.json
runs\<study_name>\experiment_comparison.csv
runs\<study_name>\candidate_comparison.csv
runs\experiment_registry.jsonl
```

横向比较已有实验：

```powershell
python -m agent.compare_experiments --metric competition_score_proxy --maximize --top-k 10
```

LLM 每轮只能输出一个原子实验计划，结构必须包含：

```json
{
  "intent": "draft|improve|debug|submit|stop",
  "hypothesis": "why this experiment should help",
  "action_type": "weight_search|finetune|code_patch|submit_best|stop",
  "params": {},
  "expected_effect": "expected metric or reliability change",
  "risk": "what can go wrong"
}
```

允许 `code_patch` 重写 `code/` 下提交源码，包括较大改动；但补丁必须进入 journal，路径不能越出 `code/`，执行后必须通过 validation/submission 校验链路。不要手工绕过 journal 直接改最终提交代码。

`code_patch` 可携带受控验证命令：

```json
{
  "params": {
    "files": [{"path": "fno_ensemble.py", "content": "..."}],
    "validation_command": ["python", "-m", "pytest", "-q"]
  }
}
```

验证命令只允许验证入口（当前为 `python` / `pytest`），失败会把节点标记为 `failed`，reviewer 会建议下一轮 `debug`。正式 autonomous CLI 默认开启 strict gate：每个 `code_patch` 必须携带 `validation_command` 或 `submission_validation_path`。

也可以直接要求本地提交校验：

```json
{
  "params": {
    "files": [{"path": "fno_ensemble.py", "content": "..."}],
    "submission_validation_path": "runs/task1-finetune-nu0.1-short-proxy-final"
  }
}
```

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
