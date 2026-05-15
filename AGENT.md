# AI4S-PDE-CNS Agent Guide

本文件是给后续 Codex、Claude Code、Cursor、OpenCode 或其他代码智能体使用的项目入口说明。进入本仓库后，先读本文件，再执行任何训练、生成代码或提交打包动作。

## 项目目标

本仓库用于参加第四届世界科学智能大赛 AI4S 智能体 CNS 挑战赛的任务 4：神经算子 PDE 智能体。

当前核心目标不是手工调一个单点模型，而是构建可审计、可迭代、可提交的 Agent 科研闭环：

- 通过 API LLM 做实验规划、诊断分析和下一步决策。
- 使用受控工具执行数据检查、训练、推理、评估和提交校验。
- 记录每次 LLM 调用、工具动作、代码变更、指标结果和提交产物，保证代码与日志可追溯。
- 借鉴 ML-Master/AIDE 的 MCTS 搜索树、draft/improve/debug 分流、实验记忆和 best-candidate 提升机制。
- 不直接照搬 ML-Master 的 Kaggle CSV 假设；本项目必须围绕 HDF5 预测、`pred.zip`、`methodology.pdf`、官方 task logs 和 code-log consistency 构建。
- 当前模型路线从官方 FNO / U-Net checkpoint ensemble 出发，继续扩展 Task 1 微调、postprocess、residual refiner，以及 Task 2 独立训练。

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
runs/         每次 Agent 实验输出、日志、manifest、临时提交；新 autonomous 实验按 task/category/date 分类
submission/   最终提交目录模板
tests/        单元测试和 smoke test
```

重要：`code/` 目录不要添加 `__init__.py`。Python 标准库也叫 `code`，把 `code/` 变成包会遮蔽标准库，导致 `pytest` 等工具异常。

## 当前数据位置

官方数据已放在：

```text
D:\Study\AI4S-PDE-CNS\data\data_and_sample_submission\train_val_test_init
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

当前系统 Python 可能是 3.13，裸 `python` 不一定包含 PyTorch/HDF5 依赖。当前本机可用的主要实验环境是 `Hwpytorch`：

```text
D:\Junao\ProgramData\anaconda3\envs\Hwpytorch\python.exe
```

已确认该环境 Python 版本为 3.10.18。当前 `configs/task1.yaml` 已通过 `python_executable` 指向该解释器。后续涉及 PyTorch、HDF5、checkpoint 推理、Task 1/Task 2 validation、final submission 打包时，优先使用这个解释器，避免系统 Python 环境缺依赖或版本不一致。

推荐命令写法：

```powershell
$PY = "D:\Junao\ProgramData\anaconda3\envs\Hwpytorch\python.exe"
& $PY -m pytest -q
& $PY -m agent.env_check
& $PY -m agent.validate_submission --path runs\<run_id>
```

如果需要新建独立环境，可参考：

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
D:\Junao\ProgramData\anaconda3\envs\Hwpytorch\python.exe -m pytest -q
```

检查环境：

```powershell
D:\Junao\ProgramData\anaconda3\envs\Hwpytorch\python.exe -m agent.env_check
```

运行 mock Agent：

```powershell
python -m agent.run --task task1 --config configs\task1.yaml
```

校验提交目录：

```powershell
D:\Junao\ProgramData\anaconda3\envs\Hwpytorch\python.exe -m agent.validate_submission --path runs\local-smoke-submission
```

打包通过校验的 run：

```powershell
python -m agent.pack_submission --run runs\<run_id>
```

默认输出：

```text
runs\<run_id>\pred.zip
```

严格代码溯源打包（只用于准备符合新官方 code-log consistency 要求的最终包）：

```powershell
python -m agent.final_submission --run-name final-strict --require-llm-code-trace --provenance-log runs\task1\autonomous\<YYYYMMDD>\<study>\planner_logs.log
```

开启该开关后，`submission.json` 会写入 `require_llm_code_trace: true`，本地 validator 会忽略 `Task1FNOWorkflow`、`Task2PersistenceWorkflow`、`bootstrap` 等合成 trace，只接受真实 LLM provider 日志中的 `code_patch` 记录来证明 `code/` 文件内容来源。严格模式不会再追加 `provider=codex` 的合成 code trace，避免官方 JSONL log 被非 LLM 调用污染。`--provenance-log` 会把 autonomous `planner_logs.log` 追加进最终 `task{N}_logs.log`。因此最终提交前，Agent 必须先通过真实 LLM 的 `code_patch` action 生成或重写提交代码，并在 provenance log 中保留对应文件内容和 hash。

严格自主性审计打包：

```powershell
python -m agent.final_submission --run-name final-strict --task1-run runs\<task1_run> --task2-run runs\<task2_run> --require-llm-code-trace --require-autonomy-audit --task1-study-dir runs\task1\autonomous\<YYYYMMDD>\<task1_study> --task2-study-dir runs\task2\autonomous\<YYYYMMDD>\<task2_study> --provenance-log runs\task1\autonomous\<YYYYMMDD>\<task1_study>\planner_logs.log --provenance-log runs\task2\autonomous\<YYYYMMDD>\<task2_study>\planner_logs.log
```

`--require-autonomy-audit` 会在生成 `pred.zip` 前检查对应 study：`planner_logs.log` 必须来自真实 LLM provider，不能是 `bootstrap/mock/static`；`journal.json` 必须包含 baseline/source 研读痕迹、至少两个带指标的实验、至少一个失败/负例分析、以及完成的 `code_patch`。Task 1 finetune 还必须能追溯到官方 Nu0.001 FNO checkpoint，并在实验中由 Agent 选择 `temporal_stride=5`；Task 2 不能引用 Task 1 数据或 checkpoint。

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
python -m agent.zero_submission --input data\data_and_sample_submission\train_val_test_init\task1_test.hdf5 --output-dir runs\task1-zero-submission --code-dir code
python -m agent.validate_submission --path runs\task1-zero-submission
```


评估 Task 1 validation prediction：

```powershell
python code\evaluate_task1.py --prediction runs\task1-val-zero\task1_val_pred.hdf5 --target data\data_and_sample_submission\train_val_test_init\task1_val.hdf5 --output runs\task1-val-zero\metrics.json
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
$env:AIGC_API_KEY="..."          # HKUST(GZ) AIGC API，优先使用
$env:HKUSTGZ_AIGC_API_KEY="..." # HKUST(GZ) AIGC API 备用变量名
$env:DEEPSEEK_API_KEY="..."
$env:KIMI_CODE_API_KEY="..."
$env:KIMI_API_KEY="..."
$env:MOONSHOT_API_KEY="..."
$env:SILICONFLOW_API_KEY="..."
```

也可以把 key 写入本地 `.env`。`.env` 不应提交到 git。

Provider 配置集中在：

```text
configs\llm_providers.yaml
```

可用 profile：

```text
hkustgz_gpt53       HKUST(GZ) AIGC OpenAI-compatible API，默认 gpt-5.3-chat
deepseek             DeepSeek 官方 API，默认 deepseek-v4-pro
deepseek_flash       DeepSeek 官方 API，默认 deepseek-v4-flash
kimi                 Kimi Code API，固定模型 kimi-for-coding
kimi_open_platform   Moonshot/Kimi 开放平台，默认 kimi-k2.6
siliconflow_glm      硅基流动 GLM profile
siliconflow_deepseek 硅基流动 DeepSeek profile
```

HKUST(GZ) profile 当前配置：

```yaml
hkustgz_gpt53:
  provider: hkustgz_gpt
  model: gpt-5.3-chat
  api_key_env:
    - AIGC_API_KEY
    - HKUSTGZ_AIGC_API_KEY
    - OPENAI_API_KEY
  base_url: https://aigc-api.hkust-gz.edu.cn/v1
  use_env_proxy: false
  request_options:
    stream: false
    max_completion_tokens: 4096
```

HKUST(GZ) 专用运行配置示例：

```yaml
llm_profile: hkustgz_gpt53
llm_profiles_path: llm_providers.yaml
env_file: .env
max_iterations: 1
time_budget_seconds: 43200
allowed_shell_commands:
  - python
  - pytest
  - git
```

该示例文件位于：

```text
configs\gpt53_hkustgz.example.yaml
```

任务配置通过 `llm_profile` 切换模型供应商：

```yaml
llm_profile: hkustgz_gpt53
llm_profiles_path: llm_providers.yaml
```

先做 API 连通性检查：

```powershell
D:\Junao\ProgramData\anaconda3\envs\Hwpytorch\python.exe -m agent.check_llm --config configs\gpt53_hkustgz.example.yaml
D:\Junao\ProgramData\anaconda3\envs\Hwpytorch\python.exe -m agent.check_llm --config configs\deepseek.example.yaml
D:\Junao\ProgramData\anaconda3\envs\Hwpytorch\python.exe -m agent.check_llm --config configs\kimi.example.yaml
D:\Junao\ProgramData\anaconda3\envs\Hwpytorch\python.exe -m agent.check_llm --config configs\siliconflow.example.yaml
```

运行 Agent 示例：

```powershell
D:\Junao\ProgramData\anaconda3\envs\Hwpytorch\python.exe -m agent.run --task task1 --config configs\gpt53_hkustgz.example.yaml
D:\Junao\ProgramData\anaconda3\envs\Hwpytorch\python.exe -m agent.run --task task1 --config configs\deepseek.example.yaml
D:\Junao\ProgramData\anaconda3\envs\Hwpytorch\python.exe -m agent.run --task task1 --config configs\kimi.example.yaml
D:\Junao\ProgramData\anaconda3\envs\Hwpytorch\python.exe -m agent.run --task task1 --config configs\siliconflow.example.yaml
```

不要让真实 API Agent 一开始就训练大模型。推荐顺序：

1. 让 Agent 只读取赛题摘要并生成实验计划。
2. 检查 `task1_logs.log` 是否是合法 JSONL。
3. 让 Agent 修改小文件或生成脚本草稿。
4. 再接入 FNO baseline 的训练/推理。

## Agent 工作流

本项目的目标 Agent 是 **PDE-Research-Agent**：它不是“聊天机器人 + 跑脚本”，而是一个受控的科研实验系统。每轮必须先观察当前 PDE 状态，再提出一个可审计实验，调用白名单工具执行，确定性评估结果，写入日志和 journal，然后决定继续扩展、debug、停止或提交。

完整闭环：

```text
Rule Guard -> Observer -> Memory -> Planner -> Tool Router -> Controlled Executor -> Evaluator -> Reviewer -> Memory/Submitter
```

各模块职责：

- `Rule Guard`：编码赛题规则、Task 1/Task 2 隔离、官方 checkpoint 约束、禁止数值求解器、训练/推理时间预算。
- `Observer`：读取 HDF5 数据、checkpoint、当前 prediction、验证指标、分段误差、耗时和失败原因。
- `Memory`：整理历史实验、最佳候选、失败分支、代码 hash、平台反馈和验证 artifacts。
- `Planner`：调用 LLM 输出一个原子实验 JSON，不直接自由发挥 shell 脚本。
- `Tool Router`：把 JSON action 映射到受控工具。
- `Controlled Executor`：只执行白名单动作，产物写入 `runs/<study>/<node_id>/...`。
- `Evaluator`：计算本地 MSE、forecast MSE、分段 proxy score、first-10-frame 校验、shape 校验、runtime。
- `Reviewer`：判断实验是否提升、是否合规、是否需要 debug、是否值得扩展。
- `Submitter`：生成 `pred.zip`、运行本地 validator、写官方格式 log 和 methodology artifacts。

每轮顺序：

1. `observe`：生成当前 task state，包括数据、模型、指标、失败和预算。
2. `remember`：压缩 journal，只保留和当前决策有关的实验记忆。
3. `plan`：LLM 只输出一个合法 action JSON。
4. `validate_action`：校验 action schema、路径、task 隔离和安全规则。
5. `route_tool`：选择对应工具。
6. `execute`：执行工具并写入确定性 artifacts。
7. `evaluate`：用本地 evaluator 计算指标，不让 LLM 猜分数。
8. `review`：决定 promote、debug、expand、stop 或 submit。
9. `record`：写 `journal.json`、`journal_report.md`、`candidate_comparison.csv`、LLM JSONL log 和 code trace。
10. `submit`：只有 validator 通过后才生成最终 `pred.zip`。

所有 LLM 调用必须通过 `LLMCallLogger`，不要直接绕过 logger 调 API。

### Action schema

LLM 每轮只能输出一个原子实验计划：

```json
{
  "intent": "draft|improve|debug|submit|stop",
  "hypothesis": "one concrete scientific reason for this experiment",
  "action_type": "one whitelisted action type",
  "params": {},
  "expected_effect": "expected metric, runtime, or reliability change",
  "risk": "what can go wrong and how it will be checked"
}
```

推荐 action 类型：

```text
inspect_data
diagnose_error
validate_baseline
weight_search
postprocess_search
finetune_checkpoint
train_refiner
baseline_zoo
evaluate_candidate
submit_best
validate_submission
code_patch
stop
```

一个 action 只能表达一个想法，例如一次权重搜索、一次分段 postprocess、一次短微调、一次 refiner 实验、一次 baseline 验证或一次提交打包。不要把多个不相关实验塞进同一个 action。

### Observer 要求

Task 1 Observer 至少要给 Planner：

- 可用数据文件、HDF5 key、shape。
- 当前预测 shape、first-10-frame 最大误差。
- overall MSE、forecast MSE、long-horizon MSE。
- 三段官方 proxy 相关指标。
- FNO、U-Net、ensemble、postprocess 的对比。
- 哪些时间段/样本误差最大。
- 当前候选是否只使用官方 checkpoint、是否微调、是否引入额外组件。
- 训练和推理耗时预算。

Task 2 Observer 至少要给 Planner：

- train/val/test shape，`nu` 是否可见。
- 是否严格没有使用 Task 1 数据和 checkpoint。
- 当前 baseline 是 persistence scaffold 还是已训练模型。
- 验证 split、推理耗时和提交 shape。

### Planner 优先级

Planner 应按以下顺序推进：

1. 复现实验和验证当前 baseline。
2. 诊断 baseline 失败位置。
3. 先做低成本搜索：ensemble、segment weights、persistence/postprocess、cached prediction blending。
4. 再做可控训练：checkpoint 短微调、residual refiner、segment-weighted loss、spectral/physics loss。
5. 再扩展模型家族：FNO variants、DeepONet、PI-DeepONet、U-Net-style predictor、Baseline Zoo。
6. 只有 shape、first-10-frame、runtime、code-log consistency、本地 validator 全部通过，才允许 submit。

### MCTS 适配原则

借鉴 ML-Master 的搜索思想，但不要照搬它的 Kaggle 执行契约。本项目中，MCTS 节点应该是结构化 PDE 实验，而不是任意生成的 Python 脚本。

- root 节点生成 baseline 或 diagnostics 分支。
- 成功节点生成 `improve` 子节点。
- 失败节点只有在错误可修时生成 `debug` 子节点。
- 多次低提升分支应终止。
- reward 来自确定性 evaluator：优先最大化官方 proxy score；没有 proxy 时最小化 validation MSE；失败、超时、非有限指标、不合规均惩罚。

## 自主实验工作流

Task 1 现在支持 AIDE/ML-Master 风格的自主实验层，但后续方向是 PDE-Research-Agent：LLM/MCTS 动态提出 action，Executor 只执行白名单工具，Evaluator 决定 reward，Journal 负责可追溯记录。

```powershell
python -m agent.run_task1_autonomous_experiment --config configs\kimi.example.yaml --study-name task1-autonomous --max-iterations 3
```

轻量 bootstrap 可先不依赖 LLM，直接生成一个保守节点，用于验证 action -> executor -> evaluator -> journal -> report 是否闭环。官方 checkpoint 路线优先使用 Nu0.001 FNO / U-Net ensemble 和 postprocess search；历史 nu0.1 路线只能作为实验参考，提交时要评估规则风险。

```powershell
D:\Junao\ProgramData\anaconda3\envs\Hwpytorch\python.exe -m agent.run_task1_autonomous_experiment --config configs\task1_mock.yaml --study-name task1-autonomous-postprocess-bootstrap --max-iterations 1 --metric competition_score_proxy --maximize --bootstrap-postprocess-search
```

该 bootstrap 会验证当前 official checkpoint ensemble/postprocess 链路；每个候选的 validation 指标写入 journal artifact 的 `candidate_results` 并摊平成 `candidate_comparison.csv`，节点 best 继续进入 `experiment_comparison.csv` 和全局 registry。

输出目录：

```text
runs\task1\autonomous\<YYYYMMDD>\<study_name>\journal.json
runs\task1\autonomous\<YYYYMMDD>\<study_name>\planner_logs.log
runs\task1\autonomous\<YYYYMMDD>\<study_name>\autonomous_summary.json
runs\task1\autonomous\<YYYYMMDD>\<study_name>\journal_report.md
runs\task1\autonomous\<YYYYMMDD>\<study_name>\experiment_results.json
runs\task1\autonomous\<YYYYMMDD>\<study_name>\experiment_comparison.csv
runs\task1\autonomous\<YYYYMMDD>\<study_name>\candidate_comparison.csv
runs\experiment_registry.jsonl
```

Task 2 autonomous 使用同样布局：

```text
runs\task2\autonomous\<YYYYMMDD>\<study_name>\
```

横向比较已有实验：

```powershell
python -m agent.compare_experiments --metric competition_score_proxy --maximize --top-k 10
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
    "submission_validation_path": "runs/<validated-run-dir>"
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

### Step 1: 固化 PDE-Research-Agent 入口

目标是形成一个统一命令入口，而不是分散脚本：

```text
agent.pde_research_agent
configs/pde_research_agent.yaml
Rule Guard + Observer + Planner + Tool Router + Executor + Evaluator + Reviewer + Submitter
```

第一版不追求复杂模型，先证明闭环可运行、日志可追溯、提交可验证。

### Step 2: 加强 Observer

补齐面向 PDE 的诊断能力：

```text
HDF5 key / shape / coordinate summary
prediction shape / first-10-frame error
overall MSE / forecast MSE / segment score proxy
sample-level and horizon-level error concentration
runtime and train-time budget report
Task 1 / Task 2 data and checkpoint isolation check
```

Observer 输出应是 compact JSON，可直接放进 Planner prompt。

### Step 3: 收敛 Action schema 和 Tool Router

将所有可执行实验收敛为白名单 action：

```text
inspect_data
diagnose_error
validate_baseline
weight_search
postprocess_search
finetune_checkpoint
train_refiner
baseline_zoo
task2_train_model
task2_submit_best
evaluate_candidate
submit_best
validate_submission
code_patch
stop
```

LLM 只能输出 action JSON，Tool Router 决定调用哪个本地工具。除 `code_patch` 外，不允许 LLM 直接写大段脚本。

### Step 4: MCTS + LLM 动态搜索

把现有静态 YAML actions 升级为动态搜索：

```text
root: baseline / diagnostics
improve: 基于当前 best node 做单点改进
debug: 修复失败节点
stop: 低收益、超时、违规风险或提交完成
```

Reward 使用本地确定性 evaluator，不让 LLM 主观判断分数。

### Step 5: Task 1 模型演进

Task 1 当前优先路径：

```text
official Nu0.001 FNO / U-Net checkpoint ensemble
segment-wise ensemble weights
persistence / postprocess search
short checkpoint fine-tuning
residual refiner / correction head
FNO variants / DeepONet / PI-DeepONet branches
```

每个实验都要比较 validation 指标、推理耗时、训练耗时和规则风险。

### Step 6: Task 2 独立训练

Task 2 必须独立：

```text
不加载 Task 1 checkpoint
不使用 Task 1 数据
使用官方 Task 2 train/val/test
训练时可利用 train/val Nu
推理时不能依赖 test Nu
```

Task 2 初期可以保留 persistence scaffold，但正式刷分需要独立训练多 Nu 模型。

### Step 7: 提交闭环

最终提交必须由 Agent 闭环生成：

```text
journal.json
journal_report.md
candidate_comparison.csv
task{N}_logs.log
task{N}_pred.hdf5
task{N}_time.csv
methodology.pdf
submission/code
pred.zip
validator result
```

如果结果无法从 action 参数、LLM log、code hash 和 artifact 追溯，不允许提交。

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
