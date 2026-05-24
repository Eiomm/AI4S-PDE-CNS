# AGENTS.md

## 0. 项目目标

本仓库位于 `/root/autodl-tmp/AI4Sv2/Task2`。目标是在 Task2 下实现一个 **MLEvolve-lite 风格的 MCGS/MCTS-lite 搜索框架**，用于 AI4S PDE Task2 多 `nu` 泛化任务。

第一版目标不是复刻完整 `third/MLEvolve`，也不是追求最高分，而是跑通最小可运行闭环：

```text
Node -> Code workspace -> Static check -> Shape check -> Cheap probe -> Metrics -> Reward -> Backup -> Next node
```

外层用 MCGS/MCTS-lite 管理 Candidate 节点，内层调用 AIDE/LLM 生成或修改 Candidate 的 solution code。GPU 训练在单卡 RTX 5090 上默认串行执行。

本文件只保留主规则、数据契约和提交契约：

- `data/Task2`：本任务数据格式、验证方式和提交契约。
- `MCGS_LITE_IMPLEMENTATION.md`：第三方代码参考和 MCGS-lite 具体实现设计。

官方赛题页（`https://competition.ai4s.com.cn/race/7/description`）的关键口径高于本地推断：Task2 提交文件是完整 200 步轨迹，其中前 10 步必须逐值复制测试输入，真正预测和评分的是第 10-199 步共 190 个未来步。

## 1. 不可破坏原则

所有实现必须遵守：

1. 不允许 Candidate solution code 读取 test `nu`，因为 `task2_test.h5` 本身不提供 `nu`。
2. 不允许使用数值求解器直接模拟 Burgers 方程来生成测试未来轨迹。
3. 不允许使用额外数据。
4. 不允许修改 runner / proxy / submission checker 主控逻辑来绕过评估。
5. 内层 AIDE/LLM 只能修改 `workspace/nodes/<candidate_id>/code/` 下的 Candidate solution code。
6. 主控系统代码只能由当前 coding agent 按本规范修改，不能交给内层 AIDE/LLM 自行改写。
7. GPU training job 默认串行；同一时间只允许一个训练/评估任务占用 GPU。
8. 第一版 cheap probe 优先，失败节点只能标记 failed，不能导致整个搜索中断。
9. 原始 `data/Task2/*.h5` 只读，不得改写、移动或覆盖。
10. Task2 按官方描述禁止使用任何公开预训练权重或 checkpoint；所有模型必须仅基于本题训练集从头训练。保守执行口径：除当前 Task2 Candidate/Trial 自己从本题数据训练出的 checkpoint 外，一律不得加载 checkpoint，包括 PDEBench 官方 checkpoint、公开代码库权重和第三方来源参数。
11. Task2 总耗时需控制在 12 小时内，测试集推理时间必须在 2 分钟以内，否则该任务为 0 分。

## 2. 权限边界

本仓库同时存在两类 agent，权限必须分清：

- **当前 coding agent**：当前在仓库中修改文件、实现主控系统、维护 `AGENTS.md` / `MCGS_LITE_IMPLEMENTATION.md`、编写 `mlevolve_lite/` 基础设施并运行验证命令的工程 agent。它可以修改主控系统和文档，但不能把自己手写的 solution code 伪装成内层 AIDE/LLM 生成的 Candidate 代码。
- **内层 AIDE/LLM**：由 MCGS-lite harness 调用，用于生成、改进、调试某个 Candidate 的 solution code。它的写权限只限于 `workspace/nodes/<candidate_id>/code/`，输出必须被 JSONL 日志记录，并能追溯到 `task2_logs.log` 中的 `response` 或 `tool_calls`。

边界规则：

- 当前 coding agent 实现或修复主控系统时，修改 `mlevolve_lite/`、测试脚本、文档和验证工具。
- 内层 AIDE/LLM 生成参赛 solution 时，只产出 Candidate code，不修改 `mlevolve_lite/`、`AGENTS.md`、`MCGS_LITE_IMPLEMENTATION.md` 或 evaluator/checker。
- 若需要人工热修 Candidate code，只能作为主控系统调试样例或临时 sandbox，不得进入正式提交；正式 Candidate 必须由内层 AIDE/LLM 生成并有日志可追溯。

## 3. Candidate 与 Trial 区分

不要把 Candidate 和 Trial 混用：

- **Candidate**：搜索图中的方案节点，代表一个 hypothesis、代码快照、父节点 lineage 和 operator。Candidate 的代码目录是 `workspace/nodes/<candidate_id>/code/`。Candidate 一旦创建，其代码快照应视为不可变；后续 debug/improve/ablate/fusion 应创建新的 child Candidate。
- **Trial**：对某个 Candidate 的一次具体执行，包括 smoke test、static/shape check、cheap probe、full train 或 final inference。Trial 记录固定配置、随机种子、运行时间、日志、metrics 和 artifacts。

存储建议：

```text
workspace/nodes/<candidate_id>/
├── node.json
├── code/
├── trials/
│   └── <trial_id>/
│       ├── config.json
│       ├── metrics.json
│       ├── logs/
│       └── artifacts/
└── metrics.json              # 汇总最佳 Trial 或最新 promoted Trial
```

规则：

- 一个 Candidate 可以有多个 Trial；某个 Trial 失败不等于 Candidate 必然失效。
- Candidate 的 `status` 描述搜索节点状态，Trial 的 `status` 描述单次运行状态。
- leaderboard/promoted 应引用具体 `candidate_id` 和 `trial_id`，避免无法复现。
- cheap probe 是 Trial，不是 Candidate；promoted 的对象是 Candidate + 其最佳合规 Trial。

## 4. 当前仓库结构

```text
Task2/
├── AGENTS.md
├── MCGS_LITE_IMPLEMENTATION.md
├── data/
│   └── Task2/
│       ├── task2_part0_train.h5
│       ├── task2_part1_train.h5
│       ├── task2_part2_train.h5
│       ├── task2_val.h5
│       ├── task2_test.h5
│       └── sample_submission/
│           ├── code/train.py
│           ├── submission.json
│           ├── task2_logs.log
│           ├── task2_pred.hdf5
│           └── task2_time.csv
└── third/
    ├── MLEvolve/
    └── aideml/
```

## 5. 实现设计文档

`AGENTS.md` 只保留赛题规则、数据/提交契约和工作方式。以下内容已经拆分到 [`MCGS_LITE_IMPLEMENTATION.md`](./MCGS_LITE_IMPLEMENTATION.md)，实现 MCGS-lite 时必须先阅读该文件：

- 如何参考 `third/aideml` 和 `third/MLEvolve` 的代码结构。
- `mlevolve_lite/` 与 `workspace/` 的目录设计。
- Node/Metrics schema、Graph DB、Operators、Selector、Scheduler、GPU Queue、Evaluator、Reward、Memory、Prompt、Seed Nodes。
- Cheap probe、promotion、验收标准、推荐实现顺序和代码质量要求。

本文件中的不可破坏原则优先级最高；实现文档不得覆盖或放宽赛题规则。

## 6. Task2 数据契约

### 6.1 训练集

训练集分 3 个 HDF5 分片：

```text
data/Task2/task2_part0_train.h5
data/Task2/task2_part1_train.h5
data/Task2/task2_part2_train.h5
```

每个分片字段一致：

```text
tensor        float32, shape (1000, 320, 256)
nu            float32, shape (1000,)
t_coordinate  float32, shape (320,), range 0.00 .. 15.95
x_coordinate  float32, shape (256,), range about 0.00049 .. 0.99658
```

合并后共有 3000 条训练轨迹，每条 320 个时间步，每步 256 个空间点。

### 6.2 验证集

```text
data/Task2/task2_val.h5
```

字段：

```text
tensor        float32, shape (100, 210, 256)
nu            float32, shape (100,)
t_coordinate  float32, shape (210,), range 0.15 .. 10.60
x_coordinate  float32, shape (256,)
```

官方兼容验证切分：

```text
val_input       = tensor[:, :10, :]       # 已观测初始条件
val_full_target = tensor[:, :200, :]      # 对齐提交 shape 的 200 步完整轨迹
val_future_190  = tensor[:, 10:200, :]    # 官方评分的 190 个未来步
```

主指标应优先估计官方评分口径，而不是只看普通 MSE：

```text
val_pred_full.shape == (N, 200, 256)
first_10_pass = max_abs(val_pred_full[:, :10, :] - val_input) <= 1e-3
future_rel_mse = segment-wise Rel-MSE over val_pred_full[:, 10:200, :]
```

同时记录：

```text
short_mse: MSE/Rel-MSE over early predicted steps
long_stat_error: long-horizon distribution/statistical error
per_nu_mse: validation MSE grouped by nu bins or exact val trajectories
worst_nu_mse: worst group/trajectory MSE
heldout_nu_mse: held-out nu split MSE if split exists
nu_estimation_mae: if solution estimates nu from first 10 steps
mse_future_200_optional: optional internal MSE over tensor[:, 10:210, :] for stability research only
```

### 6.3 测试集

```text
data/Task2/task2_test.h5
```

字段：

```text
tensor        float32, shape (1000, 10, 256)
t_coordinate  float32, shape (10,), range 0.15 .. 0.60
x_coordinate  float32, shape (256,)
```

测试集没有 `nu`。solution code 若训练阶段使用 `nu`，推理阶段必须通过观测序列估计、边缘化或隐式建模，不能读取真实 test `nu`。

提交预测必须把 `task2_test.h5` 的前 10 步原样放在 `task2_pred.hdf5/tensor[:, :10, :]`，容差 `1e-3`。`tensor[:, 10:200, :]` 才是 190 个预测时间步。

## 7. 提交契约

Task2 预测文件必须为 HDF5：

```text
task2_pred.hdf5
```

内部必须有：

```text
dataset name: tensor
shape:        (1000, 200, 256)
dtype:        float32
finite:       no NaN/Inf
first 10:     exactly copy task2_test.h5/tensor within tolerance 1e-3
predicted:    only steps 10..199 are model predictions and scored
```

时间文件：

```text
task2_time.csv
```

格式：

```csv
train_time, inference_time
1200, 60
```

节点 cheap probe 不一定要生成最终 zip，但 promoted/full 节点必须能从其 `code/train.py` 重新生成上述文件。

### 7.1 官方日志与提交要求

每个已提交任务至少包含：

```text
task2_pred.hdf5
task2_time.csv
task2_logs.log
```

`task2_logs.log` 是 Agent 科研日志，必须是 JSONL：每一行是一条合法 JSON，至少包含：

```text
timestamp
elapsed_seconds
response or tool_calls
```

日志需要能追溯 Agent 的科研闭环：调研/假设、代码修改、实验版本、失败分析、验证结论和迭代记录。提交的 `code/` 目录必须与 log 中的 LLM 调用历史相互对应，不能出现无法追溯的人工作弊式代码。

`submission.json` 的 Task2 入口使用：

```json
{
  "submission_id": "<team-or-run-id>",
  "problem_id": "PDE_Burgers",
  "code_path": "code"
}
```

## 8. Default Validation Commands

当前 coding agent 默认按以下命令验证基础契约。若某个模块尚未实现，应明确说明跳过原因。

### 8.1 数据与样例提交契约

```bash
python - <<'PY'
from pathlib import Path
import h5py
import numpy as np

root = Path("data/Task2")
for name in ["task2_part0_train.h5", "task2_part1_train.h5", "task2_part2_train.h5"]:
    with h5py.File(root / name, "r") as f:
        assert f["tensor"].shape == (1000, 320, 256), name
        assert f["nu"].shape == (1000,), name
with h5py.File(root / "task2_val.h5", "r") as f:
    assert f["tensor"].shape == (100, 210, 256)
    assert f["nu"].shape == (100,)
with h5py.File(root / "task2_test.h5", "r") as test, h5py.File(root / "sample_submission/task2_pred.hdf5", "r") as pred:
    assert pred["tensor"].shape == (1000, 200, 256)
    assert pred["tensor"].dtype == np.float32
    assert np.max(np.abs(pred["tensor"][:, :10, :] - test["tensor"][:])) <= 1e-3
print("data contract ok")
PY
```

### 8.2 MCGS-lite smoke run

实现 `mlevolve_lite` 后，默认先跑 1 轮 cheap probe：

```bash
python -m mlevolve_lite.scheduler \
  --data-dir data/Task2 \
  --workspace workspace \
  --rounds 1 \
  --cheap-epochs 1 \
  --timeout-sec 1800
```

### 8.3 Candidate code checkpoint scan

对正式 Candidate code 做保守扫描。命中不一定违规，但必须人工确认只加载了当前 Task2 Trial 自己训练出的 checkpoint：

```bash
rg -n "torch\\.load|load_state_dict|from_pretrained|checkpoint|\\.pt|\\.pth|\\.ckpt|safetensors" workspace/nodes/<candidate_id>/code
```

### 8.4 预测文件检查

```bash
PRED=workspace/nodes/<candidate_id>/trials/<trial_id>/artifacts/task2_pred.hdf5 python - <<'PY'
import os
import h5py
import numpy as np

pred_path = os.environ["PRED"]
with h5py.File("data/Task2/task2_test.h5", "r") as test, h5py.File(pred_path, "r") as pred:
    a = pred["tensor"][:]
    assert a.shape == (1000, 200, 256)
    assert a.dtype == np.float32
    assert np.isfinite(a).all()
    assert np.max(np.abs(a[:, :10, :] - test["tensor"][:])) <= 1e-3
    print(a.shape, a.dtype, float(a.min()), float(a.max()))
PY
```

### 8.5 JSONL 日志检查

```bash
LOG=workspace/nodes/<candidate_id>/trials/<trial_id>/artifacts/task2_logs.log python - <<'PY'
import json
import os

path = os.environ["LOG"]
with open(path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f, 1):
        obj = json.loads(line)
        assert "timestamp" in obj, i
        assert "elapsed_seconds" in obj, i
        assert ("response" in obj) or ("tool_calls" in obj), i
print("jsonl log ok")
PY
```

## 9. 最终提交检查

当某个 promoted 节点进入最终提交阶段，必须检查：

```text
task2_pred.hdf5 exists
dataset "tensor" exists
shape == (1000, 200, 256)
dtype == float32
np.isfinite(tensor).all()
max_abs(tensor[:, :10, :] - task2_test.h5/tensor) <= 1e-3
task2_time.csv exists
code/train.py can regenerate prediction from data/Task2
task2_logs.log is valid JSONL and contains timestamp, elapsed_seconds, response/tool_calls
logs contain config, metric, runtime, compliance status, experiment trajectory
```

推荐检查命令：

```bash
python -c "import h5py, numpy as np; f=h5py.File('task2_pred.hdf5','r'); a=f['tensor']; print(a.shape, a.dtype, np.isfinite(a[...]).all(), float(a[:].min()), float(a[:].max()))"
```

## 10. Agent 工作方式

后续 coding agent 在本仓库工作时：

- 先读本文件；涉及 MCGS-lite 实现、第三方代码参考或模块设计时，再读 `MCGS_LITE_IMPLEMENTATION.md`。
- 不再创建或依赖 `AGENT.md`。
- 用户要求实现时，直接改代码并运行最小验证，不只给方案。
- 改主控系统前说明要改哪些模块。
- 任何训练命令先用 cheap probe。
- 如果无法完成 GPU 训练，至少完成 static/shape/evaluator 层验证并说明原因。
- 最终回复必须报告改了哪些文件、跑了哪些命令、结果如何、剩余风险是什么。
