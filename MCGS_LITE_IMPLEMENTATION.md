# MCGS_LITE_IMPLEMENTATION.md

本文件承接 [`AGENTS.md`](./AGENTS.md)。`AGENTS.md` 定义赛题规则、数据契约和提交要求；本文件只描述如何参考第三方代码并实现 Task2-MCGS-lite 搜索器。若两份文档冲突，以 `AGENTS.md` 为准。

## 0. 阅读边界

- 这里的内容可以指导主控系统实现，但不能放宽数据泄漏、数值求解器、公开 checkpoint、日志 JSONL、12 小时与 2 分钟推理等官方约束。
- LLM/AIDE 只能修改 `workspace/nodes/<node_id>/code/` 下的 solution code，不能修改本文件定义的主控系统逻辑。

## 1. 需要新增的目录

实现 MCGS-lite harness 时新增：

```text
mlevolve_lite/
├── __init__.py
├── graph_db.py
├── node_schema.py
├── operators.py
├── selector.py
├── scheduler.py
├── gpu_queue.py
├── evaluator.py
├── memory.py
└── prompts/
    ├── draft.md
    ├── improve.md
    ├── debug.md
    ├── ablate.md
    └── fusion.md

workspace/
├── graph.json
├── leaderboard.json
├── promoted.json
├── gpu_queue.jsonl
├── events.jsonl
└── nodes/
    └── <node_id>/
        ├── node.json
        ├── code/
        │   └── train.py
        ├── artifacts/
        ├── logs/
        └── metrics.json
```

不要把临时权重、缓存、预测文件散落在仓库根目录。每个节点的所有可变产物都放在自己的 `workspace/nodes/<node_id>/` 下。

## 2. 第三方仓库借鉴方式

### 2.1 从 `third/aideml` 借鉴

`aideml` 的核心结构是：

- `Node` 保存 code、plan、parent、children、执行输出、metric、is_buggy。
- `Journal` 保存所有节点，并能生成历史 summary。
- `Agent.step()` 根据状态选择 `draft`、`debug` 或 `improve`。
- 每次执行后 parse output，判断 bug 和 metric。

本项目映射：

- `node_schema.py` 定义 Task2 专用 Node/Metrics。
- `graph_db.py` 替代 Journal，持久化为 `workspace/graph.json`。
- `scheduler.py` 替代 Agent.step，负责一轮搜索闭环。
- `memory.py` 生成历史 summary，喂给 LLM/AIDE prompt。

### 2.2 从 `third/MLEvolve` 借鉴

`MLEvolve` 的关键思想：

- 多 draft 分支并行探索。
- 节点有 `stage`、`local_best_node`、`visits`、`total_reward`、`branch_id`。
- 用 UCT/MCGS 管理 explore/exploit。
- 对 buggy 节点走 debug，对可运行节点走 improve。
- 停滞后触发更大粒度改动或 fusion。
- 执行后做额外 validation，例如提交文件是否存在、指标是否异常。

本项目映射：

- `selector.py` 实现 MCTS-lite UCB。
- `operators.py` 实现 `draft/improve/debug/ablate/fusion`。
- `evaluator.py` 做 shape、first-10、NaN/Inf、runtime、compliance、per-nu 指标。
- `scheduler.py` 做 backup reward 到祖先节点。
- `promoted.json` 保存可进入长训练或最终提交的 top nodes。

## 3. Node Schema

`mlevolve_lite/node_schema.py` 必须定义可 JSON 序列化的数据结构。第一版可使用 `dataclasses`。

### 3.1 Node 字段

```text
node_id: str
signature: str
parent_ids: list[str]
operator: "draft" | "improve" | "debug" | "ablate" | "fusion"
hypothesis: str
code_dir: str
checkpoint: str | null
status: "created" | "code_generated" | "static_failed" | "shape_failed" | "preflight_passed" | "running" | "cheap_probe_passed" | "promoted" | "rejected" | "failed"
visits: int
mean_score: float
best_score: float
metrics: Metrics | null
lineage: list[str]
created_at: str
updated_at: str
failure_reason: str | null
log_path: str | null
artifact_dir: str
```

### 3.2 Metrics 字段

```text
overall_mse: float | null
short_mse: float | null
long_stat_error: float | null
per_nu_mse: dict[str, float]
worst_nu_mse: float | null
heldout_nu_mse: float | null
nu_estimation_mae: float | null
runtime_sec: float | null
shape_pass: bool
first_10_pass: bool
uses_true_nu_at_test: bool
compliance_pass: bool
reward: float
```

### 3.3 状态更新规则

状态只能向前推进：

```text
created
-> code_generated
-> preflight_passed
-> running
-> cheap_probe_passed | rejected | failed
-> promoted
```

`static_failed`、`shape_failed`、`failed` 是终止状态，但不影响其他节点继续搜索。

## 4. Graph DB

`mlevolve_lite/graph_db.py` 负责读写：

```text
workspace/graph.json
workspace/leaderboard.json
workspace/promoted.json
workspace/events.jsonl
```

要求：

- 所有写入使用临时文件 + 原子 rename，避免中断导致 JSON 损坏。
- `graph.json` 至少包含 `nodes`、`edges`、`root_ids`、`last_updated`。
- `leaderboard.json` 按 reward 或 overall_mse 排序。
- `promoted.json` 只保存通过 compliance 和阈值的节点。
- 每轮 scheduler 事件追加到 `events.jsonl`，便于追踪失败原因。

## 5. Operators

`mlevolve_lite/operators.py` 实现 5 类 operator。

### 5.1 draft

创建全新方案，不依赖父节点。第一版初始化 5 个 seed nodes：

```text
multi_nu_fno_baseline
nu_estimator_concat
nu_estimator_film
mixture_of_nu_experts
conditional_fno_marginalized_nu
```

### 5.2 improve

基于一个有效父节点，提出单一可归因改进。prompt 必须要求 LLM 写清：

```text
What changes
Why this Task2/PDE setting needs it
What remains unchanged
Expected metric movement
```

### 5.3 debug

基于 failed/buggy 父节点，仅修复导致失败的最小问题。不要顺手改变模型范式。

### 5.4 ablate

对 promoted 或高分节点做消融，验证某个组件是否真的贡献 reward。例如去掉 `nu` estimator、去掉 rollout loss、去掉 FiLM conditioning。

### 5.5 fusion

从两个或多个高分节点融合已验证组件。只在多个分支有互补优势时使用，例如一个 short_mse 好、另一个 worst_nu_mse 好。

## 6. Selector

`mlevolve_lite/selector.py` 实现 MCTS-lite UCB：

```text
selection_score =
  mean_score
  + c * sqrt(log(total_visits + 1) / (node_visits + 1))
  + novelty_bonus
  - risk_penalty
```

默认：

```text
c = 1.4
novelty_bonus in [0.0, 0.2]
risk_penalty in [0.0, 0.5]
```

建议：

- `mean_score` 使用 reward，reward 越大越好。
- 对 failed/compliance-violated 节点设置极低选择分。
- 对 `visits == 0` 的 seed 节点给探索优先级。
- 如果最佳分支连续 2-3 次无改进，提升 draft/fusion/ablate 的概率。

## 7. Scheduler

`mlevolve_lite/scheduler.py` 是主控入口，每轮流程：

1. load graph。
2. 如果 graph 不存在，初始化 graph 和 5 个 seed nodes。
3. select parent node。
4. choose operator。
5. create child node workspace。
6. call AIDE/LLM 生成代码快照或 patch。
7. run static check。
8. run shape check / preflight。
9. enqueue cheap probe GPU job。
10. evaluate metrics。
11. backup reward to ancestors。
12. update graph.json / leaderboard.json。
13. promote top nodes if thresholds pass。

失败处理：

- LLM 失败：node 标记 `failed`，记录 `failure_reason`。
- static check 失败：node 标记 `static_failed`。
- shape check 失败：node 标记 `shape_failed`。
- GPU job 超时/OOM：node 标记 `failed`，继续下一轮。
- evaluator 抛错：node 标记 `failed`，保存 traceback。

第一版 CLI 建议：

```bash
python -m mlevolve_lite.scheduler \
  --data-dir data/Task2 \
  --workspace workspace \
  --rounds 1 \
  --cheap-epochs 1 \
  --timeout-sec 1800
```

## 8. GPU Queue

`mlevolve_lite/gpu_queue.py` 管理单卡串行任务。

要求：

- 同一时间只允许一个 train/evaluate job 占用 GPU。
- LLM/code generation/static check 可以并行，但第一版可以先串行实现。
- 每个 GPU job 记录：

```text
job_id
node_id
cmd
status
start_time
end_time
runtime_sec
returncode
log_path
```

日志写入：

```text
workspace/gpu_queue.jsonl
workspace/nodes/<node_id>/logs/train.log
```

如果检测到已有 job `running` 但进程不存在，应标记为 `failed_or_interrupted`，不能永久卡住队列。

## 9. Evaluator

`mlevolve_lite/evaluator.py` 第一版至少实现：

### 9.1 Static check

检查 solution code：

- 是否存在入口脚本 `train.py`。
- 是否明显读取 test `nu` 或伪造 test 未来。
- 是否调用外部数值求解器。
- 是否写入 data 原始目录。
- 是否有危险 shell 操作。

static check 是启发式，不要求完美，但要把明显违规挡住。

### 9.2 Shape check

运行轻量命令或直接检查生成文件：

- `task2_pred.hdf5` 存在。
- 包含 dataset `tensor`。
- shape 为 `(1000, 200, 256)`。
- dtype 为 `float32` 或可安全转换为 `float32`。
- 全部 finite。
- `tensor[:, :10, :]` 与 `task2_test.h5/tensor` 一致，最大绝对误差不超过 `1e-3`。

### 9.3 First-10 check

官方提交要求 Task1/Task2 输出完整 200 步轨迹。前 10 步不是预测目标，必须与测试输入完全一致；第 10-199 步才是 190 个预测步。

Evaluator 至少检查：

- 预测文件没有错误地输出 `(1000, 190, 256)` 或 `(1000, 210, 256)`。
- `task2_pred.hdf5/tensor[:, :10, :]` 与测试输入 max error `<=1e-3`。
- 如果 solution 内部生成 extended rollout artifact，不能把 extra steps 误写进正式预测文件。

### 9.4 Validation metrics

用 `task2_val.h5`：

```text
input:        tensor[:, :10, :]
full_target:  tensor[:, :200, :]
future_190:   tensor[:, 10:200, :]
pred_full:    validation artifact with shape (N, 200, 256)
```

输出：

- `overall_mse`
- `short_mse`
- `rel_mse_seg1`, `rel_mse_seg2`, `rel_mse_seg3`
- `official_score_estimate`
- `long_stat_error`
- `per_nu_mse`
- `worst_nu_mse`
- `heldout_nu_mse` if split exists
- `nu_estimation_mae` if node reports estimated nu

Task1/Task2 官方分段评分只针对 190 个预测步，分段为 0-47、47-95、95-190；前两段使用 Rel-MSE 指数分，第三段取 Lorentzian 与 Fréchet 统计距离得分的较大者。Cheap probe 可先用 `overall_mse` 和分段 Rel-MSE 近似，但 promoted 节点应输出 `official_score_estimate`。

### 9.5 Compliance check

必须输出：

```text
uses_true_nu_at_test = false
compliance_pass = shape_pass and first_10_pass and not uses_true_nu_at_test
```

任何 compliance 不通过的节点，reward 应强制为负，且不能 promoted。

## 10. Reward 设计

Reward 越大越好。第一版可用：

```text
if not compliance_pass:
    reward = -1.0
elif official_score_estimate is not None:
    reward = official_score_estimate / 100.0
elif overall_mse is None:
    reward = -0.5
else:
    reward = -log10(overall_mse + 1e-12)
    reward -= 0.1 * log1p(runtime_sec / 600)
    reward -= 0.2 * log1p(worst_nu_mse / max(overall_mse, 1e-12))
```

原则：

- 若能估计官方分段得分，优先用官方分段得分归一化为 reward。
- 没有官方分段估计时，MSE 越低 reward 越高。
- worst-nu 过差要惩罚，避免只在平均指标上好。
- 运行时间过长轻度惩罚。
- 任何 shape/compliance 失败强惩罚。

Backup：

- child reward 更新 child。
- reward 同步回传到所有 ancestor。
- ancestor 的 `visits += 1`，`mean_score` 做增量平均，`best_score` 取最大。

## 11. Memory

`mlevolve_lite/memory.py` 负责为 LLM/AIDE 构造简短历史摘要。借鉴 `aideml.Journal.generate_summary()` 和 `MLEvolve.SearchNode.fetch_child_memory()`。

摘要应包含：

```text
Attempt id
Operator
Hypothesis
Changed components
Metrics
Failure reason
Key insight
```

不要把所有代码全文塞进 prompt。默认只给 top-k 成功节点、最近失败节点和与当前 parent 相关的兄弟节点。

第一版不做 FAISS memory，不接复杂向量库。

## 12. Prompt 约束

所有 prompts 必须强调：

- 数据已在 `data/Task2`。
- 只能修改当前 child node 的 `code/` 目录。
- 最终必须产生 `task2_pred.hdf5`，内部 `tensor` shape `(1000, 200, 256)`。
- `task2_pred.hdf5/tensor[:, :10, :]` 必须复制测试输入，只有 `tensor[:, 10:200, :]` 是预测。
- 不允许读取 test `nu`。
- 不允许数值求解器。
- 不允许额外数据。
- Task2 不允许加载任何公开预训练权重/checkpoint。
- 必须支持 cheap probe 参数。
- 最后一行建议打印 `Final Validation Score: <overall_mse>`。

solution code 的最低 CLI：

```bash
python train.py \
  --data-dir ../../../data/Task2 \
  --out-dir ../artifacts \
  --epochs 1 \
  --cheap-probe
```

如果用绝对路径，必须来自 scheduler 注入参数，不要在 solution code 中硬编码个人临时目录。

## 13. Seed Nodes 设计

初始化 5 个 seed nodes 时，每个节点创建独立 workspace 和 hypothesis。

### 13.1 multi_nu_fno_baseline

假设：周期一维 PDE 适合 spectral operator；不显式用 test `nu`，只从 10 步上下文隐式推断动力学。

推荐代码方向：

- 1D FNO 或 spectral conv。
- 输入 10 帧作为 channels。
- 输出单步或多步。
- cheap probe 只训练少量样本/epoch。

### 13.2 nu_estimator_concat

假设：前 10 步包含 viscosity 信息，可先估计 `nu` proxy，再 concat 到模型输入。

注意：推理时只能估计 test `nu`，不能读取真实 test `nu`。

### 13.3 nu_estimator_film

假设：估计出的 `nu` proxy 用 FiLM/conditional normalization 调制动力学模型，比直接 concat 更稳定。

### 13.4 mixture_of_nu_experts

假设：不同 `nu` 区间动力学差异明显，用 gating network 从前 10 步选择/混合专家。

### 13.5 conditional_fno_marginalized_nu

假设：test `nu` 不可见，可对多个候选 `nu` 条件模型输出做边缘化或加权集成。

注意：候选 `nu` 来自训练分布和输入估计，不是 test 真值。

## 14. Cheap Probe 标准

Cheap probe 用于检查链路和方向，不追求最终分数。

建议默认：

```text
epochs: 1-3
train trajectories: 64-256
val trajectories: 16-100
rollout steps for quick metric: 50 and 200
timeout: 15-30 minutes
```

Cheap probe 必须生成：

```text
workspace/nodes/<node_id>/metrics.json
workspace/nodes/<node_id>/logs/train.log
```

可选生成完整 `task2_pred.hdf5`。如果 cheap probe 为节省时间只生成验证预测，shape check 可在 solution 的 smoke mode 中用 dummy/test subset 验证；但 promoted 前必须生成完整测试预测。

## 15. Promotion 规则

节点可 promoted 的最低要求：

- `compliance_pass == true`
- `shape_pass == true`
- `overall_mse` 有效
- `runtime_sec` 未超预算
- `uses_true_nu_at_test == false`
- reward 进入 leaderboard top-k 或优于当前 best

`promoted.json` 记录：

```text
node_id
overall_mse
worst_nu_mse
reward
code_dir
checkpoint
prediction_path
promoted_at
reason
```

Promotion 不代表最终提交，只代表值得进入更长训练或人工审查。

## 16. 验收标准

第一版实现必须满足：

1. 能创建 `workspace/graph.json`。
2. 能创建 5 个 seed nodes。
3. 能选 parent node。
4. 能生成 child node workspace。
5. 能对 child code 做 static check 和 shape check。
6. 能串行跑一个 cheap probe。
7. 能生成 `metrics.json`。
8. 能更新 `graph.json` 和 `leaderboard.json`。
9. 能根据 reward promote top node。
10. 所有失败都不能导致整个搜索中断，只能标记对应 node failed。

## 17. 明确不要做

第一版不要做：

- 不要引入复杂分布式系统。
- 不要接 Ray。
- 不要接 Optuna。
- 不要做完整 FAISS memory。
- 不要同时跑多个 GPU training。
- 不要让 LLM 改主控系统。
- 不要把第三方 `MLEvolve` 或 `aideml` 大规模拷贝进本项目。

## 18. 推荐实现顺序

按以下顺序落地：

1. `node_schema.py`：定义 Node/Metrics。
2. `graph_db.py`：实现 graph/leaderboard/promoted 的读写。
3. `operators.py`：先实现 seed node 创建和 child workspace 创建。
4. `selector.py`：实现 UCB parent selection。
5. `evaluator.py`：实现 static/shape/compliance/metrics。
6. `gpu_queue.py`：实现单卡串行命令运行和日志记录。
7. `scheduler.py`：串起一轮流程。
8. `prompts/`：补齐 draft/improve/debug/ablate/fusion prompt。
9. 跑 `--rounds 1 --cheap-epochs 1` 验收闭环。

## 19. 代码质量要求

- 主控代码必须是普通 Python 包，不依赖 notebook。
- 路径统一用 `pathlib.Path`。
- JSON 写入要可恢复，避免半写文件。
- 每个模块函数保持小而明确，避免把所有逻辑塞进 scheduler。
- 对外命令参数要有默认值，但不能硬编码只适用于当前机器的绝对路径。
- 异常必须落盘到 node log 或 events，不允许静默吞掉。
- 测试或 smoke command 要能在 CPU 上至少跑过 static/shape 逻辑。
