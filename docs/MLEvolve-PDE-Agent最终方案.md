# MLEvolve-PDE Agent 最终方案

## 1. 结论

本项目应将 **MLEvolve 作为科研智能体搜索框架 baseline**，而不是把它当作神经算子模型 baseline。

最终路线是：

```text
保留 AI4S-PDE-CNS 现有 PDE 领域能力
    + 引入 MLEvolve 的 MCTS / SearchNode / 多分支搜索思想
    + 用 PDE 专用 executor 和 result parser 替换 Kaggle submission 逻辑
    + 将最优节点桥接回现有 AI4S 官方提交校验与打包流程
```

对应关系：

| 层级 | 采用对象 | 角色 |
|---|---|---|
| 科学模型 baseline | FNO ensemble、DeepONetLite、UNet1D、Residual Refiner | 被搜索和优化的 PDE 模型 |
| 搜索框架 baseline | MLEvolve MCTS / MCGS 架构 | 自动提出方案、执行实验、解析指标、选择分支 |
| 比赛提交层 | `agent.submission`、`Task1FNOWorkflow`、`code/` | 生成并校验官方 `pred.zip` |

第一版不要重写整个项目，也不要另建 `AI4S-PDE-Agent/`。应在当前仓库内新增一个隔离模块，例如 `mclevolve/` 或 `agent/mcts/`，通过 adapter 接入现有 PDE 工具链。

## 2. 背景与问题

当前仓库已有一个可审计的 PDE Agent 控制层，核心能力包括：

- `agent.run_task1_autonomous_experiment`：AIDE-style 线性自主实验 loop。
- `Task1FNOWorkflow`：Task 1 FNO ensemble validation 和 submission bundle。
- `WeightedEnsembleSearch`：Task 1 ensemble weight search。
- `compute_task1_metrics`：本地 MSE、forecast MSE、long-horizon MSE、`competition_score_proxy`。
- `agent.validate_submission` / `agent.pack_submission`：官方提交结构校验和打包。

现有 loop 的限制是：它更像 `observe -> plan -> act -> review` 的线性迭代，每轮只推进一个动作。它缺少：

- 多分支并行探索。
- 基于 UCT 的系统性节点选择。
- 后期从 Top-K 候选中 exploitation。
- 分支停滞后的 evolution / fusion。
- 搜索树级别的 reward backpropagation。

MLEvolve 的价值正好在这里。它公开了较完整的 MLE 搜索架构：`SearchNode`、`Journal`、`node_selection`、`evaluation.backpropagate`、`solution_manager`、`draft/improve/debug/evolution/fusion` 多阶段 agent，以及 subprocess 执行框架。

## 3. 设计原则

1. **搜索核心可借鉴，提交逻辑必须替换**

   MLEvolve 原生面向 Kaggle / MLE-bench，默认产物是 `submission/submission.csv`，还有 grading server 和 CSV content quality check。AI4S PDE 需要的是 `task{N}_pred.hdf5`、`task{N}_time.csv`、`task{N}_logs.log`、`submission.json`、`code/` 和 `pred.zip`。因此不能只靠 dummy `submission.csv` 跑通。

2. **先 adapter，后迁移**

   第一阶段不要全量复制 MLEvolve 并深改。优先用 thin adapter 包住 MLEvolve 的搜索核心，明确哪些文件是 vendored，哪些文件是本项目 PDE 专用代码。

3. **PDE 指标由本地确定，不能依赖 LLM 猜测**

   MLEvolve 里会用 LLM 判断 metric direction 和从 stdout 解析 metric。PDE 模式下应硬编码指标方向，例如 `mse` minimize 或 `competition_score_proxy` maximize，并用正则或 JSON 读取指标，LLM 只负责总结。

4. **所有实验结果必须能回到现有 `runs/` 结构**

   搜索节点可以有自己的 workspace，但最终候选必须写入 `runs/<study>/nodes/<node_id>/`，并保留 `metrics.json`、`run_result.json`、候选代码、stdout/stderr、LLM 日志和可复现命令。

5. **官方提交桥接必须是第一阶段验收项**

   只跑出 MLEvolve `best_solution.py` 不够。第一版至少要能从 best node 调用现有 `Task1FNOWorkflow.run_test_submission(...)` 或等价流程，生成可由 `agent.validate_submission` 校验的 `pred.zip`。

## 4. 目标架构

```text
AI4S-PDE-CNS
├── agent/                          # 已有 PDE 领域层，保留
│   ├── pde_workflow.py              # Task1FNOWorkflow
│   ├── pde_metrics.py               # compute_task1_metrics
│   ├── pde_search.py                # WeightedEnsembleSearch
│   ├── pde_executor.py              # ControlledExperimentExecutor
│   ├── submission.py                # 官方提交校验/打包
│   └── run_task1_autonomous_experiment.py
│
├── mclevolve/                       # 新增，MLEvolve-inspired 搜索层
│   ├── engine/
│   │   ├── search_node.py           # vendored/adapted
│   │   ├── node_selection.py        # vendored/adapted
│   │   ├── evaluation.py            # vendored/adapted
│   │   ├── conditions.py            # vendored/adapted
│   │   ├── solution_manager.py      # adapted for PDE artifacts
│   │   ├── pde_agent_search.py      # new
│   │   └── pde_executor.py          # new
│   ├── agents/
│   │   ├── pde_draft_agent.py       # new
│   │   ├── pde_improve_agent.py     # new
│   │   ├── pde_debug_agent.py       # new
│   │   ├── pde_result_parse_agent.py# new
│   │   └── pde_code_review_agent.py # new
│   ├── llm/                         # reuse project LLM adapter or vendored thin wrapper
│   └── utils/
│
├── configs/
│   └── pde_mcts.yaml                # 新增，PDE MCTS 配置
├── scripts/
│   └── collect_mcts_results.py      # 新增，journal -> report
└── agent/
    └── run_task1_mcts_experiment.py # 新增 CLI 入口，优先于根目录 run_pde.py
```

入口建议放在 `agent/run_task1_mcts_experiment.py`，而不是仓库根目录 `run_pde.py`。这样和现有 `python -m agent.run_task1_*` 命令保持一致。

## 5. 模块迁移边界

### 5.1 可以复用或轻改的 MLEvolve 模块

| MLEvolve 模块 | 复用方式 | 注意事项 |
|---|---|---|
| `engine/search_node.py` | 复制到 `mclevolve/engine/search_node.py` | 保留 `SearchNode` / `Journal`，增加 PDE artifact 字段时要保持 JSON 可序列化 |
| `engine/node_selection.py` | 复制或改 import 后复用 | UCT、Top-K、soft switch 是领域无关逻辑 |
| `engine/evaluation.py` | 复制后小改 reward | reward 需要适配 minimize / maximize 和 PDE 节点失败语义 |
| `engine/conditions.py` | 复制后小改阈值 | 停滞检测可保留 |
| `utils/metric.py` | 复制或直接实现轻量版 | `MetricValue(maximize=False)` 对 PDE MSE 很有用 |
| `utils/serialize.py` | 复制 | 用于 journal 持久化 |
| `agents/coder/` | 可选复用 | diff patch 模式有价值，但要限制写入范围 |
| `agents/planner/` | 可选复用 | 如果复杂度过高，第一版可先用本项目现有 `ExperimentPlanner` |

### 5.2 不建议原样复用的 MLEvolve 模块

| MLEvolve 模块 | 原因 | 替代方案 |
|---|---|---|
| `engine/executor.py` | Linux 假设强，使用 `os.sched_getaffinity`；默认 CSV submission 隔离 | 写 `PDEInterpreter`，保留 subprocess + timeout，增加 Windows fallback |
| `agents/result_parse_agent.py` | 依赖 LLM 解析 metric，且强绑定 `submission.csv` | 写 `pde_result_parse_agent.py`，优先读取 `metrics.json` / stdout JSON |
| `engine/validation/*` | Kaggle grading server / CSV format check | 使用 `agent.submission.validate_submission` 和 PDE validation metrics |
| `utils/submission_fusion_utils.py` | CSV ensemble 工具 | 不用于 HDF5 PDE 预测 |
| `coldstart/*` | Kaggle task/model guidance | 第一版不用，后续可替换成 PDE model guidance |

### 5.3 直接保留的本项目 PDE 模块

| 当前模块 | 用途 |
|---|---|
| `agent/pde_workflow.py` | Task 1 FNO validation 和 official submission bundle |
| `agent/pde_metrics.py` | PDE 指标计算 |
| `agent/pde_search.py` | ensemble weight search |
| `agent/pde_executor.py` | 受控执行 action 的已有基础 |
| `agent/pde_journal.py` | 线性实验 journal，可作为 MCTS journal 的补充输出 |
| `agent/submission.py` | 官方提交校验与打包 |
| `code/` | 官方提交代码目录，MCTS 只能通过受控 patch 修改 |

## 6. PDE MCTS 节点语义

每个 `SearchNode` 表示一次可执行 PDE 实验方案。

节点核心字段：

| 字段 | 含义 |
|---|---|
| `plan` | LLM 提出的实验假设和修改说明 |
| `code` | 本轮执行代码或 patch 后的方案代码 |
| `stage` | `draft`、`improve`、`debug`、`evolution`、`fusion` |
| `metric` | 主优化指标，例如 `competition_score_proxy` maximize 或 `mse` minimize |
| `metrics` | 完整 PDE 指标字典 |
| `run_dir` | `runs/<study>/nodes/<node_id>` |
| `artifact_paths` | prediction、checkpoint、metrics、logs、zip 等路径 |
| `is_buggy` | 执行失败、指标缺失、shape 错误、NaN、OOM 等 |
| `is_valid` | PDE validation 或 official submission validation 是否通过 |

第一版推荐主指标使用：

```text
metric = competition_score_proxy
maximize = true
```

原因：当前仓库的实验比较和 leaderboard proxy 已经以 `competition_score_proxy` 为核心。若某些节点只做 MSE validation，也可以把 `mse` 作为 secondary metric，但搜索主方向先保持一致。

## 7. PDE Agent 动作设计

### 7.1 `pde_draft_agent`

职责：生成初始可执行方案。

第一版不要让它自由发明整个训练框架。可用动作应收敛在：

- 调用 `WeightedEnsembleSearch` 做权重搜索。
- 调用已有 `run_task1_baseline_zoo` 的轻量配置。
- 基于 `code/` 的受控 patch 增加小改动。
- 生成 `Task1FNOWorkflow` 可执行配置。

Prompt 必须包含：

- 输入 shape：`(N, 10, 256)`。
- 输出 shape：`(N, 200, 256)`。
- 初始 10 帧必须和官方输入一致。
- Task 1 可用 PDEBench FNO checkpoint。
- Task 2 禁用 Task 1 checkpoint 和数据。
- 禁止数值 solver 生成额外训练数据。
- 输出必须写 `metrics.json` 和标准 stdout marker。

### 7.2 `pde_improve_agent`

职责：基于成功父节点提出一个明确改进。

推荐改进方向：

- ensemble 权重微调。
- `nu0.01` / `nu0.1` 权重局部搜索。
- short fine-tune 超参：lr、steps、batch size、loss window。
- residual refiner / DeepONetLite gating。
- long-horizon loss window 调整。

要求每次只做 1 到 2 个相关变化，便于 attribution。

### 7.3 `pde_debug_agent`

职责：修复失败节点。

常见错误知识：

- HDF5 dataset key 错误。
- prediction shape 不是 `(N, 200, 256)`。
- 前 10 帧没有覆盖为 initial condition。
- CUDA OOM。
- NaN / inf loss。
- Windows path 和 PowerShell 路径问题。
- `code/` 被错误变成 Python package。

### 7.4 `pde_evolution_agent`

职责：分支停滞时做更大尺度变化。

允许：

- 从 ensemble search 切到 short fine-tune。
- 从 FNO-only 切到 baseline zoo。
- 从单模型切到 refiner / gating。

不允许：

- 未经验证直接改最终提交代码。
- 在 Task 2 使用 Task 1 checkpoint。
- 引入数值 solver 生成训练标签。

### 7.5 `pde_fusion_agent`

职责：合并不同成功分支的经验。

第一版可以只做“计划融合”，不做复杂代码自动合并。例如：

- 分支 A 的最佳 ensemble weights。
- 分支 B 的 fine-tuned `nu0.1` checkpoint。
- 分支 C 的 long-horizon loss window。

融合后生成一个新的受控实验配置，而不是让 LLM 拼接大段代码。

## 8. 执行与指标解析

### 8.1 PDEInterpreter

`PDEInterpreter` 保留 MLEvolve subprocess 隔离思想，但必须适配本项目：

- 写入 `runs/<study>/workspace/runfile_<slot>.py`。
- 每个 node 独立 `run_dir`。
- 支持 timeout。
- 捕获 stdout / stderr / return code。
- Windows 下不使用 `os.sched_getaffinity`；Linux 下可选 CPU affinity。
- GPU 并行第一版固定为 `parallel_search_num: 1`。

返回对象兼容 MLEvolve `ExecutionResult`：

```python
ExecutionResult(
    term_out=[stdout, stderr],
    exec_time=seconds,
    exc_type=None | "RuntimeError" | "TimeoutError",
    exc_info={...},
    exc_stack=[...],
)
```

### 8.2 指标输出协议

每个节点必须输出：

```text
PDE_METRICS_JSON: {"mse": 0.0016, "forecast_mse": 0.0017, "long_horizon_mse": 0.0015, "competition_score_proxy": 58.1}
PDE_PRIMARY_METRIC: 58.1
PDE_PRIMARY_METRIC_NAME: competition_score_proxy
PDE_PRIMARY_METRIC_MAXIMIZE: true
```

同时写入：

```text
runs/<study>/nodes/<node_id>/metrics.json
runs/<study>/nodes/<node_id>/run_result.json
```

`pde_result_parse_agent` 的优先级：

1. 读取 node run_dir 的 `metrics.json`。
2. 正则解析 stdout 中的 `PDE_METRICS_JSON`。
3. 正则解析 `PDE_PRIMARY_METRIC`。
4. LLM 只用于生成 `analysis` 摘要，不作为 metric 的唯一来源。

### 8.3 验证规则

Validation 节点必须检查：

- 预测 HDF5 是否存在。
- 预测 shape 是否为 `(N, 200, 256)`。
- 指标字典是否包含主指标。
- 指标是否为 finite number。

Submission 节点必须额外调用：

```powershell
python -m agent.validate_submission --path runs\<node_run_dir>
```

并确保输出：

```text
runs/<node_run_dir>/pred.zip
```

## 9. 配置建议

第一阶段配置应保守：

```yaml
study_name: task1-mcts-smoke
task: task1
metric: competition_score_proxy
maximize: true

agent:
  steps: 8
  initial_drafts: 2
  time_limit: 7200
  seed: 42
  use_global_memory: false
  use_diff_mode: true

  search:
    parallel_search_num: 1
    num_drafts: 2
    num_improves: 2
    num_bugs: 1
    max_debug_depth: 3
    top_candidates_size: 5
    branch_stagnation_threshold: 2
    topk_stagnation_threshold: 4

exec:
  timeout: 1800

pde:
  project_root: "."
  code_dir: "code"
  run_root: "runs"
  validation_target: "data/data_and_sample_submission/data_and_sample_submission/train_val_test_init/task1_val.hdf5"
  test_input: "data/data_and_sample_submission/data_and_sample_submission/train_val_test_init/task1_test.hdf5"
  methodology_path: "docs/methodology.pdf"
```

跑通后再扩大：

- `steps: 20`
- `time_limit: 14400`
- `parallel_search_num: 1`
- `use_global_memory: true`

只有确认单节点训练显存稳定后，再考虑 `parallel_search_num: 2`。

## 10. 实施阶段

### Phase 0：方案落地前检查

目标：确认 MLEvolve 引入方式和许可证风险。

任务：

- 确认 `InternScience/MLEvolve` 当前许可证状态。若无明确 license，不把大段源码复制进最终开源提交；优先使用 `third_party/MLEvolve` 作为 ignored reference。
- 在 `docs/baseline_adaptation.md` 记录 MLEvolve commit hash 和用途。
- 明确第一版只支持 Task 1 validation，不触碰 Task 2。

验收：

- 文档记录清楚 vendored/reference 边界。

### Phase 1：最小 MCTS 骨架

目标：在不运行真实训练的情况下，让 MCTS 搜索树跑通。

任务：

- 新增 `mclevolve/engine/search_node.py`、`node_selection.py`、`evaluation.py` 的最小可用版。
- 新增 `agent/run_task1_mcts_experiment.py` CLI。
- 新增 fake `PDEInterpreter`，返回固定 metrics。
- 新增 tests 验证：
  - `SearchNode` 可序列化。
  - UCT selection 能选择可扩展节点。
  - result parser 能从 `metrics.json` 得到 metric。

验收：

```powershell
pytest tests/test_task1_mcts_smoke.py -q
python -m agent.run_task1_mcts_experiment --config configs/task1_mcts_mock.yaml --max-steps 3
```

### Phase 2：接入真实 Task 1 validation

目标：MCTS 节点能调用现有 `Task1FNOWorkflow.run_validation`。

任务：

- `PDEInterpreter` 支持执行受控 action，而不是裸跑任意自由代码。
- `pde_draft_agent` 第一版只允许生成结构化 action JSON：
  - `weight_search`
  - `baseline_validate`
  - `baseline_ensemble`
  - `finetune`
  - `code_patch`
- 复用 `ControlledExperimentExecutor` 执行 action。
- `pde_result_parse_agent` 读取 `RunResult` 和 `metrics.json`。

验收：

```powershell
python -m agent.run_task1_mcts_experiment --config configs/task1_mcts_mock.yaml --max-steps 3 --bootstrap-weight-search
```

journal 中至少有 1 个成功节点，且 `best_solution` 或 `best_node` 指向一个真实 validation run。

### Phase 3：启用 improve/debug/evolution

目标：从单次 action 扩展为多分支搜索。

任务：

- 实现 `pde_improve_agent`，基于父节点 metrics 生成结构化改进 action。
- 实现 `pde_debug_agent`，基于错误信息生成修复 action。
- 实现 branch registry 和 top candidates。
- 引入 Top-K exploitation。

验收：

- 运行 8 到 10 steps。
- journal 至少包含 2 个 branch。
- 至少有一次 improve 节点基于成功父节点生成。
- top candidates 按 `competition_score_proxy` 排序。

### Phase 4：官方提交桥接

目标：最优节点可以生成官方可校验 submission bundle。

任务：

- 新增 `submit_best` MCTS action。
- 调用 `Task1FNOWorkflow.run_test_submission`。
- 调用 `agent.validate_submission`。
- 保存 `pred.zip` 路径到 best node artifacts。

验收：

```powershell
python -m agent.validate_submission --path runs\<study>\nodes\<best_submission_node>
```

通过校验，并生成：

```text
runs/<study>/nodes/<best_submission_node>/pred.zip
```

### Phase 5：报告与结果汇总

目标：将 MCTS journal 转换为可读科研演进报告。

任务：

- 新增 `scripts/collect_mcts_results.py`。
- 输出 `runs/<study>/mcts_report.md`。
- 汇总：
  - 每轮 plan。
  - action 类型。
  - metrics。
  - 父子节点关系。
  - 是否提升。
  - 失败原因。
  - 最优方案复现命令。

验收：

- `mcts_report.md` 能解释为什么 best node 胜出。
- `docs/results/task1_experiment_results.md` 可收录 MCTS 最优 run。

## 11. 测试策略

新增测试建议：

| 测试文件 | 覆盖内容 |
|---|---|
| `tests/test_mcts_search_node.py` | SearchNode、Journal、序列化 |
| `tests/test_mcts_node_selection.py` | UCT、Top-K、soft switch |
| `tests/test_task1_mcts_result_parse.py` | metrics.json / stdout marker 解析 |
| `tests/test_task1_mcts_executor.py` | fake action 执行、timeout、失败节点 |
| `tests/test_task1_mcts_submission_bridge.py` | best node 调用 submission validation |

最低验收命令：

```powershell
pytest tests/test_mcts_search_node.py tests/test_task1_mcts_result_parse.py -q
pytest tests/test_pde_autonomous.py tests/test_task1_submission_bundle.py -q
python -m agent.run_task1_mcts_experiment --config configs/task1_mcts_mock.yaml --max-steps 3
```

## 12. 风险与修正

| 风险 | 影响 | 修正 |
|---|---|---|
| MLEvolve 许可证不明确 | 复制源码可能影响后续开源/提交 | 第一版将 MLEvolve 作为 `third_party` reference，复制前确认 license；必要时重写最小 MCTS 核心 |
| 全量复制导致维护成本高 | import、配置、日志体系冲突 | 只迁移 SearchNode / selection / evaluation 最小子集 |
| Kaggle CSV 逻辑污染 PDE 流程 | dummy submission 被误判或误优化 | PDE 模式禁用 grading server 和 CSV quality check |
| LLM metric 解析不稳定 | 错误更新 best node | metric 只从 `metrics.json` / stdout marker 解析 |
| Windows 不兼容 `os.sched_getaffinity` | executor 崩溃 | Windows fallback，不做 CPU affinity |
| 自由代码生成破坏 `code/` | 官方提交不可追溯 | 第一版只允许结构化 action；`code_patch` 必须有 validation command |
| 训练时间过长 | 12 小时预算失控 | 第一版 validation-only，单节点 timeout，记录 train/inference time |
| Task 2 合规风险 | 复用 Task 1 checkpoint 导致违规 | Task 2 另起配置，强制禁用 Task 1 data/checkpoint |

## 13. 对 Claude 方案的修订意见

Claude 方案中正确的部分：

- 用 MLEvolve 替换线性 loop 的搜索能力是合理方向。
- 直接复用 `SearchNode`、`node_selection`、`evaluation` 的思路正确。
- 保留当前 PDE 领域代码是正确决策。
- `mclevolve/` 命名空间隔离是合理的。

需要修正的部分：

1. **不要一开始复制过多源码**

   `agents/coder/`、`planner/`、`llm/`、`config/` 全量复制会带来大量 import 和 provider 适配成本。第一版应优先复用本项目已有 LLM/config，必要时再引入 MLEvolve 的 coder。

2. **不要让 executor 执行任意 Python**

   AI4S 需要可审计日志和代码追溯。第一版 executor 应执行结构化 action，限制写入范围。自由代码生成可作为后续能力。

3. **指标方向不是固定 MSE minimize**

   本项目当前 best selection 多使用 `competition_score_proxy` maximize。最终配置应支持二者，但第一版建议用 `competition_score_proxy` maximize 对齐现有实验比较。

4. **官方提交验证必须前置**

   Claude 方案把 `agent.validate_submission` 放在最后检查项，但它应该成为 Phase 4 的核心交付，避免 MCTS 搜索出来的 best solution 无法提交。

5. **入口建议放进 `agent/`**

   使用 `python -m agent.run_task1_mcts_experiment` 比根目录 `run_pde.py` 更符合当前项目风格。

## 14. 推荐执行顺序

```text
Day 1:
  写最小 MCTS 数据结构和 mock runner。
  不接 LLM，不跑 GPU。

Day 2:
  接入 Task1FNOWorkflow validation。
  使用 bootstrap weight_search 生成第一个真实成功节点。

Day 3:
  接入 pde_draft / pde_improve 的结构化 action 生成。
  跑 3 到 5 steps smoke。

Day 4:
  接入 Top-K、best_solution、top_candidates 和 mcts_report。
  跑 8 到 10 steps validation-only。

Day 5:
  接入 submit_best。
  生成可通过 validate_submission 的 `pred.zip`。
```

## 15. 第一版完成标准

第一版算完成，当且仅当以下全部满足：

- `python -m agent.run_task1_mcts_experiment --config configs/task1_mcts_mock.yaml --max-steps 3` 可跑通。
- 至少一个 MCTS 节点调用真实 Task 1 validation。
- journal 中有父子节点关系、stage、metric、artifact paths。
- best node 由指标自动选出，不由人工指定。
- `mcts_report.md` 可读。
- `submit_best` 节点能生成官方结构，并通过 `agent.validate_submission`。
- 现有测试不被破坏：

```powershell
pytest -q
```

## 16. 一句话方案

最终方案不是“把 MLEvolve 改造成 PDE 项目”，而是：

```text
把 MLEvolve 的 MCTS 搜索骨架抽出来，
接到 AI4S-PDE-CNS 已有的 FNO/validation/submission 工具链上，
用结构化 action 保证合规和可审计，
用 SearchNode/UCT/Top-K/evolution/fusion 提升实验搜索能力。
```
