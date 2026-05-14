# MLEvolve-lite for AI4S PDE

## 结论

使用 MLEvolve 作为下一阶段 Agent baseline 是合理的，但本项目不应直接复制或深改完整 MLEvolve。

最小、优雅的实现方式是：

```text
保留现有 PDE/FNO 工具链
+ 引入 MLEvolve 的多分支搜索思想
+ 用 PDE 专用 action 替代 Kaggle-style 自由写代码
+ 用 pred.zip / HDF5 / 平台分数作为闭环反馈
```

## 为什么不用原版 MLEvolve 直接跑

| 原版 MLEvolve | 本项目需求 |
|---|---|
| 面向 Kaggle / MLE-bench | 面向 AI4S PDE Burgers 预测 |
| 默认生成 `submission.csv` | 必须生成 `pred.zip` |
| 代码级 fusion | 更需要预测结果级 / PDE-aware fusion |
| 依赖 Linux 多进程和 grading server | 服务器可用，但本地 Windows 不稳定 |
| LLM 可自由写完整方案 | 本项目应限制为受控实验动作 |

因此，MLEvolve 应作为搜索架构参考，而不是直接替换现有项目。

## 当前最该复用的能力

| 能力 | 是否复用 | 说明 |
|---|---|---|
| 多分支 SearchNode | 是 | 每个节点是一组实验配置或预测方案 |
| UCT / Top-K 选择 | 是 | 在探索新方案和利用好方案之间切换 |
| evolution | 是 | 分支停滞后，基于历史轨迹换策略 |
| global memory | 是 | 记录哪些实验有效、哪些失败 |
| fusion | 改造后使用 | 不做自由代码融合，改成 PDE 预测融合 |
| Kaggle CSV validation | 否 | 使用本项目 `agent.validate_submission` |

## 最小改动原则

1. 不移动 `code/`，避免破坏提交代码路径。
2. 不把 MLEvolve 全量复制进 `agent/`。
3. 不让 LLM 自由改写核心 FNO 推理代码。
4. 优先复用已有模块：
   - `agent/pde_executor.py`
   - `agent/pde_journal.py`
   - `agent/pde_search.py`
   - `agent/pde_workflow.py`
   - `agent/submission.py`
5. 所有实验输出仍放在 `runs/<study_name>/`。
6. 每次有效实验都必须记录：
   - action
   - params
   - validation metric
   - inference time
   - artifact path
   - platform score, if available

## 推荐架构

```text
Task1 MLEvolve-lite Runner
        |
        v
PDE Search Policy
  - draft
  - improve
  - debug
  - evolution
  - pde_fusion
        |
        v
ControlledExperimentExecutor
  - weight_search
  - finetune
  - baseline_train
  - baseline_ensemble
  - submit_best
        |
        v
PDE Evaluator
  - mse
  - competition_score_proxy
  - train_time
  - inference_time
  - submission validity
        |
        v
Journal / Memory / Top-K
```

## PDE 专用 action 空间

第一版只允许这些动作：

| action | 用途 |
|---|---|
| `weight_search` | 搜索 FNO / DeepONet / tail blend 权重 |
| `finetune` | 微调已有 FNO checkpoint |
| `baseline_train` | 训练轻量 DeepONet / refiner |
| `baseline_ensemble` | 组合已有 validation predictions |
| `submit_best` | 生成 `pred.zip` 并校验 |
| `import_platform_score` | 手动写入平台真实分数 |

不建议第一版开放自由 `code_patch`。如果必须开放，只允许改 `code/` 下明确文件，并强制跑提交校验。

## Fusion 的正确改法

原版 MLEvolve fusion 是：

```text
读取两个成功代码分支
让 LLM 融合代码思路
生成新代码
```

本项目应改为：

```text
读取多个成功 prediction / checkpoint
分析分段误差
选择受控融合算子
生成新的 prediction
```

推荐的 PDE fusion：

| fusion | 说明 |
|---|---|
| `convex_weight_search` | 全局权重搜索 |
| `segmentwise_blend` | 前段、中段、长时段分别加权 |
| `temporal_tail_blend` | 后 120 步引入 tail 专家 |
| `cluster_gating` | 按初始条件特征选择专家 |

## 最小文件改动建议

优先控制在 2-3 个代码文件内：

```text
agent/pde_mcts.py
  SearchNode、UCT、Top-K、branch stagnation、PDE fusion decision

agent/run_task1_mcts_experiment.py
  CLI 入口，复用 ControlledExperimentExecutor

configs/task1_mcts_mock.yaml
  smoke test 配置
```

如果当前已有同名文件，应优先扩展，不再新增平行实现。

## 第一阶段验收标准

最小闭环跑通即可：

```text
python -m agent.run_task1_mcts_experiment \
  --config configs/task1_mcts_mock.yaml \
  --max-steps 3
```

验收条件：

- 生成至少 3 个 journal node。
- 至少一个节点成功产出 validation metric。
- best node 可被选出。
- `submit_best` 能生成 `runs/<study>/pred.zip`。
- `agent.validate_submission` 通过。

## 第二阶段目标

当第一阶段稳定后，再加：

1. Top-K exploitation。
2. branch stagnation 检测。
3. PDE-aware fusion。
4. platform score import。
5. 服务器长跑配置。

## 一句话

MLEvolve 的价值不是直接带来更强模型，而是把刷榜变成一个有记忆、有分支、有选择策略的长期搜索过程。

对本项目来说，最小优雅路线是：

```text
用 MLEvolve 的搜索脑
接我们已有的 PDE 手脚
用平台分数闭环
```
