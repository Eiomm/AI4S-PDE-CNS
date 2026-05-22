# Task1 Agent Toolbox Design

本文档定义 Agent runner 可以暴露给 GPT-5.5 的工具边界。当前阶段以“简单可行、分数优先”为准。web search 已接入 executor；Optuna 是 Agent 生成代码可直接 `import optuna` 使用的 Python 库，不由 executor 代替执行；Ray Tune 和 W&B 先不用，只记录计划。工具箱不是唯一知识来源，Agent 可以结合自身内部知识提出结构、损失和训练策略；所有候选必须回到本地 validation。

## 工具优先级

1. 官方 checkpoint baseline
   - 目标：稳定生成可校验的 `task1_pred.hdf5`。
   - 必跑：官方 FNO checkpoint、官方 Unet-PF checkpoint。
   - 输出：两个 baseline 的 validation 分数、metadata、checkpoint hash、memory 记录。
   - 原因：后续 fine-tune / ensemble 必须有明确对照组。

2. 轻量验证工具
   - 目标：检查 shape、finite、前 10 帧一致、validation metric。
   - 输出：`metrics.json`、`metadata.json`、memory export。

3. Checkpoint fine-tune 与深度模型调参
   - 当前可用：Optuna，作为 Agent 生成代码中的 Python 库。
   - 暂不执行：Ray Tune。
   - 当前主方向：针对官方 checkpoint 做微调。
   - 训练数据：`data/pdebench_burgers/raw/1D_Burgers_Sols_Nu0.001.hdf5`。
   - 搜索对象：学习率、冻结/解冻策略、rollout loss 权重、teacher forcing 比例、ensemble 权重、后处理参数。
   - 执行策略：先 cheap probe，再对晋级候选做完整 validation。
   - 重要边界：Optuna 应由 Agent 在 `agent_workspace/code` 里直接 `import optuna` 使用；executor 不代替 Agent 写 objective 或执行 study。

4. 实验追踪
   - 候选：W&B。
   - 当前策略：先不用，只在 Agent 计划里记录需求。
   - 短预算实验可以只用本地 `runs/` 和 compact memory。

5. 数据检查
   - 候选：YData Profiling、Cleanlab。
   - 使用条件：发现数据异常、标签噪声、切分错误或需要做残差数据分析。
   - 对 PDE rollout 不是第一优先级。

6. 传统 AutoML baseline
   - 候选：AutoGluon、FLAML。
   - 使用条件：构造低维统计特征、残差校正或 sanity baseline。
   - 不允许替代神经网络模型生成最终 PDE 预测。

7. PDE / 神经算子生态
   - 候选：NeuralOperator、PDEBench、PhysicsNeMo。
   - 使用条件：参考网络结构、算子设计、训练接口、数据格式对齐。
   - 不允许调用数值求解器生成最终预测。

8. 知识检索
   - 候选：web search。
   - 使用条件：Agent 需要查找 FNO、Burgers、neural operator fine-tune、rollout loss、checkpoint adaptation 等公开资料来辅助决策。
   - 输出要求：检索结果只能作为设计依据，不能绕过本地 validation。

## 多轮探索设计

不采用“并发跑一堆完整实验”的策略。完整 fine-tune 成本高，而且很多候选会在早期就暴露失败。默认采用分阶段漏斗：

```text
baseline -> cheap_probe -> search -> promotion -> submission
```

- `baseline`：完整 replay 官方 FNO 和官方 Unet-PF，建立两个分数锚点。
- `cheap_probe`：小样本、短 epoch、短 rollout、少 trial，快速淘汰明显差的方向。
- `search`：当前用 Optuna 做预算受控搜索；Ray Tune 暂不执行。
- `promotion`：只把少数候选升级到完整 validation。
- `submission`：只对当前 best candidate 生成 test prediction 和 submission。

并发规则：

- 可以并发 cheap probe、ensemble 权重扫描、后处理参数扫描。
- 不默认并发多个完整 fine-tune。
- 每轮 Agent 必须给出 `parallel_trials`、预算、早停条件和晋级条件。
- executor 后续只按白名单执行命令，并把结果自动写入 memory。

## Runner 当前阶段

当前 runner 先实现最小闭环：

```text
memory_query -> prompt -> GPT-5.5 proxy -> agent_workspace/code -> runner log
```

暂不自动执行任意 shell 命令。原因是命令执行需要白名单、超时、产物校验和 memory 自动写回，应该作为下一层安全执行器单独实现。

已有三个入口：

- `scripts/task1_agent_runner.py`：负责让 GPT-5.5 生成或修改 `agent_workspace/code`。
- `scripts/task1_run_workflow.py`：负责 harness 实验闭环，跑预测后自动写 compact memory。
- `scripts/task1_tool_executor.py`：负责执行 Agent 输出的白名单 `tool_requests`。

## Executor 闭环

当前已有受控 tool executor，只允许执行白名单命令：

```text
bash scripts/run_in_env.sh scripts/task1_predict.py ...
bash scripts/run_in_env.sh scripts/task1_validate.py ...
bash scripts/run_in_env.sh scripts/memory_export.py ...
bash scripts/run_in_env.sh scripts/memory_promote.py ...
```

当前 executor 已支持 `checkpoint_replay`、`finetuned_checkpoint_replay`、`finetune_local`、`validation`、`memory_query`、`web_search`、
`data_shape_check`、`llm_log_prepare`、`submission_package`。
`optuna` 请求会被记录，并提示 Agent 应在生成代码里直接使用该库。`ray_tune`、`wandb` 按当前要求暂不执行，只记录为 deferred。

`finetune_local` 是受控本地工具：Agent 可以自主决策 `steps`、`lr`、`rollout_steps`、
样本预算和验证频率。`trainable=head|last-block-head|all` 只是快捷 preset；
Agent 也可以通过 `trainable_modules` 自由组合 FNO 模块，例如 `conv2,w2,conv3,w3,fc1,fc2`。
executor 负责参数范围和模块白名单检查，并强制 `temporal_stride=5`、`spatial_downsample=4`，
防止官方 checkpoint 微调时发生时间尺度错位。Agent 触发的实验目录统一写成
`runs/task1/<agent_run_id>__toolXX`，手工单跑实验写成 `runs/task1/<UTC timestamp>`；
描述信息写入 metadata/log，不写进目录名。

## Skill / Workflow 模块化

当前把重复科研动作拆成 7 个模块，Agent 只负责决策参数和下一步路线，具体工程动作
交给白名单 tool：

| 模块 | Tool | 产物 |
| --- | --- | --- |
| 读取赛题规则 | `memory_query` | compact memory packet |
| 检查数据 shape | `data_shape_check` | `agent_workspace/tool_outputs/data_shape/*.json` |
| 运行 baseline | `checkpoint_replay` | `runs/task1/<agent_run_id>__toolXX/` |
| 微调 checkpoint | `finetune_local` | `best.pt`、`last.pt`、`metadata.json`、`task1_time.csv` |
| 标准 replay 微调 checkpoint | `finetuned_checkpoint_replay` | `task1_pred.hdf5`、`metrics.json`、`memory_export.json` |
| 验证预测文件 | `validation` | validation JSON report |
| 整理合规 log | `llm_log_prepare` | `task1_logs.log` |
| 打包 submission | `submission_package` | `submissions/<name>/` |

这样设计的边界是：prompt 负责科研判断，tool 负责可审计执行，memory 负责长期
经验沉淀。后续某个环节出 bug 时，只修对应 tool 或 procedure，不需要让 Agent
重新学习整条 shell 工作流。

## Code Artifact Gate

为了避免 Agent 只生成 README、manifest 或说明性 helper，现在 runner 对
`code_artifacts` 做最低限度校验：

- `code_artifacts.entrypoints[].path` 必须对应 `files[].path`；
- `checkpoint_finetune` 阶段必须有 `role=train`、`is_executable=true` 的 Python 入口；
- 该训练入口必须显式包含 `temporal_stride`、`spatial_downsample`、`base_checkpoint`
  等关键标记；
- `prediction_validation` 阶段必须有 `role=validate` 的可执行 Python 入口；
- `submission_packaging` 阶段必须有 `role=package` 或 `role=predict` 的可执行 Python 入口。

这层 gate 不是为了替代实验验证，而是防止最终提交的 `code/` 与实际实验脱节。
prompt 告诉 Agent 应该生成什么，schema 要它声明入口，runner 拒绝明显不合格的
代码产物。

## Agent Loop

多轮闭环入口：

```bash
bash scripts/task1_agent_loop.sh --max-rounds 10
```

loop 只负责串联现有组件，不替代 Agent 决策：

```text
for round in 1..max_rounds:
  build round goal from memory + leaderboard + previous executor log
  task1_agent_runner.py
  task1_tool_executor.py
  record round summary under agent_workspace/loop_logs/
```

停止策略：

- `max_rounds` 是主要硬停止条件；
- 不因为 `submit_ready` 或达到阶段性分数线提前停止；
- 分数阈值只影响下一轮策略，不终止 loop；
- 默认普通工具失败也会继续进入下一轮，让 Agent 读取失败日志并修复；
  如果希望工具失败立刻停止，使用 `--stop-on-executor-failure`。

默认分数线：

- `official_unet_baseline = 13.3233129395`
- `prior_probe_score = 18.8862194195`
- `full_score = 100.0`
- `strong_score = max(prior_probe_score, 0.5 * full_score)`
- `excellent_score = 0.9 * full_score`

`full_score` 才是最终目标；`strong_score` 和 `excellent_score` 只是候选证据强度
和风险等级标签，不是路线命令，也不是优化终点。分数线只帮助 Agent 判断当前
策略的可信度、边际收益和合规风险；下一步继续探索、扩大训练、修复链路、
强化验证或准备提交，应由 Agent 基于实验反馈自主决定。

Task1 评分同时包含时间成本：

- 总分上限 150；
- 预测精度最高 75 分，等于分段预测得分最高 100 再乘以 0.75；
- 训练耗时最高 35 分，含 Agent 思考和工具执行时间；60 分钟内满分，之后按赛题档位下降；
- 推理耗时最高 40 分，0 到 120 秒线性衰减；超过 120 秒 Task1 计 0 分。

这些是工程约束，不是科研路线约束。Agent 仍应自由提出模型、损失、微调范围、
后处理、ensemble 和验证策略，但每轮 `experiment_plan` 必须说明时间预算和
精度/耗时 tradeoff。

强约束边界应保持清晰：赛题规则、输出格式、日志合规、尺度对齐、时间预算和
可追溯代码产物是硬约束；除此之外，Agent 的模型假设、训练策略、工具组合、
实验顺序和失败回退都应保留自主空间。
