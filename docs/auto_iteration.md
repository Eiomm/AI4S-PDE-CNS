# 自动实验闭环

这个项目现在可以按师兄建议跑一个最小版 `/goal` 闭环：

```text
设定目标 -> 选择策略 -> 跑 pipeline -> 终端展示过程 -> 分析结果 -> 记录经验 -> 更新 skill
```

## 推荐命令

先跑一个轻量版本，确认流程：

```bash
/data/wangjunao/miniconda3/envs/ai4s-chem-evolve/bin/python scripts/auto_iterate.py \
  --experiment goal_smoke \
  --iterations 1 \
  --strategies seed_baseline llm_diverse \
  --skip-tests
```

正式跑可以去掉 `--skip-tests`：

```bash
/data/wangjunao/miniconda3/envs/ai4s-chem-evolve/bin/python scripts/auto_iterate.py \
  --experiment goal_full \
  --iterations 2
```

如果要跑新的 agent 主循环，可以指定 `agent_main`：

```bash
/data/wangjunao/miniconda3/envs/ai4s-chem-evolve/bin/python scripts/auto_iterate.py \
  --experiment goal_agent \
  --config configs/strategies/agent.yaml \
  --iterations 1 \
  --strategies agent_main \
  --skip-tests
```

`agent_main` 使用 `runner: agent`。LLM planner 可用时会根据当前候选池、可用 action 和 memory 选择下一步；不可用或输出非法时会自动降级到 heuristic planner。

## 终端会显示什么

每个策略开始时会显示中文说明：

```text
第 1 轮 | 策略：LLM 多样性探索 (llm_diverse) | 运行名：i01_llm_diverse
参数：生成轮数=4，每轮数量=8，保留 top_k=10，模式=proxy，docking 数=0，是否调用 LLM=True
```

运行中会实时显示 harness 的关键输出。每轮结束后会显示：

```text
最近关键事件：
  生成：第 3 轮，生成器=llm_generator，数量=8
  排序完成：候选数=23，最高分=0.71
  流水线完成：候选数=23，最高分=0.71，模式=proxy

实验结果看板：当前最好综合指标=0.7412
轮次 策略             状态    提升  综合    最高    平均    合法    去重  骨架  LLM
-----------------------------------------------------------------------------------
1    固定种子基线      通过    是    0.7031  0.8124  0.7012  10/10   10    5     0
1    LLM 多样性探索    通过    是    0.7412  0.8460  0.7220  10/10   10    8     1
```

## 结果存在哪里

每次启动 `auto_iterate.py` 都会先创建一个独立实验 session：

```text
outputs/<experiment>/<session_timestamp>/
```

同一次启动里的每轮结果会放在这个 session 下：

```text
outputs/<experiment>/<session_timestamp>/runs/i01_agent_main/<run_timestamp>/
outputs/<experiment>/<session_timestamp>/runs/i02_agent_main/<run_timestamp>/
outputs/<experiment>/<session_timestamp>/runs/i03_agent_main/<run_timestamp>/
```

session 根目录会保留本次实验的最佳结果：

```text
outputs/<experiment>/<session_timestamp>/best -> runs/iXX_<strategy>/<run_timestamp>
outputs/<experiment>/<session_timestamp>/best_record.json
outputs/<experiment>/<session_timestamp>/best_result.csv
outputs/<experiment>/<session_timestamp>/best_result.log
outputs/<experiment>/<session_timestamp>/best_result.zip
```

session 内也有本次实验专属汇总：

```text
outputs/<experiment>/<session_timestamp>/experiment_index.csv
outputs/<experiment>/<session_timestamp>/experiment_index.jsonl
```

全局策略记忆仍然保存在：

```text
outputs/strategy_memory/experiment_index.csv
outputs/strategy_memory/experiment_index.jsonl
outputs/strategy_memory/best_strategies.md
outputs/strategy_memory/failed_strategies.md
```

每个具体 run 仍然有：

```text
outputs/<experiment>/<session_timestamp>/runs/<run_name>/latest/candidates.csv
outputs/<experiment>/<session_timestamp>/runs/<run_name>/latest/pipeline.log
outputs/<experiment>/<session_timestamp>/runs/<run_name>/latest/submission.zip
outputs/<experiment>/<session_timestamp>/runs/<run_name>/latest/llm_io/
```

`auto_iterate.py` 会给每个 run 注入不同的 `CHEM_EVOLVE_RUN_SEED`，避免同一个 strategy 在多轮 iteration 中完全重放同一条生成路径。需要复现实验时可以显式传入：

```bash
/data/wangjunao/miniconda3/envs/ai4s-chem-evolve/bin/python scripts/auto_iterate.py \
  --experiment goal_agent \
  --config configs/strategies/agent.yaml \
  --iterations 5 \
  --seed 20260611 \
  --skip-tests
```

## skill 输出

脚本会自动更新两份 skill：

```text
skills/ai4s-chem-evolve/SKILL.md
/home/wangjunao/.codex/skills/ai4s-chem-evolve/SKILL.md
```

之后 Codex/Claude 可以把这里的最佳策略当作初始策略，而不是每次从零开始。
