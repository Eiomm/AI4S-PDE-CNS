# AI4S Chem-Evolve Agent 架构说明

这份说明只描述当前精简后的主路径，便于新同学或评审快速理解项目。

## 目标

赛题要求 agent 根据蛋白靶点 `target.pdb` 生成小分子 `mol_smiles`，并给出合成路线 `route`。最终输出必须是 `result.zip`，CSV 列固定为：

```csv
mol_smiles,route
```

复赛是三靶点任务，容器输入为：

```text
/saisdata/target1.pdb
/saisdata/target2.pdb
/saisdata/target3.pdb
```

入口脚本按官方要求放在 `/app/run.sh`；为兼容早期本地目录，也会在直接路径不存在时读取 `/saisdata/37/target1-3.pdb`。

输出为：

```text
/saisresult/result.zip
```

其中包含 `result1.csv`、`result2.csv`、`result3.csv`。
运行开始时先检查参数、LLM API key、当前模式需要的 Vina/AiZynthFinder/SBDD 命令，以及所有 target PDB 是否存在且可解析；这些预检错误都会在清理输出前直接失败。通过运行前检查后，程序才会清理旧的 agent 托管输出：`result*.csv`、`result*.log`、`result*.zip`，以及 `generation/`、`routes/`、`docking/`、`docking_feedback/`、`llm_io/`、`work/` scratch 目录。只有全部 target 通过预检后，才开始逐 target 生成，避免留下半成品。多靶点运行时，每个 target 的中间文件写入 `work/resultN/`，避免路线、SBDD 和 docking 证据互相覆盖。清理行为写入 `output_cleanup` 日志，避免复赛环境里旧结果污染本次运行。

## 主路径

当前项目只保留一条主流程：

```text
Code/main.py 或 chem_evolve_agent.cli
  -> core.py                 组织 agent 轮次和日志
  -> runtime_tools.py         读取 PDB、调用 SBDD/Vina/AiZynthFinder
  -> chem_ops.py              RDKit、路线校验、分数、结构进化
  -> submitter.py             写 result.csv/result.log/result.zip
```

没有 legacy runner，也没有静默兜底。工具不可用时应清楚失败；候选路线不合法时拒绝该候选。

## 自主迭代进化

每个 target 会跑多轮：

1. 读取赛题描述、评分标准和数据说明。
2. 读取 `AI4S_AGENT_MEMORY_FILE` 指向的长期经验文件；默认 harness 路径是 `outputs/agent_experience.jsonl`，Docker 提交默认路径是 `data/agent_experience.jsonl`。普通记录只加载同 target signature 的历史 top candidates；`scope=global` 的精简经验种子可被任意 target 读取，但长期记忆中的 SMILES 禁止原样复用，只能作为 LLM/进化器的经验提示。
3. 如果启用 LLM，先让 LLM 基于赛题上下文、靶点 pocket、长期经验和当前 run 的 `agent_memory` 输出本轮 `agent_strategy`、优化重点、规避项和候选分子；如果显式关闭 LLM，则使用本地启发式策略。
4. 用 LLM 回合候选、外部 SBDD 工具或 RDKit 生成候选分子，并把每轮总候选池限制在 `--per-round`；本地 RDKit 生成/进化 seed 会绑定 target pocket，使多靶点离线搜索仍然按靶点分化。
5. 先用 RDKit 性质和 proxy molecule score 做轻量预筛。
6. 对预筛后进入路线预算的候选分子规划路线。
7. 用 proxy 或 Vina 计算 binding score；`competition` 模式每轮默认对当前轮最优候选追加 1 次真实 Vina 反馈，失败原因也会进入下一轮 rejection memory。
8. 按赛题形状的 molecule/route/total score 排名。
9. 把当前 top candidates 写入本次 run 的 `agent_memory`，下一轮继续用。
10. CLI 写出本次结果后，把 top candidates 追加到长期经验 JSONL；下一次实验会读到这些经验。
11. 如果是 `competition` 模式，结束前对 top candidates 调用 Vina 做最终重评分。
12. 写出 CSV/ZIP 前再次校验 `mol_smiles`、route 终产物、元素守恒和自反应等硬规则。

这样 `--rounds` 不是重复采样，而是带有可审计记忆的进化循环。日志中会看到：

默认 LLM IO 审计日志写入本次输出目录的 `llm_io/`，`agent_runtime_config` 会记录该目录；`scripts/check_llm_connectivity.py` 默认使用临时日志目录，只有显式传 `--log-dir` 时才保留连通性检查日志。
运行中终端会实时打印中文 `[agent]` 进度行，例如 `[agent][路线] 开始路线规划 | event=route_start`，并保留 `event=...` 英文事件名，覆盖 LLM 请求、候选生成、route planning、Vina feedback、最终 docking 和 round summary；harness 会把这些行保存到 `pipeline.stdout.log`。颜色默认只在交互式终端启用，也可用 `AI4S_PROGRESS_COLOR=1/0` 强制开关。最终可审计事件仍写入 `result.log`。

```text
agent_runtime_config
agent_experience_loaded
agent_strategy
agent_plan
generate
evaluate
agent_memory
agent_rejection_memory
agent_round_summary
agent_rank
```

## 工具边界

- `rdkit_property_tool`：SMILES 合法性、QED、SA、性质、结构进化和路线校验。
- `sota_sbdd_generator_tool`：外部 SBDD 生成器接口，通过 `AI4S_SBDD_GENERATOR_CMD` 接入。
- `vina_docking_tool`：Vina Python binding + OpenBabel，用于 `docking/competition` 模式。
- `aizynthfinder_route_tool`：AiZynthFinder 多步逆合成规划。

AiZynthFinder 返回 `solved=True` 时，说明路线已经拆到配置 stock 中的起始物；agent 会把这个工具证据用于 `starting_material_availability_score`。其他来源的路线仍使用本地起始物可得性启发式评分。

如果 AiZynthFinder 返回多棵 route tree，agent 会逐棵解析并使用 route 硬规则校验，选择通过硬规则且 `route_score` 最好的路线，而不是直接拿第一棵。

## 数据 prior

`app/training_code/build_benchmark_prior.py` 会从本地 benchmark actives/decoys 生成 `data/benchmarks/benchmark_prior.json`。这个文件只保存性质统计量，例如 MW、logP、QED、TPSA、SA 的 median/q1/q3，不保存 benchmark 分子字符串。agent 把它作为性质分布 prior 读入上下文，同时在 proxy search 中计算 `property_prior_score`，帮助 LLM、策略和本地搜索更贴近药物样分子空间。

## Competition 模式

`competition` 模式不是每生成一个候选就立刻 docking。它分成两段：

1. 搜索阶段：用轻量 proxy molecule score 预筛候选，再对进入路线预算的候选计算 route score，让 LLM 和 RDKit evolution 有足够反馈。
2. 终局阶段：把当前排名靠前的候选交给 Vina，使用真实 Vina binding score 重算总分。

这样更符合复赛 4 小时限制：Vina 预算集中用于最有希望的分子，同时最终提交仍然由真实 docking 分数排序。

默认复赛路线预算是：

```text
AI4S_ROUTE_LIMIT_PER_ROUND=10
AI4S_VINA_FEEDBACK_PER_ROUND=1
```

也就是每个 target 每轮最多调用 10 次 AiZynthFinder，并对当前轮最优候选做 1 次真实 Vina 反馈。这个预算是显式参数；如果调得太低导致没有候选通过路线规划，运行会失败而不是静默兜底。

没有 `.env` 时，`competition` 模式也会默认启用 AiZynthFinder：

```text
AI4S_ROUTE_ENGINE=aizynthfinder
AIZYNTHFINDER_CONFIG=data/aizynthfinder/config.yml
CHEM_EVOLVE_LLM_ENABLED=1
AI4S_ROUTE_LIMIT_PER_ROUND=10
AI4S_VINA_FEEDBACK_PER_ROUND=1
```

这是显式默认值，不是静默兜底。如果 AiZynthFinder 不可用，运行会清楚失败。

## 输出目录

单靶点本地运行会生成：

```text
result.csv
result.log
result.zip
```

三靶点复赛运行会生成：

```text
result1.csv
result1.log
result2.csv
result2.log
result3.csv
result3.log
result.zip
```

最终提交包 `result.zip` 只包含 `result1.csv`、`result2.csv`、`result3.csv`。每个 target 的 log 留在输出目录用于人工审计，不额外打包中间 zip；最终 zip 组装完成后，每个 `resultN.log` 会追加 `final_submit` 事件，记录 zip 路径和成员列表。

## 分数和硬规则

本地分数贴近赛题：

```text
molecule_score = 0.8 * binding_score + 0.1 * validity_score + 0.1 * sa_score
route_score    = 0.55 * route_validity
               + 0.30 * starting_material_availability
               + 0.05 * step_penalty
               + 0.05 * convergence
               + 0.05 * balance
total          = 0.60 * molecule_score + 0.40 * route_score
```

硬零规则：

- 分子无效则 molecule score 为 0。
- route 最终产物不等于 `mol_smiles` 则 route score 为 0。
- 元素不守恒则 route score 为 0。
- 自反反应 `A>>A` 直接拒绝。

`scripts/inspect_result_zip.py` 不只检查文件存在，还会检查 zip 成员合同、CSV 列、route 最终产物、元素守恒和自反反应等硬零规则。不传期望成员时默认按单靶点 `result.csv + result.log` 检查。

## 常用命令

本地快速验证：

```bash
python -m chem_evolve_agent.cli \
  --targets examples/target.pdb \
  --out runs/smoke \
  --rounds 2 \
  --per-round 8 \
  --top-k 5 \
  --mode proxy \
  --runner agent
```

复赛入口：

```bash
python Code/main.py
```

检查工具和数据：

```bash
python scripts/check_tools.py
python scripts/check_tools.py --require-sbdd
python scripts/check_data.py
python scripts/check_llm_connectivity.py
```
