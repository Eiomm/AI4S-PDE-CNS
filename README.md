# Chem-Evolve Agent

这是 AI4S CNS 小分子设计任务的精简版 agent。当前代码只保留一条主路径：

```text
target.pdb
  -> pocket summary
  -> competition/scoring context
  -> LLM + molecule generation tools
  -> strict route planning with AiZynthFinder
  -> proxy scoring during search, or direct Vina scoring in docking mode
  -> long-term experience memory + elite memory + structure evolution
  -> final Vina reranking in competition mode
  -> result.csv / result.log / result.zip
```

没有 legacy runner，没有静默兜底。工具不可用或候选没有可验证路线时，候选会被拒绝；如果没有任何合格候选，整次运行失败。

## 代码结构

```text
chem_evolve_agent/
  cli.py             命令行入口
  core.py            agent 主流程
  chem_ops.py        SMILES、路线校验、分数定义、内置 RDKit 生成
  runtime_tools.py   PDB 读取、工具检查、外部 SBDD 生成器、Vina 调用
  llm.py             LiteLLM/OpenAI-compatible JSON 调用
  models.py          Candidate、Route、Score
  submitter.py       结果 CSV/LOG/ZIP 写出
```

## 本地运行

```bash
cd /data/wangjunao/AI4S
export CHEM_EVOLVE_LLM_ENABLED=1
/data/wangjunao/miniconda3/envs/ai4s-chem-evolve/bin/python -m chem_evolve_agent.cli \
  --targets examples/target.pdb \
  --out runs/smoke \
  --rounds 1 \
  --per-round 8 \
  --top-k 5 \
  --mode proxy \
  --runner agent
```

检查提交包：

```bash
/data/wangjunao/miniconda3/envs/ai4s-chem-evolve/bin/python scripts/inspect_result_zip.py runs/smoke/result.zip
```

跑 harness：

```bash
CHEM_EVOLVE_LLM_ENABLED=1 bash scripts/run_harness_once.sh --name smoke --target examples/target.pdb --rounds 1 --per-round 8 --top-k 5 --mode proxy --skip-tests
```

直接跑真实 `competition` 或 `docking` 模式：

```bash
# competition：搜索阶段 proxy_search，最终 Vina 重排
bash scripts/run_real_mode.sh --mode competition --target examples/target.pdb

# docking：候选评估阶段直接 Vina 打分
bash scripts/run_real_mode.sh --mode docking --target examples/target.pdb
```

默认输出到 `outputs/competition_real/latest` 或 `outputs/docking_real/latest`。小规模验证可以临时关闭 LLM：

```bash
CHEM_EVOLVE_LLM_ENABLED=0 bash scripts/run_real_mode.sh --mode competition --target examples/target.pdb --rounds 1 --per-round 2 --top-k 1 --docking-limit 1
```

运行中终端会实时打印中文 `[agent]` 进度行，采用轻量分组样式，例如 `[agent][LLM] 开始调用 LLM | event=llm_start`、`[agent][路线] 开始路线规划 | event=route_start`、`[agent][路线] 路线规划通过 | event=route_accept`、`[agent][路线] 路线规划拒绝 | event=route_reject`、`[agent][Vina] 开始 Vina 反馈 | event=vina_feedback_start` 和 `[agent][轮次] 本轮完成 | event=round_done`。LLM 等待期间默认每 30 秒打印一次 `[agent][LLM] 等待 LLM 返回` 心跳；可用 `AI4S_LLM_HEARTBEAT_SECONDS=10` 调快，或设为 `0` 关闭。这些实时行也会写入本次 run 的 `pipeline.stdout.log`；最终可审计结构化日志仍写入 `result.log`/`pipeline.log`。如果只想保留最终日志，可以设置 `AI4S_PROGRESS_STDERR=0` 关闭实时打印。颜色默认只在交互式终端启用；通过 `AI4S_PROGRESS_COLOR=1` 可以强制开启，`AI4S_PROGRESS_COLOR=0` 可以关闭。

如果只想离线验证 `result.zip` 合同，可以手动设 `CHEM_EVOLVE_LLM_ENABLED=0`。正常 agent 运行默认应使用 API 推理。

真实主链路 smoke：

```bash
bash scripts/run_real_competition_smoke.sh
```

这个 smoke 默认启用 `CHEM_EVOLVE_LLM_ENABLED=1`，并使用 AiZynthFinder 路线与 Vina 复排；只做离线结构验证时才手动覆盖 `CHEM_EVOLVE_LLM_ENABLED=0`。

这会用最小规模跑真实 `competition` 模式，强制经过 AiZynthFinder 路线规划和 Vina 终局重评分，并检查日志中出现 `competition_dock` 和 `route_source=aizynthfinder`。

复赛容器入口：

```bash
bash scripts/run_competition_final.sh
```

运行开始时会先检查参数、LLM API key、当前模式需要的 Vina/AiZynthFinder/SBDD 命令，以及所有 target PDB 是否存在且可解析；这些预检错误都会在清理输出前直接失败。通过预检后，程序才会清理输出目录中由本 agent 管理的旧 `result*.csv`、`result*.log`、`result*.zip`，以及 `generation/`、`routes/`、`docking/`、`docking_feedback/`、`llm_io/`、`work/` scratch 目录。只有全部 target 通过预检后才开始逐靶点生成；多靶点模式只生成最终 `result.zip`，不会留下中间 `result1.zip/result2.zip/result3.zip`。
每个 `result*.log` 也会写入 `agent_runtime_config`，记录运行模式、LLM 模型、LLM 审计日志目录、路线引擎、路线预算、docking 预算和输出清理结果；只记录 API key 是否存在，不写入 key 明文。默认 LLM IO 日志写到本次输出目录的 `llm_io/`，不会写到仓库根目录的 `runs/llm_io/`。

Docker 镜像默认使用当前稳态复赛配置：`AGENT_MODE=competition`、`CHEM_EVOLVE_LLM_ENABLED=1`、`AGENT_ROUNDS=8`、`AGENT_PER_ROUND=32`、`AGENT_TOP_K=10`、`AGENT_DOCKING_LIMIT=10`、`AI4S_ROUTE_ENGINE=aizynthfinder`、`AI4S_ROUTE_LIMIT_PER_ROUND=10`、`AI4S_VINA_FEEDBACK_PER_ROUND=1`。Dockerfile 默认使用清华 conda/pip 镜像源安装依赖，可通过 `DOCKER_CONDA_CHANNEL_ALIAS`、`DOCKER_PIP_INDEX_URL` 覆盖。镜像不包含 API key；运行时需要通过环境变量传入。Docker 专用 LLM 配置模板见 `configs/docker_llm.env.example`，建议用 `AI4S_AGENT_API_KEY_ENVS=DOCKER_LLM_API_KEY` 隔离本地测试 key。原始 benchmark actives/decoys 不进入镜像，只保留聚合 prior。构建时会把 `app/training_code` 复制到赛题审查路径 `/app/training_code`。

也可以按天池样例风格构建推送：

```bash
export DOCKER_REGISTRY=registry.cn-shenzhen.aliyuncs.com/ai4s-junao
bash docker_build.sh v1
```

## 工具

查看当前工具：

```bash
/data/wangjunao/miniconda3/envs/ai4s-chem-evolve/bin/python scripts/check_tools.py
/data/wangjunao/miniconda3/envs/ai4s-chem-evolve/bin/python scripts/check_tools.py --require-sbdd
/data/wangjunao/miniconda3/envs/ai4s-chem-evolve/bin/python scripts/check_tools.py --strict
```

普通检查只报告当前工具状态；`--require-sbdd` 用于确认外部 SBDD 生成器已经接好，会用 `examples/target.pdb` 实际调用一次生成器并要求至少 1 个 RDKit-valid SMILES；`--strict` 要求所有工具都可用。SBDD 检查不只看 `AI4S_SBDD_GENERATOR_CMD` 是否存在，还会确认命令里的可执行文件能找到。

可用工具边界：

- `rdkit_property_tool`：SMILES 合法性、性质、QED、SA。
- `sota_sbdd_generator_tool`：外部 MolCRAFT/DiffGui/DiffSBDD/TargetDiff 风格 pocket-conditioned 3D 生成器；通过 `AI4S_SBDD_GENERATOR_CMD` 接入。
- `vina_docking_tool`：Vina Python binding + OpenBabel，`--mode docking/competition` 时实际使用。
- `aizynthfinder_route_tool`：AiZynthFinder MCTS/template 逆合成路线工具；唯一正式路线规划主路径。

`aizynthfinder_route_tool` 返回 solved route 时，agent 会把“已拆到 AiZynthFinder stock 起始物”作为 `starting_material_availability_score=1.0` 的证据；非 AiZynthFinder 路线仍按本地起始物启发式评分。
当 AiZynthFinder 返回多棵 route tree 时，agent 会选择通过硬规则且 `route_score` 最好的路线，不会只取第一棵。

外部工具落地方式见：

```bash
docs/tool_implementation.md
```

准备/检查工具：

```bash
bash scripts/setup_external_tools.sh
/data/wangjunao/miniconda3/envs/ai4s-chem-evolve/bin/python scripts/check_tools.py
```

检查数据：

```bash
/data/wangjunao/miniconda3/envs/ai4s-chem-evolve/bin/python scripts/check_data.py
```

`scripts/check_data.py` 会同时检查赛题文档、复赛入口 `Code/main.py`、`Code/README.md`、Dockerfile training-code 打包点、`app/training_code/README.md`、benchmark prior 和本地验证数据。

赛前一键检查：

```bash
bash scripts/check_competition_ready.sh
AI4S_REQUIRE_SBDD=1 bash scripts/check_competition_ready.sh
AI4S_RUN_REAL_COMPETITION_SMOKE=1 bash scripts/check_competition_ready.sh
```

默认一键检查会跑数据/交付物检查、普通工具检查、LLM 连通性检查和快速离线提交 smoke。`AI4S_REQUIRE_SBDD=1` 会把外部 SBDD probe 作为硬门槛；`AI4S_RUN_REAL_COMPETITION_SMOKE=1` 会额外跑 AiZynthFinder + Vina 的真实 competition smoke。

agent 默认会读取：

```text
docs/competition_race5_description.md
docs/sota_tools_and_scoring.md
data/README.md
data/benchmarks/benchmark_prior.json
```

然后再调用 API 推理候选分子。`benchmark_prior.json` 是聚合性质统计，不是固定候选库；agent 也会把它转成 `property_prior_score` 写入候选评估日志，并作为 proxy search 的一部分。

agent 还会维护跨实验长期经验文件。输出目录位于 `outputs/` 下时，harness/CLI 默认写到：

```text
outputs/agent_experience.jsonl
```

如果输出目录不在 `outputs/` 下，例如 `/tmp/smoke_xxx`，默认 memory 会放在本次 `out_dir/agent_experience.jsonl`，避免临时实验互相污染。每次成功生成候选后，CLI 会把本次 top candidates 的 SMILES、路线来源、binding 来源、总分和分项分数追加进去。下一次运行同一个 target signature 时，`core.py` 会在第 1 轮开始前读取这些历史经验，并把它们作为 `long_term_memory` 合并进 `agent_memory`；LLM prompt 和 RDKit evolution generator 都能看到这些经验。长期 memory 只提供经验，不会绕过 RDKit、route、SA、binding 和提交前硬规则校验。

Docker 镜像默认读取：

```text
data/agent_experience.jsonl
```

这个文件是精简后的全局经验种子，允许随镜像上传。为了避免它变成固定答案库，agent 会拒绝原样复用长期经验中的 SMILES，只允许 LLM 和 RDKit evolution generator 基于这些经验做改造。

可以显式指定长期经验文件：

```bash
python -m chem_evolve_agent.cli \
  --targets examples/target.pdb \
  --out outputs/api_proxy/manual \
  --rounds 1 \
  --per-round 8 \
  --top-k 5 \
  --mode proxy \
  --memory-file outputs/agent_experience.jsonl
```

也可以通过环境变量控制：

```bash
AI4S_AGENT_MEMORY_FILE=outputs/agent_experience.jsonl
AI4S_AGENT_MEMORY_LIMIT=10
```

## 自主迭代进化

`--rounds` 现在代表真实的迭代轮数。每一轮开始时，agent 会把长期经验文件里同 target signature 的 top 历史候选和当前 run 已评估候选合并成 `agent_memory`；每一轮结束后，当前 run 的最高分候选也会进入下一轮 memory。下一轮会同时：

- 如果启用 LLM，把赛题说明、评分标准、数据 prior、靶点 pocket、长期经验和当前 run 优秀分子交给 LLM，让它输出本轮 `agent_strategy`、优化重点、规避项和候选分子。
- 把最近被拒绝候选的原因也交给 LLM，例如 `invalid_smiles`、`AiZynthFinder found no solved route`、`route_score_zero`、`round_vina_feedback_failed`，让下一轮能避开失败方向。
- 如果显式关闭 LLM，使用本地启发式策略继续跑可复现 smoke。
- 调用 RDKit evolution generator，对优秀分子做小的可校验结构扰动；本地生成/进化 seed 会绑定 target pocket，避免多靶点离线搜索完全同质。
- 保留 internal RDKit generator，继续提供基础多样性，但它只作为显式生成工具参与候选池，不替代 LLM/SBDD/AiZynthFinder/Vina。

`--per-round` 表示每轮进入预筛的总候选池大小，不是每个生成工具各自的数量。每轮候选会先用 RDKit 性质和 proxy molecule score 做轻量排序，再把路线规划预算花在更有希望的候选上。复赛默认 `AI4S_ROUTE_LIMIT_PER_ROUND=10`，也就是每个 target 最多 `rounds * 10` 次 AiZynthFinder 路线搜索；这个值可以显式调高或调低。
`competition` 模式默认 `AI4S_VINA_FEEDBACK_PER_ROUND=1`，每轮会把当前轮最优候选用真实 Vina 打一次分并写回 `agent_memory`，让下一轮 LLM/进化器看到真实 binding 反馈；如果这次 Vina feedback 失败，失败原因会进入 `agent_rejection_memory`。每轮结束都会写 `agent_round_summary`，记录本轮生成、筛选、路线评估、接受、拒绝和当前最优分子。最终重排会复用已经 Vina 过的候选，不重复 docking。

所有候选仍然必须通过路线、SA、validity、binding/proxy binding 和硬零规则，不能靠记忆直接复制最终答案。
写出 `result*.csv` 和最终 `result.zip` 前，submitter 会再次校验 route 终产物、元素守恒、自反应和候选分数，防止坏路线进入提交包。

`competition` 模式采用两阶段评估：迭代搜索阶段用 proxy binding 快速探索和进化，所有候选先经过路线与分数筛选；运行结束前只对当前 top candidates 调用 Vina 做最终重评分。这样 Vina 预算会花在更有希望的候选上，而不是花在最早生成的候选上。

如果没有 `.env`，`competition` 模式也会自动设置：

```text
AI4S_ROUTE_ENGINE=aizynthfinder
AIZYNTHFINDER_CONFIG=data/aizynthfinder/config.yml
CHEM_EVOLVE_LLM_ENABLED=1
AI4S_ROUTE_LIMIT_PER_ROUND=10
AI4S_VINA_FEEDBACK_PER_ROUND=1
```

也就是说复赛入口和本地 CLI 都使用 AiZynthFinder 路线规划；旧的窄 template route 工具已经从主路径删除。

更完整的中文架构说明见：

```text
docs/agent_architecture_cn.md
```

## 分数定义

本地目标贴近赛题：

```text
molecule_score = 0.8 * binding_score + 0.1 * validity_score + 0.1 * sa_score
route_score    = 0.55 * route_validity
               + 0.30 * starting_material_availability
               + 0.05 * step_penalty
               + 0.05 * convergence
               + 0.05 * balance
total          = 0.60 * molecule_score + 0.40 * route_score
```

硬规则：

- 分子无效：`molecule_score = 0`
- route 最终产物不等于 `mol_smiles`：`route_score = 0`
- route 元素不守恒：`route_score = 0`
- 自反反应 `A>>A`：`route_score = 0`

`scripts/inspect_result_zip.py` 会检查 zip 成员合同、CSV 列、route 最终产物，以及这些 route hard-zero 规则；不传期望成员时默认按单靶点 `result.csv + result.log` 检查。

## SOTA 调研

见 `docs/sota_tools_and_scoring.md`。
