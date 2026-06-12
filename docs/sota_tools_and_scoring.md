# SOTA 方法、工具边界和评分定义

## 赛题要求

AI4S Race 5 要求 agent 根据一个或多个 `target.pdb` 生成小分子和合成路线。提交文件是 `result.zip`，其中 CSV 必须包含：

```csv
mol_smiles,route
```

复赛总分权重是 molecule:route = 6:4。分子分内部强调 Vina binding，路线分强调路线合法性、起始原料可得性、步数、收敛性和元素守恒。

## SOTA 调研结论

### 1. Pocket-conditioned 3D 生成

最相关的 SOTA 方向是 structure-based drug design (SBDD) 的 pocket-conditioned 3D molecule generation。

MolCRAFT 是 ICML 2024 方法，核心是把 SBDD 放进连续参数空间，并用 noise-reduced sampling 缓解离散/连续混合空间导致的无效构象问题。论文报告其达到 reference-level Vina Score，并优于强基线。适合抽象成 `sota_sbdd_generator_tool`，输入 pocket，输出 SMILES/3D ligand。

DiffGui 是 2025 Nature Communications 方法，使用 target-conditioned E(3)-equivariant diffusion，同时引入 bond diffusion 和 property guidance。它的价值在于生成时同时考虑原子、键、binding affinity 和 drug-like properties，正好对应本赛题 molecule score 的方向。

DiffSBDD 是 Nature Computational Science 2024 方法，把 SBDD 表述为 3D conditional generation，用 SE(3)-equivariant diffusion 根据 protein pocket 生成 ligand。它的实现适合做外部 generator command。

因此本项目不把这些大模型硬塞进主仓库，而是提供一个清晰工具接口：

```text
AI4S_SBDD_GENERATOR_CMD
```

外部命令读取：

```text
AI4S_TARGET_PDB
AI4S_POCKET_JSON
AI4S_OUTPUT_SMILES
AI4S_LIMIT
```

这样 MolCRAFT、DiffGui、DiffSBDD、TargetDiff 都可以作为独立 tool 接入 agent。

### 2. 逆合成规划

AiZynthFinder 是实际工程里最方便保留的工具。它基于 Monte Carlo tree search，把目标分子递归拆成可购买 precursor，并用反应模板策略网络指导搜索。它是可审计、开源、适合比赛日志的 route tool。

RetroSynFormer 代表更新的 sequence/Decision Transformer 路线规划方向。它把多步逆合成看成序列决策，并用 reward 约束 building block、route depth 等因素。当前仓库不直接实现它，路线主路径先保持一个真实工具：

```text
aizynthfinder_route_tool
future: retrosynformer_route_tool
```

当前精简核心使用 AiZynthFinder 做路线规划；AiZynthFinder 失败时直接拒绝候选，不再保留窄范围 template smoke 工具。

### 3. Binding 评分

赛题明确强调 AutoDock Vina binding score。因此：

- `--mode proxy` 只用于本地快速 smoke。
- `--mode docking` 会直接用 Vina 评分候选。
- `--mode competition` 搜索阶段用 proxy binding 快速进化，最终对 top candidates 调用 Vina 重评分。
- 如果 Vina/OpenBabel 不可用，运行失败，不再降级到 proxy。

## 本项目当前工具

| 工具 | 当前状态 | 主路径作用 |
|---|---|---|
| RDKit | 保留 | SMILES、性质、SA、结构进化和路线校验 |
| SOTA SBDD external command | 新增接口 | 接 MolCRAFT/DiffGui/DiffSBDD/TargetDiff |
| Vina Python binding + OpenBabel | 已接主路径 | docking mode 直接评分；competition mode 最终重评分 |
| AiZynthFinder | 已接主路径 | `AI4S_ROUTE_ENGINE=aizynthfinder` 时做 MCTS/template 逆合成 |
| fpocket | 移出主路径 | 当前 pocket summary 直接从 PDB 坐标生成 |
| REINVENT4 | 移出主路径 | 当前优先 SBDD 3D generator |

## 分数定义

```text
molecule_score = 0.8 * binding_score + 0.1 * validity_score + 0.1 * sa_score
route_score    = 0.55 * route_validity_score
               + 0.30 * starting_material_availability_score
               + 0.05 * step_penalty_score
               + 0.05 * convergence_score
               + 0.05 * balance_score
total          = 0.60 * molecule_score + 0.40 * route_score
```

硬零规则：

- `validity_score = 0` 时，`molecule_score = 0`
- route 最终产物不等于设计分子时，`route_score = 0`
- 元素不守恒时，`route_score = 0`
- 自反反应时，`route_score = 0`

## 参考来源

- MolCRAFT: Structure-Based Drug Design in Continuous Parameter Space, ICML 2024, PMLR.
- DiffGui: Target-aware 3D molecular generation based on guided equivariant diffusion, Nature Communications 2025.
- DiffSBDD: Structure-based drug design with equivariant diffusion models, Nature Computational Science 2024.
- AiZynthFinder GitHub and Journal of Cheminformatics paper.
- AutoDock Vina official project and paper.
