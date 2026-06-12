# 外部工具实现说明

这份文档说明当前三个外部工具怎样真正落地。原则很简单：选中的工具必须真实可用；不可用就报错，不做静默兜底。

## 1. SOTA SBDD Generator

用途：接入 MolCRAFT、DiffGui、DiffSBDD、TargetDiff 这类 pocket-conditioned 3D 分子生成模型。

当前项目不把大模型代码塞进主仓库，而是保留稳定接口：

```bash
export AI4S_SBDD_GENERATOR_CMD=/path/to/sbdd_adapter.sh
```

agent 调用该命令时会提供这些环境变量：

```text
AI4S_TARGET_PDB      当前 target pdb
AI4S_POCKET_JSON     pocket center/box/summary
AI4S_OUTPUT_SMILES   生成器必须写出的 csv/txt
AI4S_LIMIT           本轮需要的候选数
```

输出要求：

```csv
smiles
COc1ccc(NC(C)=O)cc1
...
```

adapter 至少要写出一个 RDKit-valid SMILES。空文件或全非法 SMILES 会让 agent 直接失败，不会静默改用内部生成器。

TargetDiff 的官方采样入口是：

```bash
python scripts/sample_for_pocket.py configs/sampling.yml --pdb_path pocket10.pdb
```

所以实际 adapter 要做三件事：

1. 从 `AI4S_TARGET_PDB` 或 `AI4S_POCKET_JSON` 准备模型需要的 pocket PDB。
2. 调用模型采样脚本。
3. 从模型输出中抽取 SMILES，写到 `AI4S_OUTPUT_SMILES`。

## 2. Vina Docking Tool

当前实现已经接入：

- `obabel`：SMILES/PDB 转 PDBQT
- `vina` Python binding：实际 docking 和 affinity 计算

检查：

```bash
python scripts/check_tools.py
python scripts/check_tools.py --require-sbdd
python scripts/check_tools.py --strict
```

普通检查只报告当前状态；`--require-sbdd` 会在 `AI4S_SBDD_GENERATOR_CMD` 缺失、命令不可执行，或 probe 无法产出 RDKit-valid SMILES 时失败；`--strict` 要求所有外部工具都就绪。SBDD probe 默认使用 `examples/target.pdb` 和 `--probe-limit 1`。

运行 `--mode docking` 或 `--mode competition` 时，如果 `python:vina` 或 `obabel` 不可用，运行直接失败。区别是：`docking` 模式逐候选直接 docking；`competition` 模式先用 proxy 搜索，结束前只对当前 top candidates 做 Vina 重评分。

可调环境变量：

```bash
export AI4S_VINA_CPU=1
export AI4S_VINA_EXHAUSTIVENESS=8
```

## 3. AiZynthFinder Route Tool

当前实现已经接入 `aizynthcli`。必须准备 public data 或你自己的策略模型/stock。

公共数据安装：

```bash
mkdir -p data/aizynthfinder
download_public_data data/aizynthfinder
export AIZYNTHFINDER_CONFIG=data/aizynthfinder/config.yml
export AI4S_ROUTE_ENGINE=aizynthfinder
```

agent 对每个候选分子调用：

```bash
aizynthcli --config "$AIZYNTHFINDER_CONFIG" --smiles target.smi --output aizynth_output.json.gz
```

然后从 `trees` 字段解析路线，转换成比赛需要的：

```text
reactant1.reactant2>>product,intermediate.reactant3>>final_mol
```

路线仍会被本项目再次校验：

- final product 必须等于 `mol_smiles`
- 不能元素不守恒
- 不能 `A>>A`
- route score 必须大于 0

## 当前建议

本项目默认：

```bash
CHEM_EVOLVE_LLM_ENABLED=1
AI4S_ROUTE_ENGINE=aizynthfinder
```

也就是说 agent 每轮先读赛题/评分上下文、靶点 pocket 和历史高分候选，让 LLM 输出本轮策略、优化重点、规避项和候选分子；再用 RDKit、AiZynthFinder、proxy/Vina 按硬规则筛选和排名。复赛 `competition` 模式会把 Vina 预算集中到最终 top candidates。
