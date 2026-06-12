# Data 目录说明

这个目录只放 agent 运行所需的数据和本地验证数据，不放固定答案库。

跨实验沉淀的候选经验在 `outputs/` 实验目录下默认写入 `outputs/agent_experience.jsonl`；临时输出目录会写入本次 `out_dir/agent_experience.jsonl`，避免不同 `/tmp` 实验互相污染。准备 Docker 提交时，可以把筛选后的经验种子放在 `data/agent_experience.jsonl` 随镜像上传；它是 agent 自己评估过的经验，不是 benchmark actives/decoys。运行时会把它作为 LLM/进化器的经验提示，且禁止原样复用长期记忆中的 SMILES，所有候选仍必须重新通过 RDKit、路线、SA、binding 和提交前校验。

## 当前数据状态

```text
data/
  benchmarks/manifest.yaml     本地验证 benchmark 清单
  benchmarks/benchmark_prior.json  聚合性质 prior，不含 benchmark 分子字符串
  agent_experience.jsonl       可选全局经验种子，不是固定答案库
  aizynthfinder/                AiZynthFinder public model/stock 数据
```

## 比赛输入

比赛环境不会从这里读取最终 target。复赛容器入口读取：

```text
/saisdata/target1.pdb
/saisdata/target2.pdb
/saisdata/target3.pdb
```

本地旧布局 `/saisdata/37/target1-3.pdb` 仍可被入口兼容读取。

本地 smoke 用：

```text
examples/target.pdb
```

## AiZynthFinder 数据

AiZynthFinder 需要 policy model 和 stock。用官方工具下载：

```bash
download_public_data data/aizynthfinder
```

完成后应有：

```text
data/aizynthfinder/config.yml
```

`.env` 中应配置：

```bash
AIZYNTHFINDER_CONFIG=data/aizynthfinder/config.yml
AI4S_ROUTE_ENGINE=aizynthfinder
```

## Benchmark 数据

`data/benchmarks/manifest.yaml` 是本地验证清单，不是比赛答案。里面的 actives/decoys 只能用于：

- 比较策略
- 调参
- 回归测试

不能把活性分子作为最终固定输出库打包进比赛镜像。

`data/benchmarks/benchmark_prior.json` 由 `app/training_code/build_benchmark_prior.py` 生成，只保存性质统计量，例如 MW、logP、QED、TPSA、SA 的 median/q1/q3。它用于给 agent 和 LLM 一个性质分布 prior，并参与 proxy search 的 `property_prior_score`；不保存也不输出 benchmark 分子字符串。

## 审计命令

```bash
python scripts/check_data.py
```

这个脚本会检查：

- 赛题说明文档是否存在
- 本地 target 是否存在
- benchmark manifest 中列出的文件是否真的落盘
- AiZynthFinder config 是否已经准备好
