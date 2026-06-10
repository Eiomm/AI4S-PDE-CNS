# AI4S 小分子生成任务说明

这份项目的目标很朴素：给定一个蛋白结构 `target.pdb`，自动生成一批可能能结合这个靶点的小分子，并把结果打包成比赛需要的提交文件。

## 输入是什么

- 官方给的靶点结构：`target.pdb`
- 文件里主要是蛋白原子坐标，没有现成配体
- 当前最小链路会用蛋白整体坐标估一个兜底 pocket，后续可以替换成更专业的口袋预测或 docking 准备流程

## 输出是什么

比赛需要的核心输出是一个 zip：

- `result.csv`：两列，`mol_smiles` 和 `route`
- `result.log`：运行过程日志
- `result.zip`：包含上面两个文件

为了本地查看更直观，脚本还会额外复制三份简单名字：

- `candidates.csv`：候选分子结果
- `pipeline.log`：完整流水线事件日志
- `submission.zip`：可以检查或提交的压缩包别名

zip 里面仍然保留比赛要求的 `result.csv` 和 `result.log`，所以不会破坏提交格式。

复赛容器里的最终提交格式是三靶点版本：

```text
/saisdata/37/target1.pdb -> result1.csv
/saisdata/37/target2.pdb -> result2.csv
/saisdata/37/target3.pdb -> result3.csv
/saisresult/result.zip   -> 包含 result1.csv, result2.csv, result3.csv
```

本地排序使用复赛权重：

```text
total = 0.60 * molecule_score + 0.40 * route_score
```

## 当前最小链路

当前实现的是一个能跑通的最小闭环：

1. 读取 `target.pdb`
2. 从蛋白坐标估计 pocket 中心和盒子大小
3. 用种子分子、片段拼接、突变和可选 LiteLLM 生成 SMILES
4. 用 RDKit 过滤非法 SMILES 并统一 canonical SMILES
5. 用轻量代理评分排序
6. 生成兜底 retrosynthesis route，保证 route 最终产物和 SMILES 一致
7. 可选尝试 docking；没有 Vina 或 PDBQT 时会记录原因并继续
8. 写出 `result.csv/log/zip`
9. 用检查脚本验证 zip 格式和 route 一致性

这条链路不是最终高分方案，但它有一个重要作用：保证工程从输入到输出是闭合的，之后每次改生成器、评分器、docking 或 LLM 提示词，都能马上测试有没有把提交格式弄坏。

## 一键运行

```bash
conda activate ai4s-chem-evolve
bash scripts/run_harness_once.sh --name official --target target.pdb
```

结果会进入：

```text
outputs/official/<timestamp>/
outputs/official/latest -> 最新一次运行
```

## 后台 tmux 运行

```bash
bash scripts/run_tmux_training.sh official
bash scripts/run_tmux_training.sh official --status
tmux attach -t ai4s-chem-official
```

如果要跑更重一点的参数：

```bash
AGENT_ROUNDS=8 AGENT_PER_ROUND=64 AGENT_TOP_K=20 bash scripts/run_tmux_training.sh final
```

## 后续增强方向

- 用更可靠的 pocket 检测替换当前 centroid fallback
- 准备 receptor PDBQT，让 Vina docking 真正参与排序
- 增加 LLM 反思生成策略，把失败分子和高分片段写入 memory
- 接入更严格的合成可行性工具
- 多目标时沿用 `result1.csv/result2.csv/result3.csv` 的比赛格式
