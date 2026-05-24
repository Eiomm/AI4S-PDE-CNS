# AI4S智能体CNS挑战赛：神经算子 PDE 智能体赛题总结

> 更新时间：2026-05-22  
> 官网入口：https://competition.ai4s.com.cn/race/7/description  
> 说明：官网页面为动态渲染，公开检索到的正文信息有限。本总结综合了官网入口、公开赛道介绍，以及已整理过的赛题细则，目的是作为项目知识库/README/Agent Prompt 的基础材料。

---

## 1. 一句话总结

这个赛题不是单纯训练一个 PDE 预测模型，而是要求构建一个**自主科研 Agent**：它需要围绕神经算子模型，在 PDEBench/Burgers/KS 等标准科学数据上，完成“理解赛题 → 改进模型/损失/训练策略 → 自动实验验证 → 记录科研过程 → 生成合规提交物”的闭环。

更直白地说：

> 主办方想看的不是“你手工调出一个好 checkpoint”，而是“你的 Agent 是否能像一个小型科研工程师一样，自动做实验、自动迭代、自动解释为什么这么改”。

---

## 2. 赛题定位

| 维度 | 内容 |
|---|---|
| 大赛 | 第四届世界科学智能大赛 / AI4S 智能体 CNS 挑战赛 |
| 任务方向 | 神经算子自动改进 / 神经算子 PDE 智能体 |
| 核心对象 | 用神经算子建模复杂动态物理系统，重点是 PDE 时空预测 |
| 代表基线 | FNO、DeepONet、ICON；也可参考 PI-DeepONet 等物理约束神经算子/算子学习方法 |
| 数据基础 | PDEBench 标准科学数据集；初赛重点是 1D Burgers，附加题为 Kuramoto–Sivashinsky 方程 |
| 组织意图 | 推动神经算子从“人工调参的静态模型优化”走向“Agent 驱动的自主科研闭环” |

---

## 3. 主办方真正想考察什么

公开介绍里反复强调：参赛智能体需要具备完整科研闭环能力，而不是只输出一个模型权重。可以拆成四类能力：

1. **文献解析与逻辑解构**  
   Agent 要能理解论文、代码、数学公式、数据流、训练逻辑，而不是只会调用训练脚本。

2. **瓶颈诊断与假设提出**  
   Agent 要能从日志、指标、预测误差中判断问题，比如：短期误差大、长程漂移、统计量不稳、泛化到未知物理参数失败等。

3. **自主设计与代码演进**  
   Agent 要能提出并实现改进，比如改网络结构、改损失函数、改训练 schedule、加物理约束、修复 shape/schema 问题。

4. **实验验证与科学迭代**  
   Agent 要能跑实验、比较结果、判断是否晋级/停止/回滚，并把整个过程记录成可追溯日志。

---

## 4. 初赛核心任务概览

### Task-1：固定物理环境下的 1D Burgers 预测

| 项目 | 内容 |
|---|---|
| 任务类型 | 固定粘性系数下的 PDE 时空预测 |
| 数据文件 | `1D_Burgers_Sols_Nu0.001.hdf5` |
| 数据规模 | 约 10000 个样本，每个样本 200 个时间步，空间长度 1024 |
| 目标 | 在固定物理环境下做短期精准预测，并兼顾中长期统计稳定性 |
| 官方 checkpoint | `1D_Burgers_Sols_Nu0.001_FNO.pt`、`1D_Burgers_Sols_Nu0.001_Unet-PF-20.pt` |
| 常见降采样设置 | `reduced_resolution_t=5`、`reduced_resolution=4` |
| 直观含义 | 时间从 200 压到 40，空间从 1024 压到 256；模型的一步预测相当于真实物理时间上的 5 步 |

重点理解：

- Task-1 更像“先把闭环跑通”的任务。
- 适合先复现官方 FNO/UNet baseline，再做轻量改进。
- 不要一上来追求复杂架构，先保证数据读取、shape、预测文件、日志、时间统计全部合规。

---

### Task-2：多物理参数泛化

| 项目 | 内容 |
|---|---|
| 任务类型 | 多粘性系数 Burgers 泛化预测 |
| 训练集 | 包含多个 Nu/粘性系数，以及对应的 Nu 值 |
| 测试集 | 不提供 Nu，只给初始条件 |
| 难点 | 模型要从初始条件和早期演化中隐式识别物理参数，不能依赖测试集显式 Nu |
| 目标 | 在未知物理参数下保持预测精度和长程稳定性 |

Task-2 的关键不是“多训练几个 Nu 就完事”，而是：

- 测试阶段没有 Nu，所以模型不能把 Nu 当作必需输入；
- 可以考虑让模型从前若干帧中学习隐式参数表征；
- 可以考虑多分支模型、condition encoder、latent parameter inference、统计特征辅助等方向；
- 但任何方案都必须遵守禁止使用额外数据和禁止调用数值求解器的规则。

---

### Task-3：Kuramoto–Sivashinsky 方程附加题

| 项目 | 内容 |
|---|---|
| 任务系统 | Kuramoto–Sivashinsky, KS 方程 |
| 方程特性 | 非线性、强混沌、对初始条件敏感，长程预测极难 |
| 输入 | 前 20 步，shape 通常为 `(20, 256)` |
| 输出 | 完整 400 步，shape 通常为 `(400, 256)` |
| 测试规模 | 预测文件通常为 `(1000, 400, 256)` |
| 参数 | 训练/验证集含 `lambda2`，通常来自 `Uniform[1.0, 1.5]`；测试集不提供 `lambda2` |
| 硬约束 | 输出前 20 步必须和输入一致，容差约 `1e-3` |
| 预训练限制 | 不允许使用公开预训练权重，需要从提供的 KS 训练数据 scratch 训练 |

Task-3 的核心难点：

- KS 是混沌系统，点对点长期 MSE 很容易崩；
- 长程预测更应该关注统计结构、谱能量、稳定性、滚动误差，而不只是单步误差；
- 需要特别小心输出前 20 步原样拷贝，否则格式上直接吃亏；
- 测试集不提供 `lambda2`，所以也需要隐式参数识别能力。

---

## 5. 明确禁止/限制事项

| 规则 | 解释 |
|---|---|
| 禁止调用数值求解器 | 不能在测试时用传统 PDE solver 生成答案；否则可能导致任务零分 |
| 禁止使用额外数据 | 不能引入赛题外数据训练或增强 |
| Task-3 禁止公开预训练权重 | KS 附加题需要从官方训练数据 scratch 训练 |
| 测试集缺失物理参数 | Task-2/Task-3 测试阶段不能依赖显式 Nu 或 lambda2 |
| 推理时间限制 | 需要关注全量测试推理时间，例如 Task-2 可能要求非常短的推理时长 |
| 提交文件必须合规 | shape、HDF5 key、CSV 字段、日志 JSONL 都要严格检查 |

一句话：

> 不要靠“偷偷求解 PDE”拿分，也不要让 Agent 生成一堆看起来很厉害但无法提交的文件。

---

## 6. 提交物要求整理

建议最终目录统一为：

```text
submission/
├── submission.json
├── methodology.pdf
├── task1_pred.hdf5
├── task1_time.csv
├── task1_logs.log
├── task2_pred.hdf5
├── task2_time.csv
├── task2_logs.log
├── task3_pred.hdf5
├── task3_time.csv
├── task3_logs.log
└── code/
    ├── README.md
    ├── configs/
    ├── scripts/
    ├── models/
    ├── agent/
    └── ...
```

### 6.1 每个任务至少 3 类文件

| 文件 | 作用 |
|---|---|
| `task{N}_pred.hdf5` | 预测结果文件 |
| `task{N}_time.csv` | 记录训练时间和推理时间，一般包含 `train_time,inference_time` |
| `task{N}_logs.log` | Agent 科研日志，每行是合法 JSON |

### 6.2 额外全局文件

| 文件/目录 | 作用 |
|---|---|
| `submission.json` | 提交元信息，例如队伍名、problem_id、code_path |
| `methodology.pdf` | 方法说明文档 |
| `code/` | 代码目录，不能为空，并且应与日志中记录的 Agent 生成/修改过程对应 |

### 6.3 日志要求

`task{N}_logs.log` 建议使用 JSON Lines，每一行类似：

```json
{"timestamp":"2026-05-22T10:00:00+08:00","elapsed_seconds":12.3,"response":"...","tool_calls":[...]}
```

核心不是“随便写个 log”，而是要能证明：

- Agent 做了哪些决策；
- 为什么提出这个实验；
- 调用了哪些工具/脚本；
- 改了哪些代码；
- 实验结果如何；
- 为什么 promote/reject/repair/stop。

---

## 7. 预测文件 shape 重点

| 任务 | 预测 shape | 输入一致性约束 |
|---|---:|---|
| Task-1 | `(N, 200, 256)` | 前 10 步通常需要与输入保持一致 |
| Task-2 | `(N, 200, 256)` | 前 10 步通常需要与输入保持一致 |
| Task-3 | `(1000, 400, 256)` | 前 20 步必须与输入一致，容差约 `1e-3` |

提交前一定要做：

```text
1. HDF5 key 检查
2. dtype 检查，建议 float32
3. shape 检查
4. NaN/Inf 检查
5. 前若干输入步一致性检查
6. 推理时间统计检查
7. 文件命名检查
8. submission.json 检查
```

---

## 8. 评价维度理解

公开介绍中提到，挑战赛会从三个方向综合评估：

| 评价维度 | 对 PDE 任务的具体含义 |
|---|---|
| 科学性能 | 预测精度、泛化能力、长程稳定性、统计结构是否合理 |
| 探索效率与计算经济性 | Agent 是否少走弯路，训练/推理是否高效，是否避免无意义大实验 |
| 演进逻辑严密性 | Agent 的诊断、假设、代码修改、实验结论是否透明、可追溯、可信 |

所以这个赛题的得分逻辑大概率不是单一 leaderboard 分数，而是：

```text
最终模型效果 + 自动科研过程质量 + 资源效率 + 提交合规性
```

---

## 9. Baseline 为什么重要

你之前困惑“为什么官方给 baseline”。这里要分清：

| baseline 作用 | 解释 |
|---|---|
| 复现起点 | 先保证数据、训练、推理、提交链路能跑通 |
| 对照组 | 后续所有改进都要证明比 baseline 好，不能凭感觉说有效 |
| Agent 参考对象 | Agent 可以基于 baseline 诊断瓶颈，比如长程漂移、过平滑、边界误差 |
| 合规锚点 | 官方 checkpoint/官方代码更容易保证输入输出格式正确 |

重点：

> baseline 不是限制你只能用它，而是给你一个最低可复现参照系。真正的比赛是让 Agent 在合规范围内自动改进它。

---

## 10. 对我们项目的推荐实现思路

### 10.1 不要一开始追求“大而全 Agent”

建议先做一个简单但稳定的闭环：

```text
赛题规则读取
→ baseline 复现
→ preflight 检查
→ 小步训练
→ 评估
→ 生成结构化实验报告
→ 根据规则 promote/reject/repair
→ 写入 task_logs.log
→ 生成提交文件
```

Agent 的价值不在于“每一步都自由发挥”，而在于：

- 能遵守 hard constraints；
- 能自动记录；
- 能基于指标做保守决策；
- 能避免无限调参和无效实验。

---

### 10.2 推荐实验状态机

```text
proposal
  ↓
preflight
  ↓
short_train
  ↓
evaluate
  ↓
promote / reject / repair / stop
  ↓
memory
```

每个状态都要有明确输入输出：

| 状态 | 作用 |
|---|---|
| `proposal` | Agent 提出一个原子实验假设 |
| `preflight` | import/shape/forward/backward/1-batch speed 检查 |
| `short_train` | 小步训练，避免一上来烧满资源 |
| `evaluate` | 统一指标评估，和 baseline 对比 |
| `promote` | 效果好，进入更长训练或提交候选 |
| `reject` | 效果差，归档并停止 |
| `repair` | 代码或 schema 问题，进入修复 |
| `memory` | 把经验写入知识库，供下一轮使用 |

---

### 10.3 推荐代码模块

```text
project/
├── configs/
│   ├── competition_contract.yaml      # 赛题硬规则
│   ├── task1.yaml
│   ├── task2.yaml
│   └── task3.yaml
├── agent/
│   ├── prompts/                       # system/task/action schema prompt
│   ├── planner.py                     # 生成实验计划
│   ├── executor.py                    # 执行训练/评估/修复
│   ├── verifier.py                    # 中立检查输出、代码、指标
│   └── memory.py                      # 实验记忆
├── data/
├── models/
├── scripts/
│   ├── train_task1.py
│   ├── infer_task1.py
│   ├── evaluate_task1.py
│   └── make_submission.py
├── runs/
│   └── task1/exp_xxx/
├── submission/
└── README.md
```

---

## 11. 最容易踩坑的地方

| 坑 | 后果 | 建议 |
|---|---|---|
| HDF5 读取太慢 | GPU 等数据，训练效率极差 | 缓存文件句柄、预切片、DataLoader 优化 |
| Agent 输出 JSON 不合 schema | 执行器无法解析 | system prompt 强约束 + schema validator |
| 只看短期 MSE | 长程预测崩掉 | 加入 long rollout、统计量、谱指标 |
| Task-2 依赖 Nu 输入 | 测试时无 Nu，直接失效 | 学隐式参数，不要把 Nu 作为唯一条件 |
| Task-3 前 20 步没复制输入 | 格式硬伤 | 预测后强制 overwrite 前 20 步 |
| 日志不可追溯 | methodology 和 code 对不上 | 所有 Agent 决策、工具调用、文件修改写 JSONL |
| 手工改代码太多 | 不符合 Agent 生成历史 | 用 patch/action 记录修改过程 |
| 一上来跑大实验 | 资源浪费，debug 慢 | 先 preflight + short_train |

---

## 12. 最小可行方案 MVP

如果现在要快速推进，建议顺序是：

1. **先锁死提交规范**  
   写 `validate_submission.py`，检查 HDF5 shape、CSV 字段、log JSONL、submission.json。

2. **先复现 Task-1 baseline**  
   让 FNO/UNet 官方 checkpoint 或官方结构跑出一个合法 `task1_pred.hdf5`。

3. **做统一评估脚本**  
   指标先简单：短期 MSE、长程 MSE、rollout 稳定性、NaN 检查、速度。

4. **做 Agent 日志代理**  
   所有 LLM 请求和回复写入 `task{N}_logs.log`，每行合法 JSON。

5. **再加 Agent planner**  
   让 Agent 每轮只提出一个原子实验，不要让它一次改十个地方。

6. **最后再做模型创新**  
   例如 physics loss、spectral loss、multi-step rollout loss、latent parameter encoder、lightweight FNO 改进等。

---

## 13. 可以写进 methodology.pdf 的核心表述

可以这样概括你的方案：

> 我们构建了一个面向神经算子 PDE 预测任务的自主科研 Agent。系统将赛题硬规则、任务输入输出规范、模型训练脚本、评估器、提交检查器和实验记忆统一到一个闭环中。Agent 每轮生成一个原子实验假设，经 preflight 检查后执行短训与评估，并根据统一 leaderboard 决定 promote、reject 或 repair。整个过程以 JSONL 形式记录，保证模型改进、实验结果与最终提交文件之间可追溯。模型侧以 FNO/UNet/DeepONet 等 baseline 为起点，重点优化多步 rollout 稳定性、物理一致性和未知物理参数泛化能力。

---

## 14. 我对赛题的最终判断

这个赛题表面上是 PDE/神经算子预测，实际上更像是：

```text
科学机器学习任务
+ 自动实验工程系统
+ Agent 规划与日志治理
+ 合规提交工程
```

所以你的优先级应该是：

```text
提交规范 > baseline 复现 > 评估闭环 > 日志可追溯 > Agent 小步自动迭代 > 模型创新
```

不要一开始就陷入“我要发明一个全新神经算子”的焦虑。对初赛来说，更现实的高分路线是：

> 用稳定工程把 baseline 跑扎实，再让 Agent 围绕少数可靠改进做自动实验，并把过程记录得漂亮、可信、可复现。
