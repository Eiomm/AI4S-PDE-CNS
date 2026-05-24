# Task2 MCGS-lite 交接文档

> 交接时间: 2026-05-24  
> 当前状态: 系统骨架完整，LLM 搜索链路已通，最佳节点约 79/150 分

---

## 1. 已完成的修改

### 核心文件修改

| 文件 | 改动内容 |
|------|---------|
| `mlevolve_lite/operators.py` | Seed bootstrap + LLM 调用 + clip 防发散 + checkpoint 保存 + `task2_logs.jsonl` 记录 |
| `mlevolve_lite/scheduler.py` | 主控闭环 + `--use-llm` / `--auto-full-train` / `--llm-model` CLI 参数 |
| `mlevolve_lite/evaluator.py` | 150 分制官方评分估计 + checkpoint 扫描 + static/shape/compliance 检查 |
| `mlevolve_lite/llm_backend.py` | OpenAI-compatible 后端（DeepSeek / gpt.ge） |
| `mlevolve_lite/code_extractor.py` | Markdown 代码块提取 |
| `mlevolve_lite/memory.py` | Prompt 组装（含 parent metrics） |
| `mlevolve_lite/prompts/improve.md` | 强制 seg3 改进 + nu 条件 + residual prediction |
| `configs/agent_gpt55.yaml` | GPT-5.5 配置 |
| `tests/test_mlevolve_lite.py` | 4 个单元测试 |

### 关键发现

1. **clip 是最关键的改进**: `torch.clamp(out, -3.5, 3.5)` 让 MSE 从 29.74 → 0.435
2. **DeepSeek-chat > GPT-5.5**: DeepSeek 会生成 FNO + multi-step loss，GPT-5.5 偏保守（但支持 `reasoning_content`）
3. **nu 条件输入有效但不够强**: NuEstimator + concat 能降低部分 nu 的 MSE，但 worst_nu_mse 仍在 3.5-4.5
4. **seg3 仍是最大短板**: 长程 105-200 步 rel_mse 约 1.2-1.4，丢 20-30/50 分
5. **DataLoader num_workers 死锁**: 600k 样本 + num_workers=4 会导致训练卡住，已改回 num_workers=0

---

## 2. 历史最佳节点

### 节点 `improve_9ca86ae2`（5轮搜索第2轮，已删除）
- **MSE**: 0.293
- **官方估计**: ~71/150（旧100分制）
- **架构**: FNO + multi-step rollout loss
- **位置**: 原 workspace 已删除，代码未保存

### 节点 `improve_931de6bd`（当前最佳，可用）
- **MSE**: 0.435
- **官方估计**: ~79/150
- **架构**: FNO + NuEstimator + concat conditioning
- **位置**: `workspace/nodes/improve_931de6bd/`

### 节点 `improve_cd12051f`（GPT-5.5 生成）
- **MSE**: 0.347
- **架构**: CNN（无 FNO）

---

## 3. 已知问题

### P0: full train 训练卡住/极慢
- **现象**: GPU 利用率 4%，20 epochs 预计 10-20 分钟，但日志无输出
- **根因**: 模型太小（hidden=48, 4 blocks）+ DataLoader 单线程
- **临时方案**: 去掉 num_workers，接受慢速训练
- **根本方案**: 增大模型（hidden=128, modes=32, 6-8 blocks）+ batch_size=128/256

### P1: 150 分制评分是近似公式
- 基于猜测的分段指数衰减 + Lorentzian/Fréchet 近似
- 真实官方公式未知，可能有 ±15 分偏差

### P2: 没有 tool 调用
- LLM 是单轮问答模式，不能读取数据、运行测试、查看 metrics
- 需要时可在 `llm_backend.py` 中加入 function calling

### P3: task2_logs.log 的双重含义
- 官方要求 `task2_logs.log` 记录 LLM 调用历史
- 当前 `train.py` 也生成同名的训练日志
- 已解决: 主控系统写入 `workspace/task2_logs.jsonl` 记录 LLM 调用

---

## 4. 常用命令

```bash
# 运行单元测试
cd /root/autodl-tmp/AI4Sv2/Task2
python -m pytest tests/test_mlevolve_lite.py -v

# 1轮 cheap probe 搜索（DeepSeek）
cd /root/autodl-tmp/AI4Sv2/Task1 && source .env && cd /root/autodl-tmp/AI4Sv2/Task2
python -m mlevolve_lite.scheduler \
  --rounds 1 --cheap-epochs 1 --timeout-sec 1800 \
  --use-llm --llm-model deepseek-chat

# 5轮搜索
python -m mlevolve_lite.scheduler \
  --rounds 5 --cheap-epochs 1 --timeout-sec 1800 \
  --use-llm --llm-model deepseek-chat

# GPT-5.5 测试
python -m mlevolve_lite.scheduler \
  --rounds 1 --cheap-epochs 1 --timeout-sec 1800 \
  --use-llm --llm-model gpt-5.5 --llm-base-url https://api.gpt.ge/v1

# 手动运行某个节点的 train.py
python workspace/nodes/<node_id>/code/train.py \
  --data-dir data/Task2 \
  --out-dir workspace/nodes/<node_id>/artifacts_full \
  --epochs 20 --batch-size 32 --lr 1e-3
```

---

## 5. 下一步优先级（推荐）

| 优先级 | 任务 | 预期收益 |
|--------|------|---------|
| P0 | **修复 full train 速度** — 增大模型 + batch_size | 训练从 20min → 5min，GPU 利用率 4% → 60% |
| P1 | **让 LLM 生成更大的模型** — prompt 中明确要求 hidden≥128, modes≥32 | 分数可能 79 → 100+ |
| P2 | **跑 10 轮搜索筛选最佳** | 找到历史最优架构 |
| P3 | **对最佳节点 full train (20-50 epochs)** | 释放模型潜力 |
| P4 | **加 time embedding / residual prediction** | 重点改进 seg3 |
| P5 | **精确官方评分公式** | 需要查找或逆向工程 |

---

## 6. 提交前检查清单

- [ ] `task2_pred.hdf5`: shape (1000, 200, 256), float32, finite
- [ ] `task2_pred.hdf5/tensor[:, :10, :]`: 与 test 输入误差 ≤ 1e-3
- [ ] `task2_time.csv`: train_time, inference_time
- [ ] `task2_logs.log`: JSONL，含 timestamp + elapsed_seconds + response/tool_calls
- [ ] `code/train.py`: 可由 LLM 调用历史追溯
- [ ] checkpoint 扫描: 无外部预训练权重
- [ ] 推理时间 < 2 分钟
- [ ] 无 test nu 读取
- [ ] 无数值求解器

---

## 7. 关键文件位置

```
/root/autodl-tmp/AI4Sv2/Task2/
├── mlevolve_lite/          # 主控系统
├── workspace/              # 搜索产物
│   ├── graph.json
│   ├── leaderboard.json
│   ├── promoted.json
│   ├── task2_logs.jsonl    # LLM 调用日志（提交证据）
│   └── nodes/
│       └── improve_931de6bd/  # 当前最佳节点
├── configs/agent_gpt55.yaml
└── HANDOVER.md             # 本文件
```

---

## 8. API 密钥

```bash
cd /root/autodl-tmp/AI4Sv2/Task1
source .env
# DEEPSEEK_API_KEY, VAPI_API_KEY, SILICONFLOW_API_KEY 等
```

---

*交接完成。系统已可运行，核心瓶颈是模型规模太小导致 GPU 利用率低和 seg3 长程预测弱。*
