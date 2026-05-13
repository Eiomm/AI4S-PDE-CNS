# AI4S-PDE-CNS 变更记录

本文档用于按固定格式记录代码变更、实验命令、验证结果和提交决策。后续每次涉及代码或实验流程的修改，都应追加到本文档。

## 2026-05-13：DeepONetLite 后段窗口训练

### 变更摘要

| 字段 | 内容 |
| --- | --- |
| 变更编号 | `tail-window-deeponet-lite-20260513` |
| 任务 | Task 1 |
| 目标 | 将 `DeepONetLite` 训练为 `t >= 120` 的长时段专家模型，并通过 validation-only temporal tail blending 与当前 FNO ensemble 融合。 |
| 是否生成提交包 | 否 |
| 当前保留提交包 | `runs/task1-finetune-nu0.1-short-proxy-final/pred.zip` |
| 本次变更后的最佳验证结果 | `runs/task1-zoo-tail120-deeponet-v2/temporal_tail_blend_deeponet_lite` |

### 代码变更

| 文件 | 变更 | 目的 |
| --- | --- | --- |
| `agent/task1_baseline_train.py` | 新增 `normalize_loss_window(output_steps, loss_start_step, loss_end_step)`。 | 在训练开始前校验监督训练时间窗口。 |
| `agent/task1_baseline_train.py` | 为 `train_task1_baseline` 新增 `loss_start_step` 和 `loss_end_step` 参数。 | 支持只在指定预测时间段上计算训练损失。 |
| `agent/task1_baseline_train.py` | 将监督损失范围从 `prediction[:, 10:, :]` 改为 `prediction[:, loss_start:loss_end, :]`。 | 支持长时段专家训练，同时保持完整预测输出 shape 不变。 |
| `agent/run_task1_baseline_zoo.py` | 新增 CLI 参数 `--loss-start-step` 和 `--loss-end-step`。 | 通过 Baseline Zoo runner 暴露后段窗口训练能力。 |
| `tests/test_task1_baseline_zoo_cli.py` | 新增 loss window 校验和配置记录测试。 | 确认窗口配置被记录，非法窗口会失败。 |
| `README.md` | 新增 DeepONetLite 后段专家复现实验命令。 | 方便在本机或远程 GPU 服务器复现。 |
| `docs/results/task1_experiment_results.md` | 更新实验结果排序表。 | 记录新的 validation best 结果。 |

### 实验命令

#### 失败变体：v1

```powershell
D:\Junao\ProgramData\anaconda3\envs\Hwpytorch\python.exe -m agent.run_task1_baseline_zoo ^
  --study-name task1-zoo-tail120-deeponet-v1 ^
  --models fno_ensemble,deeponet_lite ^
  --max-samples 12000 ^
  --steps 5000 ^
  --batch-size 4 ^
  --lr 0.001 ^
  --hidden 128 ^
  --device cuda ^
  --loss-start-step 120 ^
  --loss-end-step 200 ^
  --checkpoint-override nu0.1=runs\task1-finetune-nu0.1-lr3e-6-short-proxy\best.pt ^
  --fno-weight nu0.001=0.0 ^
  --fno-weight nu0.01=0.085 ^
  --fno-weight nu0.1=0.915 ^
  --fno-weight nu1.0=0.0
```

#### 提升变体：v2

```powershell
D:\Junao\ProgramData\anaconda3\envs\Hwpytorch\python.exe -m agent.run_task1_baseline_zoo ^
  --study-name task1-zoo-tail120-deeponet-v2 ^
  --models fno_ensemble,deeponet_lite ^
  --max-samples 12000 ^
  --steps 8000 ^
  --batch-size 4 ^
  --lr 0.0003 ^
  --hidden 96 ^
  --device cuda ^
  --loss-start-step 120 ^
  --loss-end-step 200 ^
  --checkpoint-override nu0.1=runs\task1-finetune-nu0.1-lr3e-6-short-proxy\best.pt ^
  --fno-weight nu0.001=0.0 ^
  --fno-weight nu0.01=0.085 ^
  --fno-weight nu0.1=0.915 ^
  --fno-weight nu1.0=0.0
```

#### 组合搜索

```powershell
D:\Junao\ProgramData\anaconda3\envs\Hwpytorch\python.exe -m agent.run_task1_combo_search ^
  --study-dir runs\task1-zoo-tail120-deeponet-v2 ^
  --target data\Task1\task1_val.hdf5
```

#### 结果台账更新

```powershell
python -m agent.update_experiment_results ^
  --runs-root runs ^
  --output docs\results\task1_experiment_results.md ^
  --csv runs\experiment_results_summary.csv ^
  --json runs\experiment_results_summary.json
```

### 验证结果

| Run | 模型或融合方式 | Proxy | MSE | Forecast MSE | Long-Horizon MSE | Segment3 RMSE | 决策 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `task1-zoo-tail120-deeponet-v1` | `temporal_tail_blend_deeponet_lite` | 58.75300740 | 0.001548390245 | 0.001629884469 | 0.000851877602 | 0.02918694232 | 不采用 |
| `task1-zoo-tail120-deeponet-v2` | `temporal_tail_blend_deeponet_lite` | 59.06335202 | 0.001520397756 | 0.001600418691 | 0.000792946046 | 0.02815929768 | 作为当前 validation best |
| `task1-zoo-medium-finetuned-fno` | `temporal_tail_blend_deeponet_lite` | 58.99420380 | 0.001526509685 | 0.001606852300 | 0.000805813266 | 0.02838685022 | 上一版 validation best |
| `current-final-proxy` | FNO ensemble baseline | 58.18577979 | 0.001603422098 | 0.001687812735 | 0.000967734135 | 0.03110842547 | fallback baseline |

### 选定融合配置

| 字段 | 内容 |
| --- | --- |
| 融合类型 | `temporal_tail_blend` |
| 基座模型 | `fno_ensemble` |
| 后段模型 | `deeponet_lite` |
| 切换时间步 | `105` |
| 后段权重 | `0.55` |
| 输出文件 | `runs/task1-zoo-tail120-deeponet-v2/temporal_tail_blend_deeponet_lite/task1_val_pred.hdf5` |

### 验证记录

| 检查项 | 命令 | 结果 |
| --- | --- | --- |
| 单元测试和集成测试 | `python -m pytest -q` | `105 passed` |
| 组合搜索输出 | `agent.run_task1_combo_search` | `runs/task1-zoo-tail120-deeponet-v2/combo_search_summary.json` |
| 结果台账 | `agent.update_experiment_results` | `docs/results/task1_experiment_results.md` 已更新 |
| Git 记录 | `git log -1 --oneline` | `6cd7776 feat: add tail-window baseline training` |

### 结论

| 项目 | 结论 |
| --- | --- |
| 后段窗口训练是否有效 | 在更保守的 `v2` 设置下有效。 |
| 主要原因 | 较低学习率和较小 hidden size 比激进的 `v1` 设置更适合当前 DeepONetLite 后段专家训练。 |
| 提交状态 | 本次实验没有生成新的 `pred.zip`。 |
| 下一步建议 | 在更强服务器上使用更大样本数和多个随机种子复跑该策略，并继续通过相同 combo-search gate 验证。 |
