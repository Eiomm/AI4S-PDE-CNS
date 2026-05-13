# Task 1 根源改进实验记录

## 目标

从赛题根源重新发散，不把已有 FNO ensemble 或 DeepONetLite tail 结果视为绝对中心。优先验证是否存在比继续微调更根本的提升方向。

## 改进点列表

| 编号 | 改进点 | 核心假设 | 当前状态 |
| --- | --- | --- | --- |
| 1 | PDEBench Analog / kNN Forecasting | 如果测试初值能在 PDEBench 全量轨迹中找到相似历史轨迹，直接复用相似轨迹未来段可能优于神经网络 rollout。 | 已实验，预测本身较弱，不进入融合。 |
| 2 | `nu` 识别与分池 | 前 10 帧可估计 Burgers 黏性系数 `nu`，再按 `nu` 选择模型、数据池或融合权重。 | 已做初步估计，可用于诊断，但直接分池 analog 无收益。 |
| 3 | Time-dependent blending | 不使用固定全局权重，而是按时间段动态融合 FNO、tail expert、retrieval 结果。 | 已有 `temporal_tail_blend`，当前最有效。 |
| 4 | TFNO / PINO 主模型增强 | 当前最大瓶颈仍是主模型整体预测能力，需训练更强 neural operator，再使用物理残差约束降低长时漂移。 | 待服务器实验。 |

## 实验 1：Analog / kNN Forecasting

### 实现变更

| 文件 | 变更 |
| --- | --- |
| `agent/task1_analog_forecast.py` | 新增 Analog/kNN validation runner。 |
| `agent/task1_analog_forecast.py` | 支持扫描 PDEBench raw HDF5，按前 10 帧找 top-k 相似轨迹。 |
| `agent/task1_analog_forecast.py` | 支持 `feature_mode=initial` 和 `feature_mode=initial_gradient`。 |
| `agent/task1_analog_forecast.py` | 新增 `estimate_burgers_nu`，用于从前 10 帧估计 `nu`。 |
| `tests/test_task1_analog_forecast.py` | 新增 kNN 精确匹配测试、`nu` 估计稳定性测试、特征振幅保留测试。 |

### 关键修正

| 问题 | 原因 | 修正 |
| --- | --- | --- |
| 首轮 analog 结果极差 | 逐样本标准化抹掉了均值和振幅信息。 | 默认关闭 `normalize_per_sample`，保留振幅信息。 |
| 混合 `nu` 检索误选严重 | 初值相似不等于未来动力学一致。 | 增加 `nu` 估计和单 `nu` 数据池验证。 |

### 实验命令

#### 混合数据池：top1，1000 candidates/file，逐样本归一化

```powershell
D:\Junao\ProgramData\anaconda3\envs\Hwpytorch\python.exe -m agent.task1_analog_forecast ^
  --run-dir runs\task1-root-analog-v1\analog_top1_1000 ^
  --target data\Task1\task1_val.hdf5 ^
  --top-k 1 ^
  --max-candidates-per-file 1000 ^
  --feature-mode initial_gradient ^
  --normalize-per-sample
```

#### 混合数据池：top1，1000 candidates/file，保留振幅

```powershell
D:\Junao\ProgramData\anaconda3\envs\Hwpytorch\python.exe -m agent.task1_analog_forecast ^
  --run-dir runs\task1-root-analog-v1\analog_top1_1000_raw ^
  --target data\Task1\task1_val.hdf5 ^
  --top-k 1 ^
  --max-candidates-per-file 1000 ^
  --feature-mode initial_gradient
```

#### 混合数据池：top3，5000 candidates/file，初值特征

```powershell
D:\Junao\ProgramData\anaconda3\envs\Hwpytorch\python.exe -m agent.task1_analog_forecast ^
  --run-dir runs\task1-root-analog-v1\analog_top3_5000_initial ^
  --target data\Task1\task1_val.hdf5 ^
  --top-k 3 ^
  --max-candidates-per-file 5000 ^
  --feature-mode initial
```

#### `Nu0.001` 单数据池：top3，全量 10000 candidates

```powershell
D:\Junao\ProgramData\anaconda3\envs\Hwpytorch\python.exe -m agent.task1_analog_forecast ^
  --run-dir runs\task1-root-analog-v1\analog_top3_nu0001_full_initial ^
  --target data\Task1\task1_val.hdf5 ^
  --raw-hdf5 data\pdebench_burgers\raw\1D_Burgers_Sols_Nu0.001.hdf5 ^
  --top-k 3 ^
  --max-candidates-per-file 10000 ^
  --feature-mode initial
```

## 实验结果

| Run | Proxy | MSE | Forecast MSE | Long-Horizon MSE | Segment3 RMSE | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `analog_top1_1000` | 9.96103728 | 0.1674317803 | 0.1762439792 | 0.1615684912 | 0.4019558324 | 逐样本归一化不可用。 |
| `analog_top1_1000_raw` | 27.59695419 | 0.0123919857 | 0.0130441955 | 0.0068514989 | 0.0827737817 | 保留振幅后明显改善，但仍弱。 |
| `analog_top3_5000_initial` | 31.71952504 | 0.0103329678 | 0.0108768082 | 0.0036211976 | 0.0601763875 | 当前 analog 最好，但远弱于 FNO。 |
| `analog_top3_5000_initial_gradient` | 31.71512926 | 0.0103361717 | 0.0108801807 | 0.0036235300 | 0.0601957636 | 梯度特征未带来额外收益。 |
| `analog_top3_nu0001_full_initial` | 26.75542841 | 0.0204453987 | 0.0215214723 | 0.0075630652 | 0.0869658854 | 单 `Nu0.001` 数据池变差。 |
| `analog_top1_nu0001_full_initial` | 23.06969169 | 0.0285080911 | 0.0300085169 | 0.0136289563 | 0.1167431211 | 单邻居更差。 |

## `nu` 估计观察

| 数据 | 估计结果 |
| --- | --- |
| `task1_val` | 估计值集中在约 `0.001` 附近。 |
| raw `Nu0.001` | 估计均值约 `0.00103`。 |
| raw `Nu0.01` | 估计均值约 `0.00344`。 |
| raw `Nu0.1` | 估计均值约 `0.0319`。 |
| raw `Nu1.0` | 估计均值约 `0.41`。 |

说明：该估计器可以区分数量级，但高 `nu` 存在系统性缩放偏差。因此它适合做诊断或 gating feature，不适合直接作为唯一分池规则。

## 融合验证

### FNO + Analog 权重扫描

| Analog Weight | Proxy | MSE | Long-Horizon MSE | Segment3 RMSE |
| ---: | ---: | ---: | ---: | ---: |
| 0.00 | 58.18577979 | 0.0016034221 | 0.0009677341 | 0.0311084255 |
| 0.05 | 57.76312752 | 0.0016410049 | 0.0009533879 | 0.0308769804 |
| 0.10 | 56.82680542 | 0.0017205766 | 0.0009545174 | 0.0308952652 |
| 0.15 | 55.45530012 | 0.0018421372 | 0.0009711226 | 0.0311628408 |
| 0.20 | 53.75014335 | 0.0020056867 | 0.0010032036 | 0.0316733895 |

结论：analog 预测不应以正权重进入当前融合；最佳 analog 权重为 `0.0`。

## 当前结论

| 改进点 | 当前判断 |
| --- | --- |
| Analog/kNN 直接预测 | 不采用。结果远弱于 FNO，且融合权重扫描显示正权重会降低 proxy。 |
| Analog/kNN 特征 | 保留。可作为后续 gating 的候选特征来源。 |
| `nu` 估计 | 保留。适合作为诊断、分组或模型选择特征，但不能单独决定数据池。 |
| 继续大规模检索 | 暂缓。除非增加更强的形状对齐、时间位移对齐或 learned embedding，否则全量检索收益有限。 |
| 下一步优先级 | 转向 TFNO/PINO 主模型增强，继续保留 `DeepONetLite` tail specialist。 |

## 验证记录

| 检查项 | 命令 | 结果 |
| --- | --- | --- |
| Analog 单元测试 | `python -m pytest tests/test_task1_analog_forecast.py -q` | `3 passed` |
| 结果台账更新 | `python -m agent.update_experiment_results ...` | `docs/results/task1_experiment_results.md` 已更新 |
