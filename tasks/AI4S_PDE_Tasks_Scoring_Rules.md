# AI4S PDE Tasks Scoring Rules

本文件只保留评分相关内容，供 Agent / prompt / 本地 evaluator 参考。

## 1. Task 1 评分（最高 150 分）

Task 1 总分由三部分组成：

```text
Task1_total = prediction_score * 0.75 + training_time_score + inference_time_score
```

其中：

- `prediction_score`：Task 1/2 通用分段预测分，范围 0-100。
- `training_time_score`：训练耗时分，最高 35。
- `inference_time_score`：推理耗时分，最高 40。

### 1.1 训练耗时分

训练耗时包含 Agent 思考时间。

| 训练耗时 | 分数 |
|---:|---:|
| <= 60 min | 35 |
| <= 120 min | 25 |
| <= 300 min | 20 |
| <= 500 min | 10 |
| > 500 min | 0 |

### 1.2 推理耗时分

推理耗时指完整测试集 rollout 时间。

| 推理耗时 | 分数 |
|---:|---:|
| 0 min | 40 |
| 0-2 min | 从 40 线性递减到 0 |
| > 2 min | 该任务得 0 分 |

线性段可写作：

```text
inference_time_score = 40 * max(0, 1 - inference_seconds / 120)
```

如果 `inference_seconds > 120`，Task 1 该任务直接为 0。

## 2. Task 2 评分（最高 150 分）

Task 2 只按预测精度计分：

```text
Task2_total = prediction_score * 1.5
```

其中 `prediction_score` 使用 Task 1/2 通用分段预测分，范围 0-100。

约束：

- 训练时间不计入分数，但总时长需控制在 12 小时以内。
- 如果推理时间超过 2 分钟，则 Task 2 总分为 0。

## 3. Task 1/2 通用分段预测分

Task 1/2 都预测完整 200 步轨迹，其中前 10 步是输入条件，不作为预测评分主体。评分只针对后 190 个预测时间步：

```text
future = tensor[:, 10:200, :]   # shape: (N, 190, 256)
```

190 个预测步分为三段：

| 段 | 未来窗口索引 | 原始轨迹索引 | 权重 | 评分公式 |
|---|---:|---:|---:|---|
| 第 1 段 | 0-47 | 10-56 | 25% | `100 * exp(-20 * Rel-MSE_1)` |
| 第 2 段 | 47-95 | 57-104 | 25% | `100 * exp(-10 * Rel-MSE_2)` |
| 第 3 段 | 95-190 | 105-199 | 50% | `max(Lorentzian, Frechet)` |

### 3.1 Rel-MSE

```text
Rel-MSE = sum((pred - true)^2) / (sum(true^2) + eps)
```

建议本地实现使用小常数：

```text
eps = 1e-8 或 1e-12
```

### 3.2 第 1 段

```text
Rel-MSE_1 = Rel-MSE(pred[:, 10:57, :], true[:, 10:57, :])
score1    = 100 * exp(-20 * Rel-MSE_1)
```

### 3.3 第 2 段

```text
Rel-MSE_2 = Rel-MSE(pred[:, 57:105, :], true[:, 57:105, :])
score2    = 100 * exp(-10 * Rel-MSE_2)
```

### 3.4 第 3 段

第 3 段使用长期预测评分：

```text
segment3_pred = pred[:, 105:200, :]
segment3_true = true[:, 105:200, :]

RMSE_3     = sqrt(mean((segment3_pred - segment3_true)^2))
Lorentzian = 100 / (1 + 10 * RMSE_3)
Frechet    = 50 * exp(-(FD ** 2))
score3     = max(Lorentzian, Frechet)
```

其中 `FD` 是预测轨迹与真实轨迹统计分布的 Frechet distance。官方具体 FD 实现未在本地代码中给出时，本地 evaluator 可以先使用 `Lorentzian` 作为确定性 proxy，但文档和 prompt 必须保留完整官方公式：

```text
score3 = max(Lorentzian, Frechet)
```

### 3.5 Task 1/2 分段预测总分

```text
prediction_score = 0.25 * score1 + 0.25 * score2 + 0.50 * score3
```

范围：

```text
0 <= prediction_score <= 100
```

## 4. Task 3 评分（附加题，满分 350 分体系）

Task 3 本身使用专用分段预测分。训练时间不计入评分，但需控制在 12 小时以内。推理时间超过 2 分钟则 Task 3 得分为 0。

提交 Task 3 时，总分按两种方案分别计算，取较高者作为最终总分。

### 4.1 方案 A：Task 1 + Task 2 + Task 3

```text
Total_A = Task1_total + Task2_total + Task3_prediction_score * 0.5
```

其中：

```text
Task1_total <= 150
Task2_total <= 150
Task3_prediction_score * 0.5 <= 50
Total_A <= 350
```

方案 A 需要同时提交 Task 1、Task 2、Task 3。

### 4.2 方案 B：Task 1 + Task 3

```text
Total_B = Task1_total + Task3_prediction_score * 2
```

其中：

```text
Task1_total <= 150
Task3_prediction_score * 2 <= 200
Total_B <= 350
```

方案 B 只需要 Task 1 和 Task 3。若三项均已提交，则方案 A/B 都适用，最终取较高者。

## 5. Task 3 专用分段预测分

Task 3 给定前 20 个观测步，预测完整 400 步轨迹。评分只针对后 380 个预测时间步：

```text
future = tensor[:, 20:400, :]   # shape: (N, 380, 256)
```

380 个预测步分为三段：

| 段 | 原始轨迹步范围 | 未来窗口索引 | 物理时间 | 权重 | 评分公式 |
|---|---:|---:|---:|---:|---|
| 第 1 段 | 20-49 | 0-30 | t in [10, 24.5] | 25% | `100 * exp(-20 * Rel-MSE_1)` |
| 第 2 段 | 50-199 | 30-180 | t in [25, 99.5] | 25% | `100 * exp(-10 * Rel-MSE_2)` |
| 第 3 段 | 200-399 | 180-380 | t in [100, 199.5] | 50% | `max(Lorentzian, Frechet)` |

### 5.1 第 1 段

```text
Rel-MSE_1 = Rel-MSE(pred[:, 20:50, :], true[:, 20:50, :])
score1    = 100 * exp(-20 * Rel-MSE_1)
```

### 5.2 第 2 段

```text
Rel-MSE_2 = Rel-MSE(pred[:, 50:200, :], true[:, 50:200, :])
score2    = 100 * exp(-10 * Rel-MSE_2)
```

### 5.3 第 3 段

```text
segment3_pred = pred[:, 200:400, :]
segment3_true = true[:, 200:400, :]

RMSE_3     = sqrt(mean((segment3_pred - segment3_true)^2))
Lorentzian = 100 / (1 + 10 * RMSE_3)
Frechet    = 50 * exp(-(FD ** 2))
score3     = max(Lorentzian, Frechet)
```

### 5.4 Task 3 分段预测总分

```text
Task3_prediction_score = 0.25 * score1 + 0.25 * score2 + 0.50 * score3
```

范围：

```text
0 <= Task3_prediction_score <= 100
```

## 6. 关键注意事项

- Task 1/2 的本地普通 weighted MSE 只是 proxy，不是官方分数。
- Task 1/2/3 的官方方向都是：前两段用 Rel-MSE 指数分，第三段用 `max(Lorentzian, Frechet)`。
- 如果没有官方 Frechet distance 实现，本地可以用 Lorentzian-only 估分，但 prompt 里必须保留完整 `max(Lorentzian, Frechet)` 公式。
- Task 1/2 推理超过 2 分钟会导致对应任务为 0。
- Task 3 推理超过 2 分钟会导致 Task 3 为 0。

## 7. 最小示例代码

下面代码可用于本地 validation 估分。注意：如果没有官方 Frechet distance 实现，`frechet_score_fn=None` 时会使用 Lorentzian-only 作为本地 proxy。

```python
import math
import numpy as np


EPS = 1e-12


def rel_mse(pred, true, eps=EPS):
    pred = pred.astype(np.float64, copy=False)
    true = true.astype(np.float64, copy=False)
    return float(np.sum((pred - true) ** 2) / (np.sum(true ** 2) + eps))


def exp_score(error, alpha):
    return 100.0 * math.exp(-alpha * error)


def lorentzian_score(pred, true):
    rmse = float(np.sqrt(np.mean((pred.astype(np.float64) - true.astype(np.float64)) ** 2)))
    return 100.0 / (1.0 + 10.0 * rmse)


def long_horizon_score(pred, true, frechet_score_fn=None):
    lorentzian = lorentzian_score(pred, true)
    if frechet_score_fn is None:
        return lorentzian
    frechet = float(frechet_score_fn(pred, true))
    return max(lorentzian, frechet)


def task12_prediction_score(pred_full, true_full, frechet_score_fn=None):
    """
    Task 1/2 official-like segmented prediction score.

    pred_full, true_full: shape (N, 200, 256)
    first 10 frames are observed input; score uses frames 10:200.
    Returns score in [0, 100], higher is better.
    """
    assert pred_full.shape == true_full.shape
    assert pred_full.shape[1:] == (200, 256)

    seg1_pred, seg1_true = pred_full[:, 10:57, :], true_full[:, 10:57, :]
    seg2_pred, seg2_true = pred_full[:, 57:105, :], true_full[:, 57:105, :]
    seg3_pred, seg3_true = pred_full[:, 105:200, :], true_full[:, 105:200, :]

    e1 = rel_mse(seg1_pred, seg1_true)
    e2 = rel_mse(seg2_pred, seg2_true)
    s1 = exp_score(e1, alpha=20.0)
    s2 = exp_score(e2, alpha=10.0)
    s3 = long_horizon_score(seg3_pred, seg3_true, frechet_score_fn)

    total = 0.25 * s1 + 0.25 * s2 + 0.50 * s3
    return {
        "prediction_score": float(total),
        "score1": float(s1),
        "score2": float(s2),
        "score3": float(s3),
        "rel_mse1": float(e1),
        "rel_mse2": float(e2),
    }


def task1_total_score(prediction_score, train_seconds, inference_seconds):
    if inference_seconds > 120:
        return 0.0

    train_minutes = train_seconds / 60.0
    if train_minutes <= 60:
        train_score = 35.0
    elif train_minutes <= 120:
        train_score = 25.0
    elif train_minutes <= 300:
        train_score = 20.0
    elif train_minutes <= 500:
        train_score = 10.0
    else:
        train_score = 0.0

    infer_score = 40.0 * max(0.0, 1.0 - inference_seconds / 120.0)
    return 0.75 * prediction_score + train_score + infer_score


def task2_total_score(prediction_score, inference_seconds):
    if inference_seconds > 120:
        return 0.0
    return 1.5 * prediction_score


def task3_prediction_score(pred_full, true_full, frechet_score_fn=None):
    """
    Task 3 segmented prediction score.

    pred_full, true_full: shape (N, 400, 256)
    first 20 frames are observed input; score uses frames 20:400.
    Returns score in [0, 100], higher is better.
    """
    assert pred_full.shape == true_full.shape
    assert pred_full.shape[1:] == (400, 256)

    seg1_pred, seg1_true = pred_full[:, 20:50, :], true_full[:, 20:50, :]
    seg2_pred, seg2_true = pred_full[:, 50:200, :], true_full[:, 50:200, :]
    seg3_pred, seg3_true = pred_full[:, 200:400, :], true_full[:, 200:400, :]

    e1 = rel_mse(seg1_pred, seg1_true)
    e2 = rel_mse(seg2_pred, seg2_true)
    s1 = exp_score(e1, alpha=20.0)
    s2 = exp_score(e2, alpha=10.0)
    s3 = long_horizon_score(seg3_pred, seg3_true, frechet_score_fn)

    total = 0.25 * s1 + 0.25 * s2 + 0.50 * s3
    return {
        "task3_prediction_score": float(total),
        "score1": float(s1),
        "score2": float(s2),
        "score3": float(s3),
        "rel_mse1": float(e1),
        "rel_mse2": float(e2),
    }


def final_score_scheme_a(task1_score, task2_score, task3_prediction):
    return min(350.0, task1_score + task2_score + 0.5 * task3_prediction)


def final_score_scheme_b(task1_score, task3_prediction):
    return min(350.0, task1_score + 2.0 * task3_prediction)
```
