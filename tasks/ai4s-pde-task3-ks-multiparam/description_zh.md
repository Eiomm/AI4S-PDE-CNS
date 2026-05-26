# Task 3：Kuramoto-Sivashinsky 多参数预测（中文版）

本文件是 `description.md` 的中文说明版，用于阅读和 prompt 参考；原始英文版仍以 `description.md` 为准。

## 1. 任务目标

Task 3 是 AI4S PDE 神经算子挑战赛中的 KS 方程附加题。

需要根据每条测试轨迹的前 20 个观测时间步，预测完整 400 步轨迹：

```text
输入:  (20, 256)
输出:  (400, 256)
提交:  task3_pred.hdf5, shape = (100, 400, 256)
```

脚本必须在当前工作目录生成两个独立文件：

```text
task3_pred.hdf5             # 只包含预测 tensor，不放属性或额外数据集
task3_inference_time.txt    # 只包含全测试集 rollout 推理时间，单位秒
```

同时需要在 stdout 单独打印：

```text
INFERENCE_TIME=<seconds>
```

不要自己创建 `task3_time.csv`，打包脚本会根据日志和 `task3_inference_time.txt` 组装。

## 2. KS 方程与难点

KS 方程形式：

```text
u_t + u·u_x + λ₂·u_xx + u_xxxx = 0
```

这是典型混沌 PDE。误差会随时间指数增长，本任务设置下 Lyapunov time 大约是 10-20 个模型步。因此：

- 短期可以做 pointwise prediction；
- 中期要延缓 rollout 发散；
- 长期不应只看逐点 MSE，更应该匹配统计分布和能量谱。

官方评分也体现了这一点：后段使用 `max(Lorentzian, Frechet)`。

## 3. 硬约束

违反任一条可能导致该任务为 0 分。

1. **必须从 scratch 训练**
   不能加载任何公开预训练权重、PDEBench 权重、Task1 checkpoint、HuggingFace 权重或其他 `.pt` 文件。模型参数必须随机初始化，只能使用 `data/KS_train.hdf5` 训练。

2. **`data/KS_val.hdf5` 只能用于评估**
   不能把 val tensor 放进 loss、optimizer step 或 backward。`lambda2` 可以用于分层统计，但不能参与梯度训练。

3. **测试前 20 步必须原样复制**
   `task3_pred.hdf5["tensor"][:, :20, :]` 必须和 `KS_test.hdf5["tensor"]` 一致，容差 `1e-3`。

4. **测试时没有显式 `lambda2`**
   `KS_test.hdf5` 不包含 `lambda2`。推理路径可以从前 20 步估计隐变量，但不能假设测试集提供 `lambda2`。

5. **时间约束**
   训练总时长（含 Agent 思考时间）不超过 12 小时。完整测试集推理时间必须小于等于 2 分钟，否则 Task 3 得 0。

## 4. 数据格式

```text
data/KS_train.hdf5
data/KS_val.hdf5
data/KS_test.hdf5
```

注意：Task 3 的 HDF5 坐标 key 使用连字符，和 Task 1 一样：

```text
t-coordinate
x-coordinate
```

不是 Task 2 的下划线形式。

数据结构：

```text
KS_train.hdf5
  tensor        float32  (2000, 400, 256)
  lambda2       float32  (2000,)
  t-coordinate  float32  (400,)
  x-coordinate  float32  (256,)

KS_val.hdf5
  tensor        float32  (100, 400, 256)
  lambda2       float32  (100,)
  t-coordinate  float32  (400,)
  x-coordinate  float32  (256,)

KS_test.hdf5
  tensor        float32  (100, 20, 256)
  t-coordinate  float32  (20,)
  x-coordinate  float32  (256,)
  没有 lambda2
```

物理参数：

```text
空间点数 N = 256
存储时间步 dt = 0.5
总步数 = 400
lambda2 ~ Uniform[1.0, 1.5]
```

`lambda2` 越小，系统通常更混沌，长期预测更难。

## 5. 当前推荐模型

当前推荐的更强 baseline 是 **lambda2-agnostic 1D FNO**：

```text
模型: FNO1d
initial_step = 20
modes = 32
width = 96
extra_channels = 0
训练方式: 从 scratch 训练
```

也就是说，当前默认推荐是：

```text
lambda2-agnostic FNO
```

原因：

- 测试集不提供 `lambda2`；
- 前 20 帧已经包含较丰富的动力学信息；
- 显式估计 `lambda2` 如果不准，可能反而污染主模型；
- `lambda2`-agnostic FNO 更稳，失败面更小；先用 `modes=32,width=96`，如果推理或训练预算吃紧，再退回 `modes=24,width=64`。

可以在验证集中按 `lambda2` 分层看误差。如果低 `lambda2` 或高 `lambda2` 区间明显更差，再考虑加入 `lambda2` 估计器或条件输入。

## 6. 可选 `lambda2` 策略

### 6.1 忽略 `lambda2`

训练一个统一模型，只从前 20 帧隐式识别动力学状态。

优点：

- 简单；
- 推理稳定；
- 不依赖显式参数；
- 当前推荐作为首选 baseline。

### 6.2 辅助头估计 `lambda2`

训练一个小 encoder/head：

```text
lambda2_hat = g(u[:, :20, :])
```

训练时用 train/val 的 `lambda2` 标签监督。测试时先估计 `lambda2_hat`，再输入主模型或作为条件。

适用场景：

- validation 按 `lambda2` 分层后发现某些参数区间明显失败；
- 当前模型对不同混沌强度泛化不足。

### 6.3 `lambda2` 作为额外通道

把 `lambda2` 或 `lambda2_hat` broadcast 成 `(256,)` 空间 map，作为 FNO 的 `extra_channels`。

优点是表达能力更强，缺点是如果 `lambda2_hat` 不准，会影响每个空间位置的特征，风险更高。

## 7. 推荐训练 recipe

基础训练流程：

1. 读取 `KS_train.hdf5`。
2. 只用训练集统计量做归一化：

```text
u_norm = (u - train_mean) / train_std
```

3. 建立 FNO：

```text
FNO1d(modes=32, width=96, initial_step=20, extra_channels=0)
```

4. 随机采样 rollout window：

```text
t0w ~ Uniform[0, n_time - 20 - H)
window = batch[:, t0w:t0w+20, :]
target = batch[:, t0w+20:t0w+20+H, :]
```

5. 使用 horizon curriculum：

```text
HORIZON_SCHEDULE = [1]*5 + [5]*10 + [10]*10 + [20]*10 + [40]*5
```

6. 使用 AdamW + cosine scheduler：

```text
EPOCHS = 40
BATCH = 16
LR = 5e-4
weight_decay = 1e-5
```

7. 必须做梯度裁剪：

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```

8. 最小增强 loss：

```text
loss = MSE + 0.20 * spectral_loss + 0.05 * gradient_loss
```

其中 `spectral_loss` 使用 `log1p(|rfft(u)|^2)`，`gradient_loss` 使用空间一阶差分。

9. 维护 EMA，并在 validation 上比较 raw weights 和 EMA weights：

```text
EMA decay = 0.995
选择 official_like_segment_score 更高的权重用于 test rollout
```

10. 推理后强制复制前 20 步：

```python
test_pred[:, :20, :] = test_initial
```

## 8. 本地验证和评分

`KS_val.hdf5` 只能用于评估。正确流程：

```text
val 前 20 步 -> 模型 rollout -> val_pred
val_pred vs val 完整真值 -> local validation score
```

官方 Task 3 评分只看后 380 个预测步：

```text
future = tensor[:, 20:400, :]
```

分段：

| 段 | 原始步范围 | 未来窗口索引 | 物理时间 | 权重 | 公式 |
|---|---:|---:|---:|---:|---|
| 第 1 段 | 20-49 | 0-30 | t in [10, 24.5] | 25% | `100 * exp(-20 * Rel-MSE_1)` |
| 第 2 段 | 50-199 | 30-180 | t in [25, 99.5] | 25% | `100 * exp(-10 * Rel-MSE_2)` |
| 第 3 段 | 200-399 | 180-380 | t in [100, 199.5] | 50% | `max(Lorentzian, Frechet)` |

第三段：

```text
RMSE_3     = sqrt(mean((pred[:, 200:400] - true[:, 200:400])^2))
Lorentzian = 100 / (1 + 10 * RMSE_3)
Frechet    = 50 * exp(-(FD ** 2))
score3     = max(Lorentzian, Frechet)
```

如果本地没有官方 Frechet distance 实现，可以先用 Lorentzian-only 做确定性 lower-bound proxy，但 prompt 和日志中仍应保留完整 `max(Lorentzian, Frechet)` 公式。

普通 weighted MSE 只能作为 debug proxy，不应作为唯一 best-model 标准。

## 9. 推理效率

测试集只有 100 条轨迹，shape 是：

```text
(100, 20, 256) -> (100, 400, 256)
```

推荐：

```text
device = cuda
batch_size = 100
一次性 batch 完 100 条测试样本
```

可选加速：

- `torch.compile(model)`：在最终 eval 后尝试；
- `model.half()`：风险较高，必须先在 val 上确认不降分；
- 避免逐样本循环推理。

## 10. 常见失败模式

| 现象 | 原因 | 修复 |
|---|---|---|
| `KeyError: t_coordinate` | 复制了 Task2 loader | Task3 使用 `t-coordinate` |
| 测试读取 `lambda2` 报错 | `KS_test.hdf5` 没有 `lambda2` | 测试推理必须 lambda2-free |
| 被判使用预训练 | 加载了 `.pt` | Task3 必须 random init |
| 前 20 步不一致 | 没有强制复制输入 | `pred[:, :20] = test_initial` |
| 长期发散 | 只用 1-step 或 5-step loss | curriculum 至少推到 H=20 |
| 预测幅值爆炸 | 没有梯度裁剪 | 加 `clip_grad_norm_` |
| 推理超过 2 分钟 | 没 batch / 没 GPU | GPU + batch_size=100 |
| train loss 低但 val 差 | 只采样前段轨迹 | 随机采样 `t0w` |
| 归一化后分数异常 | metric 前没反归一化 | `pred = pred * std + mean` |
| seg1/seg2 好，seg3 差 | 长期统计结构不对 | spectral loss、ensemble、distribution loss |

## 11. 可能进一步提分的 trick

以下是基于当前 prompt 的额外建议；其中 spectral loss、gradient loss、EMA 和 `H=40` 已作为最小增强 baseline 写入英文 `description.md`。

### 11.1 保存 best-by-validation checkpoint 和 prediction

每次 validation 后保存：

```text
best_model_state
best_val_score
best_pred
```

不要让后续 debug 轮覆盖已经更好的输出。

### 11.2 加 spectral loss

KS 后段更看统计分布和能量谱，建议在训练 loss 中加入：

```text
MSE(|rfft(pred)|^2, |rfft(true)|^2)
```

推荐权重：

```text
0.05 - 0.3
```

短期以 MSE 为主，长期逐步提高 spectral 权重。

### 11.3 多 horizon loss

不要只训练固定 H。可以同时优化多个 horizon：

```text
H in {1, 5, 10, 20, 40}
```

或者 late epoch 加到 `H=30/40`，但注意显存和训练时间。

### 11.4 scheduled sampling / free rollout validation

训练时逐渐减少 teacher forcing，让模型更适应自回归误差累积。

validation 时必须跑完整 400 步 rollout，不要只看短 horizon loss。

### 11.5 按 lambda2 分层评估

在 val 上分桶：

```text
lambda2 in [1.0,1.15), [1.15,1.3), [1.3,1.5]
```

如果某一段明显差，再考虑：

- 增加训练采样权重；
- 加 lambda2 estimator；
- 加条件 FNO extra channel。

### 11.6 EMA / SWA

对混沌 rollout，EMA 往往比最后一个 epoch 稳：

```text
EMA decay = 0.995 或 0.999
```

也可以试 SWA，但要验证完整 rollout 分数。

### 11.7 seed ensemble

测试集只有 100 条，推理很快。可以训练 3 个不同 seed 的模型，预测后平均：

```text
pred = mean(pred_seed_1, pred_seed_2, pred_seed_3)
```

通常能改善长期统计稳定性，但会增加训练时间。

### 11.8 轻微噪声增强

训练输入 window 可以加很小的高斯噪声，增强 rollout 鲁棒性：

```text
noise_std = 0.005 - 0.02 in normalized space
```

不要对 target 加噪声。

### 11.9 导数/梯度 loss

KS 中空间结构重要，可以加一阶差分或谱梯度误差：

```text
MSE(dx(pred), dx(true))
```

权重建议小一些，例如 `0.02 - 0.1`。

### 11.10 后段统计校准

如果 pointwise 后段必然失同步，可以监控：

- mean/std；
- energy spectrum；
- spatial autocorrelation；
- amplitude histogram。

训练目标中适当加入这些统计损失，可能更贴近 Frechet 分支。
