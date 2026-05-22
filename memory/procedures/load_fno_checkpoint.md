# load_fno_checkpoint

目的：加载官方 Nu0.001 FNO checkpoint，并保持 PDEBench checkpoint 的结构兼容。

步骤：

1. 使用 `torch.load(..., map_location=device, weights_only=False)` 读取 checkpoint。
2. 若 payload 含 `model_state_dict`，取该字段；否则直接视为 state dict。
3. 从 `fc0.weight` 和 `conv0.weights1` 推断 `width`、`initial_step`、`modes`。
4. 构造 `FNO1d`，加载 state dict，切换到 `eval()`。

注意：

- Task1 官方最终路线只允许 `checkpoints/official/nu0.001_fno.pt`。
- 非 Nu0.001 FNO checkpoint 不能混入最终 Task1 提交路线。
