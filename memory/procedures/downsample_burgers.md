# downsample_burgers

目的：fine-tune 阶段需要把 PDEBench 原始 Burgers 数据对齐到 Task1 官方 FNO 的时间尺度。

当前状态：Phase 1 未实现 fine-tune，只记录流程。

关键规则：

1. Task1 输入固定为前 10 帧，空间大小 256。
2. 官方 FNO checkpoint 与 `reduced_resolution_t=5` 对齐。
3. fine-tune 时不要使用 raw adjacent frames 的 `temporal_stride=1` 训练官方 FNO descendant。
4. PDEBench 原始空间分辨率为 1024，Task1 输出空间分辨率为 256，需要稳定下采样。
5. `reduced_resolution=4`，所以空间下采样必须从 1024 点取到 256 点。
6. 模型的第 1 个预测步对应原始数据向前 5 个时间索引，不是向前 1 个索引。
7. fine-tune 的前 10 帧窗口应类似原始索引 `0, 5, 10, ..., 45`；第一个监督目标应是原始索引 `50`。

实现检查清单：

- dataloader 必须显式保存 `temporal_stride=5` 和 `spatial_downsample=4` 到 run metadata。
- 如果训练配置里出现 `temporal_stride=1` 且 base checkpoint 是官方 Task1 checkpoint，应直接报错。
- 验证集和测试集已经是 reduced scale，不要再次做时间/空间下采样。
