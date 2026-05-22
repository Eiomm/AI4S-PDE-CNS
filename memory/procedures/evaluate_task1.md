# evaluate_task1

目的：对 validation prediction 计算 Task1 proxy 和基础 MSE 指标。

输入：

- prediction HDF5，dataset key 优先 `tensor`；
- validation target `data/task1_val.hdf5`，dataset key `tensor`。

输出指标：

- `mse`
- `initial_mse`
- `forecast_mse`
- `long_horizon_mse`
- `segment1_rel_mse`
- `segment2_rel_mse`
- `segment3_rmse`
- `competition_score_proxy`

硬校验：

- shape 必须是 `(N, 200, 256)`；
- 前 10 帧必须与输入初始条件一致；
- 所有值必须 finite。
