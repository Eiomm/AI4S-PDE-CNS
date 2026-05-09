# AI4S Race 7 Summary

Official page: https://competition.ai4s.com.cn/race/7/description

Race: AI4S智能体CNS挑战赛——神经算子PDE智能体

First version project focus: auditable Agent framework and submission compliance.

Key rules captured from the official page:

- Task 1: fixed-physics 1D Burgers prediction. Official PDEBench checkpoint fine-tuning is allowed.
- Task 2: multi-physics 1D Burgers prediction. Train from scratch; do not use Task 1 data or checkpoint.
- Input: first 10 time steps.
- Output: 200 time steps and 256 spatial points; time steps 10-199 are predicted.
- Task bundle: `task{N}_pred.hdf5`, `task{N}_time.csv`, `task{N}_logs.log`.
- Logs must be JSONL for each LLM call and include `timestamp` and `elapsed_seconds`.
- Total Task 2 Agent/training span must stay within 12 hours.
- Numerical solver generated extra training data is forbidden.

Known official data packages:

- `data_and_sample_submission.zip`
- `task_log_sample.zip`
