# make_submission

目的：从一次已校验 run 生成 Task1 submission 目录。

必需文件：

- `code/`
- `submission.json`
- `task1_pred.hdf5`
- `task1_time.csv`
- `task1_logs.log`

`submission.json` 至少包含：

```json
{
  "submission_id": "...",
  "problem_id": "PDE_Burgers",
  "code_path": "code"
}
```

兼容旧 sample submission：

- 可以保留 `methodology` 和 `submission` 字段；
- log 可以是普通文本，不强制 JSONL；
- `task1_pred.hdf5` dataset key 必须是 `tensor`。

最终提交注意：

- `code/` 需要经过 GPT-5.5 Agent + official proxy 生成或修改，确保可追溯。
