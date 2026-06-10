# Output Layout Notes

当前 `outputs/` 里混在一起的目录，本质上有四类：

- `verify_*`: 连通性、tmux、中文显示等验证运行。
- `goal_*`: 自动迭代产生的真实策略实验。
- `strategy_memory/`: 跨实验的经验表、最佳策略、失败策略。
- `final*`: 模拟或正式提交包的生成记录。

建议后续沿用这个逻辑：`experiment -> run -> artifacts/logs/metrics/config`。

```text
outputs/
  strategy_memory/
  index/
    run_index.csv
    run_index.jsonl
  <run_group>/
    latest -> <timestamp>
    <timestamp>/
      candidates.csv
      pipeline.log
      submission.zip
      test.log
      inspect.log
      run.env
      harness.done
      llm_io/
      docking/
```

刷新索引：

```bash
python scripts/index_outputs.py
```

查看工具是否可用：

```bash
python scripts/check_tools.py
```

现在先不移动旧目录，避免破坏已有 `latest` 链接和 strategy memory 中保存的路径。等索引稳定后，可以再加一个归档脚本，把 `verify_*` 批量移到 `outputs/archive/verification/`。
