# Codex 工程级 Harness Prompt

下面这段提示词可以直接交给 Codex，用来按“生成 -> 测试 -> 复查 -> 修复 -> 记录经验 -> 继续迭代”的闭环改这个项目。

```text
你现在在 /data/wangjunao/AI4S 项目里工作。请以工程闭环方式迭代 AI4S 小分子生成 pipeline。

目标：
1. 每轮都必须保持最小链路可运行：读取 target.pdb，生成候选 SMILES，写出 result.csv/result.log/result.zip，并通过 scripts/inspect_result_zip.py。
2. 优先提升候选分子的合理性、可合成性、评分排序和日志可追踪性。
3. 不要破坏比赛提交格式。单目标 zip 内应包含 result.csv 和 result.log；多目标保留 result1.csv/result2.csv/result3.csv。
4. 本地可读产物使用简单名字：candidates.csv、pipeline.log、submission.zip。

每一轮按照这个顺序执行：
1. 读取当前代码和最近一次 outputs/<name>/latest 的日志，判断上轮失败点或可提升点。
2. 选择一个小而明确的改动，不要一次改太多模块。
3. 实现改动。
4. 运行：
   - conda run -n ai4s-chem-evolve python -m pytest -v
   - bash scripts/run_harness_once.sh --name codex_iter --target target.pdb
5. 复查输出：
   - outputs/codex_iter/latest/candidates.csv 是否非空
   - outputs/codex_iter/latest/submission.zip 是否通过 inspect
   - outputs/codex_iter/latest/pipeline.log 是否记录关键阶段
6. 如果失败，先修复失败，再重新运行同样检查。
7. 如果成功，把本轮经验写入 outputs/engineering_memory.md，说明：
   - 改了什么
   - 为什么改
   - 测试结果
   - 下一轮建议

约束：
- 使用现有项目风格，不做无关重构。
- 优先小步提交式改动。
- 任何新增依赖都要说明原因，并更新 environment.yml 或 pyproject.toml。
- 如果需要联网查资料，只参考官方文档或明确可信的上游项目。
- 每次结束时给出本轮产物路径和验证结果。
```

当前项目已经把这个闭环落成了脚本：

```bash
bash scripts/run_harness_once.sh --name codex_iter --target target.pdb
```

需要后台长跑时：

```bash
bash scripts/run_tmux_training.sh codex_iter
bash scripts/run_tmux_training.sh codex_iter --status
```
