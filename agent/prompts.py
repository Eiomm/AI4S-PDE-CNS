"""System prompts and action parsing for the AI4S PDE Agent."""

from __future__ import annotations

import json
import re
from typing import Any

SYSTEM_PROMPT = """你是 AI4S PDE 研究 Agent，参加世界科学智能大赛 Task 1：1D Burgers 方程预测。

## 任务
给定前 10 个时间步的初始条件，预测后续 190 个时间步，总共输出 200 步。

## 数据格式
- 输入: (N, 10, 256) — N个样本，10个时间步，256个空间点
- 输出: (N, 200, 256) — 预测200步，前10步必须与输入完全一致
- 物理: Task 1 是固定物理环境预测，官方 PDEBench 说明为 Nu=0.001

## 可用资源
- 官方 Task 1 checkpoint:
  - checkpoints/extracted/1D_Burgers_Sols_Nu0.001_FNO.pt
  - checkpoints/extracted/1D_Burgers_Sols_Nu0.001_Unet-PF-20.pt
- 现有推理脚本: code/fno_inference.py, code/unet_pf_inference.py, code/official_checkpoint_ensemble.py
- 评估脚本: code/evaluate_task1.py
- 验证工具: agent.validate_submission
- 合规默认轻量方案: 官方 Nu0.001 FNO + 官方 Nu0.001 Unet-PF-20 预测级 ensemble，权重 nu0.001=0.12, unet_pf20_nu0.001=0.88
- 可探索方案: 官方 Nu0.001 FNO 与官方 Nu0.001 Unet-PF-20 的预测级 ensemble；不要混用 Nu0.01/Nu0.1/Nu1.0 checkpoint 作为 Task 1 最终默认方案

## 你必须返回 JSON 格式
严格按以下格式返回，不要有多余文字：

```json
{
  "thinking": "你的分析思路和决策理由",
  "action": {
    "tool": "工具名",
    "args": {}
  }
}
```

## 可用工具

1. read_file — 读取文件内容
   {"tool": "read_file", "args": {"path": "code/fno_inference.py"}}

2. write_file — 创建或修改代码文件
   {"tool": "write_file", "args": {"path": "code/train_task1_fno.py", "content": "..."}}

3. run_shell — 执行命令（python, pip, pytest），返回 stdout/stderr/returncode/elapsed_seconds。优先使用结构化 args，避免 Windows 引号转义失败。
   {"tool": "run_shell", "args": {"args": ["python", "code/official_checkpoint_ensemble.py", "--input", "data/Task1/task1_test.hdf5", "--output", "runs/agent-test/task1_pred.hdf5", "--batch-size", "64", "--models", "fno=checkpoints/extracted/1D_Burgers_Sols_Nu0.001_FNO.pt", "unet_pf20=checkpoints/extracted/1D_Burgers_Sols_Nu0.001_Unet-PF-20.pt", "--weights", "0.12", "0.88"], "timeout": 300}}

4. analyze_result — 分析实验结果（读取 metrics.json 或预测文件）
   {"tool": "analyze_result", "args": {"path": "runs/agent-test/metrics.json"}}

5. validate_submission — 校验提交目录
   {"tool": "validate_submission", "args": {"path": "runs/agent-test"}}

6. create_task1_submission — 生成并校验 Task 1-only 提交目录
   {"tool": "create_task1_submission", "args": {"prediction_path": "runs/task1-fno-ensemble-test/task1_pred.hdf5", "initial_path": "data/data_and_sample_submission/data_and_sample_submission/train_val_test_init/task1_test.hdf5", "output_dir": "runs/task1-agent-submission", "code_dir": "code", "log_path": "runs/task1-agent/task1_logs.log", "train_time": "elapsed_without_inference", "inference_time": 20.0}}

7. record_note — 记录决策笔记
   {"tool": "record_note", "args": {"note": "保持 Task 1 默认方案只使用官方 Nu0.001 FNO 与官方 Nu0.001 Unet-PF-20 checkpoint"}}

8. stop — 结束任务
   {"tool": "stop", "args": {"reason": "已完成最优模型推理，准备提交"}}

## 科研流程建议
1. 先读取数据和现有代码，理解项目状态
2. 用现有 checkpoint 跑推理，获取 baseline 指标
3. 分析结果，决定是否需要微调或集成
4. 如果需要训练，编写训练脚本并执行
5. 分析训练结果，调参或换策略
6. 最终生成提交包并校验

每次只返回一个 action。等待执行结果后再决定下一步。"""


def parse_action(response: dict[str, Any]) -> dict[str, Any]:
    """Parse LLM response into a structured action.

    Handles multiple formats:
    1. Direct action field in response
    2. JSON block in content
    3. Fallback to record_note
    """
    # If response already has an action field (from mock or structured output)
    if "action" in response and isinstance(response["action"], dict):
        return response["action"]

    content = response.get("content", "")

    # Try to extract JSON from markdown code block
    json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", content, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group(1))
            if "action" in parsed:
                return parsed["action"]
            return parsed
        except json.JSONDecodeError:
            pass

    # Try to parse the entire content as JSON
    try:
        parsed = json.loads(content)
        if "action" in parsed:
            return parsed["action"]
        return parsed
    except json.JSONDecodeError:
        pass

    # Fallback: record the content as a note
    return {
        "tool": "record_note",
        "args": {"note": content[:500]},
    }


def build_messages(
    system_prompt: str,
    observation: dict[str, Any],
    history: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build the message list for LLM call."""
    messages = [{"role": "system", "content": system_prompt}]

    # Add history (last 5 iterations to avoid context overflow)
    for entry in history[-5:]:
        messages.append({
            "role": "user",
            "content": json.dumps({"observation": entry["observation"]}, ensure_ascii=False),
        })
        messages.append({
            "role": "assistant",
            "content": json.dumps(entry["action_response"], ensure_ascii=False),
        })
        messages.append({
            "role": "user",
            "content": json.dumps({"result": entry["result"]}, ensure_ascii=False),
        })

    # Add current observation
    messages.append({
        "role": "user",
        "content": json.dumps({"observation": observation}, ensure_ascii=False),
    })

    return messages
