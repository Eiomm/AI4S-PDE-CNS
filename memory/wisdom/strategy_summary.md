# Task1 Strategy Summary

当前稳定策略：

1. 先用官方 FNO / Unet-PF checkpoint replay 建立可复现 baseline。
2. 所有候选都必须先通过 shape、finite、前 10 帧一致性校验。
3. 官方 FNO 原始 checkpoint 合规但分数低，只适合 sanity check。
4. 后续高分路线应围绕 Unet-PF、official ensemble/postprocess、或官方 FNO clean-lineage fine-tune 展开。
5. Memory 只记录已执行、有指标、有 artifact 的经验；不要保存 LLM 的纯猜想。
6. 最终提交 `code/` 必须来自 `agent_workspace/code`，由 GPT-5.5 Agent 经官方 proxy 生成或修改；当前 `src/` 只作为 harness。
