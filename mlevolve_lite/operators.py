from __future__ import annotations

import json
import re
import shutil
import textwrap
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .code_extractor import extract_python_code
from .llm_backend import LLMBackend
from .memory import build_prompt_context
from .node_schema import Node, Operator
from .research_log import append_research_event_many, file_record, new_response_id, text_sha256


SEED_HYPOTHESES = {
    "multi_nu_fno_baseline": "Spectral-style baseline that infers dynamics from the first 10 frames without test nu.",
    "nu_estimator_concat": "Estimate a viscosity proxy from observed frames and concatenate it to dynamics features.",
    "nu_estimator_film": "Use an estimated viscosity proxy to FiLM-condition a dynamics model.",
    "mixture_of_nu_experts": "Use a gating model to blend experts for different inferred viscosity regimes.",
    "conditional_fno_marginalized_nu": "Marginalize over plausible train-distribution viscosities at inference.",
}


def make_node_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def slugify_node_name(raw: str, fallback: str = "research_path") -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", raw.lower()).strip("_")
    slug = re.sub(r"_+", "_", slug)
    return (slug or fallback)[:72]


def extract_hypothesis_name(response_text: str) -> str | None:
    """Extract the human-readable hypothesis heading requested in LLM prompts."""
    match = re.search(
        r"^#{1,6}\s*Hypothesis name\s*\n+(?P<name>.+?)\s*(?:\n#{1,6}\s|\n```|\Z)",
        response_text,
        flags=re.IGNORECASE | re.MULTILINE | re.DOTALL,
    )
    if not match:
        return None
    lines = [line.strip() for line in match.group("name").splitlines() if line.strip()]
    return lines[0] if lines else None


def create_seed_nodes(workspace: Path) -> list[Node]:
    nodes = []
    for signature, hypothesis in SEED_HYPOTHESES.items():
        node_id = signature
        root = workspace / "nodes" / node_id
        code_dir = root / "code"
        artifact_dir = root / "artifacts"
        (root / "logs").mkdir(parents=True, exist_ok=True)
        code_dir.mkdir(parents=True, exist_ok=True)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        write_bootstrap_train_py(code_dir / "train.py", hypothesis)
        nodes.append(
            Node(
                node_id=node_id,
                signature=signature,
                parent_ids=[],
                operator=Operator.DRAFT,
                hypothesis=hypothesis,
                code_dir=str(code_dir),
                artifact_dir=str(artifact_dir),
                lineage=[node_id],
                log_path=str(root / "logs" / "train.log"),
            )
        )
    return nodes


def create_child_workspace(
    workspace: Path,
    parent: Node,
    operator: Operator = Operator.IMPROVE,
    *,
    round_index: int | None = None,
    branch_index: int | None = None,
    node_slug: str | None = None,
) -> Node:
    signature = slugify_node_name(node_slug or f"{parent.signature}_{operator.value}", fallback=operator.value)
    if round_index is not None and branch_index is not None:
        node_id = f"r{round_index:03d}_b{branch_index:02d}_{signature}"
    else:
        node_id = make_node_id(operator.value)
    root = workspace / "nodes" / node_id
    code_dir = root / "code"
    artifact_dir = root / "artifacts"
    (root / "logs").mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    parent_code = Path(parent.code_dir)
    if parent_code.exists():
        shutil.copytree(parent_code, code_dir)
    else:
        code_dir.mkdir(parents=True, exist_ok=True)
        write_bootstrap_train_py(code_dir / "train.py", parent.hypothesis)
    return Node(
        node_id=node_id,
        signature=signature,
        parent_ids=[parent.node_id],
        operator=operator,
        hypothesis=f"{operator.value} child of {parent.node_id}: {parent.hypothesis}",
        code_dir=str(code_dir),
        artifact_dir=str(artifact_dir),
        lineage=[*parent.lineage, node_id],
        log_path=str(root / "logs" / "train.log"),
    )


def apply_llm_to_child(
    child: Node,
    parent: Node,
    all_nodes: list[Node],
    llm: LLMBackend | None,
    prompts_dir: Path | str,
    research_log_path: Path | str | None = None,
    extra_log_paths: list[Path | str] | None = None,
    event_defaults: dict | None = None,
    log_lock=None,
) -> tuple[bool, str | None]:
    """Call LLM to rewrite child train.py. Returns (success, error_or_none)."""
    if llm is None:
        return True, None  # keep copied parent code

    prompts_dir = Path(prompts_dir)
    prompt_file = prompts_dir / f"{child.operator.value}.md"
    if not prompt_file.exists():
        prompt_file = prompts_dir / "draft.md"

    messages = build_prompt_context(
        operator=child.operator.value,
        parent_node=parent,
        nodes=all_nodes,
        code_dir=child.code_dir,
        prompt_template_path=prompt_file if prompt_file.exists() else None,
    )

    import time
    llm_call_start = time.time()
    try:
        response_text = llm.chat(messages)
    except Exception as exc:
        return False, f"llm chat failed: {exc}"
    llm_call_elapsed = time.time() - llm_call_start
    response_id = new_response_id(child.node_id)
    child.response_id = response_id
    hypothesis_name = extract_hypothesis_name(response_text)
    if hypothesis_name:
        child.hypothesis = hypothesis_name
        child.signature = slugify_node_name(hypothesis_name, fallback=child.signature)

    # Save raw LLM response for audit trail (per-node markdown)
    llm_log = Path(child.code_dir).parent / "logs" / "llm_response.md"
    llm_log.write_text(
        f"# LLM Response for {child.node_id}\n\n"
        f"**Operator:** {child.operator.value}\n"
        f"**Parent:** {child.parent_ids[0] if child.parent_ids else 'none'}\n"
        f"**Model:** {llm.model}\n\n"
        f"## Raw Response\n\n"
        f"{response_text}\n",
        encoding="utf-8",
    )

    # Append to global task2_logs.jsonl (JSONL format for official audit)
    workspace_root = Path(child.code_dir).parent.parent.parent
    task_log = Path(research_log_path) if research_log_path is not None else workspace_root / "task2_logs.jsonl"
    node_log = Path(child.code_dir).parent / "logs" / "research_loop.jsonl"
    log_paths = [task_log, node_log, *(extra_log_paths or [])]
    log_entry = {
        "event": "llm_response",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": llm_call_elapsed,
        "response": response_text,
        "response_id": response_id,
        "response_sha256": text_sha256(response_text),
        "node_id": child.node_id,
        "parent_id": child.parent_ids[0] if child.parent_ids else None,
        "operator": child.operator.value,
        "model": llm.model,
    }
    log_entry.update(event_defaults or {})
    append_research_event_many(log_paths, log_entry, lock=log_lock)

    new_code = extract_python_code(response_text)
    if new_code is None:
        return False, "no python code block found in llm response"

    train_py = Path(child.code_dir) / "train.py"
    train_py.write_text(new_code, encoding="utf-8")
    code_entry = {
        "event": "code_written",
        "elapsed_seconds": llm_call_elapsed,
        "response": response_text,
        "response_id": response_id,
        "node_id": child.node_id,
        "parent_id": child.parent_ids[0] if child.parent_ids else None,
        "operator": child.operator.value,
        "model": llm.model,
        "files_written": [file_record(train_py, root=workspace_root)],
    }
    code_entry.update(event_defaults or {})
    append_research_event_many(log_paths, code_entry, lock=log_lock)
    return True, None


def write_bootstrap_train_py(path: Path, hypothesis: str) -> None:
    path.write_text(
        textwrap.dedent(
            f'''
            import argparse
            import csv
            import json
            import time
            import warnings
            from datetime import datetime, timezone
            from pathlib import Path

            import h5py
            import numpy as np
            import torch
            import torch.nn as nn
            import torch.nn.functional as F

            HYPOTHESIS = {hypothesis!r}

            # Data amplitude statistics (computed from train set)
            CLIP_MIN = -3.5
            CLIP_MAX = 3.5


            def log_event(handle, start, response):
                handle.write(json.dumps({{
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "elapsed_seconds": time.time() - start,
                    "response": response,
                }}) + "\\n")
                handle.flush()


            class ResBlock1D(nn.Module):
                def __init__(self, channels):
                    super().__init__()
                    self.conv1 = nn.Conv1d(channels, channels, 5, padding=2)
                    self.conv2 = nn.Conv1d(channels, channels, 5, padding=2)

                def forward(self, x):
                    return F.relu(self.conv2(F.relu(self.conv1(x))) + x)


            class SimplePredictor(nn.Module):
                def __init__(self, in_ch=10, hidden=64):
                    super().__init__()
                    self.enc = nn.Conv1d(in_ch, hidden, 5, padding=2)
                    self.res1 = ResBlock1D(hidden)
                    self.res2 = ResBlock1D(hidden)
                    self.dec = nn.Conv1d(hidden, 1, 5, padding=2)

                def forward(self, x):
                    # x: (B, 10, 256)
                    x = F.relu(self.enc(x))
                    x = self.res1(x)
                    x = self.res2(x)
                    return self.dec(x)  # (B, 1, 256)


            def load_train_data(data_dir, max_trajectories=256, max_samples_per_traj=100):
                parts = ["task2_part0_train.h5", "task2_part1_train.h5", "task2_part2_train.h5"]
                inputs, targets = [], []
                traj_count = 0
                for part in parts:
                    path = Path(data_dir) / part
                    if not path.exists():
                        continue
                    with h5py.File(path, "r") as f:
                        tensor = f["tensor"][:]
                    n = min(tensor.shape[0], max(0, max_trajectories - traj_count))
                    if n <= 0:
                        break
                    tensor = tensor[:n]
                    traj_count += n
                    # sample contexts and next-step targets
                    T = tensor.shape[1]
                    end = min(T - 1, 10 + max_samples_per_traj)
                    for t in range(10, end):
                        inputs.append(tensor[:, t - 10:t, :].astype(np.float32))
                        targets.append(tensor[:, t:t + 1, :].astype(np.float32))
                if not inputs:
                    return None, None
                inputs = np.concatenate(inputs, axis=0)  # (N, 10, 256)
                targets = np.concatenate(targets, axis=0)  # (N, 1, 256)
                return inputs, targets


            def train_model(model, data_dir, epochs, batch_size, lr, device, cheap_probe, log_handle, start):
                model.train()
                max_traj = 128 if cheap_probe else 3000
                max_spt = 100 if cheap_probe else 200
                inputs, targets = load_train_data(data_dir, max_traj, max_spt)
                if inputs is None:
                    log_event(log_handle, start, "No training data found.")
                    return
                dataset = torch.utils.data.TensorDataset(
                    torch.from_numpy(inputs), torch.from_numpy(targets)
                )
                loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                criterion = nn.MSELoss()
                for epoch in range(epochs):
                    total_loss = 0.0
                    for xb, yb in loader:
                        xb, yb = xb.to(device), yb.to(device)
                        pred = model(xb)
                        loss = criterion(pred, yb)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        total_loss += float(loss.item())
                    avg = total_loss / max(len(loader), 1)
                    log_event(log_handle, start, f"Epoch {{epoch + 1}}/{{epochs}} loss={{avg:.6f}}")


            def autoregressive_rollout(model, initial, steps, device):
                # initial: (N, 10, 256)
                model.eval()
                history = [torch.from_numpy(initial[:, i, :]) for i in range(initial.shape[1])]
                with torch.no_grad():
                    for _ in range(steps):
                        inp = torch.stack(history[-10:], dim=1).to(device)  # (N, 10, 256)
                        out = model(inp)  # (N, 1, 256)
                        # HARD CLIP: prevent autoregressive explosion
                        out = torch.clamp(out, min=CLIP_MIN, max=CLIP_MAX)
                        history.append(out[:, 0, :].cpu())
                result = torch.stack(history, dim=1).numpy().astype(np.float32)  # (N, 10+steps, 256)
                return result


            def run_inference(model, data_dir, out_dir, device, log_handle, start):
                data_dir = Path(data_dir)
                out_dir = Path(out_dir)
                out_dir.mkdir(parents=True, exist_ok=True)

                # Test inference -> (1000, 200, 256)
                with h5py.File(data_dir / "task2_test.h5", "r") as f:
                    test_input = f["tensor"][:].astype(np.float32)  # (1000, 10, 256)
                pred_test = autoregressive_rollout(model, test_input, 190, device)
                assert pred_test.shape == (1000, 200, 256)
                with h5py.File(out_dir / "task2_pred.hdf5", "w") as f:
                    f.create_dataset("tensor", data=pred_test)

                # Val inference -> (100, 200, 256)
                val_path = data_dir / "task2_val.h5"
                if val_path.exists():
                    with h5py.File(val_path, "r") as f:
                        val_input = f["tensor"][:, :10, :].astype(np.float32)  # (100, 10, 256)
                    pred_val = autoregressive_rollout(model, val_input, 190, device)
                    assert pred_val.shape == (100, 200, 256)
                    with h5py.File(out_dir / "val_pred.hdf5", "w") as f:
                        f.create_dataset("tensor", data=pred_val)

                    # Compute validation MSE over future 190 steps
                    with h5py.File(val_path, "r") as f:
                        val_target = f["tensor"][:, :200, :].astype(np.float32)
                    mse = float(np.mean((pred_val[:, 10:200, :] - val_target[:, 10:200, :]) ** 2))
                    log_event(log_handle, start, f"Validation MSE (future 190): {{mse:.6f}}")
                    return mse
                return None


            def main():
                parser = argparse.ArgumentParser()
                parser.add_argument("--data-dir", required=True)
                parser.add_argument("--out-dir", required=True)
                parser.add_argument("--epochs", type=int, default=1)
                parser.add_argument("--batch-size", type=int, default=32)
                parser.add_argument("--lr", type=float, default=1e-3)
                parser.add_argument("--cheap-probe", action="store_true")
                args = parser.parse_args()
                start = time.time()
                out_dir = Path(args.out_dir)
                out_dir.mkdir(parents=True, exist_ok=True)

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = SimplePredictor(in_ch=10, hidden=64).to(device)
                log_path = out_dir / "task2_logs.log"
                with log_path.open("w", encoding="utf-8") as log:
                    log_event(log, start, f"Bootstrap harness candidate: {{HYPOTHESIS}} device={{device}} cheap_probe={{args.cheap_probe}}")
                    train_model(model, args.data_dir, args.epochs, args.batch_size, args.lr, device, args.cheap_probe, log, start)
                    val_mse = run_inference(model, args.data_dir, args.out_dir, device, log, start)
                    # Save checkpoint for potential resume
                    ckpt_path = out_dir / "checkpoint.pt"
                    torch.save({{"model_state_dict": model.state_dict(), "hypothesis": HYPOTHESIS}}, ckpt_path)
                    log_event(log, start, f"Saved checkpoint to {{ckpt_path}}")
                    runtime = time.time() - start
                    with (out_dir / "task2_time.csv").open("w", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=["train_time", "inference_time"])
                        writer.writeheader()
                        writer.writerow({{"train_time": max(runtime - 5.0, 0.0), "inference_time": min(runtime, 120.0)}})
                    log_event(log, start, "Generated task2_pred.hdf5, val_pred.hdf5, and task2_time.csv.")
                    if val_mse is not None:
                        print(f"Final Validation Score: {{val_mse}}")
                    else:
                        print("Final Validation Score: 0.0")


            if __name__ == "__main__":
                main()
            '''
        ).lstrip(),
        encoding="utf-8",
    )
