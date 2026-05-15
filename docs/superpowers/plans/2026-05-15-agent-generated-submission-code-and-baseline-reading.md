# Agent-Generated Submission Code and Baseline Reading Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make each autonomous study produce traceable Agent-generated code, then merge Task 1 and Task 2 artifacts into one official `submission/` layout with a shared `code/` directory.

**Architecture:** Add a submission workspace layer between per-task autonomous runs and final packaging. Add a baseline-reading layer that lets the Agent inspect `third_party/baseline` repositories, summarize PDEBench/FNO/DeepONet/PI-DeepONet code, and use those summaries as Planner context before proposing code evolution.

**Tech Stack:** Python 3.10, existing `agent/pde_*` framework, HDF5 validation, JSONL LLM logs, pytest.

---

### Task 1: Add Submission Workspace Aggregator

**Files:**
- Create: `agent/submission_workspace.py`
- Test: `tests/test_submission_workspace.py`
- Modify later: `agent/final_submission.py`

- [ ] **Step 1: Write tests for combining two task run directories**

```python
def test_submission_workspace_merges_task_runs_with_shared_code(tmp_path):
    task1 = tmp_path / "runs" / "task1" / "study"
    task2 = tmp_path / "runs" / "task2" / "study"
    for run, task in [(task1, "task1"), (task2, "task2")]:
        (run / "code").mkdir(parents=True)
        (run / "code" / f"{task}_infer.py").write_text(f"print('{task}')\n", encoding="utf-8")
        (run / f"{task}_pred.hdf5").write_bytes(b"fake")
        (run / f"{task}_time.csv").write_text("train_time,inference_time\n1,1\n", encoding="utf-8")
        (run / f"{task}_logs.log").write_text('{"timestamp":"2026-05-15T00:00:00+08:00","elapsed_seconds":1,"response":{}}\n', encoding="utf-8")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "methodology.pdf").write_bytes(b"%PDF-1.4\n")

    output = build_submission_workspace(
        output_dir=tmp_path / "runs" / "combined",
        task_runs={"task1": task1, "task2": task2},
        methodology_path=tmp_path / "docs" / "methodology.pdf",
    )

    assert (output / "task1_pred.hdf5").is_file()
    assert (output / "task2_pred.hdf5").is_file()
    assert (output / "code" / "task1_infer.py").is_file()
    assert (output / "code" / "task2_infer.py").is_file()
```

- [ ] **Step 2: Implement `build_submission_workspace`**

```python
def build_submission_workspace(*, output_dir: str | Path, task_runs: Mapping[str, str | Path], methodology_path: str | Path) -> Path:
    output = Path(output_dir)
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    code_out = output / "code"
    code_out.mkdir()
    for task, run_path in task_runs.items():
        run = Path(run_path)
        for suffix in ("pred.hdf5", "time.csv", "logs.log"):
            source = run / f"{task}_{suffix}"
            if not source.exists():
                raise FileNotFoundError(source)
            shutil.copy2(source, output / source.name)
        merge_code_dir(run / "code", code_out)
    write_submission_json(output / "submission.json")
    shutil.copy2(methodology_path, output / "methodology.pdf")
    return output
```

- [ ] **Step 3: Add collision policy**

If both task runs contain the same relative code path with different content, fail with a clear error:

```text
Shared code collision: code/model.py differs between task1 and task2 runs.
```

- [ ] **Step 4: Run tests**

```powershell
& $PY -m pytest tests/test_submission_workspace.py -q
```

Expected: tests pass.

### Task 2: Add Agent Code Generation Workspace

**Files:**
- Create: `agent/code_generation_workspace.py`
- Test: `tests/test_code_generation_workspace.py`
- Modify: `agent/pde_executor.py`

- [ ] **Step 1: Test that a real LLM `code_patch` creates a per-run code snapshot**

```python
def test_code_patch_writes_snapshot_and_manifest(tmp_path):
    result = apply_agent_code_patch(
        code_root=tmp_path / "runs" / "task1" / "study" / "code",
        files=[{"path": "model.py", "content": "class Model: pass\n"}],
        provenance_record={"provider": "hkustgz_gpt", "model": "gpt-5.3-chat"},
    )
    assert (result.code_root / "model.py").read_text(encoding="utf-8") == "class Model: pass\n"
    assert (result.code_root / "code_manifest.json").is_file()
```

- [ ] **Step 2: Modify `code_patch` executor**

When the Planner outputs `action_type=code_patch`, write files to:

```text
runs/<task>/autonomous/<date>/<study>/nodes/<node_id>/code/
```

Then use that code snapshot for validation and later submission packaging, instead of silently relying on repository-level `code/`.

- [ ] **Step 3: Preserve repository `code/` as library source**

The repository-level `code/` remains the developer working copy and baseline implementation. Submitted `code/` must come from Agent-generated snapshots when `require_llm_code_trace=true`.

- [ ] **Step 4: Run tests**

```powershell
& $PY -m pytest tests/test_code_generation_workspace.py tests/test_pde_autonomous.py -q
```

Expected: code patches remain path-safe and produce traceable manifests.

### Task 3: Add Baseline Reader for `third_party/baseline`

**Files:**
- Create: `agent/baseline_reader.py`
- Test: `tests/test_baseline_reader.py`
- Modify: `agent/pde_observer.py`
- Modify: `agent/pde_planner.py`

- [ ] **Step 1: Test baseline repo indexing**

```python
def test_baseline_reader_indexes_known_repos(tmp_path):
    root = tmp_path / "third_party" / "baseline"
    (root / "PDEBench" / "pdebench" / "models" / "fno").mkdir(parents=True)
    (root / "PDEBench" / "pdebench" / "models" / "fno" / "fno.py").write_text("class FNO1d: pass\n", encoding="utf-8")
    index = index_baseline_repos(root)
    assert "PDEBench" in index
    assert any("fno.py" in item["path"] for item in index["PDEBench"]["files"])
```

- [ ] **Step 2: Implement baseline indexing**

Index only relevant files first:

```text
third_party/baseline/PDEBench/pdebench/models/fno/*.py
third_party/baseline/PDEBench/pdebench/models/pinn/*.py
third_party/baseline/PDEBench/pdebench/models/unet/*.py
third_party/baseline/neuraloperator/**/*.py
third_party/baseline/deeponet/**/*.py
third_party/baseline/Physics-informed-DeepONets/**/*.ipynb
```

Skip data, checkpoints, `.npy`, `.mat`, and generated caches.

- [ ] **Step 3: Add code summarization records**

For each selected file, create compact records:

```json
{
  "repo": "PDEBench",
  "path": "pdebench/models/fno/fno.py",
  "symbols": ["FNO1d", "SpectralConv1d"],
  "relevance": "Task1 FNO checkpoint architecture and reduced resolution training",
  "agent_use": "compare with local code/train_task1_fno_finetune.py"
}
```

- [ ] **Step 4: Expose summaries to Observer**

Add `baseline_context` to `observe_research_context()`:

```json
{
  "baseline_context": {
    "PDEBench": "...",
    "neuraloperator": "...",
    "deeponet": "...",
    "Physics-informed-DeepONets": "..."
  }
}
```

- [ ] **Step 5: Force Planner to cite baseline source**

For `code_patch`, `finetune_checkpoint`, `task2_train_model`, and `train_refiner`, require:

```json
"params": {
  "source_method": "PDEBench/FNO",
  "source_files": ["third_party/baseline/PDEBench/pdebench/models/fno/fno.py"]
}
```

- [ ] **Step 6: Run tests**

```powershell
& $PY -m pytest tests/test_baseline_reader.py tests/test_pde_research_capabilities.py -q
```

Expected: Planner prompt includes baseline context and source citation requirements.

### Task 4: Connect Task Logs to Research Logs

**Files:**
- Modify: `agent/task_log_export.py`
- Modify: `agent/task1_submission.py`
- Modify: `agent/task2_submission.py`
- Test: `tests/test_pde_method_library_and_logs.py`

- [ ] **Step 1: Export official task log from autonomous study**

Add command:

```powershell
& $PY -m agent.task_log_export --study-dir runs\task1\autonomous\20260515\<study> --output-path runs\<submission>\task1_logs.log --task task1 --code-dir runs\task1\autonomous\20260515\<study>\nodes\<best_node>\code
```

- [ ] **Step 2: Include four official stages**

Each exported log must include JSONL sections:

```text
problem_understanding
literature_method_trace
bottleneck_diagnosis
code_evolution
experiment_tracking
```

- [ ] **Step 3: Add validator test**

Check every line has:

```text
timestamp
elapsed_seconds
response
```

Expected: official log format passes local validation.

### Task 5: Wire Final Combined Submission

**Files:**
- Modify: `agent/final_submission.py`
- Modify: `agent/submission.py`
- Test: `tests/test_final_submission.py`

- [ ] **Step 1: Add CLI arguments**

```powershell
& $PY -m agent.final_submission `
  --run-name final-agent-generated `
  --task1-run runs\task1\autonomous\20260515\<study>\nodes\<node>\submission `
  --task2-run runs\task2\autonomous\20260515\<study>\nodes\<node>\submission `
  --require-llm-code-trace `
  --provenance-log runs\task1\autonomous\20260515\<study>\planner_logs.log `
  --provenance-log runs\task2\autonomous\20260515\<study>\planner_logs.log
```

- [ ] **Step 2: Merge task artifacts**

Use `build_submission_workspace()` to produce:

```text
runs/final-agent-generated/
  submission.json
  task1_pred.hdf5
  task1_time.csv
  task1_logs.log
  task2_pred.hdf5
  task2_time.csv
  task2_logs.log
  methodology.pdf
  code/
```

- [ ] **Step 3: Validate and pack**

Run:

```powershell
& $PY -m agent.validate_submission --path runs\final-agent-generated
& $PY -m agent.pack_submission --run runs\final-agent-generated
```

Expected:

```text
runs/final-agent-generated/pred.zip
```

### Task 6: Task2 AutoDL Improvement Loop

**Files:**
- Modify: `agent/run_task2_autonomous_experiment.py`
- Modify: `agent/pde_executor.py`
- Modify: `code/train_task2_models.py`
- Test: `tests/test_task2_autonomous.py`

- [ ] **Step 1: Add Task2 search templates**

Allow Planner to choose:

```text
minifno_nu
minifno
unet
all
```

with controlled knobs:

```text
epochs, hidden_channels, modes, lr, batch_size, sample_limit, nu_aux_weight
```

- [ ] **Step 2: Require train-from-scratch evidence**

Every Task2 training node must record:

```json
{
  "forbidden_task1_checkpoint_check": true,
  "train_files": ["data/Task2/task2_part0_train.h5", "..."],
  "test_nu_used": false
}
```

- [ ] **Step 3: Auto-submit only if selected beats persistence**

If `selected` is null, `task2_submit_best` must package persistence scaffold and mark it as scaffold. If selected exists, it must package the trained checkpoint prediction.

- [ ] **Step 4: Run smoke and tests**

```powershell
& $PY -m agent.run_task2_autonomous_experiment --config configs\closed_loop_hkustgz.yaml --study-name task2-smoke --max-iterations 1 --bootstrap-train
& $PY -m pytest tests/test_task2_autonomous.py tests/test_task2_train_from_scratch.py -q
```

Expected: one Task2 node completes, journal records metrics, and no Task1 data/checkpoint leakage is detected.

### Task 7: Full Verification

**Files:**
- No new files.

- [ ] **Step 1: Run full tests**

```powershell
& $PY -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 2: Run one Task1 and one Task2 autonomous smoke**

```powershell
& $PY -m agent.run_task1_autonomous_experiment --config configs\closed_loop_hkustgz.yaml --study-name task1-smoke --max-iterations 1 --metric competition_score_proxy --maximize --bootstrap-postprocess-search
& $PY -m agent.run_task2_autonomous_experiment --config configs\closed_loop_hkustgz.yaml --study-name task2-smoke --max-iterations 1 --bootstrap-train
```

Expected: both write under classified `runs/task*/autonomous/YYYYMMDD/...`.

- [ ] **Step 3: Build final strict submission**

```powershell
& $PY -m agent.final_submission --run-name final-agent-generated --require-llm-code-trace --provenance-log runs\task1\autonomous\<date>\<study>\planner_logs.log --provenance-log runs\task2\autonomous\<date>\<study>\planner_logs.log
```

Expected: final validator passes only if submitted `code/` has real LLM `code_patch` provenance.
