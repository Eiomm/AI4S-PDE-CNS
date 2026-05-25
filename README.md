# AI4S-PDE-CNS

Team **USAIL** entry for the 4th World Scientific Intelligence Challenge —
Track 4 (AI4S Agent CNS, PDE neural-operator track).

The repository wires a ReAct-style auto-research agent (Claude Opus 4.7
through an OpenAI-compatible proxy) around the AIDE workflow, runs it on
three PDE-prediction tasks end-to-end, and packages the results into the
competition's required submission layout.

## Layout

```text
tasks/                                    # what the agent sees
├── ai4s-pde-task1-burgers-fixed/         # fixed Nu=0.001 Burgers, FNO checkpoint allowed
│   ├── description.md                    # full agent brief, with inline FNO + recipe
│   ├── config.yaml
│   ├── burgers_FNO/                      # PDEBench checkpoints (.pt, gitignored)
│   ├── data/                             # test+val+training hdf5 (gitignored)
│   ├── src/ai4sv2_task1/                 # bundled FNO loader/rollout (optional)
│   └── run/                              # sandbox runs land here at execution
├── ai4s-pde-task2-burgers-multinu/       # multi-Nu Burgers, from scratch
│   ├── description.md
│   ├── config.yaml
│   └── data/                             # 3 part_train.h5 + val.h5 + test.h5 (gitignored)
└── ai4s-pde-task3-ks-multiparam/         # KS chaotic system, from scratch
    ├── description.md
    ├── config.yaml
    └── data/                             # KS_train / KS_val / KS_test .hdf5 (gitignored)

dslighting/                               # vendored AIDE framework
├── services/llm/executor.py              # patched to accept Claude's ```json fences
└── ... (workflow / monitoring / debug / ...)

scripts/                                  # one-click drivers + packagers
├── run_ai4s_aide_task.py                 # underlying AIDE runner
├── aide_task1_claude_one_click.sh        # per-task one-click (Claude through gpt.ge)
├── aide_task2_claude_one_click.sh
├── aide_task3_claude_one_click.sh
├── aide_task1_gpt55_one_click.sh         # gpt-5.5 variant (kept for ablation)
├── build_task_submission.py              # per-task pred/log/csv packager
├── build_task1_submission.py             # task-1 specific legacy variant
├── build_final_submission.py             # aggregate 3 tasks + code/ + methodology
├── build_methodology_pdf.py              # render methodology.pdf via fpdf2
├── run_all_and_submit.sh                 # full pipeline: 3 AIDE runs + assemble
├── rsync_to_hpc_parallel.sh              # 7-stream parallel upload to HPC
├── methodology.pdf                       # rendered methodology document
├── submission.json                       # {"submission_id":"usail",...}
├── task1_finetune_smoketest.py           # verified-recipe reproducer
├── task1_finetune_inspect.py             # diagnostic helper
└── export_dslighting_llm_io.py           # normalize debug payloads

task_log_sample/openai-log/proxy.py       # official LLM call proxy (writes JSONL)
src/ai4sv2_task1/                         # repo-side bundled task1 helpers
tests/test_task1_core.py                  # smoke tests
```

Large artifacts are deliberately not tracked (see `.gitignore`):
`.venv/`, `outputs/`, `submission/`, `submission.zip`, every `*.hdf5` /
`*.h5` / `*.pt` (training data, checkpoints, predictions), and the two
contest-provided sample-submission bundles
(`data_and_sample_submission/`, `task3_data_sample_submission/`).

## Architecture in one paragraph

Each task is solved by an autonomous **ReAct loop**: the LLM produces a
*Plan* + a complete Python script, the script runs in a hermetic
sandbox (the *Act*), a judge LLM consumes the result and returns a
structured *Observation*, and the search policy decides whether to
improve, debug, or redraft. Every LLM call passes through a logging
proxy that writes a contest-format JSONL log, giving a complete audit
trail from the agent's decisions to the final code. The methodology
document at `scripts/methodology.pdf` describes this in more detail.

## Environment

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt \
  -i https://pypi.tuna.tsinghua.edu.cn/simple
```

`.env` must define the model and key settings consumed by the AIDE
runner:

```text
AI4S_AGENT_MODEL       # e.g. claude-opus-4-7
AI4S_AGENT_BASE_URL    # http://127.0.0.1:8080/v1
APIFOX_GPT_GE_API_KEY  # or OPENAI_API_KEY / VAPI_API_KEY
```

## Running one task

```bash
# Task 1: fixed-Nu Burgers, FNO checkpoint allowed
bash scripts/aide_task1_claude_one_click.sh

# Task 2: multi-Nu Burgers, train from scratch
bash scripts/aide_task2_claude_one_click.sh

# Task 3: Kuramoto-Sivashinsky, train from scratch
bash scripts/aide_task3_claude_one_click.sh
```

Each script:

1. starts the official logging proxy on its own port (default
   `8080/8081/8082` — overridable via `AI4S_PROXY_PORT`)
2. invokes `scripts/run_ai4s_aide_task.py <N>` with `--max-iterations`,
   `--keep-workspace`, and the structured-debug flags
3. writes the run to `outputs/aide_task{N}_claude/<stamp>/`
   (`workspace/.../sandbox/task{N}_pred.hdf5`,
   `workspace/.../sandbox/task{N}_inference_time.txt`,
   `llm_io/llm-*.jsonl`, `aide_stdout_stderr.log`, ...)

## Building the submission

After all three tasks have at least one successful run:

```bash
bash scripts/run_all_and_submit.sh                  # run 3 tasks + assemble + zip
SKIP_RUN=1 bash scripts/run_all_and_submit.sh       # just assemble (skip the agent)
SKIP_TASK3=1 bash scripts/run_all_and_submit.sh     # tasks 1+2 only
PARALLEL=1 bash scripts/run_all_and_submit.sh       # run all 3 agents concurrently
```

This produces the official submission tree:

```text
submission/
├── submission.json
├── methodology.pdf
├── task1_pred.hdf5
├── task1_time.csv
├── task1_logs.log
├── task2_pred.hdf5
├── task2_time.csv
├── task2_logs.log
├── task3_pred.hdf5
├── task3_time.csv
├── task3_logs.log
└── code/
    ├── task1/step01_<hash>.py    # every LLM-generated sandbox script,
    ├── task1/step02_<hash>.py    # in execution order, hash matches
    ├── task1/step03_<hash>.py    # an LLM call in task{N}_logs.log
    ├── task2/...
    └── task3/...
```

…and `submission.zip` alongside it (≈ 800 MB at full three-task scale).

### Timing details

- `train_time`  = wall-clock from the first to the last LLM call in
  `llm_io/llm-*.jsonl` (includes all agent thinking time, as required).
- `inference_time` = the agent's own `time.perf_counter()` around the
  test-set rollout, persisted both as `task{N}_inference_time.txt` and
  printed to stdout as `INFERENCE_TIME=<sec>`. The packager prefers
  the txt file, falls back to stdout, and re-measures locally as a
  last resort. Task 2 and Task 3 also enforce the 2-minute hard cap
  with an `assert inference_time < 120` in the agent script.

## What worked for Task 1 (verified)

Zero-shot rollout of the released FNO checkpoint scores
`weighted_score ≈ 0.4457` on `task1_val.hdf5`. The agent identified
that this baseline diverges past frame ~50 of the 190-step rollout
because the checkpoint was trained with 1-step loss only. A two-epoch
fine-tune on the PDEBench training file with a 5-step rollout loss,
gradient clipping `clip_grad_norm_=1.0`, AdamW `lr=1e-4`, and a random
temporal window drops `weighted_score` to **0.000340** — a **>1300×**
improvement — in about 70 seconds on CPU. The full reproducer is
`scripts/task1_finetune_smoketest.py`; the recipe is also inlined in
`tasks/ai4s-pde-task1-burgers-fixed/description.md`.

## Running on HPC

```bash
# parallel rsync to HPC account 2 via the VPN-tunnelled ssh proxy
bash scripts/rsync_to_hpc_parallel.sh

# remote location
~/projects/AI4S-PDE-CNS/
```

The HPC has a Python 3.12 environment; recreate the venv there before
running anything (the local `.venv/` is gitignored and is macOS-only
binary).

## Logging proxy details

The proxy at `task_log_sample/openai-log/proxy.py` intercepts every
`/v1/chat/completions` call, writes a JSONL entry with the contest's
required fields (`timestamp`, `elapsed_seconds`, `request`, `response`
or `tool_calls`), and forwards the request to the upstream OpenAI-
compatible endpoint. We patched `dslighting/services/llm/executor.py`
to strip ` ```json … ``` ` markdown fences from response bodies before
Pydantic validation, since Claude through the OpenAI shim wraps
structured-output JSON in fences by default — without the patch the
judge step crashes with `LLM-002 invalid JSON`.

## License & attribution

Internal challenge submission. The `dslighting/` framework is vendored;
the patches we applied are intentionally minimal and called out in the
methodology document.
