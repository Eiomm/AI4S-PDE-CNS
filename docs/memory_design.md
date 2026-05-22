# Task1 Long-Term Agent Memory Design

## Goal

Memory is not a chat transcript. It is a compact research index that lets the
next Agent retrieve only the few facts needed for the next experiment:

- what has been tried;
- what worked;
- what failed and should not be repeated;
- which artifact/checkpoint/prediction proves the claim;
- which candidate is currently best under a named metric.

Raw logs, full code diffs, full training histories, and full LLM conversations
remain in run directories. Long-term memory stores structured summaries and
pointers.

## External Design Basis

The design follows four mainstream patterns:

1. **Event trace vs. memory split**: OpenAI Agents SDK uses sessions for
   conversation history and tracing for auditable runs; our official proxy logs
   are trace/audit, not prompt memory.
2. **Working / episodic / semantic memory split**: common in LangGraph/LangMem
   and Generative Agents. Current-run state is working memory; experiment
   records are episodic; stable lessons are semantic.
3. **Virtual context / archival memory**: MemGPT-style systems keep large memory
   outside the model context and retrieve relevant chunks on demand.
4. **Reflection / lesson consolidation**: Reflexion-style agents convert trial
   outcomes into short verbal lessons. We do the same, but only from measured
   experiments.

2025-specific findings that shape the implementation:

- Memory quality control matters more than just adding more records. Recent
  empirical work shows that agents tend to follow retrieved experiences; bad
  or misaligned memories can propagate errors. Therefore every memory record
  must carry validation status, decision, and promotability.
- Episodic memory is especially important for long-horizon agents. For us,
  "episode" means one experiment attempt with hypothesis, code/config delta,
  metrics, artifacts, and review.
- Benchmarks now evaluate memory by retrieval, test-time learning,
  long-range understanding, and selective forgetting. Our memory APIs should
  expose exactly these operations: query, append, promote/update, and retire.
- Production memory systems such as Mem0 and Zep emphasize dynamic extraction,
  consolidation, retrieval, temporal/graph structure, and lower token cost than
  full-context replay. We should start with schema-first JSONL, then add
  embeddings/graph indexes only for retrieval over compact summaries.

## Repository Layout

```text
memory/
├── contract/
│   └── task1_rules.yaml
├── working/
│   ├── current_context.json
│   └── latest_error.log
├── episodic/
│   └── runs.jsonl
├── findings/
│   ├── validated_findings.yaml
│   ├── negative_findings.yaml
│   └── metric_leaderboard.csv
├── failures/
│   ├── failure_bank.yaml
│   └── bug_signatures.yaml
├── procedures/
│   ├── load_fno_checkpoint.md
│   ├── downsample_burgers.md
│   ├── evaluate_task1.md
│   └── make_submission.md
└── wisdom/
    └── strategy_summary.md
```

This follows the user's external survey:

- MLEvolve: each experiment is a searchable node with plan, code/config delta,
  metrics, success/failure, and compliance status.
- ML-Master HCC: transient execution traces are distilled into stable
  long-term knowledge.
- A-MEM / Zettelkasten: memories carry tags/topics and can later be linked.
- MemoryOS / MemOS: memory has storage, update, retrieval, generation APIs and
  remains inspectable/editable.
- MemP: procedural memory is first-class, so repeated operations become
  reusable procedures.
- RMM / Graphiti / Mem0: retrieval quality, temporal validity, provenance, and
  hybrid metadata retrieval matter.

Run-local memory:

```text
runs/task1/<UTC timestamp>/
├── metadata.json
├── metrics.json
├── task1_logs.log
└── memory_export.json
```

`memory_export.json` is the small, validated summary that can be appended to
`memory/episodic/runs.jsonl`.

## Record Types

### Experiment Record

One completed experiment or failed attempt.

```json
{
  "schema": "task1_experiment_record_v1",
  "record_id": "task1:20260521:official_fno_val",
  "task": "task1",
  "created_at": "2026-05-21T00:00:00Z",
  "route": "official_fno",
  "hypothesis": "Official FNO baseline establishes replay parity.",
  "changes": {
    "model": "fno",
    "checkpoint": "checkpoints/official/nu0.001_fno.pt",
    "ensemble": null,
    "postprocess": null
  },
  "metrics": {
    "competition_score_proxy": 5.138304068561983,
    "mse": 0.4238491429584984,
    "forecast_mse": 0.44615699258789276,
    "long_horizon_mse": 0.7622751701625805
  },
  "validation": {
    "shape": [100, 200, 256],
    "first_ten_match": true,
    "finite": true
  },
  "artifacts": {
    "run_dir": "runs/task1/official_fno_val",
    "prediction": "runs/task1/official_fno_val/task1_pred.hdf5",
    "metrics": "runs/task1/official_fno_val/metrics.json",
    "checkpoint_sha256": "51fce18b715f8507383171fe3993ee88969c5e0d0aa44c37ff3fe137001fa480"
  },
  "decision": "baseline",
  "promotable": false,
  "quality": {
    "artifact_exists": true,
    "metrics_verified": true,
    "log_trace_available": true,
    "memory_review": "keep"
  },
  "tags": ["official", "fno", "baseline"]
}
```

### Lesson Record

Only stable cross-run conclusions. These are tiny.

```json
{
  "schema": "task1_lesson_record_v1",
  "lesson_id": "task1:lesson:stride5",
  "task": "task1",
  "kind": "training_rule",
  "statement": "Fine-tuning official FNO must use temporal_stride=5 to match the checkpoint time scale.",
  "evidence_records": ["task1:20260520:stage1_scaleup"],
  "confidence": "high",
  "do_not_repeat": [
    "Do not fine-tune raw adjacent frames with temporal_stride=1 for official FNO descendants."
  ]
}
```

### Candidate Record

Small current-best board. This prevents searching every old run.

```json
{
  "schema": "task1_candidate_record_v1",
  "slot": "best_clean_official_lineage",
  "record_id": "task1:20260520:stage1_scaleup",
  "metric": "competition_score_proxy",
  "value": 81.3140561104237,
  "lineage_root": "checkpoints/official/nu0.001_fno.pt",
  "submit_ready": false,
  "blockers": ["needs full replay under current Task1 root", "needs official proxy code trace"]
}
```

## Write Policy

Write long-term memory only after an observable event:

- prediction completed;
- validation metrics written;
- submission validation passed or failed;
- Agent-generated code patch applied and tested;
- candidate promoted or rejected.

Never write:

- raw LLM conversations;
- full logs;
- full training curves;
- large arrays;
- speculative ideas without metrics.

## Read Policy

The Agent prompt receives a compact retrieval packet, not the whole memory.
Current implementation intentionally reads only four sources:

1. `memory/contract/task1_rules.yaml`
2. `memory/episodic/runs.jsonl`
3. `memory/findings/metric_leaderboard.csv`
4. `memory/wisdom/strategy_summary.md`

It does not include failures/procedures/findings YAML by default. Those files
remain human-readable references until we need them.

Example packet:

```json
{
  "best_candidates": [
    {"slot": "best_clean_official_lineage", "value": 81.3140561104237, "blockers": ["needs replay"]}
  ],
  "relevant_lessons": [
    {"lesson_id": "task1:lesson:stride5", "statement": "..."}
  ],
  "recent_failures": [
    {"route": "official_fno", "reason": "low proxy baseline; use only as sanity check"}
  ],
  "retrieval_budget": {
    "max_records": 8,
    "max_chars": 6000
  }
}
```

Retrieval order:

1. hard constraints and current best candidate;
2. exact route/tag matches;
3. most recent failures for the same edit type;
4. semantic search over `lessons.jsonl` only if needed.

Prompt budget policy:

- hard constraints: always included, max 1200 chars;
- current best board: max 5 candidates;
- relevant lessons: max 6 records;
- recent failures: max 5 records;
- experiment examples: max 3 records, only if same route/edit type;
- total memory packet: default 6000 chars, hard max 10000 chars.

This prevents the old failure mode where the full knowledge base is pasted into
the model and consumes the useful context window.

## Implementation Phases

Phase 1:

- Write JSONL registry and small JSON candidate board.
- Add `scripts/memory_export.py`, `scripts/memory_query.py`,
  `scripts/memory_promote.py`.
- Retrieval is deterministic filtering by task, route, tags, metric.
- Add memory quality labels: `keep`, `retire`, `promote_candidate`,
  `do_not_repeat`.

Script responsibilities:

- `memory_export.py`: turn one run directory into `memory_export.json`; with
  `--append-registry`, append it to `memory/episodic/runs.jsonl`.
- `memory_query.py`: return a small prompt-ready retrieval packet under a
  strict record/character budget.
- `memory_promote.py`: update `memory/findings/metric_leaderboard.csv`; this is only a
  candidate board update, not final submission approval.

Phase 2:

- Add `embeddings.sqlite` for semantic lesson search.
- Embed only `lesson.statement`, `hypothesis`, `decision`, and tags.
- Never embed raw logs or full code.
- Add optional temporal edges:
  `parent_record_id`, `derived_from`, `supersedes`, `contradicts`.

Phase 3:

- Integrate GPT-5.5 Agent:
  - prompt receives retrieval packet;
  - generated code patch is logged through OpenAI proxy;
  - successful run exports compact memory;
  - promotion requires memory record plus live artifact validation.

## Guardrails

- Memory cannot promote a checkpoint by itself.
- Any record referencing a missing artifact is downgraded to historical context.
- Task1 memory cannot be loaded by Task2/Task3 except through common method
  notes that contain no checkpoint paths.
- Human statements can enter memory as constraints or hypotheses, not as
  measured results.
