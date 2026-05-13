# Baseline Adaptation Notes

This project keeps the current lightweight AI4S PDE agent as the control plane and uses
external agent frameworks as references instead of copying them into the submission path.

## Local Reference Repositories

The following repositories are cloned under `third_party/` for local inspection and
adaptation experiments:

- `third_party/dslighting` at `47d8c0d`
- `third_party/ML-Master` at `6a7bf55`

They are ignored by git because they are external source trees. Keep production
submission code under `code/`, agent orchestration under `agent/`, and experiment output
under each `runs/<experiment>/` directory.

## Packaging Rule

Every experiment that is ready for submission should produce its compressed artifact at:

```text
runs/<experiment>/pred.zip
```

Do not create a separate central submission directory under `runs/` for normal
experiments. The experiment directory is the single source of truth for prediction,
timing, logs, copied submission code, methodology, and the final zip.

## DSLighting-Inspired Baseline Layer

Use DSLighting as the guide for task-oriented workflow organization:

- `TaskSpec`: task id, input data path, validation target path, output shape, and timing rules.
- `Workflow`: ordered stages such as prepare data, run inference, evaluate, package, and record.
- `AgentResult`: success flag, score/metrics, duration, output path, and error details.

The first local adaptation should be a small `Task1FNOWorkflow` wrapper around the
existing FNO ensemble code. It should make repeated validation experiments easier without
changing the current agent framework heavily.

## ML-Master-Inspired Search Layer

Use ML-Master as the guide for long-horizon experiment search:

- Separate exploration from reasoning.
- Cache experiment outcomes in a compact memory file.
- Feed only the strongest findings back into the next planning step.

For this project, the first memory artifact should be:

```text
runs/<experiment>/experiment_memory.json
```

It should summarize model/checkpoint choices, weights, validation metrics, command lines,
and conclusions. Later experiments can read prior memory before deciding whether to try
fine-tuning, ensembling, or Task 2 training.

## Next Baseline Target

The current best Task 1 validation result is the weighted FNO ensemble:

```text
mse: 0.001679162949145986
forecast_mse: 0.0017675399464694265
long_horizon_mse: 0.001573041832931417
```

The next baseline should improve from this point, not from the zero-train baseline.
Recommended order:

1. Stabilize `Task1FNOWorkflow` with repeatable validation and packaging.
2. Add a small search over ensemble weights using validation MSE as the objective.
3. Add short Task 1 fine-tuning only if the validation workflow is reliable.
4. Keep Task 2 isolated and train from scratch.
