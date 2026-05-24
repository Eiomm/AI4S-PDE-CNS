Draft a new Task2 candidate as the first step of a scientific research loop:

[hypothesis] -> [code] -> [experiment] -> [metrics] -> [next hypothesis]

The scheduler will run your generated `train.py` and feed metrics back into later prompts. Your response is the scientific record for this code change, so make the hypothesis clear.

## Task2 Problem Summary
- Burgers' equation on a periodic 1D spatial domain with 256 points.
- Train: 3000 trajectories x 320 timesteps. Val: 100 trajectories x 210 timesteps.
- Test: 1000 trajectories x 10 initial timesteps; predict steps 10..199.
- Train/val include viscosity `nu`; test does not. Infer or marginalize `nu` from the first 10 frames.

## Hard Constraints
- Use only data from `data/Task2`.
- Do not read test `nu`.
- Do not use numerical PDE solvers to generate future test trajectories.
- Do not load public pretrained weights or external checkpoints.
- Produce `task2_pred.hdf5` with dataset `tensor` shaped `(1000, 200, 256)`, `float32`, finite.
- Copy `task2_test.h5/tensor` into `tensor[:, :10, :]` within `1e-3`.
- CLI must work: `python train.py --data-dir ../../../data/Task2 --out-dir ../artifacts --epochs 1 --cheap-probe`.
- End by printing: `Final Validation Score: <overall_mse>`.

## Candidate Autonomy
- Put lightweight AutoML/search inside the generated code. Expose `AUTO_ML_SEARCH_SPACE`, and when practical support `--automl-trials` so the candidate can tune a small set of architecture/training choices itself.
- The scheduler will not hardcode detailed trial parameters. The generated code should pick sensible defaults for cheap probes and fuller training.
- Keep logging in JSONL with `timestamp`, `elapsed_seconds`, and either `response` or `tool_calls`.

## Output Structure
Your response must start with:

### Hypothesis name
<short, human-readable node name>

### Hypothesis
<what mechanism should improve Task2 performance>

### Experiment design
<what the generated code will do and what metric movement would support or falsify it>

Then output the complete `train.py` inside a single Python code block.
