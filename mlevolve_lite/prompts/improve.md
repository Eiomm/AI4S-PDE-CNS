Improve one existing Task2 candidate as a scientific research loop:

[hypothesis] -> [code] -> [experiment] -> [metrics] -> [next hypothesis]

The scheduler will run your generated `train.py` and record metrics. Your job is to propose the next hypothesis and write the code that tests it.

## Task2 Constraints
- Use only data from `data/Task2`.
- Do not read test `nu`; infer or marginalize viscosity from the first 10 test frames.
- Do not use numerical PDE solvers to generate future test trajectories.
- Do not load public pretrained weights or external checkpoints.
- Produce `task2_pred.hdf5` with dataset `tensor` shaped `(1000, 200, 256)`, `float32`, finite.
- Copy `task2_test.h5/tensor` into `tensor[:, :10, :]` within `1e-3`.
- CLI must work: `python train.py --data-dir ../../../data/Task2 --out-dir ../artifacts --epochs 1 --cheap-probe`.
- End by printing: `Final Validation Score: <overall_mse>`.

## Research Instructions
- Start from the parent metrics and recent attempts. Name the concrete failure mode you are testing.
- Choose one meaningful path that differs from previous attempts. Avoid edits that only tweak one scalar value.
- Put lightweight AutoML/search inside the generated code. Expose `AUTO_ML_SEARCH_SPACE`, and when practical support `--automl-trials` so the code can choose among a few architecture/training variants.
- Keep experiment runtime cheap under `--cheap-probe`; the search should be small and robust.
- Write JSONL reasoning/metrics from `train.py` with `timestamp`, `elapsed_seconds`, and either `response` or `tool_calls`.

## Output Structure
Your response must start with:

### Hypothesis name
<short, human-readable node name>

### Feedback used
<which parent metrics or failures motivated this>

### Experiment design
<what code path will test the hypothesis and what metric movement would support it>

Then output the complete replacement `train.py` inside a single Python code block.
