# Training Code Notes

The current submission is inference-first. No supervised model training is required to run the agent, but the image includes the reproducible preprocessing code used to build the benchmark descriptor prior that guides property-aware generation.

## Runtime components

1. Pocket-conditioned molecule generation through an external SOTA SBDD command when `AI4S_SBDD_GENERATOR_CMD` is set.
2. Optional LLM SMILES generation through LiteLLM.
3. RDKit internal generation for local smoke runs.
4. Strict route validation with no self-reaction routes.
5. Vina/OpenBabel scoring for docking and competition mode.
6. Submission packaging into `/saisresult/result.zip`.

The official online runner does not execute the files under `/app/training_code`; they are included as review material.

## Environment

The inference and preprocessing environment is defined by the repository root files copied into the Docker image:

- `Dockerfile`
- `environment.docker.yml`
- `requirements-docker-core.txt`
- `requirements-docker-retro.txt`

Main tools used by the agent:

- Python 3.10
- RDKit
- AutoDock Vina
- OpenBabel
- AiZynthFinder
- LiteLLM for optional API-based LLM calls

## Benchmark prior preprocessing

Build the aggregate property prior from local DUD-E validation data:

```bash
python app/training_code/build_benchmark_prior.py \
  --manifest data/benchmarks/manifest.yaml \
  --output data/benchmarks/benchmark_prior.json
```

Input data used for this preprocessing step:

- Local processed DUD-E benchmark target folders listed in `data/benchmarks/manifest.yaml`
- Per-target `target.pdb`
- Per-target `actives.smi` and `decoys.smi`
- Per-target `metadata.json`

Output:

- `data/benchmarks/benchmark_prior.json`

The output contains median/q1/q3 descriptor summaries for benchmark actives and decoys. It intentionally does not store benchmark molecular strings and is not a fixed candidate library. The agent reads it as context for property-aware generation.

To keep the submitted training-code review package small and avoid packaging fixed benchmark molecule libraries, the image includes this preprocessing script and the aggregate prior, but not the raw DUD-E molecule files.

## Image packaging

During Docker build, the repository `app/training_code` directory is copied to:

```text
/app/training_code
```

Current contents:

```text
/app/training_code/README.md
/app/training_code/build_benchmark_prior.py
```

This directory is far below the 5 GB review limit.

Keep API keys out of source control.
