# Task 1 Active Experiment Results

- Updated: 2026-05-14
- Scope: compliant Task 1 experiments using only the official `Nu0.001_FNO` and `Nu0.001_Unet-PF-20` checkpoints.
- Ranking below is sorted by MSE. The current best compliant local candidate is a segment-wise official checkpoint blend with a persistence stabilizer.
- Archived exploratory runs are intentionally omitted from the active ranking because they used assets outside the current final-submission policy.

| Rank | Run | Model | Weights | Proxy | MSE | Forecast MSE | Long MSE | Segment3 RMSE | Status |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `task1-official-ensemble-optimization/persist_piecewise_best_score` | official checkpoint ensemble + persistence | segment FNO `0.17/0.03/0.11`, persistence alpha `0.89/0.95/0.41` | `18.86073947` | `0.0315804837` | `0.03324261438` | `0.03868998039` | `0.1966976878` | active |
| 2 | `task1-mcts-validation-smoke/current-final-best-mse` | official checkpoint ensemble | `nu0.001=0.12`, `unet_pf20_nu0.001=0.88` | `13.03828949` | `0.0582305179` | `0.06129528198` | `0.09366471407` | `0.3060469148` | active |
| 3 | `task1-mcts-validation-smoke/proxy-score-blend` | official checkpoint ensemble | `nu0.001=0.04`, `unet_pf20_nu0.001=0.96` | `13.83318215` | `0.0611382528` | `0.06435605561` | `0.09708481833` | `0.3115843679` | active |
| 4 | `task1-mcts-validation-smoke/unet-only` | official checkpoint ensemble | `nu0.001=0.00`, `unet_pf20_nu0.001=1.00` | `13.39284111` | `0.0648512424` | `0.06826446572` | `0.10280756547` | `0.3206361887` | active |
| 5 | `_verify_official_ensemble/task1_val_fno_only` | official FNO only | `nu0.001=1.00`, `unet_pf20_nu0.001=0.00` | `5.13830407` | `0.4238491430` | `0.44615699259` | `0.76227517016` | `0.8730837131` | diagnostic |

## Historical Results Policy

The previous full table contained strong local exploratory results, but those rows are not active ranking evidence for the compliant track.

Use archived runs only for diagnosis; do not use them for final Task 1 packaging unless the rules explicitly permit those checkpoints and fine-tuned assets.

## Final Packaging Selection

The current conservative final package should use Rank 1:

```text
Task 1 segment FNO weights: 0.17, 0.03, 0.11
Task 1 persistence segment alpha: 0.89, 0.95, 0.41
Task 2: persistence scaffold only
Output: runs/final-official-ensemble-postprocess-task2-persistence/pred.zip
```
