# Live Model Validation Report

- Generated at: `2026-03-13T14:37:12.475344+00:00`
- Generation model: `gpt-4.1-mini`
- Judge model: `gpt-4.1-mini`
- Dataset: `/Users/justin/Documents/New project/repo_inspect/artifacts/live_runtime_probe_tasks.json`

## Dataset

- Train tasks per round: `4`
- Train rounds: `1`
- Expanded train tasks: `4`
- Holdout tasks: `4`

## Aggregate Metrics

| Split | Policy | Mean objective reward | Contract pass rate | Mean path length | Branch HHI |
| --- | --- | ---: | ---: | ---: | ---: |
| train | adaptive_full | 0.7508 | n/a | 2.0000 | 0.2500 |
| train | frozen_forest | 0.7633 | n/a | 2.0000 | 0.2500 |
| train | direct_model | 0.5650 | n/a | 1.0000 | 1.0000 |
| holdout | adaptive_full | 0.7240 | 1.0000 | 2.0000 | 0.3750 |
| holdout | frozen_forest | 0.7115 | 1.0000 | 2.0000 | 0.2500 |
| holdout | direct_model | 0.5462 | 1.0000 | 1.0000 | 1.0000 |

## Holdout Comparisons

- Adaptive vs frozen objective gain: `0.0125`
- Adaptive vs direct objective gain: `0.1778`
- Frozen vs direct objective gain: `0.1653`

## Pairwise Judge Summary

- `adaptive_full__vs__frozen_forest`: left win rate `0.0`, right win rate `0.25`, ties `3/4`
- `adaptive_full__vs__direct_model`: left win rate `1.0`, right win rate `0.0`, ties `0/4`
- `frozen_forest__vs__direct_model`: left win rate `0.75`, right win rate `0.25`, ties `0/4`

## Notes

- This evaluation uses a real closed-source model for generation and judging, but the task set is curated rather than benchmark-standard.
- Explicit human feedback loops were not exercised because no real user labels were available.
- Pairwise judging uses an LLM judge, so those results should be read alongside the objective heuristic metrics.
