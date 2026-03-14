# Live Model Validation Report

- Generated at: `2026-03-13T14:31:14.657478+00:00`
- Generation model: `gpt-4.1-mini`
- Judge model: `gpt-4.1-mini`
- Dataset: `/Users/justin/Documents/New project/repo_inspect/examples/live_eval_tasks.json`

## Dataset

- Train tasks per round: `12`
- Train rounds: `2`
- Expanded train tasks: `24`
- Holdout tasks: `12`

## Aggregate Metrics

| Split | Policy | Mean objective reward | Contract pass rate | Mean path length | Branch HHI |
| --- | --- | ---: | ---: | ---: | ---: |
| train | adaptive_full | 0.7557 | n/a | 2.0000 | 0.1910 |
| train | frozen_forest | 0.7432 | n/a | 2.0000 | 0.1319 |
| train | direct_model | 0.5655 | n/a | 1.0000 | 1.0000 |
| holdout | adaptive_full | 0.6922 | 1.0000 | 2.0000 | 0.2361 |
| holdout | frozen_forest | 0.6964 | 1.0000 | 2.0000 | 0.2222 |
| holdout | direct_model | 0.5732 | 1.0000 | 1.0000 | 1.0000 |

## Holdout Comparisons

- Adaptive vs frozen objective gain: `-0.0042`
- Adaptive vs direct objective gain: `0.119`
- Frozen vs direct objective gain: `0.1232`

## Pairwise Judge Summary

- `adaptive_full__vs__frozen_forest`: left win rate `0.1667`, right win rate `0.1667`, ties `8/12`
- `adaptive_full__vs__direct_model`: left win rate `0.75`, right win rate `0.0833`, ties `2/12`
- `frozen_forest__vs__direct_model`: left win rate `0.5`, right win rate `0.25`, ties `3/12`

## Notes

- This evaluation uses a real closed-source model for generation and judging, but the task set is curated rather than benchmark-standard.
- Explicit human feedback loops were not exercised because no real user labels were available.
- Pairwise judging uses an LLM judge, so those results should be read alongside the objective heuristic metrics.
