# Codex Routing Divergence Benchmark

- Generated at: `2026-03-14T13:42:17.691868+00:00`
- Generation model: `gpt-4.1-mini`
- Judge model: `gpt-4.1-mini`
- Dataset: `/Users/justin/Documents/New project/repo_inspect/artifacts/routing_micro_probe_dataset.json`
- Train sibling candidates per task: `2`
- Holdout sibling candidates per task: `1`

## Aggregate Metrics

| Split | Policy | Mean objective reward | Mean path length | Branch HHI |
| --- | --- | ---: | ---: | ---: |
| train | full_adaptive | 0.7667 | 3.0000 | 0.5000 |
| train | frozen | 0.7958 | 3.0000 | 0.5000 |
| holdout | full_adaptive | 0.7667 | 2.0000 | 0.5000 |
| holdout | frozen | 0.7333 | 2.0000 | 0.5000 |

## Divergence Summary

- Holdout objective gain (adaptive - frozen): `0.0334`
- Selected-branch divergence: `1/2`
- Activated-path divergence: `1/2`
- Mean reward delta on divergent tasks: `0.0667`
- Mean reward delta on non-divergent tasks: `0.0`

## Pairwise Judge

- Adaptive wins: `0`
- Frozen wins: `2`
- Ties: `0`
- Mean score adaptive: `7.0`
- Mean score frozen: `9.0`

## By Aspect

| Aspect | n | Adaptive mean | Frozen mean | Delta | Branch diffs | Adaptive branches | Frozen branches |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| code_consistency_audit | 1 | 0.6583 | 0.6583 | 0.0000 | 0 | {'verification_constraint_checker': 1} | {'verification_constraint_checker': 1} |
| planning_risk | 1 | 0.8750 | 0.8083 | 0.0667 | 1 | {'planner_risk_allocator': 1} | {'planner_timeline_optimizer': 1} |

## Divergent Holdout Tasks

| Task | Aspect | Adaptive branch | Frozen branch | Adaptive | Frozen | Delta |
| --- | --- | --- | --- | ---: | ---: | ---: |
| holdout_divergence_planning_risk_admin | planning_risk | planner_risk_allocator | planner_timeline_optimizer | 0.8750 | 0.8083 | 0.0667 |

## Top Learned Weight Deltas

| Branch | Adaptive weight | Frozen weight | Delta |
| --- | ---: | ---: | ---: |
| planner_risk_allocator | 0.9664 | 0.9300 | 0.0364 |

## Usage

- adaptive_full backend usage: `{'call_count': 16, 'ok_calls': 16, 'error_calls': 0, 'prompt_tokens': 3942, 'completion_tokens': 8162, 'total_tokens': 12104, 'total_latency_ms': 121778.001, 'mean_latency_ms': 7611.125}`
- frozen backend usage: `{'call_count': 16, 'ok_calls': 16, 'error_calls': 0, 'prompt_tokens': 3515, 'completion_tokens': 8540, 'total_tokens': 12055, 'total_latency_ms': 131923.205, 'mean_latency_ms': 8245.2}`
- judge backend usage: `{'call_count': 2, 'ok_calls': 2, 'error_calls': 0, 'prompt_tokens': 2432, 'completion_tokens': 321, 'total_tokens': 2753, 'total_latency_ms': 5541.591, 'mean_latency_ms': 2770.796}`

## Notes

- This benchmark fixes the parent branch per task so it measures sibling adaptation rather than full-tree macro routing.
- Training evaluates multiple sibling leaves per task to let adaptive update from the best-performing leaf under that parent.
- Holdout executes only the policy-chosen sibling leaf, using current learned weights and sibling memory under the forced parent.
- Pairwise judging uses an LLM judge and should be read together with the objective verifier score.
