# Codex Routing Divergence Benchmark

- Generated at: `2026-03-14T06:55:40.475729+00:00`
- Generation model: `gpt-4.1-mini`
- Judge model: `gpt-4.1-mini`
- Dataset: `/Users/justin/Documents/New project/repo_inspect/artifacts/routing_micro_probe_dataset.json`
- Train sibling candidates per task: `2`
- Holdout sibling candidates per task: `1`

## Aggregate Metrics

| Split | Policy | Mean objective reward | Mean path length | Branch HHI |
| --- | --- | ---: | ---: | ---: |
| train | full_adaptive | 0.7521 | 3.0000 | 0.4062 |
| train | frozen | 0.7667 | 3.0000 | 0.3750 |
| holdout | full_adaptive | 0.7667 | 2.0000 | 0.5000 |
| holdout | frozen | 0.7333 | 2.0000 | 0.5000 |

## Divergence Summary

- Holdout objective gain (adaptive - frozen): `0.0334`
- Selected-branch divergence: `2/2`
- Activated-path divergence: `2/2`
- Mean reward delta on divergent tasks: `0.0333`
- Mean reward delta on non-divergent tasks: `0.0`

## Pairwise Judge

- Adaptive wins: `1`
- Frozen wins: `1`
- Ties: `0`
- Mean score adaptive: `8.0`
- Mean score frozen: `8.0`

## By Aspect

| Aspect | n | Adaptive mean | Frozen mean | Delta | Branch diffs | Adaptive branches | Frozen branches |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| code_consistency_audit | 1 | 0.6583 | 0.6583 | 0.0000 | 1 | {'verification_consistency_auditor': 1} | {'verification_constraint_checker': 1} |
| planning_risk | 1 | 0.8750 | 0.8083 | 0.0667 | 1 | {'planner_risk_allocator': 1} | {'planner_timeline_optimizer': 1} |

## Divergent Holdout Tasks

| Task | Aspect | Adaptive branch | Frozen branch | Adaptive | Frozen | Delta |
| --- | --- | --- | --- | ---: | ---: | ---: |
| holdout_divergence_planning_risk_admin | planning_risk | planner_risk_allocator | planner_timeline_optimizer | 0.8750 | 0.8083 | 0.0667 |
| holdout_divergence_code_consistency_backfill | code_consistency_audit | verification_consistency_auditor | verification_constraint_checker | 0.6583 | 0.6583 | 0.0000 |

## Top Learned Weight Deltas

| Branch | Adaptive weight | Frozen weight | Delta |
| --- | ---: | ---: | ---: |
| planner_risk_allocator | 0.9740 | 0.9300 | 0.0440 |

## Usage

- adaptive_full backend usage: `{'call_count': 34, 'ok_calls': 34, 'error_calls': 0, 'prompt_tokens': 7419, 'completion_tokens': 15952, 'total_tokens': 23371, 'total_latency_ms': 215788.712, 'mean_latency_ms': 6346.727}`
- frozen backend usage: `{'call_count': 28, 'ok_calls': 28, 'error_calls': 0, 'prompt_tokens': 5647, 'completion_tokens': 12408, 'total_tokens': 18055, 'total_latency_ms': 154035.148, 'mean_latency_ms': 5501.255}`
- judge backend usage: `{'call_count': 2, 'ok_calls': 2, 'error_calls': 0, 'prompt_tokens': 1978, 'completion_tokens': 358, 'total_tokens': 2336, 'total_latency_ms': 4798.139, 'mean_latency_ms': 2399.07}`

## Notes

- This benchmark fixes the parent branch per task so it measures sibling adaptation rather than full-tree macro routing.
- Training evaluates multiple sibling leaves per task to let adaptive update from the best-performing leaf under that parent.
- Holdout executes only the policy-chosen sibling leaf, using current learned weights and sibling memory under the forced parent.
- Pairwise judging uses an LLM judge and should be read together with the objective verifier score.
