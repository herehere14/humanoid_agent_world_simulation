# Codex Routing Divergence Benchmark

- Generated at: `2026-03-14T13:35:59.892835+00:00`
- Generation model: `gpt-4.1-mini`
- Judge model: `gpt-4.1-mini`
- Dataset: `/Users/justin/Documents/New project/repo_inspect/artifacts/codex_routing_divergence_quick_probe_dataset.json`
- Train sibling candidates per task: `2`
- Holdout sibling candidates per task: `1`

## Aggregate Metrics

| Split | Policy | Mean objective reward | Mean path length | Branch HHI |
| --- | --- | ---: | ---: | ---: |
| train | full_adaptive | 0.7625 | 3.0000 | 0.3750 |
| train | frozen | 0.8500 | 3.0000 | 0.5000 |
| holdout | full_adaptive | 0.7792 | 2.0000 | 0.5000 |
| holdout | frozen | 0.7091 | 2.0000 | 0.5000 |

## Divergence Summary

- Holdout objective gain (adaptive - frozen): `0.0701`
- Selected-branch divergence: `2/4`
- Activated-path divergence: `2/4`
- Mean reward delta on divergent tasks: `0.1734`
- Mean reward delta on non-divergent tasks: `-0.0333`

## Pairwise Judge

- Adaptive wins: `2`
- Frozen wins: `2`
- Ties: `0`
- Mean score adaptive: `8.0`
- Mean score frozen: `7.0`

## By Aspect

| Aspect | n | Adaptive mean | Frozen mean | Delta | Branch diffs | Adaptive branches | Frozen branches |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| code_consistency_audit | 1 | 0.6583 | 0.7250 | -0.0667 | 0 | {'verification_constraint_checker': 1} | {'verification_constraint_checker': 1} |
| general_consistency_audit | 1 | 0.7583 | 0.7583 | 0.0000 | 0 | {'verification_constraint_checker': 1} | {'verification_constraint_checker': 1} |
| planning_risk | 2 | 0.8500 | 0.6766 | 0.1734 | 2 | {'planner_risk_allocator': 2} | {'planner_timeline_optimizer': 2} |

## Divergent Holdout Tasks

| Task | Aspect | Adaptive branch | Frozen branch | Adaptive | Frozen | Delta |
| --- | --- | --- | --- | ---: | ---: | ---: |
| holdout_divergence_planning_risk_ledger | planning_risk | planner_risk_allocator | planner_timeline_optimizer | 0.8750 | 0.5950 | 0.2800 |
| holdout_divergence_planning_risk_admin | planning_risk | planner_risk_allocator | planner_timeline_optimizer | 0.8250 | 0.7583 | 0.0667 |

## Top Learned Weight Deltas

| Branch | Adaptive weight | Frozen weight | Delta |
| --- | ---: | ---: | ---: |
| planner_risk_allocator | 0.9571 | 0.9300 | 0.0271 |
| verification_constraint_checker | 0.9782 | 0.9700 | 0.0082 |

## Usage

- adaptive_full backend usage: `{'call_count': 21, 'ok_calls': 21, 'error_calls': 0, 'prompt_tokens': 4742, 'completion_tokens': 10556, 'total_tokens': 15298, 'total_latency_ms': 151789.022, 'mean_latency_ms': 7228.049}`
- frozen backend usage: `{'call_count': 20, 'ok_calls': 20, 'error_calls': 0, 'prompt_tokens': 3799, 'completion_tokens': 10466, 'total_tokens': 14265, 'total_latency_ms': 181093.887, 'mean_latency_ms': 9054.694}`
- judge backend usage: `{'call_count': 4, 'ok_calls': 4, 'error_calls': 0, 'prompt_tokens': 4213, 'completion_tokens': 747, 'total_tokens': 4960, 'total_latency_ms': 13129.974, 'mean_latency_ms': 3282.494}`

## Notes

- This benchmark fixes the parent branch per task so it measures sibling adaptation rather than full-tree macro routing.
- Training evaluates multiple sibling leaves per task to let adaptive update from the best-performing leaf under that parent.
- Holdout executes only the policy-chosen sibling leaf, using current learned weights and sibling memory under the forced parent.
- Pairwise judging uses an LLM judge and should be read together with the objective verifier score.
