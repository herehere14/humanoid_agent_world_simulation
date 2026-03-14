# Codex Routing Divergence Benchmark

- Generated at: `2026-03-14T13:48:54.160054+00:00`
- Generation model: `gpt-4.1-mini`
- Judge model: `gpt-4.1-mini`
- Dataset: `/Users/justin/Documents/New project/repo_inspect/artifacts/codex_routing_divergence_quick_probe_dataset.json`
- Train sibling candidates per task: `2`
- Holdout sibling candidates per task: `1`

## Aggregate Metrics

| Split | Policy | Mean objective reward | Mean path length | Branch HHI |
| --- | --- | ---: | ---: | ---: |
| train | full_adaptive | 0.8333 | 3.0000 | 0.3750 |
| train | frozen | 0.8208 | 3.0000 | 0.3750 |
| holdout | full_adaptive | 0.7333 | 2.0000 | 0.5000 |
| holdout | frozen | 0.7386 | 2.0000 | 0.5000 |

## Divergence Summary

- Holdout objective gain (adaptive - frozen): `-0.0053`
- Selected-branch divergence: `2/4`
- Activated-path divergence: `2/4`
- Mean reward delta on divergent tasks: `-0.0106`
- Mean reward delta on non-divergent tasks: `0.0`

## Pairwise Judge

- Adaptive wins: `3`
- Frozen wins: `1`
- Ties: `0`
- Mean score adaptive: `8.5`
- Mean score frozen: `7.5`

## By Aspect

| Aspect | n | Adaptive mean | Frozen mean | Delta | Branch diffs | Adaptive branches | Frozen branches |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| code_consistency_audit | 1 | 0.6583 | 0.6583 | 0.0000 | 0 | {'verification_constraint_checker': 1} | {'verification_constraint_checker': 1} |
| general_consistency_audit | 1 | 0.7583 | 0.7583 | 0.0000 | 0 | {'verification_constraint_checker': 1} | {'verification_constraint_checker': 1} |
| planning_risk | 2 | 0.7583 | 0.7689 | -0.0106 | 2 | {'planner_risk_allocator': 2} | {'planner_timeline_optimizer': 2} |

## Divergent Holdout Tasks

| Task | Aspect | Adaptive branch | Frozen branch | Adaptive | Frozen | Delta |
| --- | --- | --- | --- | ---: | ---: | ---: |
| holdout_divergence_planning_risk_ledger | planning_risk | planner_risk_allocator | planner_timeline_optimizer | 0.7583 | 0.8083 | -0.0500 |
| holdout_divergence_planning_risk_admin | planning_risk | planner_risk_allocator | planner_timeline_optimizer | 0.7583 | 0.7294 | 0.0289 |

## Top Learned Weight Deltas

| Branch | Adaptive weight | Frozen weight | Delta |
| --- | ---: | ---: | ---: |
| planner_risk_allocator | 0.9664 | 0.9300 | 0.0364 |
| verification_consistency_auditor | 0.9698 | 0.9500 | 0.0198 |

## Usage

- adaptive_full backend usage: `{'call_count': 21, 'ok_calls': 21, 'error_calls': 0, 'prompt_tokens': 5234, 'completion_tokens': 11088, 'total_tokens': 16322, 'total_latency_ms': 182112.283, 'mean_latency_ms': 8672.013}`
- frozen backend usage: `{'call_count': 20, 'ok_calls': 20, 'error_calls': 0, 'prompt_tokens': 4404, 'completion_tokens': 11593, 'total_tokens': 15997, 'total_latency_ms': 187333.363, 'mean_latency_ms': 9366.668}`
- judge backend usage: `{'call_count': 4, 'ok_calls': 4, 'error_calls': 0, 'prompt_tokens': 4997, 'completion_tokens': 678, 'total_tokens': 5675, 'total_latency_ms': 11250.304, 'mean_latency_ms': 2812.576}`

## Notes

- This benchmark fixes the parent branch per task so it measures sibling adaptation rather than full-tree macro routing.
- Training evaluates multiple sibling leaves per task to let adaptive update from the best-performing leaf under that parent.
- Holdout executes only the policy-chosen sibling leaf, using current learned weights and sibling memory under the forced parent.
- Pairwise judging uses an LLM judge and should be read together with the objective verifier score.
