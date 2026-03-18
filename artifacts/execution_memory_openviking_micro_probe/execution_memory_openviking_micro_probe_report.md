# Codex Routing Divergence Benchmark

- Generated at: `2026-03-14T13:36:44.181873+00:00`
- Generation model: `gpt-4.1-mini`
- Judge model: `gpt-4.1-mini`
- Dataset: `/Users/justin/Documents/New project/repo_inspect/artifacts/routing_micro_probe_dataset.json`
- Train sibling candidates per task: `2`
- Holdout sibling candidates per task: `1`

## Aggregate Metrics

| Split | Policy | Mean objective reward | Mean path length | Branch HHI |
| --- | --- | ---: | ---: | ---: |
| train | full_adaptive | 0.7667 | 3.0000 | 0.5000 |
| train | frozen | 0.7667 | 3.0000 | 0.5000 |
| holdout | full_adaptive | 0.6833 | 2.0000 | 0.5000 |
| holdout | frozen | 0.5200 | 2.0000 | 0.5000 |

## Divergence Summary

- Holdout objective gain (adaptive - frozen): `0.1633`
- Selected-branch divergence: `1/2`
- Activated-path divergence: `1/2`
- Mean reward delta on divergent tasks: `0.3766`
- Mean reward delta on non-divergent tasks: `-0.05`

## Pairwise Judge

- Adaptive wins: `1`
- Frozen wins: `0`
- Ties: `1`
- Mean score adaptive: `9.0`
- Mean score frozen: `6.5`

## By Aspect

| Aspect | n | Adaptive mean | Frozen mean | Delta | Branch diffs | Adaptive branches | Frozen branches |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| code_consistency_audit | 1 | 0.6083 | 0.6583 | -0.0500 | 0 | {'verification_constraint_checker': 1} | {'verification_constraint_checker': 1} |
| planning_risk | 1 | 0.7583 | 0.3817 | 0.3766 | 1 | {'planner_risk_allocator': 1} | {'planner_timeline_optimizer': 1} |

## Divergent Holdout Tasks

| Task | Aspect | Adaptive branch | Frozen branch | Adaptive | Frozen | Delta |
| --- | --- | --- | --- | ---: | ---: | ---: |
| holdout_divergence_planning_risk_admin | planning_risk | planner_risk_allocator | planner_timeline_optimizer | 0.7583 | 0.3817 | 0.3766 |

## Top Learned Weight Deltas

| Branch | Adaptive weight | Frozen weight | Delta |
| --- | ---: | ---: | ---: |
| planner_risk_allocator | 0.9664 | 0.9300 | 0.0364 |

## Usage

- adaptive_full backend usage: `{'call_count': 16, 'ok_calls': 16, 'error_calls': 0, 'prompt_tokens': 3482, 'completion_tokens': 7186, 'total_tokens': 10668, 'total_latency_ms': 100440.326, 'mean_latency_ms': 6277.52}`
- frozen backend usage: `{'call_count': 16, 'ok_calls': 16, 'error_calls': 0, 'prompt_tokens': 3150, 'completion_tokens': 7165, 'total_tokens': 10315, 'total_latency_ms': 115154.863, 'mean_latency_ms': 7197.179}`
- judge backend usage: `{'call_count': 2, 'ok_calls': 2, 'error_calls': 0, 'prompt_tokens': 2018, 'completion_tokens': 338, 'total_tokens': 2356, 'total_latency_ms': 5718.419, 'mean_latency_ms': 2859.209}`

## Notes

- This benchmark fixes the parent branch per task so it measures sibling adaptation rather than full-tree macro routing.
- Training evaluates multiple sibling leaves per task to let adaptive update from the best-performing leaf under that parent.
- Holdout executes only the policy-chosen sibling leaf, using current learned weights and sibling memory under the forced parent.
- Pairwise judging uses an LLM judge and should be read together with the objective verifier score.
