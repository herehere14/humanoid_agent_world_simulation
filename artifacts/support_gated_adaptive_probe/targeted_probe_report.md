# Support-Gated Adaptive Probe

## Aggregate

- full_adaptive: train=0.7550, holdout=0.7675, delta_vs_frozen=+0.0000
- frozen: train=0.7612, holdout=0.7675
- memory_only: train=0.7488, holdout=0.7550, delta_vs_frozen=-0.0125
- weight_only: train=0.7612, holdout=0.7675, delta_vs_frozen=+0.0000

## Optimizer Summary

- full_adaptive: updates=0/8, allowed=0, max_branches_per_event=0, mean_total_delta=0.0000
- frozen: updates=0/8, allowed=0, max_branches_per_event=0, mean_total_delta=0.0000
- memory_only: updates=0/8, allowed=0, max_branches_per_event=0, mean_total_delta=0.0000
- weight_only: updates=0/8, allowed=0, max_branches_per_event=0, mean_total_delta=0.0000

## Holdout Branches

### full_adaptive
- holdout_planning_recovery: planner_risk_allocator (0.8050)
- holdout_code_pr_review: verification_constraint_checker (0.7550)
- holdout_general_tradeoff_note: retrieval_evidence_tracer (0.7050)
- holdout_planning_risk_register: planner_risk_allocator (0.8050)

### frozen
- holdout_planning_recovery: planner_risk_allocator (0.7550)
- holdout_code_pr_review: verification_constraint_checker (0.8050)
- holdout_general_tradeoff_note: retrieval_evidence_tracer (0.7050)
- holdout_planning_risk_register: planner_risk_allocator (0.8050)

### memory_only
- holdout_planning_recovery: planner_risk_allocator (0.8050)
- holdout_code_pr_review: verification_constraint_checker (0.7550)
- holdout_general_tradeoff_note: retrieval_evidence_tracer (0.7050)
- holdout_planning_risk_register: planner_risk_allocator (0.7550)

### weight_only
- holdout_planning_recovery: planner_risk_allocator (0.7550)
- holdout_code_pr_review: verification_constraint_checker (0.8050)
- holdout_general_tradeoff_note: retrieval_evidence_tracer (0.7050)
- holdout_planning_risk_register: planner_risk_allocator (0.8050)
