# Targeted Restructure Probe

## Aggregate

- full_adaptive: train=0.7488, holdout=0.7550
- frozen: train=0.7738, holdout=0.7675
- memory_only: train=0.7675, holdout=0.7800
- weight_only: train=0.7550, holdout=0.7550

## Holdout Branches

### full_adaptive
- holdout_planning_recovery: planner_risk_allocator (0.8050)
- holdout_code_pr_review: verification_constraint_checker (0.7550)
- holdout_general_tradeoff_note: retrieval_evidence_tracer (0.7050)
- holdout_planning_risk_register: planner_risk_allocator (0.7550)

### frozen
- holdout_planning_recovery: planner_risk_allocator (0.8050)
- holdout_code_pr_review: verification_constraint_checker (0.8050)
- holdout_general_tradeoff_note: retrieval_evidence_tracer (0.7050)
- holdout_planning_risk_register: planner_risk_allocator (0.7550)

### memory_only
- holdout_planning_recovery: planner_risk_allocator (0.8050)
- holdout_code_pr_review: verification_constraint_checker (0.8050)
- holdout_general_tradeoff_note: retrieval_evidence_tracer (0.7550)
- holdout_planning_risk_register: planner_risk_allocator (0.7550)

### weight_only
- holdout_planning_recovery: planner_risk_allocator (0.7550)
- holdout_code_pr_review: verification_constraint_checker (0.8050)
- holdout_general_tradeoff_note: retrieval_evidence_tracer (0.7050)
- holdout_planning_risk_register: planner_risk_allocator (0.7550)
