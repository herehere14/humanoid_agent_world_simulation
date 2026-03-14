# Full vs Frozen Execution Probe

- full_adaptive: train=0.7612, holdout=0.7675
- frozen: train=0.7550, holdout=0.7550
- delta_vs_frozen: +0.0125

## Holdout Branches

### full_adaptive
- holdout_planning_recovery: planner_risk_allocator (0.8050)
- holdout_code_pr_review: verification_constraint_checker (0.8050)
- holdout_general_tradeoff_note: retrieval_evidence_tracer (0.7050)
- holdout_planning_risk_register: planner_risk_allocator (0.7550)

### frozen
- holdout_planning_recovery: planner_risk_allocator (0.7550)
- holdout_code_pr_review: verification_constraint_checker (0.7550)
- holdout_general_tradeoff_note: retrieval_evidence_tracer (0.7550)
- holdout_planning_risk_register: planner_risk_allocator (0.7550)
