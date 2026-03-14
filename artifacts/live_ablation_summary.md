# Live Ablation Summary

## Artifacts

- JSON report: `artifacts/live_model_ablation_validation/live_model_ablation_report.json`
- Markdown report: `artifacts/live_model_ablation_validation/live_model_ablation_report.md`

## Setup

- Model: `gpt-4.1-mini`
- Judge model: `gpt-4.1-mini`
- Dataset: `examples/live_eval_tasks.json`
- Train: 12 tasks x 2 rounds = 24 episodes per policy
- Holdout: 12 tasks
- Policies:
  - `full_adaptive`: weights + memory
  - `frozen`: no learning
  - `memory_only`: memory updates only
  - `weight_only`: weight updates only

## Holdout Objective Results

| Policy | Mean objective reward |
| --- | ---: |
| full_adaptive | 0.6922 |
| frozen | 0.6922 |
| weight_only | 0.6922 |
| memory_only | 0.6797 |

## Main Read

- `memory_only` is the only policy that is clearly worse on the main objective metric.
- `full_adaptive`, `frozen`, and `weight_only` are tied on the aggregate holdout objective score.
- That means the learned memory component is the strongest candidate for what is hurting the current system.
- Weight updates alone did not improve the aggregate score, but they also did not degrade it relative to frozen.

## Pairwise Judge Read

- `full_adaptive` vs `frozen`: `1` win, `4` losses, `7` ties
- `weight_only` vs `frozen`: `1` win, `3` losses, `8` ties
- `memory_only` vs `frozen`: `3` wins, `1` loss, `8` ties

## Interpretation

- The heuristic objective says memory hurts and weight updates are neutral.
- The pairwise judge does not cleanly agree; it mildly prefers `memory_only` over `frozen` on judged quality despite the lower heuristic score.
- The safest conclusion is:
  - the memory pathway is the only component that clearly lowers the benchmark's objective score
  - the weight-update pathway does not show a measurable aggregate gain, but it is less likely to be the main source of regression

## Supporting Detail

- Train objective means:
  - `frozen`: `0.7494`
  - `weight_only`: `0.7474`
  - `full_adaptive`: `0.7432`
  - `memory_only`: `0.7390`
- No policy triggered prompt rewrites, candidate creation, or branch promotion in this run.
- Learned branch weights in `full_adaptive` and `weight_only` converged to very similar top branches, while `memory_only` kept original branch weights and relied only on replay bias.
