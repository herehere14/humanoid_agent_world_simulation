# Live Validation Combined Summary

## Artifacts

- Core benchmark JSON: `artifacts/live_model_validation_core/live_model_validation_report.json`
- Core benchmark Markdown: `artifacts/live_model_validation_core/live_model_validation_report.md`
- Runtime-assisted probe JSON: `artifacts/live_model_validation_assisted_probe/live_model_validation_report.json`
- Runtime-assisted probe Markdown: `artifacts/live_model_validation_assisted_probe/live_model_validation_report.md`

## Setup

- Generation model: `gpt-4.1-mini`
- Judge model: `gpt-4.1-mini`
- Branch execution used the real OpenAI API through the new `OpenAIChatBackend`
- Baselines:
  - `adaptive_full`: prompt-forest with online adaptation
  - `frozen_forest`: same prompt-forest, adaptation disabled
  - `direct_model`: same closed-source model, direct prompt only

## Core Benchmark

- Dataset: `examples/live_eval_tasks.json`
- Train: 12 tasks x 2 rounds = 24 adaptive/frozen training episodes
- Holdout: 12 unseen tasks
- Agent runtimes: disabled for the main benchmark to isolate routing/adaptation value and keep the run bounded

### Core Holdout Results

| Policy | Mean objective reward | Contract pass rate | Pairwise vs direct |
| --- | ---: | ---: | --- |
| adaptive_full | 0.6922 | 1.0000 | 9 wins, 1 loss, 2 ties |
| frozen_forest | 0.6964 | 1.0000 | 6 wins, 3 losses, 3 ties |
| direct_model | 0.5732 | 1.0000 | baseline |

### Core Adaptive vs Frozen

- Objective delta: `-0.0042` in favor of `frozen_forest`
- Pairwise judge: `adaptive_full` won 2, lost 2, tied 8
- Training trend:
  - `adaptive_full`: `+0.0146` from first third to last third
  - `frozen_forest`: `-0.0104` from first third to last third
- Interpretation: adaptation showed some training-time movement, but it did not turn into a clear holdout win over the frozen prompt-forest.

### Core Behavioral Notes

- The prompt-forest structure itself clearly helped versus a direct single prompt on the same model.
- The online adaptation loop did not materially outperform the frozen forest on this curated real-task set.
- Both forest policies reached perfect contract pass rate on the strict-format holdout tasks; the direct baseline also passed these tasks, so contract routing was not the differentiator here.
- The adaptive router heavily favored `verification` and `json_lock`, and often selected lock-style leaves on non-contract tasks. That looks like a real routing pathology worth fixing before making stronger learning claims.
- No prompt rewrites, candidate creation, or branch growth triggered in the real benchmark.

## Runtime-Assisted Probe

- Dataset: 4 train tasks + 4 holdout tasks
- Agent runtimes: enabled
- Purpose: exercise the live evaluator/advisor API path without paying the full wall-clock cost on the whole benchmark

### Probe Holdout Results

| Policy | Mean objective reward | Contract pass rate |
| --- | ---: | ---: |
| adaptive_full | 0.7240 | 1.0000 |
| frozen_forest | 0.7115 | 1.0000 |
| direct_model | 0.5462 | 1.0000 |

### Probe Adaptive vs Frozen

- Objective delta: `+0.0125` in favor of `adaptive_full`
- Pairwise judge: `adaptive_full` won 0, lost 1, tied 3
- Interpretation: the runtime-assisted path showed a small heuristic gain, but still not a decisive qualitative win over the frozen forest on the judged outputs.

### Probe Runtime Cost

- Adaptive evaluator runtime usage:
  - 8 calls
  - 8,138 total tokens
  - 27.38 seconds total latency
- Adaptive optimizer advisor runtime usage:
  - 4 calls
  - 3,515 total tokens
  - 19.54 seconds total latency

## Bottom Line

- The real-model evidence supports this claim:
  - the prompt-forest architecture is meaningfully better than a plain direct prompt against the same closed-source model on this dataset
- The real-model evidence does not yet support this stronger claim:
  - the current online adaptation loop is reliably better than the frozen prompt-forest baseline
- The project's current value appears to come more from the branch library, routing structure, and contract-aware prompting than from learned online adaptation.
