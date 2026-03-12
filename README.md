# Prompt Forest Agent

A **Random-Forest-inspired prompt-branch agent** and **reward-guided prompt routing architecture**.

This project implements a non-parametric adaptive layer for agent systems:
- No base-model weight updates
- No LoRA/fine-tuning
- Only prompt templates, branch weights, routing preferences, and memory are adapted

It is designed as a standalone MVP and as an OpenClaw-compatible control layer for future integration.

## Research framing

This project is positioned as:
- A multi-branch adaptive agent architecture
- An ensemble prompt-routing framework
- A reward-shaped memory-and-routing agent
- A prompt-level policy optimization framework

It is **not** an AGI claim.

## Core idea

For each task:
1. Root router classifies task type and activates a sequential branch path.
2. Branches execute one-by-one (hierarchical): macro branch -> niche sub-branch.
3. Judge scores branch outputs.
4. Aggregator selects the final output (default: leaf node output).
5. Evaluator Agent (Agent 1) emits structured reward and failure signals.
6. Optimizer Agent (Agent 2) performs constrained local updates to active path branches only.
7. Memory records trajectories and biases future routing.

## Architecture

```
OpenClaw-compatible runtime adapter
-> Prompt routing layer
-> Branch execution layer
-> Evaluator/Judge layer (Agent 1)
-> Optimizer layer (Agent 2)
-> Memory layer
```

### Hierarchical forest layout

- Root node
- Layer 1 (macro): 6 branches (`analytical`, `planner`, `retrieval`, `critique`, `verification`, `creative`)
- Layer 2 (niche): 12 specialized sub-branches (2 under each macro branch)

Routing is sequential, one node per layer.

### Main modules

- `router/`: task classification and branch activation
- `branches/`: specialized prompt branches with weight and prompt state
- `backend/`: abstract LLM backend + mock backend
- `evaluator/`: reward scoring
- `agents/evaluator_agent.py`: structured optimization signal generation
- `agents/optimizer_agent.py`: constrained branch-local adaptation + candidate lifecycle
- `memory/`: trajectory and performance memory
- `aggregator/`: selection/merge strategies
- `adapters/openclaw_adapter.py`: OpenClaw-style trajectory interoperability
- `experiments/`: benchmark runner

## Two-agent adaptation

### Agent 1: Evaluator / Judge
Outputs:
- `reward_score`
- `failure_reason`
- `confidence`
- `suggested_improvement_direction`
- branch-level feedback for activated branches

### Agent 2: Branch Optimizer / Prompt Updater
Allowed actions:
- Slight branch-weight updates (bounded)
- Local prompt rewrite for weak branches
- Candidate branch propose/trial/promote/archive under strict rules
- Path-level reward propagation (leaf reward is propagated upstream with discount)

Not allowed:
- Global architecture rewrite
- Unlimited branch creation
- Global uncontrolled prompt mutation

## Candidate branch lifecycle

Candidate branches are created only when:
- repeated failures occur,
- active branches underperform,
- no active candidate already exists,
- branch cap and duplication checks pass.

Lifecycle: `propose -> trial -> promote/archive`.

## OpenClaw compatibility notes

The design borrows RL-style loop concepts (trajectory logging, judge signal, policy-like adaptation) inspired by OpenClaw-RL, while restricting optimization to the prompt-routing-memory layer.

Compatibility points:
- `PromptForestEngine.openclaw_ingest(event)` accepts structured trajectory-like events
- `OpenClawAdapter.process_trajectory()` wraps external runtime events
- Tool events, outputs, branch activity, and evaluator signals are accepted as structured metadata

## Project structure

```
openclaw_closedsourcemodel_RL/
  configs/
  examples/
  artifacts/
  src/prompt_forest/
    core/
    agents/
    branches/
    router/
    evaluator/
    memory/
    rewards/
    aggregator/
    backend/
    adapters/
    experiments/
  tests/
```

## Install

```bash
cd /Users/justin/Documents/New\ project/openclaw_closedsourcemodel_RL
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -e .[dev]
```

## Run

### 1) Single task

```bash
prompt-forest run-task \
  --task "Calculate derivative of x^2" \
  --task-type math \
  --metadata '{"expected_keywords": ["derivative", "2x"], "required_substrings": ["confidence"]}'
```

### 2) Benchmark demo

```bash
prompt-forest benchmark --dataset examples/demo_tasks.json --rounds 6
```

### 3) OpenClaw-style event

```bash
prompt-forest openclaw-event --event-file examples/openclaw_event.json
```

### 4) Full demo script

```bash
PYTHONPATH=src python examples/run_demo.py
```

### 5) Candidate branch trial demo

```bash
PYTHONPATH=src python examples/candidate_trial_demo.py
```

### 6) RL learning validation (adaptive vs frozen)

```bash
PYTHONPATH=src python examples/run_rl_validation.py
```

or via CLI:

```bash
prompt-forest rl-validate --episodes 240 --seeds 11,17,19,23,29,31,37,41,43,47
```

## Logged artifacts

- `artifacts/events.jsonl`: per-task routing, branch rewards, optimization events
- `artifacts/memory_records.jsonl`: memory trajectories
- `artifacts/benchmark_summary.json`: benchmark metrics over rounds
- `artifacts/demo_report.json`: final demo summary
- `artifacts/candidate_trial_report.json`: candidate branch creation/trial state snapshot
- `artifacts/rl_validation_report.json`: multi-seed adaptive-vs-frozen learning evidence

## What the MVP demonstrates

- Selective multi-branch activation by task type
- Branch-level reward assignment
- Branch-local updates (weights, prompt variants)
- Memory-influenced future routing
- Evaluator/optimizer separation
- Candidate branch creation policy with trial lifecycle

## Testing

```bash
pytest
```

Focused RL-learning tests:

```bash
PYTHONPATH=src pytest tests/test_learning_dynamics.py
```

## Future research extensions

1. Replace mock backend with real OpenAI-compatible and local model backends.
2. Add external verifier plugins (unit tests, symbolic solvers, retrieval validators).
3. Add contextual bandit or Thompson sampling router policy.
4. Add richer evaluator modes (pairwise judge, consistency checks, uncertainty calibration).
5. Add branch prompt versioning and A/B rollback dashboards.
6. Add multi-objective reward (accuracy, latency, token cost, safety).
7. Integrate directly into OpenClaw runtime loop as an adaptation middleware service.
