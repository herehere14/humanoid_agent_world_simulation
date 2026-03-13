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

### 1b) Interactive chat with the RL agent

```bash
PYTHONPATH=src python -m prompt_forest.cli chat --task-type auto --visibility full
```

Use `/type <task_type>` to pin type and `/auto` to restore auto-routing.
Use `--visibility minimal|eval|opt|full` to control how much of Agent 1/Agent 2 internals are shown.

### 1c) Inspect recent evaluator/optimizer traces

```bash
prompt-forest inspect-events --limit 8 --visibility full
```

This prints per-task routing path, branch-level evaluation signals, and optimizer updates (weight delta, advantage, decay, rewrites, candidate lifecycle actions).

### 1d) Enable real API-backed Evaluator Agent and Optimizer Agent

By default, Agent 1 and Agent 2 run with deterministic local logic.  
To run them as model-backed agents, enable `agent_runtimes` in config and set API keys.

Config section:

```json
"agent_runtimes": {
  "evaluator": {
    "enabled": true,
    "provider": "openai_compatible",
    "model": "gpt-4.1-mini",
    "api_key_env": "OPENAI_API_KEY",
    "base_url": "https://api.openai.com/v1"
  },
  "optimizer": {
    "enabled": true,
    "provider": "openai_compatible",
    "model": "gpt-4.1-mini",
    "api_key_env": "OPENAI_API_KEY",
    "base_url": "https://api.openai.com/v1"
  }
}
```

Environment variable example:

```bash
export OPENAI_API_KEY="your_key_here"
PYTHONPATH=src python -m prompt_forest.cli --config configs/runtime_openai_example.json chat --task-type auto --visibility full
```

Supported runtime providers:
- `openai_compatible` (OpenAI-compatible chat completions endpoint)
- `gemini` (Google Gemini generateContent endpoint)

Ready-to-use examples:
- `configs/runtime_openai_example.json`
- `configs/runtime_gemini_example.json`

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

### 7) Detailed hierarchical validation (learning + ablations)

```bash
PYTHONPATH=src python examples/run_detailed_validation.py
```

or via CLI:

```bash
prompt-forest detailed-validate --episodes 240 --seeds 11,17,19,23,29
```

This detailed validation now also reports branch-growth metrics:
- average number of newly created branches
- hierarchy depth gain
- reward trend during growth-probe runs

### 8) Multi-round auto-improvement loop (anti-bias objective)

```bash
PYTHONPATH=src python examples/run_auto_improve.py
```

or via CLI:

```bash
prompt-forest auto-improve \
  --rounds 3 \
  --candidates 12 \
  --episodes 140 \
  --final-episodes 220 \
  --seeds 3,5,7,11 \
  --final-seeds 11,17,19,23,29,31,37,41
```

The auto-improver only promotes configurations that improve holdout performance while penalizing instability, task-type unfairness, and branch-collapse concentration.

### 9) Continuous improvement cycles (hours-scale)

```bash
PYTHONPATH=src python examples/run_continuous_improve.py
```

or via CLI:

```bash
prompt-forest continuous-improve \
  --cycles 12 \
  --rounds-per-cycle 2 \
  --candidates 6 \
  --episodes 180 \
  --final-episodes 220 \
  --seeds 11,17,19,23,29,31,37,41 \
  --final-seeds 11,17,19,23,29,31,37,41 \
  --patience 3
```

This loop runs repeated auto-improvement cycles, executes regression tests after each cycle, accepts only improving/no-bias candidates, and rolls back regressions automatically.

## Logged artifacts

- `artifacts/events.jsonl`: per-task routing, branch rewards, optimization events
- `artifacts/memory_records.jsonl`: memory trajectories
- `artifacts/benchmark_summary.json`: benchmark metrics over rounds
- `artifacts/demo_report.json`: final demo summary
- `artifacts/candidate_trial_report.json`: candidate branch creation/trial state snapshot
- `artifacts/rl_validation_report.json`: multi-seed adaptive-vs-frozen learning evidence
- `artifacts/detailed_validation_report.json`: branch inventory, improvement metrics, and branch-ablation effects
- `artifacts/auto_improve/auto_improve_summary.json`: multi-round search log and selected best config
- `artifacts/auto_improve/final_validation_report.json`: final unbiased validation for promoted config
- `artifacts/continuous_improve/continuous_improve_summary.json`: multi-cycle long-run optimization trace with acceptance/rollback decisions

For quick human-readable visibility:
- `prompt-forest run-task ... --visibility full`
- `prompt-forest chat --visibility full`
- `prompt-forest inspect-events --limit 10 --visibility full`

## What the MVP demonstrates

- Selective multi-branch activation by task type
- Branch-level reward assignment
- Branch-local updates (weights, prompt variants)
- Memory-influenced future routing
- Evaluator/optimizer separation
- Candidate branch creation policy with trial lifecycle
- Optional API-backed Evaluator/Optimizer agents with strict JSON contracts
- Multi-candidate branch spawning for deeper random-forest-like specialization

## Testing

```bash
pytest
```

Focused RL-learning tests:

```bash
PYTHONPATH=src pytest tests/test_learning_dynamics.py
```

Branch growth tests:

```bash
PYTHONPATH=src pytest tests/test_branch_growth.py
```

## Future research extensions

1. Replace mock backend with real OpenAI-compatible and local model backends.
2. Add external verifier plugins (unit tests, symbolic solvers, retrieval validators).
3. Add contextual bandit or Thompson sampling router policy.
4. Add richer evaluator modes (pairwise judge, consistency checks, uncertainty calibration).
5. Add branch prompt versioning and A/B rollback dashboards.
6. Add multi-objective reward (accuracy, latency, token cost, safety).
7. Integrate directly into OpenClaw runtime loop as an adaptation middleware service.
