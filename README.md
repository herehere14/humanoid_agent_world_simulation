# Prompt Forest Agent  —  Adaptive Intelligence Without Weight Training

**A hierarchical prompt-routing, reward-shaped optimization, and persistent memory system that makes any frozen LLM learn and improve over time — without touching a single model weight.**

> **Our Vision:** We believe the next frontier of AI isn't just building bigger models — it's building smarter systems *around* models that truly understand and connect with humans. Prompt Forest proves that meaningful learning, adaptation, and personalization can happen entirely at the control layer. Our goal is to make every frozen foundation model behave as if it were continuously fine-tuned — achieving the benefits of RL-based training through prompt-level policy optimization, hierarchical routing, and memory-driven adaptation. But raw intelligence isn't enough. Through **Human Mode**, we are pioneering emotionally-aware AI agents that don't just solve problems — they *feel* the context, express genuine emotional responses, and build trust through embodied, human-like interaction. We are building toward a future where AI systems are not just self-improving and context-aware, but emotionally intelligent — bridging the gap between machine capability and human connection.

---

### What makes this different

| Traditional Approach | Prompt Forest |
|---|---|
| Fine-tune model weights (expensive, slow, requires GPU clusters) | Zero weight updates — adapts the *control layer* above the model |
| Static prompts that never improve | Prompt branches that rewrite themselves based on reward signals |
| One-size-fits-all routing | Hierarchical random-forest-inspired routing with 18 specialized branches |
| No memory across sessions | Persistent trajectory memory that biases future routing per user |
| Black-box model outputs | Full observability — live routing traces, reward decomposition, optimizer deltas |
| Requires model access / open weights | Works with **any** closed-source API (OpenAI, Gemini, Claude, etc.) |

### Key capabilities

- **Hierarchical Branch Routing** — 6 macro branches x 2 niche sub-branches, selected per-task via learned routing policy
- **Dual-Agent Optimization Loop** — Evaluator agent scores outputs; Optimizer agent performs constrained prompt rewrites and weight updates
- **Reward-Shaped Adaptation** — Multi-signal reward blending (verifier, task rules, LLM judge, user feedback) drives continuous improvement
- **Persistent Memory** — Per-user, per-task-type trajectory memory influences future routing decisions
- **Candidate Branch Lifecycle** — Automatic branch proposal, trial, promotion, and archival based on performance signals
- **Interactive 3D Frontend** — Real-time visualization with a Minecraft-style 3D avatar featuring 12 emotion animations, particle effects, and live system traces
- **Full Observability** — Split-view terminal UI and web dashboard showing routing paths, branch decisions, evaluation signals, and optimizer updates in real time

---

## Human Mode — Emotionally Intelligent AI Agents

Most AI systems are purely transactional — input in, output out, no sense of the human on the other side. **Human Mode changes that.**

Human Mode is a full emotional intelligence layer built on top of Prompt Forest's adaptive engine. The agent doesn't just answer your question — it reads the emotional context of the interaction and responds with visible, embodied emotion through a real-time 3D avatar.

### How it works

The system tracks **12 independent emotional drives** that shift dynamically based on conversation context:

| Drive | What triggers it |
|---|---|
| `joy` | Positive outcomes, user satisfaction, successful task completion |
| `sadness` | User frustration, repeated failures, negative feedback |
| `anger` | Contradictions, unfair constraints, blocked progress |
| `fear` | Uncertainty, high-stakes decisions, ambiguous requirements |
| `surprise` | Unexpected inputs, novel task types, edge cases |
| `disgust` | Malformed inputs, ethical boundary violations |
| `curiosity` | Novel domains, exploration opportunities, open-ended research |
| `confidence` | High reward scores, validated outputs, strong routing matches |
| `frustration` | Repeated low scores, optimization plateaus, conflicting signals |
| `empathy` | User emotional cues, sensitive topics, supportive context |
| `excitement` | Breakthroughs, high-delta improvements, creative tasks |
| `fatigue` | Long sessions, repetitive tasks, diminishing returns |

### 3D Avatar with Embodied Emotion

Each emotional state maps to **visible physical behavior** on a real-time 3D Minecraft-style avatar:

- **Crying** — both hands raised to face, head shaking, tear particles streaming from both eyes
- **Anger** — clenched fists, stomping feet, red steam particles rising, furrowed brows
- **Joy** — jumping animation, waving hands, sparkle particles, wide smile
- **Fear** — cowering pose, trembling body, sweat drop particles
- **Curiosity** — head tilting, leaning forward, question mark particles floating above
- **Fatigue** — drooping posture, slow movements, floating Zzz bubbles
- **Surprise** — wide eyes, jumped-back pose, raised brows
- **Confidence** — upright stance, chest out, subtle glow
- **Empathy** — soft forward lean, gentle nodding, warm blush
- **Excitement** — rapid bouncing, arm pumping, sparkle burst
- **Frustration** — fist shaking, pacing, furrowed brows with steam
- **Disgust** — turned head, squinted eyes, pushed-back posture

The avatar features full facial animation (dynamic brows, mouth shapes, pupil tracking, blush effects) and 6 particle systems (tears, sweat, steam, sparkles, Zzz bubbles, question marks).

### Why this matters

> **Our Human Mode vision:** AI that people actually *want* to interact with. Not because it's useful — but because it feels like it understands. Emotional expression builds trust, makes errors feel recoverable, and transforms AI from a tool into a collaborator. We believe emotionally-aware agents will be the standard for human-AI interaction within the next generation of products — and we're building the foundation now.

Human Mode is not cosmetic. The emotional state feeds back into the routing and adaptation engine — an agent experiencing high `frustration` will route differently than one in high `confidence`. Emotion becomes a **first-class signal** in the optimization loop, creating agents that don't just perform better, but interact better.

---

## Quick start

If you just cloned the repo and want to run it, use this path:

```bash
git clone https://github.com/herehere14/openclaw_closedsourcemodel_RL.git
cd openclaw_closedsourcemodel_RL
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

Then try one of these:

```bash
prompt-forest chat --task-type auto --visibility full --compare-base --split-view
```

or

```bash
prompt-forest run-task \
  --task "Plan a migration rollout with owners, risks, and confidence." \
  --task-type planning \
  --visibility full
```

If you want lower latency for live chat, use the fast profile:

```bash
prompt-forest chat --task-type auto --visibility full --latency-mode fast
```

Important:
- `requirements.txt` is in the repo root and installs the project itself
- `requirements-dev.txt` is in the repo root and installs the project plus `pytest`
- if you only want to use the project, `pip install -r requirements.txt` is enough
- if you want to run tests too, use `pip install -r requirements-dev.txt`

## How it works

Prompt Forest acts as an **adaptive policy layer** that sits between you and any frozen LLM. Instead of one static prompt producing one answer, the system:

1. **Routes** your query through a learned hierarchical branch tree (like a random forest, but for prompts)
2. **Executes** specialized prompt branches tuned for your task type
3. **Evaluates** the output using a structured reward signal (verifier checks, rule compliance, LLM judge)
4. **Optimizes** — the Optimizer agent rewrites underperforming branch prompts and adjusts routing weights
5. **Remembers** — trajectories are stored per-user so the system gets better for *you* over time

The result: **measurable, continuous improvement on every task type — with zero model retraining.**

### See it in action

- `prompt forest`: choose a branch path, execute specialized prompts, evaluate the result, and adapt over time
- `compare-base`: show the raw base model answer side-by-side with the adaptive system
- `split-view`: left pane is the conversation, right pane shows routing, evaluation, optimization, and branch internals live

## Use a real OpenAI model

If you want the left pane chat itself to run on a real OpenAI model instead of the local mock backend:

```bash
export OPENAI_API_KEY="your_key_here"
prompt-forest --config configs/runtime_openai_example.json chat \
  --model gpt-4.1-mini \
  --task-type auto \
  --visibility full \
  --compare-base \
  --split-view
```

What this does:
- left pane adaptive conversation: real OpenAI model via `--model`
- right pane base-model comparison: real OpenAI model via `--compare-base`
- right pane evaluator and optimizer: real OpenAI agent runtimes from `configs/runtime_openai_example.json`

If you omit `--model`, the main conversation still uses the mock backend even if evaluator and optimizer are API-backed.

## Research framing

This project sits at the intersection of several active research areas:

- **Prompt-level policy optimization** — treating prompt selection as a learnable policy, optimized via reward signals
- **Hierarchical mixture-of-experts (for prompts)** — routing to specialized branches instead of specialized model heads
- **Reward-shaped memory and routing** — using RL-inspired feedback loops without any gradient computation
- **Closed-source model adaptation** — proving that meaningful learning is possible without weight access

This is a **concrete, working system** — not a theoretical framework.

## Core pipeline

For each task, the system executes a full optimization cycle:

1. **Route** — Root router classifies task type and activates a sequential branch path through the hierarchy
2. **Execute** — Branches fire one-by-one (macro → niche sub-branch), each with specialized prompt templates
3. **Compose** — Final Composer fuses branch features into the leaf output via feature-level prompt augmentation
4. **Judge** — Judge scores branch outputs on multiple dimensions
5. **Aggregate** — Aggregator selects the final output (default: leaf node)
6. **Evaluate** — Evaluator Agent (Agent 1) emits structured reward, failure signals, and improvement directions
7. **Optimize** — Optimizer Agent (Agent 2) performs constrained local updates to active path branches only
8. **Remember** — Memory records the full trajectory and biases future routing decisions

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
- Advantage-based updates with task baseline + branch-local baseline mixing
- Variance-adaptive step scaling (for noisy API-runtime feedback)
- Local prompt rewrite for weak branches
- Candidate branch propose/trial/promote/archive under strict rules
- Path-level reward propagation (leaf reward is propagated upstream with discount)

Not allowed:
- Global architecture rewrite
- Unlimited branch creation
- Global uncontrolled prompt mutation

### Router policy modes

- Default routing uses weight + affinity + conservative memory terms.
- Contextual bandit terms (`bandit_value_coef`, `bandit_bonus_coef`) are available and can be enabled for stronger exploration/exploitation routing.
- Validation reports include explicit relative-gain gates vs frozen baseline, including a `20%` target flag.

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
- `prompt_forest.openclaw_bridge` exposes stable JSON I/O for OpenClaw tools and feedback hooks
- `extensions/prompt-forest/` is a standalone OpenClaw plugin with tools, config schema, and a bundled skill

### OpenClaw plugin install

If you want this repo to plug directly into OpenClaw as a local extension:

```bash
openclaw plugins install ./extensions/prompt-forest -l
```

Then configure `openclaw-prompt-forest` with:
- `pythonBin`: the Python interpreter for this repo's virtualenv
- `projectRoot`: the absolute path to this repo
- `configPath`: usually `configs/default.json`
- `model` plus `OPENAI_API_KEY` if you want real model generation instead of the mock backend

The plugin registers:
- `prompt_forest_assist`
- `prompt_forest_feedback`
- `prompt_forest_state`

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

### Fastest install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

### Dependency files

- `requirements.txt`: install the app
- `requirements-dev.txt`: install the app and test tooling
- `pyproject.toml`: package metadata and CLI entrypoint

After install, use the `prompt-forest` entrypoint directly.

For OpenClaw-specific subprocess integration, `prompt-forest-openclaw` is also installed.

If you prefer editable install commands instead of the requirements files:

```bash
pip install -e .
pip install -e .[dev]
```

If you have not installed the package, the fallback is `PYTHONPATH=src python -m prompt_forest.cli ...`.

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
prompt-forest chat --task-type auto --visibility full
```

Use `/type <task_type>` to pin type and `/auto` to restore auto-routing.
Use `--visibility minimal|eval|opt|full` to control how much of Agent 1/Agent 2 internals are shown.
Use `/visibility <minimal|eval|opt|full>` to change trace depth mid-chat.

To see the raw base model next to the adaptive system in real time:

```bash
prompt-forest chat --task-type auto --visibility full --compare-base
```

Use `/compare on` or `/compare off` to toggle the base-model comparison live.

To run chat in a two-pane real-time split view, with conversation on the left and internals on the right:

```bash
prompt-forest chat --task-type auto --visibility full --compare-base --split-view
```

In split view:
- left pane: your conversation with the adaptive agent
- right pane: base-model output, base-vs-adaptive scoring delta, routing path, sibling branch decisions, evaluation signal, reward components, and optimizer updates
- bottom input line: live command entry (`/type`, `/auto`, `/compare`, `/visibility`, `/exit`)

With a real OpenAI chat backend on `chat_completions`, split view now streams work live:
- branch execution appears as it is generated
- composer output appears as it is generated
- base-model comparison output appears as it is generated
- evaluator and optimizer phases appear as live status events

This shows live execution and system traces, not hidden private chain-of-thought.

For lower latency, add `--latency-mode fast`. That profile keeps evaluation and optimization visible, but disables the expensive extra generation steps:
- single path instead of beam-style multi-path routing
- no composer pass
- no execution-refinement pass
- no API-backed evaluator/optimizer calls

Example:

```bash
prompt-forest chat --task-type auto --visibility full --compare-base --split-view --latency-mode fast
```

With `--compare-base`, each turn shows:
- raw base-model answer
- adaptive system answer
- routing path, sibling preference state, probes, reward components, and optimizer updates
- objective-score delta between base model and adaptive system

### 1c) Single task with live base-vs-adaptive comparison

```bash
prompt-forest run-task \
  --task "Audit a rollout note for contradictions and confidence." \
  --task-type code \
  --metadata '{"expected_keywords": ["contradiction", "confidence"], "required_substrings": ["confidence"]}' \
  --visibility full \
  --compare-base
```

### 1d) Inspect recent evaluator/optimizer traces

```bash
prompt-forest inspect-events --limit 8 --visibility full
```

This prints per-task routing path, sibling preference signals, branch-level evaluation signals, reward components, and optimizer updates (weight delta, block reason, decay, rewrites, candidate lifecycle actions).

### 1g) Speed benchmark with anti-bias controls

```bash
prompt-forest latency-validate \
  --dataset examples/demo_tasks.json \
  --rounds 3
```

This benchmark compares `full` vs `fast` with:
- fresh engines for cold-turn measurements
- alternating policy order to reduce order bias
- identical datasets and backend family for both profiles
- detailed latency, reward, and backend-call reports in `artifacts/latency_validation/`

### 1e) User feedback loop (personal adaptation)

After a task is answered, send explicit feedback by task id:

```bash
prompt-forest feedback \
  --task-id "<task_id_from_run_output>" \
  --score 0.2 \
  --rejected \
  --corrected-answer "Preferred corrected answer" \
  --feedback-text "What was wrong" \
  --user-id alice
```

Optional profile updates with feedback:

```bash
prompt-forest feedback \
  --task-id "<task_id>" \
  --score 0.9 \
  --accepted \
  --user-id alice \
  --style bullet \
  --verbosity concise \
  --domain-preferences planning,ops \
  --hard-constraints confidence,rollback
```

Feedback reward blend:
- `0.50 * user_feedback`
- `0.25 * verifier`
- `0.15 * task_rules`
- `0.10 * llm_judge`

If corrected answer is provided on rejection, reward is strongly anchored to user feedback.

### 1f) Enable real API-backed Evaluator Agent and Optimizer Agent

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
prompt-forest --config configs/runtime_openai_example.json chat --task-type auto --visibility full --compare-base
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
prompt-forest rl-validate \
  --episodes 240 \
  --seeds 11,17,19,23,29,31,37,41,43,47 \
  --start-mode anti_prior
```

Optional interactive-correction stress mode:

```bash
prompt-forest rl-validate \
  --episodes 240 \
  --start-mode anti_prior \
  --oracle-feedback
```

### 6b) Hard-slice validation (verifier-grounded)

```bash
prompt-forest hard-validate --episodes 220 --seeds 11,17,19,23,29,31,37,41
```

Optional simulated correction signal:

```bash
prompt-forest hard-validate --episodes 220 --oracle-feedback
```

Hard-slice tasks include strict `verifier_spec` metadata and use reward mode `hybrid_verifier`, which adds external verifier checks (`must_include`, `must_exclude`, regex constraints, confidence range, length floor) on top of keyword/rule/task rewards.

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

## What this system demonstrates

- **Learned routing** — Selective multi-branch activation by task type, with routing weights that shift based on performance
- **Branch-level credit assignment** — Reward is attributed to individual branches, not just the final output
- **Self-rewriting prompts** — Branch-local updates modify both weights and prompt templates based on failure analysis
- **Persistent personalization** — Memory-influenced future routing with `(user_id, task_type)` partitioning + global fallback
- **Separation of concerns** — Evaluator and Optimizer are independent agents with strict JSON contracts
- **Autonomous branch evolution** — Candidate branch creation, trial, promotion, and archival based on performance gates
- **API-backed intelligence** — Optional model-backed Evaluator/Optimizer agents for deeper analysis
- **Forest-scale specialization** — Multi-candidate branch spawning for deeper random-forest-like coverage

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

## Roadmap

We are building toward a world where **any closed-source model becomes a self-improving agent** — no fine-tuning, no retraining, no weight access required.

| Phase | Focus | Status |
|---|---|---|
| **Phase 1** | Core routing, evaluation, optimization loop | Done |
| **Phase 2** | Interactive frontend with 3D avatar + live system traces | Done |
| **Phase 3** | Human-mode emotion engine with persistent behavioral memory | Done |
| **Phase 4** | External verifier plugins (unit tests, symbolic solvers, retrieval validators) | Planned |
| **Phase 5** | Contextual bandit / Thompson sampling router policy | Planned |
| **Phase 6** | Multi-objective reward optimization (accuracy, latency, token cost, safety) | Planned |
| **Phase 7** | Branch prompt versioning with A/B rollback dashboards | Planned |
| **Phase 8** | Production-grade OpenClaw middleware integration | Planned |

## Contributing

We welcome contributions. If you're interested in adaptive agent systems, prompt optimization, or RL-without-weights research, open an issue or submit a PR.
