# Prompt Forest Agent

A **hierarchical prompt-routing and online adaptation layer** for LLM agents.

This repository does **not** fine-tune model weights. Instead, it improves behavior by learning at the **control layer** above the model: routing, branch selection, prompt specialization, branch-local updates, reward shaping, and memory.

In other words:

- **Base model** = the frozen foundation model that generates text
- **Prompt Forest** = the adaptive policy layer that decides *how to use* the model
- **Evaluator / Judge** = the component that turns outcomes into structured feedback
- **Optimizer** = the component that updates branch weights, branch prompts, candidate branches, and future routing bias

This makes the repo best understood as a:

- **prompt-level policy optimization system**
- **hierarchical mixture-of-prompts architecture**
- **reward-shaped routing and memory engine**
- **closed-source-model adaptation framework without weight training**

---

## Why this repo exists

Most agent systems treat the base model as fixed and rely on static prompting.
This repo asks a different question:

> If the base model is frozen, can we still make the *system* learn over time?

This project answers **yes** by adapting:

- which branch path gets activated
- how strongly branches are preferred
- how branch prompts evolve locally
- how failures produce new candidate branches
- how memory biases future routing decisions

So the learning is **real**, but it happens in the **external control policy**, not inside the neural weights of the model.

---

## What this repo is

Prompt Forest is a **Random-Forest-inspired, hierarchical prompt-branch architecture** that executes a task through specialized branches, scores the result, and then performs constrained local adaptation.

At a high level, every task goes through this loop:

1. **Task classification**: infer or read the task type (`math`, `planning`, `factual`, `code`, `creative`, `general`)
2. **Hierarchical routing**: choose a path through the forest
3. **Branch execution**: run specialized prompts branch-by-branch
4. **Composition / aggregation**: combine or select the final answer
5. **Evaluation**: produce reward, confidence, and failure signals
6. **Optimization**: update only the active path and related local state
7. **Memory write-back**: store performance signals to bias future routing

The result is an **adaptive agent shell** around a frozen model.

---

## What this repo is *not*

This repo is **not**:

- full neural network training
- backpropagation through model parameters
- LoRA, PEFT, or gradient descent on transformer weights
- AGI
- unrestricted self-rewrite

The system is intentionally constrained.
It performs **bounded, local, interpretable adaptation** rather than opaque global model mutation.

---

## ML analogy: how to think about it

This project becomes much easier to understand if you compare it to machine learning components.

### 1. Branches are like expert sub-networks

Each branch is a specialized reasoning mode:

- `analytical`
- `planner`
- `retrieval`
- `critique`
- `verification`
- `creative`

with niche children underneath them.

These are similar to **experts** in a mixture-of-experts system, except here the experts are **prompt programs**, not separate trained neural modules.

### 2. The router is like a gating network

The router scores branches using:

- learned branch weight
- task-type affinity
- memory bias
- exploration / exploitation logic
- contract-specific routing bonuses and penalties

That makes the router conceptually similar to a **gating function** in mixture-of-experts, contextual bandits, or policy routing.

### 3. Memory acts like an external learned state

Instead of storing everything in weights, the system stores outcomes in memory and uses them later to bias routing.

That means memory here plays a role similar to:

- an external value table
- retrieval-augmented policy bias
- persistent task-conditioned routing prior

### 4. Evaluator signals are like rewards and supervision

The evaluator produces signals such as:

- reward score
- confidence
- failure reason
- suggested improvement direction
- branch-level feedback

This is similar to a combination of:

- reward modeling
- process supervision
- structured critique
- error labeling

### 5. The optimizer is “gradient-like” but not true backprop

The optimizer updates the active path using reward and advantage-style signals.
That is similar in spirit to reinforcement learning or policy improvement.

But it is **not true backpropagation** because:

- there is no computation graph through transformer weights
- no gradient is propagated through hidden activations
- no parameter tensor is optimized with SGD/Adam
- no end-to-end differentiable training loop exists here

A precise way to describe it is:

> Prompt Forest performs **discrete, structured, policy-layer adaptation** that is *analogous* to gradient-based improvement, but it does **not** train the foundation model’s neurons directly.

---

## Relation to neurons, backpropagation, and deep learning

A useful comparison:

### Standard neural training

In a neural network:

- neurons produce activations
- errors are computed at the output
- gradients are backpropagated through layers
- weights are updated continuously
- learning is distributed across parameters

### Prompt Forest training

In this repo:

- branches produce prompted outputs
- a judge/evaluator computes structured outcome signals
- reward is propagated **across the active branch path**
- branch weights and prompts are updated locally
- routing preferences and memory are adjusted over time

So the mapping looks like this:

| Neural net concept | Prompt Forest analogue |
|---|---|
| neuron / feature channel | branch / prompt module |
| gating / expert routing | hierarchical router |
| activation path | activated branch path |
| loss / reward signal | evaluator score + feedback |
| backprop through weights | local reward propagation through path |
| learned parameters | branch weights, prompt state, memory bias |
| hidden state shaping | explicit routing + external memory shaping |

The key scientific distinction is:

> Neural nets learn by **changing internal parameters**. Prompt Forest learns by **changing external control structure**.

That is the most important conceptual point to keep accurate in the README.

---

## Core architecture

```text
User task
  -> task typing / root routing
  -> hierarchical prompt path selection
  -> branch execution
  -> composer / aggregator
  -> evaluator / judge
  -> optimizer
  -> memory update
  -> next-turn routing bias
```

### Forest layout

- **Root node**
- **Layer 1 (macro branches)**:
  - `analytical`
  - `planner`
  - `retrieval`
  - `critique`
  - `verification`
  - `creative`
- **Layer 2 (niche branches)**:
  - specialized child branches beneath each macro branch

Routing is **sequential and hierarchical**.
A task activates a path rather than broadcasting equally to all branches.

---

## Backend architecture

The backend layer is one of the strongest parts of this repo because it separates **control logic** from **generation source**.

### `LLMBackend`

The abstract backend interface defines the generation contract used by the rest of the system.
It supports:

- `generate(...)`
- `generate_stream(...)`

This means the forest logic is backend-agnostic.
The adaptive control layer can sit on top of different text generation providers.

### `MockLLMBackend`

This backend simulates task-branch affinity using a deterministic quality matrix plus controlled noise.

Why it matters:

- gives you a reproducible environment for testing routing behavior
- lets you verify whether the adaptive layer is actually learning better branch selection
- makes it possible to observe adaptation without paying API cost

This is effectively a **simulated environment** for agent-policy research.

### `DomainShiftBackend`

This backend intentionally shifts which branch is best for which task.
That forces the routing/optimization layer to adapt.

Why it matters:

- creates a benchmark where the default affinity prior is wrong
- tests whether the system can recover under distribution shift
- is conceptually similar to validating robustness under **non-stationary task distributions**

### `OpenAIChatBackend`

This backend connects the same adaptive control stack to a real OpenAI-compatible chat completion API.
It tracks:

- latency
- usage
- error counts
- retries
- call logs

Why it matters:

- shows that Prompt Forest is not just a toy simulator
- allows the same adaptive routing layer to run on top of a real closed-source model
- makes the repo a practical **meta-controller for proprietary LLMs**

### Architectural significance of the backend split

This separation is important because it means:

- the **base model can stay frozen**
- the **adaptive intelligence lives in the orchestration layer**
- the same policy logic can be tested in simulation and then transferred to a real API model

That design is unusually clean and should be highlighted in the README.

---

## Routing logic

The router is not a trivial keyword switch.
It combines several signals:

- branch weight
- task-type affinity priors
- memory-derived success bias
- visit counts for exploration
- candidate status penalties
- contract-specific bonuses / penalties
- decayed exploration schedule
- beam-style path pruning in the hierarchical router

This means routing is best described as:

> a lightweight, interpretable, non-neural policy over a forest of prompt experts.

The hierarchical router also supports **contract-aware routing**, which is important for structured outputs such as JSON / CSV / code-patch / bullet-constrained responses.

---

## Two-agent adaptation loop

### Agent 1: Evaluator / Judge

Agent 1 converts behavior into structured optimization signals.
Typical outputs include:

- `reward_score`
- `failure_reason`
- `confidence`
- `suggested_improvement_direction`
- branch-level feedback

This makes Agent 1 similar to a hybrid of:

- reward model
- verifier
- critic
- process judge

### Agent 2: Optimizer / Prompt Updater

Agent 2 performs **constrained local adaptation**.
It can:

- update branch weights within bounds
- use advantage-style updates
- scale steps based on variance / noise
- locally rewrite weak branch prompts
- create candidate branches under strict rules
- propagate path-level reward upstream

It cannot:

- rewrite the entire architecture freely
- create unlimited branches
- mutate the whole system without constraints

This controlled design is one of the repo’s strongest research choices because it keeps adaptation **observable, bounded, and debuggable**.

---

## Candidate branch lifecycle

A powerful idea in this repo is that new branches do not appear arbitrarily.
They follow a lifecycle:

```text
propose -> trial -> promote / archive
```

Candidate branches are only created when:

- failures repeat
- active branches underperform
- no competing active candidate already exists
- branch caps and duplication checks pass

This is interesting from an ML perspective because it resembles **controlled architecture search**, but at the prompt-program level instead of the neural-parameter level.

---

## Memory and personalization

Memory is not just a chat log.
It is part of the learning signal.

The system stores trajectories and uses them to bias future routing.
That gives you:

- persistent adaptation across tasks
- user-conditioned routing tendencies
- path-level performance memory
- explicit feedback integration

In the personal adaptation loop, the user can provide:

- score
- accepted / rejected signal
- corrected answer
- feedback text
- style preferences
- verbosity preferences
- domain preferences
- hard constraints

This is a strong design choice because it turns user interaction into a usable **online learning channel**.

---

## Comparison to OpenClaw-RL

This repo is inspired by the RL framing around OpenClaw-style systems, but it is more precise to say:

- **OpenClaw-RL-style idea**: learn from trajectories, feedback, and next-step outcomes
- **This repo’s specialization**: restrict learning to the prompt-routing-memory-control layer

So rather than training the model itself, this system trains the **decision structure around the model**.

That makes it especially relevant for:

- closed-source foundation models
- API-only deployment settings
- interpretable adaptation research
- fast experimentation without full model retraining

---

## Why this matters scientifically

This repo explores an important research direction:

> Can we get meaningful online adaptation from frozen closed-source models by learning in the orchestration layer instead of the parameter layer?

That question matters because model-weight training is often:

- expensive
- slow
- opaque
- infrastructure-heavy
- impossible for API-only models

Prompt Forest shows an alternative:

- keep the foundation model frozen
- move learning into routing, memory, branching, evaluation, and local prompt updates
- make the system adaptive without touching proprietary weights

This is not a replacement for neural training.
But it is a serious and useful **systems-level learning approach**.

---

## Repository structure

```text
openclaw_closedsourcemodel_RL/
├── artifacts/
├── configs/
├── docs/
├── examples/
├── tests/
├── pyproject.toml
└── src/prompt_forest/
    ├── adapters/
    ├── agents/
    ├── aggregator/
    ├── backend/
    ├── branches/
    ├── core/
    ├── evaluator/
    ├── memory/
    ├── rewards/
    └── router/
```

Important modules:

- `core/engine.py` — main orchestration engine
- `router/hierarchical_router.py` — hierarchical path selection
- `backend/base.py` — backend abstraction
- `backend/mock.py` — deterministic simulation backend
- `backend/simulated.py` — domain-shift backend for adaptation testing
- `backend/openai_chat.py` — real OpenAI-compatible backend
- `agents/evaluator_agent.py` — structured evaluator logic
- `agents/optimizer_agent.py` — local adaptation logic
- `memory/store.py` — persistent routing / performance memory

---

## Installation

### Fast install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

### Editable install

```bash
pip install -e .
pip install -e .[dev]
```

### Package / CLI

The project installs a CLI entrypoint:

```bash
prompt-forest
```

If you have not installed the package, fallback usage is:

```bash
PYTHONPATH=src python -m prompt_forest.cli ...
```

---

## Running the system

### Single task

```bash
prompt-forest run-task \
  --task "Calculate derivative of x^2" \
  --task-type math \
  --metadata '{"expected_keywords": ["derivative", "2x"], "required_substrings": ["confidence"]}'
```

### Interactive chat

```bash
prompt-forest chat --task-type auto --visibility full
```

### Compare adaptive output vs base model

```bash
prompt-forest chat --task-type auto --visibility full --compare-base
```

### Split-view mode

```bash
prompt-forest chat --task-type auto --visibility full --compare-base --split-view
```

### Fast latency mode

```bash
prompt-forest chat --task-type auto --visibility full --compare-base --split-view --latency-mode fast
```

### Inspect traces

```bash
prompt-forest inspect-events --limit 8 --visibility full
```

### User feedback loop

```bash
prompt-forest feedback \
  --task-id "<task_id>" \
  --score 0.2 \
  --rejected \
  --corrected-answer "Preferred corrected answer" \
  --feedback-text "What was wrong" \
  --user-id alice
```

### Benchmark

```bash
prompt-forest benchmark --dataset examples/demo_tasks.json --rounds 6
```

### Latency validation

```bash
prompt-forest latency-validate \
  --dataset examples/demo_tasks.json \
  --rounds 3
```

---

## Using a real OpenAI-compatible model

```bash
export OPENAI_API_KEY="your_key_here"
prompt-forest --config configs/runtime_openai_example.json chat \
  --model gpt-4.1-mini \
  --task-type auto \
  --visibility full \
  --compare-base \
  --split-view
```

This setup can use:

- a real OpenAI-compatible model for the main adaptive conversation
- a real base-model comparison stream
- model-backed evaluator / optimizer runtimes

Supported runtime provider examples in the repo include:

- `openai_compatible`
- `gemini`

---

## Best one-line summary

> **Prompt Forest is an adaptive control layer for frozen LLMs: a hierarchical prompt-routing, memory, evaluation, and local-optimization system that learns without changing model weights.**

---

## Strong research framing

A stronger and more accurate way to position the project is:

> Prompt Forest explores **online systems-level learning for closed-source LLMs** by shifting adaptation from neural parameters to interpretable control structure: branch routing, local prompt evolution, reward-guided optimization, and persistent memory.

That framing is more accurate than saying it does standard backpropagation, and more advanced than calling it only “prompt engineering.”

---

## Future directions

High-value future extensions:

- better contextual bandit routing
- learned meta-routing over branch histories
- formal regret-style evaluation of routing policy
- richer candidate-branch architecture search
- better verifier reward decomposition
- more principled offline / online evaluation splits
- branch pruning and compression strategies
- transfer of learned routing priors across users or domains

---

## Final conceptual takeaway

The deepest idea in this repo is simple:

**You do not always need to train the neurons to make the system learn.**

Sometimes it is enough to train the **policy around the neurons**.

That is what Prompt Forest is doing.
