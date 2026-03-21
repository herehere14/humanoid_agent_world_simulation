# Humanoid Agent World Simulation

**A computational framework for emergent multi-agent social dynamics — where hundreds of autonomous agents develop persistent emotional states, form complex relationships, and collectively shape a living world through deterministic cognition and narrative intelligence.**

---

> We are building toward autonomous agents that don't just act — they *feel*, remember, adapt, and influence each other in ways that produce emergent social phenomena indistinguishable from human communities. This system demonstrates that believable agency at scale doesn't require an LLM call per agent per tick — it requires the right cognitive architecture.

---

## Overview

This project implements a **population-scale agent simulation engine** capable of running hundreds of autonomous agents with continuous emotional state, persistent memory, wound psychology, relationship dynamics, and emergent behavioral cascades — all at sub-second tick speeds with near-zero inference cost for state computation.

Each agent is driven by **HeartState**, a real-time emotional cognition engine that maintains six continuous psychological variables, processes events through learned semantic embeddings, and produces deterministic behavioral decisions. When combined with selective LLM narration, the system generates rich multi-day narratives where factory workers organize protests, relationships fracture under stress, and community resilience emerges organically from individual agent states.

### Key Results

| Metric | Value |
|---|---|
| **Agent scale** | 300 agents, 8 districts, 25 locations |
| **Tick speed** | 1,000 agents in 52 seconds (pure state computation) |
| **State differentiation** | Cohen's d = −14.92 (laid-off vs. bystander emotional divergence) |
| **Relationship complexity** | 3,655 relationship pairs formed organically over 10 simulated days |
| **Narration cost** | 300-agent, 10-day full narrative: $0.13 (973 LLM calls) |
| **Heart vs. LLM blind test** | 51.2% overall win rate; 62% in recovery-phase emotional consistency |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    World Simulation                      │
│                                                         │
│  ┌──────────┐   ┌──────────┐   ┌───────────────────┐   │
│  │ Scheduled │   │ Location │   │   Relationship    │   │
│  │  Events   │──▶│  Engine  │──▶│     Store         │   │
│  └──────────┘   └──────────┘   │ (trust, warmth,   │   │
│                                │  resentment,       │   │
│       ┌────────────────────┐   │  familiarity)      │   │
│       │   SharedBrain      │   └───────────────────┘   │
│       │ (SBERT + Anchors)  │            │               │
│       └────────┬───────────┘            │               │
│                │                        │               │
│   ┌────────────▼────────────────────────▼───────────┐   │
│   │              Per-Agent Tick Loop                 │   │
│   │                                                  │   │
│   │  1. Event Embedding (SBERT encode)              │   │
│   │  2. HeartState Update (numpy, <1ms)             │   │
│   │  3. Wound Decay & Application                   │   │
│   │  4. Action Selection (deterministic rules)      │   │
│   │  5. Interaction Resolution (pair matching)      │   │
│   │  6. Emotional Contagion (location-based)        │   │
│   │  7. Memory Recording                            │   │
│   └─────────────────────────────────────────────────┘   │
│                         │                               │
│              ┌──────────▼──────────┐                    │
│              │  Selective LLM      │                    │
│              │  Narration Layer    │                    │
│              │  (async, ranked by  │                    │
│              │   drama score)      │                    │
│              └─────────────────────┘                    │
└─────────────────────────────────────────────────────────┘
```

---

## HeartState Engine

Each agent maintains a continuous emotional state with six real-valued dimensions, updated every simulated hour through semantic similarity against learned emotion anchors:

| Variable | Range | Function |
|---|---|---|
| **Arousal** | 0.0 – 1.0 | Activation level — drives urgency and social broadcast strength |
| **Valence** | 0.0 – 1.0 | Positive/negative affect — primary driver of behavioral mode |
| **Tension** | 0.0 – 1.0 | Accumulated stress — triggers confrontation and withdrawal thresholds |
| **Impulse Control** | 0.0 – 1.0 | Self-regulation capacity — decays under sustained distress |
| **Energy** | 0.0 – 1.0 | Physical/psychological reserves — depleted by high arousal, restored by rest |
| **Vulnerability** | 0.0 – 1.0 | Emotional exposure — amplifies contagion susceptibility and wound depth |

### Wound System

Traumatic events create persistent **wounds** — `(impact, decay_rate)` tuples that continuously drag valence downward across simulated days:

- **Targeted events** (e.g., being personally fired): decay rate 0.996 (~10-day half-life)
- **Ambient events** (e.g., witnessing a factory closure): decay rate 0.992 (~3-day half-life)
- Wounds stack multiplicatively — an agent who loses their job, then witnesses a colleague collapse, carries compounding emotional damage

### Deterministic Action Selection

Actions are selected through priority-ordered rules evaluated against HeartState thresholds — no randomness, no LLM inference:

```
COLLAPSE > LASH_OUT > CONFRONT > FLEE > WITHDRAW > SEEK_COMFORT >
RUMINATE > VENT > CELEBRATE > HELP_OTHERS > WORK > IDLE
```

Each action has behavioral consequences: `WITHDRAW` and `FLEE` move the agent home; `SEEK_COMFORT` relocates to the nearest warm relationship; `LASH_OUT` generates ripple events affecting nearby agents.

### Emotional Contagion

Agents at the same location influence each other's emotional state:
- **Broadcast strength** scales with arousal (highly activated agents spread emotion further)
- **Susceptibility** scales with vulnerability and relationship warmth
- Creates emergent phenomena: panic cascades in crowds, calming effects of trusted relationships

---

## Simulation Features

### Multi-District World

The 300-agent simulation spans 8 districts with 25 distinct locations:

| District | Locations | Primary Agents |
|---|---|---|
| Industrial | Factory floor, loading docks, break room | Factory workers, dock workers |
| Downtown | Office towers, courthouse, restaurants | Professionals, government workers |
| University | Lecture halls, library, student union | Students, academics |
| Market | Market square, food stalls, workshops | Vendors, artisans |
| Waterfront | Docks, fish market, harbor bar | Dock workers, fishers |
| Government Hill | City hall, municipal offices | Government workers, officials |
| Suburbs North/South | Residential areas, parks, schools | Community members, families |
| Central | Central park, town square | Cross-district convergence point |

### Relationship Tracking

Sparse-storage relationship system tracking four dimensions per agent pair:

- **Trust** — accumulated through positive interactions, eroded by conflict
- **Warmth** — emotional closeness, drives comfort-seeking behavior
- **Resentment** — directional (A→B independent of B→A), triggers confrontation
- **Familiarity** — interaction count, influences partner selection priority

Relationships are seeded through workplace proximity, neighborhood adjacency, cross-district family ties, and rival pairs — then evolve organically through simulation.

### Event Cascades & Ripple Mechanics

Events don't exist in isolation. The system implements cascading event chains:

1. **Scheduled events** inject crises at specific ticks and locations
2. **Ripple detection** analyzes post-tick state — crowd distress (avg valence < 0.25 with 8+ agents) or extreme individual actions (LASH_OUT, COLLAPSE) generate new events
3. **Generated events** affect nearby agents at the next tick, creating chain reactions
4. **Location overrides** move agents to rallies, protests, and town halls on specific days

Example cascade from the 300-agent simulation:
```
Day 2: Chemical leak at industrial district
  → 45 factory workers wounded (valence drops to 0.26)
  → 3 agents LASH_OUT, generating witness-trauma events
Day 3: Factory shutdown announced
  → Targeted layoffs deepen wounds (10-day decay)
  → Workers begin organizing (CONFRONT actions spike)
Day 4: Protest march — 75 agents converge on central park
  → Cross-district encounters between workers and officials
  → Emotional contagion spreads distress to bystanders
Day 7: Community rally — 150 agents
  → Support networks form (warmth relationships spike)
  → Recovery phase begins for connected agents
```

### Selective Narration

Not every agent interaction needs LLM narration. The system ranks moments by **drama score**:

```
drama = vulnerability + (1 - valence) + arousal + cross_district_bonus
```

Per tick, only the highest-drama interactions receive LLM narration:
- Top 12 interactions (by combined drama of both agents)
- Top 8 event reactions (agents directly affected by scheduled events)
- Top 5 solo moments (isolated high-distress agents)
- Crowd scenes (20+ agents at one location)
- District pulse summaries (every 6 simulated hours)

This selective approach keeps narration costs under $0.15 for a full 300-agent, 10-day simulation while focusing narrative attention where it matters most.

---

## Learned Brain (Phase 1)

The emotional classification backbone uses a **GRU-based sequence encoder** trained on the EmpatheticDialogues dataset:

| Component | Detail |
|---|---|
| **Architecture** | GRU encoder → 32-dimensional latent state |
| **Training data** | EmpatheticDialogues (25K conversations, 32 emotion labels) |
| **Embedding** | SBERT (all-MiniLM-L6-v2) 384-dim sentence embeddings |
| **Top-1 accuracy** | 44% on 32 emotion labels |
| **Top-5 accuracy** | 82% |
| **Anchor system** | 32 emotion anchors with precomputed SBERT embeddings for real-time cosine similarity |

The brain provides fast, learned emotion classification that feeds HeartState updates — replacing hardcoded keyword matching with semantic understanding.

---

## RL-Optimized Emotional Planning (Phase 2)

Phase 2 addresses the core challenge: LLMs generate similar prose regardless of the emotional context passed to them. Three components work together to force behavioral differentiation:

### Reward Model
- MLP: `[latent_state(32) || response_embedding(384)]` → predicted judge score
- Trained on 1,200 scored examples from blind GPT-4o evaluation
- Enables Best-of-N selection (N=4 candidates at varied temperatures)

### Prompt Strategy Policy
- 8 behavioral strategies: `raw_explosive`, `cold_controlled`, `sarcastic_bitter`, `anxious_scattered`, `defeated_minimal`, `warm_engaged`, `confident_direct`, `cautious_measured`
- Policy network selects strategy based on brain latent state
- Trained with REINFORCE using reward model scores

### Conversation Arc Planner
- Single LLM call plans 6-turn emotional trajectory at conversation start
- Per-turn targets: `{target_valence, target_intensity, preferred_strategy, arc_note}`
- Soft bias on policy selection for narrative consistency

---

## Evaluation Suite

### Scale Benchmark
Measures pure computational throughput and state differentiation without LLM calls:
- 1,000 agents simulated in 52 seconds
- Validates that targeted agents show statistically significant emotional divergence from bystanders

### Heart vs. LLM Blind Comparison
GPT-4o judges Heart-narrated vs. pure-LLM agent moments in blind A/B tests:
- Heart wins 51.2% overall
- Heart wins 62% during recovery phases (where persistent state tracking matters most)
- Pure LLM wins 52% during acute crisis (where dramatic prose without state tracking is sufficient)

### Character Identity Blind Test
Tests whether agents are distinguishable from each other — the core question of whether the system produces *characters* or *interchangeable narration*.

---

## Quick Start

```bash
git clone https://github.com/herehere14/humanoid_agent_world_simulation.git
cd humanoid_agent_world_simulation
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run the 30-Agent Simulation

```bash
export OPENAI_API_KEY="your_key_here"
PYTHONPATH=src python examples/learned_brain/world_sim/run_narrative.py
```

Produces a 15-day narrative across a small town with layoff crisis, community response, and recovery arc. ~540 LLM calls, ~$0.06.

### Run the 300-Agent Simulation

```bash
PYTHONPATH=src python examples/learned_brain/world_sim/run_large_narrative.py
```

Full multi-district simulation with 300 agents, 8 districts, 45 scheduled events, protest marches, cross-district encounters, and emergent relationship networks. ~970 LLM calls, ~$0.13.

### Run Evaluations

```bash
# Scale benchmark (no API key needed)
PYTHONPATH=src python examples/learned_brain/world_sim/eval/scale_benchmark.py

# Heart vs LLM comparison
PYTHONPATH=src python examples/learned_brain/world_sim/eval/eval_heart_vs_llm.py

# Full evaluation suite
PYTHONPATH=src python examples/learned_brain/world_sim/eval/run_all.py
```

---

## Project Structure

```
├── src/prompt_forest/
│   └── brain/                      # Neural cognitive architecture
│       ├── unified_predictor.py    # Main prediction interface
│       ├── brain_predictor.py      # Brain-specific prediction logic
│       ├── sequence_encoder.py     # GRU sequence encoding
│       ├── latent_state.py         # 32-dim latent state management
│       ├── transition_model.py     # State transition dynamics
│       ├── prospect_learner.py     # Prospect-theoretic learning
│       └── controller.py          # Decision control logic
│
├── examples/learned_brain/
│   ├── model.py                    # GRU emotion classifier
│   ├── learned_brain_engine.py     # Brain engine integration
│   ├── heart_engine.py             # HeartState computation engine
│   ├── reward_model.py             # Phase 2 reward model
│   ├── prompt_policy.py            # Phase 2 behavioral strategy policy
│   ├── arc_planner.py              # Phase 2 conversation arc planning
│   ├── phase2_pipeline.py          # Full Phase 2 pipeline integration
│   │
│   └── world_sim/                  # World simulation engine
│       ├── world.py                # Simulation loop, tick engine
│       ├── world_agent.py          # Agent definition, HeartState, SharedBrain
│       ├── action_table.py         # Deterministic action selection rules
│       ├── relationship.py         # Sparse relationship storage
│       ├── contagion.py            # Emotional contagion mechanics
│       ├── scenarios.py            # 30-agent scenario builder
│       ├── scenarios_large.py      # 300-agent multi-district builder
│       ├── run_narrative.py        # 30-agent narrative runner
│       ├── run_large_narrative.py  # 300-agent narrative runner
│       ├── dashboard.py            # Agent/world state dashboard
│       └── eval/                   # Evaluation suite
│           ├── scale_benchmark.py
│           ├── eval_heart_vs_llm.py
│           ├── judge_comparison.py
│           └── character_identity_blind_test.py
│
├── examples/
│   ├── narrative.md                # 30-agent simulation output (4,257 lines)
│   └── city_narrative.md           # 300-agent simulation output (7,258 lines)
│
├── tests/                          # 33 test files
└── configs/                        # Runtime configurations
```

---

## Technical Foundations

This system draws from several research domains:

- **Affective computing** — continuous emotional state representation with decay dynamics and wound persistence
- **Multi-agent simulation** — emergent social phenomena from local interaction rules at population scale
- **Sentence embeddings** — SBERT-based semantic similarity for event-to-emotion mapping without task-specific training
- **Reinforcement learning** — reward-shaped prompt policy optimization for LLM behavioral differentiation
- **Computational social science** — relationship dynamics, emotional contagion, and collective behavior modeling

---

## Roadmap

| Phase | Description | Status |
|---|---|---|
| **Phase 1** | Learned brain — GRU emotion encoder, SBERT anchors | Complete |
| **Phase 2** | RL-optimized emotional planning — reward model, prompt policy, arc planner | Complete |
| **Phase 3** | World simulation — 300-agent multi-district engine with narration | Complete |
| **Phase 4** | Personality engine — PersonalityDNA, behavioral directives, character-specific prompting | Planned |
| **Phase 5** | Dynamic world events — agent actions reshape environment, economic systems, political shifts | Planned |
| **Phase 6** | Long-term memory — multi-week persistent memory with emotional significance decay | Planned |

---

## License

Research project — see repository for details.

## Contributing

We welcome contributions in multi-agent systems, affective computing, emergent narrative generation, and computational social simulation. Open an issue or submit a PR.
