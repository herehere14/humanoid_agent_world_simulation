# Humanoid Agent World Simulation

[中文说明](README.zh-CN.md)

> A decision-grade AI world model for macro shocks, policy design, enterprise strategy, and high-fidelity synthetic societies.

Humanoid Agent World Simulation is a research platform for simulating how people, firms, regulators, and social systems react under pressure. Instead of forecasting only top-line metrics, it models the micro decisions that create those metrics: fear, trust, debt, supply chains, layoffs, policy votes, rumor spread, institutional response, and second-order economic cascades.

This repository combines four layers:

- `Heart Engine`: continuous emotional and psychological state for each agent
- `Ripple Engine`: traceable cause-and-effect propagation through the economy
- `LLM Agency`: named leaders and decision-makers who can change world state directly
- `Prompt Forest Runtime`: adaptive routing, evaluation, and frontend/backend tooling around the broader agent stack

## Why This Matters

Most simulation systems are either:

- too abstract to explain *why* outcomes happen, or
- too expensive to scale because they require an LLM call for every agent on every tick

This project takes a different path. The deterministic layer carries the full world forward every tick, while the LLM layer is used selectively where executive decisions, political judgment, or high-salience moments actually matter.

That makes it useful for:

- investors testing how shocks propagate through industries
- enterprises pressure-testing layoffs, pricing, and supply-chain moves
- policy teams evaluating stimulus, lockdowns, or rate decisions
- game studios building NPC economies that feel genuinely alive
- researchers exploring emergent behavior from psychologically grounded agents

## Validation Snapshot

| Signal | Current repo evidence |
| --- | --- |
| Real-economy scenario scale | `817` agents built by `scenarios_real_economy.py` |
| Named LLM decision-makers | `32` leaders across firms and institutions |
| Organizational fabric | `2,736` structural links in the real-economy builder |
| Historical validation suite | `51 / 61` checks passed, `86.7%` direction accuracy in `examples/artifacts/historical_validation.json` |
| COVID historical validation | `10 / 12` checks passed in the public artifact |
| Measured real-data comparison | `81.2%` direction accuracy across `16` metric comparisons in `examples/artifacts/real_data_validation.json` |
| Financial-crisis output artifact | `artifacts/financial_crisis_2008_sim.json` |

Project positioning materials describe the COVID scenario as roughly `91%` behaviorally similar to history by composite scenario scoring. The public repository currently includes the more conservative automated validation artifacts above, which are a better source of truth for technical diligence.

## What The System Actually Simulates

### 1. Heart Engine

Every agent carries a continuously updated internal state:

- arousal
- valence
- tension
- impulse control
- energy
- vulnerability

The engine also tracks persistent wounds, coping style, attachment patterns, threat lens, and subjective interpretation. Two workers facing the same shock do not behave the same way, because the model keeps psychological state continuous instead of resetting each turn.

### 2. Ripple Engine

Decisions are not treated as isolated events. They become causal chains.

Example:

```text
CEO freezes hiring
-> team loses confidence
-> overtime gets cut
-> household spending falls
-> local merchants see lower demand
-> more employers tighten budgets
```

The system stores this as explicit propagation through named actors and organizational links instead of vague "macro pressure" alone.

### 3. LLM Agency

The highest-leverage actors can make real decisions:

- Jensen Huang analog at `NovaTech`
- central bank leadership at `Federal_Reserve`
- Treasury leadership, Congress, CDC
- banking, retail, energy, and pharma executives

These decisions are structured, validated, and injected back into the simulation. The LLM is not a narrator floating above the world; it is a participant with power.

### 4. Real Economy Layer

The real-economy builder contains:

- named companies such as `NovaTech`, `ApexDevices`, `CloudScale`, `MegaMart`, `RetailGiant`, `CostPlus`, `FirstBank`, `PetroMax`, and `PharmaCore`
- institutions including the `Federal_Reserve`, `US_Congress`, `Treasury_Dept`, and `CDC`
- hundreds of employees, workers, students, retirees, gig workers, and small business owners
- supply, employment, regulation, and governance links that turn isolated choices into economy-wide effects

## Example Use Cases

### Capital Allocation And Investing

Simulate rate hikes, oil shocks, bank stress, consumer pullbacks, or supply bottlenecks before capital is committed. The value is not just directional output, but a legible chain of why the system moved.

### Corporate Strategy

Ask questions like:

- What happens if we cut headcount by 20%?
- What if a competitor overinvests during a downturn?
- How does a reputational crisis spread from consumers to regulators to suppliers?

### Policy And Public Sector Planning

Model lockdowns, stimulus packages, emergency messaging, or financial rescues. Stress-test who absorbs damage first, where trust breaks, and which interventions stabilize the system fastest.

### Games And Virtual Worlds

Use the same stack to create NPCs with memory, fear, exhaustion, debt pressure, coalition behavior, and location-based emotional contagion, rather than scripted barks.

## Repository Map

```text
api_server.py                               FastAPI + SSE backend for the UI
frontend/                                   React/Vite frontend and 3D world-viewer work
src/prompt_forest/                          Adaptive routing, evaluation, memory, orchestration
examples/learned_brain/world_sim/           Simulation engine, scenarios, LLM agency, validation
examples/artifacts/                         Historical validation outputs and evaluation data
artifacts/                                  Runtime reports, benchmark outputs, experiment logs
docs/architecture.md                        Earlier architecture notes for the prompt-forest layer
```

## Quick Start

### Python environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pip install fastapi uvicorn openai
```

This is still a research repository, so some simulation paths may require additional ML dependencies from your local environment.

### Launch the UI + backend

```bash
./start.sh
```

Backend: `http://localhost:8000`  
Frontend: `http://localhost:3000`

### Inspect the real-economy builder

```bash
PYTHONPATH=examples/learned_brain python - <<'PY'
from world_sim.scenarios_real_economy import build_real_economy

world, agent_meta, fabric = build_real_economy(seed=42)
print("agents:", len(world.agents))
print("org_links:", len(fabric.links))
PY
```

### Run historical validation

```bash
PYTHONPATH=examples/learned_brain python examples/learned_brain/world_sim/eval/historical_validation.py
PYTHONPATH=examples/learned_brain python examples/learned_brain/world_sim/eval/real_data_validation.py
```

### Run the legacy large-town narrative sim

```bash
export OPENAI_API_KEY="your_key_here"
PYTHONPATH=src python examples/learned_brain/world_sim/run_large_narrative.py
```

## Strategic Direction

The near-term roadmap is to make this more useful as a world-modeling product, not just a research demo:

- scale from hundreds of agents to `100,000+`
- expand the real economy with more firms, agencies, and geopolitical actors
- connect live external feeds such as FRED and market data
- deepen the 3D world viewer into an operator dashboard
- let agents browse, gather information, and update their beliefs from the live web
- support multi-crisis stacking such as inflation + layoffs + energy shocks + policy conflict

## Investment Thesis

If this category works, it becomes more than a simulator. It becomes infrastructure for:

- enterprise scenario planning
- synthetic training environments for AI agents
- next-generation economic intelligence products
- believable game and virtual-world populations
- policy rehearsal before real-world rollout

The long-term implication is straightforward: software will not just answer questions about the world. It will model the world well enough to let teams test futures before they pay for them.

## License

MIT
