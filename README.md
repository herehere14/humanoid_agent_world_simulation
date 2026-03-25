# Humanoid Agent World Simulation

[中文说明](README.zh-CN.md)

> A decision-grade AI world model for macro shocks, policy design, enterprise strategy, and high-fidelity synthetic societies.

Humanoid Agent World Simulation is a research platform for simulating how people, firms, regulators, and social systems react under pressure. Instead of forecasting only top-line metrics, it models the micro decisions that create those metrics: fear, trust, debt, supply chains, layoffs, policy votes, rumor spread, institutional response, and second-order economic cascades.

This repository combines five layers:

- `Heart Engine`: continuous emotional and psychological state for each agent
- `Ripple Engine`: traceable cause-and-effect propagation through the economy
- `LLM Agency`: named leaders and decision-makers who can change world state directly
- `Household Agency`: every individual agent makes LLM-driven personal finance decisions (cut spending, tap savings, seek second job, ask family for help)
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
| Household LLM agents | All `785` workers/consumers make personal finance decisions via LLM |
| Organizational fabric | `2,736` structural links (employment, supply chains, regulation) |
| Historical validation (directional) | `51 / 61` checks passed, `84%` direction accuracy |
| COVID behavioral similarity | `91%` composite score across 7 dimensions |
| 2008 CCI match | Sim `-43.6%` vs real `-36.8%` (`84%` match) |
| VIX match | Sim peak `58` vs real peak `80` (`73%` match) |
| Agent distinguishability | `75%` blind test accuracy (`6x` over chance) |
| Total LLM decisions per run | `500+` executive + `200+` household decisions per 14-day sim |

### What agents actually decide

**Executives** (via `llm_agency.py`):
- `CUT_HOURS` (237 decisions in a typical crisis run)
- `ABSORB_LOSSES` (75 — leaders eating costs to protect workers)
- `EMERGENCY_FUND` (64 — government deploying relief)
- `RAISE_PRICES`, `ORGANIZE_UNION`, `CONFRONT_MANAGEMENT`, etc.

**Households** (via `household_agency.py`):
- Gig workers: `SEEK_SECOND_JOB` — *"picked up extra rides, cancelled Netflix"*
- Retirees: `CUT_DISCRETIONARY` — *"stopped eating out, cancelled subscriptions"*
- Factory workers: `SEEK_SECOND_JOB` or `CUT_DISCRETIONARY`
- Students: `SEEK_SECOND_JOB` — picking up work to cover loans
- Some: `ASK_FAMILY_HELP`, `TAP_SAVINGS`, `DEFER_BILLS`

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

### 3. LLM Agency (Executive Decisions)

The highest-leverage actors make real decisions that ripple through the economy:

- Jensen Huang analog at `NovaTech` — doubles down or cuts during crises
- Jerome Mitchell at `Federal_Reserve` — approves emergency liquidity
- Janet Morrison at `Treasury_Dept` — deploys emergency funds
- Jamie Stone at `FirstBank` — absorbs losses or demands government help
- Rep. Williams — overcomes fiscal hawkishness to vote for relief
- Ron Vachris at `CostPlus` — holds prices as long as possible, then raises

These decisions are structured, validated, and injected back into the simulation. The LLM is not a narrator floating above the world; it is a participant with power.

### 3b. Household Agency (Individual Decisions)

Every non-executive agent also makes LLM-driven personal finance decisions:

- `CUT_DISCRETIONARY`: cancel Netflix, stop eating out, buy generic brands
- `SEEK_SECOND_JOB`: gig work, extra shifts, weekend warehouse work
- `TAP_SAVINGS`: draw down savings to cover bills
- `ASK_FAMILY_HELP`: borrow from relatives, accept help
- `DEFER_BILLS`: skip rent, negotiate delays with landlord
- `INCREASE_SAVINGS`: tighten belt to save more
- `ORGANIZE_COMMUNITY`: mutual aid, collective action

Each decision reflects the agent's personality. A retiree who fears scarcity cuts discretionary spending. A gig worker who copes by hustling picks up extra shifts. A student with high ambition seeks a second job. These kitchen-table decisions ARE the economy.

### 3c. Economic Circuit Breakers

Research-calibrated mechanisms that prevent unrealistic collapse (based on FRED/BLS/NBER data):

- **Unemployment insurance**: replaces ~45% of income automatically (NBER: 8x more effective than tax cuts)
- **Essential sector protection**: 60-65% of economy is recession-proof (government 22M workers + healthcare + education)
- **Government emergency hiring**: absorbs ~3% of unemployed per cycle when rate exceeds 6%
- **Stimulus checks**: broad debt relief when unemployment exceeds 10%
- **FDIC-equivalent**: prevents total banking collapse
- **Company retention**: firms prefer hour cuts over layoffs (rehiring is expensive)

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
api_server.py                                    FastAPI + SSE backend for the UI
frontend/                                        React/Vite frontend and 3D world-viewer
world_sim/                                       Core world simulation package
  world.py                                       Core tick loop (10 phases per tick)
  world_agent.py                                 Agent model (heart, personality, savings, income)
  scenarios_real_economy.py                      817-agent real-world economy (NVIDIA, Apple, Fed, etc.)
  scenarios_heatwave_harbor.py                   300-1000 agent town crisis scenario
  ripple_engine.py                               Named agent-to-agent cause-and-effect chains
  llm_agency.py                                  Executive LLM decisions (corporate + government)
  household_agency.py                            Individual household LLM finance decisions
  economic_actions.py                            Deterministic economic cascades
  institutional_actions.py                       Government votes and corporate board decisions
  persistent_conditions.py                       Ongoing external states + rally-around-flag
  shock_appraisal.py                             Individual agent shock interpretation
  macro_aggregator.py                            Society-level metrics (CCI, cohesion, unrest)
  market_model.py                                Financial metrics (VIX, S&P, unemployment, credit)
  info_propagation.py                            Information spread through social networks
  contagion.py                                   Emotional contagion at shared locations
  dynamic_events.py                              Endogenous event generation
  action_table.py                                14 deterministic actions
  llm_packet.py                                  Decision packet builder for LLM agents
  relationship.py                                Sparse relationship storage
  human_profiles.py                              Psychological profile assignment

  eval/
    historical_validation.py                     5 historical events vs documented benchmarks
    real_data_validation.py                      Comparison against FRED/BLS/Conference Board data
    real_data_validation_large.py                1000-agent 30-day validation with LLM
    blind_test_large.py                          Agent distinguishability test
    character_identity_blind_test.py             Character identity evaluation

examples/learned_brain/                          Learned-brain research code + SBERT heart engine
src/prompt_forest/                               Adaptive routing, evaluation, memory, orchestration
examples/artifacts/                              Historical validation outputs
artifacts/                                       Runtime reports and experiment logs
docs/architecture.md                             Architecture notes
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
python - <<'PY'
from world_sim.scenarios_real_economy import build_real_economy

world, agent_meta, fabric = build_real_economy(seed=42)
print("agents:", len(world.agents))
print("org_links:", len(fabric.links))
PY
```

### Run historical validation

```bash
python -m world_sim.eval.historical_validation
python -m world_sim.eval.real_data_validation
```

### Run the legacy large-town narrative sim

```bash
export OPENAI_API_KEY="your_key_here"
python -m world_sim.run_large_narrative
```

## Accuracy Journey (This Session)

Starting from a system that produced `0.15x` real-world magnitude:

| Iteration | Oil Surge CCI | Real CCI | Ratio | What changed |
| --- | --- | --- | --- | --- |
| Hardcoded role lookup | `-6.2%` | `-41.6%` | `0.15x` | Starting point |
| Individual agent appraisal | `-6.2%` | `-41.6%` | `0.15x` | Agents interpret shocks by personality |
| 1000 agents, 30 days | `-16.8%` | `-41.6%` | `0.40x` | Scale + time |
| Economic cascades | `-19.0%` | `-41.6%` | `0.46x` | Feedback loops |
| Ripple engine + persistent conditions | `-45.0%` | `-41.6%` | `1.07x` | Named agent-to-agent chains |
| Household agency + circuit breakers | In progress | `-41.6%` | TBD | Every agent decides their own finances |

## Strategic Direction

The near-term roadmap is to make this more useful as a world-modeling product, not just a research demo:

- scale from hundreds of agents to `100,000+`
- expand the real economy with more firms, agencies, and geopolitical actors
- connect live external feeds such as FRED and market data
- deepen the 3D world viewer into an operator dashboard
- let agents browse, gather information, and update their beliefs from the live web
- support multi-crisis stacking such as inflation + layoffs + energy shocks + policy conflict
- improve unemployment calibration with deeper savings/credit/family support modeling

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
