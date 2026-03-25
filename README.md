# AI Agents World Simulator / we priortise scale, with our heart and ripple engine, we are able to simulate a huge world / economy with less than $1 token cost per run (about 20000 agents)

[中文说明](README.zh-CN.md)

> **Inject a real-world shock. Watch 10000 named AI agents — CEOs, workers, retirees, gig drivers etc — make individual decisions that cascade through companies, governments, and households. See what happens to the economy.**

```
>>> Financial crises hit

  [Day 3] Darren Foster (PetroMax CEO): CUT_HOURS
    "To avoid layoffs, we're reducing shifts across the board."
    → 35 PetroMax workers lose income

  [Day 3] Olivia (gig worker): SEEK_SECOND_JOB
    "Picked up extra rides and food-delivery shifts, cancelled Netflix."

  [Day 4] Janet Morrison (Treasury Secretary): EMERGENCY_FUND
    "We are authorizing a targeted emergency liquidity facility."
    → 40 distressed citizens receive relief

  [Day 5] Ron Vachris (CostPlus CEO): RAISE_PRICES
    "I won't let anyone here lose their job, but prices have to go up."
    → 15 customers pay more for essentials

  CCI: 81 → 45 (-44%)  |  Real 2008 CCI: -42%  |  Match: 95%
  unemployment rate: 3% to 14% | Real 2008 unemployment rate: 10% peak crises| over 70% match
```

**Every decision is made by a named individual with a specific personality, and every decision has concrete consequences for other named individuals. These LLM agents will
influence non-LLM agents through our heart and ripple engine, broaden the effects in the world**


---

## What Makes This Different

| Feature | This Project | Typical Agent Sims |
|---------|:---:|:---:|
| Agents have persistent psychology | Heart engine with wounds, coping, attachment | Stateless or shifting mood depends on their actions |
| Decisions create named ripple chains | Gavin raises prices → Rosa pays more → she cuts spending -> butterfly effects | Aggregate "pressure" numbers |
| LLM agents make REAL decisions | Jensen Huang decides to invest; that actually hires workers | LLM generates dialogue other LLM reads and interacts with |
| Household agents control their own money | Each person decides: cut spending, tap savings, find second job | Top-down income assignment |
| Calibrated from real economic data | BLS, FRED, NBER, IMF — every parameter has a source | Guessed parameters |
| Named companies and institutions | NovaTech, Federal Reserve, CostPlus, FirstBank etc | "company_1", "gov_agent_3" |

---

## Start Here: Key Files

| File | What It Does | Lines |
|------|-------------|:-----:|
| **[`world_sim/world.py`](world_sim/world.py)** | **Core simulation loop** — 10 phases per tick: heart update, actions, interactions, economic cascades, ripple chains, LLM decisions, macro aggregation | ~400 |
| **[`world_sim/scenarios_real_economy.py`](world_sim/scenarios_real_economy.py)** | **Real-world economy builder** — 817 agents: NVIDIA, Apple, Microsoft, Amazon, Walmart, JPMorgan, Federal Reserve, Congress, Treasury, CDC + 320 workers/consumers | ~600 |
| **[`world_sim/llm_agency.py`](world_sim/llm_agency.py)** | **Executive LLM decisions** — CEOs/CFOs decide: cut workers, absorb losses, raise prices, lobby government. Decisions ripple through org fabric | ~500 |
| **[`world_sim/household_agency.py`](world_sim/household_agency.py)** | **Household LLM decisions** — Every individual decides their own finances: cut spending, tap savings, seek second job, ask family for help | ~450 |
| **[`world_sim/ripple_engine.py`](world_sim/ripple_engine.py)** | **Cause-and-effect chains** — When Gavin raises prices, Rosa/Barrett/Hector each individually pay more. 2,736 organizational links | ~400 |
| **[`world_sim/world_agent.py`](world_sim/world_agent.py)** | **Agent model** — Heart state, personality (threat lens, coping style, self-story), savings, income, employment status, memory | ~900 |
| **[`world_sim/calibrated_economy.py`](world_sim/calibrated_economy.py)** | **Research-calibrated economics** — Every parameter from BLS/FRED/NBER data. Sector risk caps, fiscal multipliers, savings distribution | ~300 |
| **[`data/economic_simulation_research.md`](data/economic_simulation_research.md)** | **533 lines of real economic research** — household spending, corporate cascades, labor markets, price transmission, government response timelines | 533 |

---

## Live Simulation Results

### COVID-19 Pandemic — 91% Behavioral Similarity

| Metric | Score |
|--------|:-----:|
| Corporate response pattern (absorb → cut hours) | **100%** |
| Government response (stimulus + restrictions) | **100%** |
| Worker response (organize, confront, seek work) | **100%** |
| Sector impact (retail cut, finance absorbed) | **100%** |
| Decision diversity (11 types, all 3 pillars active) | **100%** |
| Trajectory shape (rapid decline → deceleration → floor) | **87%** |
| CCI magnitude (-53.8% vs real -35.4%) | **69%** |

### 2008 Financial Crisis

| Metric | Simulation | Real 2008 | Match |
|--------|:---:|:---:|:---:|
| CCI drop | -44% | -36.8% | 84% |
| VIX peak | 58 | 80 | 73% |
| Credit stress | maxed | LIBOR-OIS 365bp | 100% |
| Corporate: hour cuts dominant | 237 decisions | Widespread | 100% |
| Government: emergency funds | 81 decisions | TARP + ARRA | 100% |

### Agent Distinguishability — 75% (6x over chance)

In a blind test, a judge model correctly identified which agent said what 75% of the time from dialogue alone. Daria (dock worker) was 100% identifiable. Each agent has a unique voice driven by their psychology.

---

## The Three Pillars

### 1. Government

Named officials examples make policy decisions via LLM:
- **Jerome Mitchell** (Fed Chair, fears chaos, needs control) → approves emergency liquidity
- **Janet Morrison** (Treasury Secretary, fears scarcity, needs usefulness) → deploys emergency funds
- **Rep. Williams** (Budget Chair, fears scarcity, needs justice) → overcomes fiscal hawkishness to vote for relief
- **Dr. Mandy Cohen** (CDC Director, fears chaos, needs truth) → organizes mutual aid


### 2. Firms

Named executives examples at named companies make business decisions via LLM:
- **Jensen Huang** (NovaTech CEO) → doubles down on AI investment during downturn
- **Jamie Stone** (FirstBank CEO) → hoards capital, demands government support
- **Ron Vachris** (CostPlus CEO) → holds prices as long as possible to protect customers
- **Daniel Pinto** (FirstBank Co-President) → absorbs losses to protect his team

### 3. Individuals (Households) examples

Every worker/consumer makes personal finance decisions via LLM:
- **Olivia** (gig worker): *"Picked up extra rides, cancelled Netflix"* → `SEEK_SECOND_JOB`
- **Priya** (retiree): *"Stopped eating out, cancelled subscriptions"* → `CUT_DISCRETIONARY`
- **Brian** (factory worker): *"Picking up weekend overtime"* → `SEEK_SECOND_JOB`
- **Carlos** (retiree): *"Asked my daughter if I could borrow until pension clears"* → `ASK_FAMILY_HELP`

---

## Economic Circuit Breakers (Research-Calibrated)

Based on real FRED/BLS/NBER data — prevents unrealistic economic collapse:

| Mechanism | Real Data | Effect in Sim |
|-----------|-----------|--------------|
| Unemployment insurance | Replaces 43% of wages (DOL) | Income floor for laid-off agents |
| Essential sector protection | 60-65% of economy recession-proof | Healthcare/gov/teacher agents stay employed |
| Fiscal multiplier | 1.0-1.5x during recession (IMF) | Government spending amplified |
| Sector risk caps | Only 15-20% of workforce loses jobs (BLS) | Prevents 100% unemployment |
| Savings buffers | 55% have 3 months saved; 30% paycheck-to-paycheck (Fed) | Realistic depletion timeline |
| Corporate retention | Companies prefer hour cuts over layoffs | Income reduces before employment ends |

---

## Repository Map

```
world_sim/                          CORE SIMULATION ENGINE
  world.py                          10-phase tick loop
  world_agent.py                    Agent model (heart + personality + economy)
  scenarios_real_economy.py         817 agents: named companies + institutions
  llm_agency.py                     Executive LLM decisions
  household_agency.py               Household LLM decisions
  ripple_engine.py                  Named cause-and-effect chains
  calibrated_economy.py             Research-calibrated parameters
  economic_calibration.py           All parameter values with sources
  shock_appraisal.py                Individual agent shock interpretation
  macro_aggregator.py               Society-level metrics
  market_model.py                   Financial metrics (VIX, S&P, unemployment)
  persistent_conditions.py          Ongoing crises + rally-around-flag
  institutional_actions.py          Government/corporate board decisions
  economic_actions.py               Deterministic economic cascades
  info_propagation.py               Information spread via social networks
  contagion.py                      Emotional contagion at locations
  dynamic_events.py                 Endogenous event generation
  world_information.py              6 shock types (oil, banking, COVID, etc.)
  action_table.py                   14 deterministic agent actions
  relationship.py                   Sparse relationship storage
  human_profiles.py                 Psychological profiles
  eval/                             Validation against real historical data

data/
  economic_simulation_research.md   533 lines of real-world economic research

api_server.py                       FastAPI backend + SSE streaming
frontend/                           React/3D world viewer
src/prompt_forest/                  Adaptive routing + evaluation engine
artifacts/                          Simulation outputs + validation reports
```

## Quick Start

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pip install fastapi uvicorn openai sentence-transformers

# Build the real economy (no API key needed)
python -c "
from world_sim.scenarios_real_economy import build_real_economy
world, meta, fabric = build_real_economy()
print(f'Agents: {len(world.agents)}')
print(f'Org links: {len(fabric.links)}')
print(f'LLM leaders: {sum(1 for m in meta.values() if m.get(\"is_llm_agent\"))}')
"

# Run with LLM decisions (needs OpenAI API key)
export OPENAI_API_KEY="your-key"
python -c "
from world_sim.scenarios_real_economy import build_real_economy
from world_sim.llm_agency import LLMAgencyEngine
world, meta, fabric = build_real_economy()
world.initialize()
for _ in range(48): world.tick()  # baseline
world.ingest_information('Oil prices surge 100%')
llm = LLMAgencyEngine(api_key='$OPENAI_API_KEY', model='gpt-5-mini', fabric=fabric)
for _ in range(72):
    world.tick()
    for evt in llm.tick(world, max_calls=3):
        print(f'{evt.actor_name} -> {evt.target_name}: {evt.action}')
"

# Launch UI
./start.sh
```

## Accuracy Journey

| Version | CCI Drop | Real 2008 | Ratio | What Changed |
|---------|:---:|:---:|:---:|-------------|
| Hardcoded lookup | -6% | -42% | 0.15x | Starting point |
| Individual appraisal | -6% | -42% | 0.15x | Agents interpret by personality |
| 1000 agents, 30 days | -17% | -42% | 0.40x | Scale + time |
| Economic cascades | -19% | -42% | 0.46x | Feedback loops |
| Ripple engine + persistence | -45% | -42% | 1.07x | Named agent chains |
| Calibrated economy | -44% | -42% | 1.05x | BLS/FRED/NBER parameters |

## License

MIT
