# Humanoid Agent World Simulation

**A human-response simulation engine where micro behavior compounds into macro reality.**

[中文 README](README.zh-CN.md)

If oil prices double, a pandemic breaks out, a major employer starts layoffs, or a brand is exposed for child labor, what happens next?

This repo is building a world where economically embedded AI people react like humans: they carry emotion, memory, coping style, social pressure, debt, loyalty, rivalry, and private narratives. Their local decisions then propagate through neighborhoods, firms, institutions, and information networks to produce measurable macro outcomes.

This is not a prompt-loop toy where every agent is re-improvised from scratch. It is a **causal human-world simulator** built around:

- `Heart Engine`: persistent private state, appraisal, motives, masking, and memory
- `Ripple Engine`: shock propagation, relationship change, faction dynamics, and macro aggregation
- `LLM Agency`: selective high-salience decision-making for key actors and critical scenes

## Why This Matters

Most multi-agent demos can produce plausible dialogue. Far fewer can answer:

- Why did this specific person react this way?
- What are they hiding?
- Which relationships changed because of it?
- How do thousands of local reactions turn into unemployment stress, consumer confidence collapse, protest waves, or institutional trust shifts?

That is the problem this project is targeting.

The long-term vision is a **scalable human response simulator** for:

- market and commodity shock testing
- policy scenario analysis
- corporate and brand crisis simulation
- game worlds with believable NPC societies
- training, strategy, and synthetic foresight systems

## What Makes This Different

### Normal LLM-only world simulations

Most LLM agent worlds are fundamentally prompt loops:

- expensive at scale
- weak on long-horizon memory
- shallow on causal continuity
- hard to inspect
- characters tend to collapse into the same voice

### Humanoid Agent World Simulation

This system separates **state** from **surface generation**.

- The simulator owns truth: emotion, memory, relationships, pressure, and world causality.
- LLMs are used selectively for active, high-salience agents, not as the whole substrate.
- The result is lower cost, better inspectability, and stronger long-horizon consistency.

## Current Headline Results

| Metric | Current result |
| --- | --- |
| Historical macro validation | **83.6%** overall accuracy |
| Historical direction accuracy | **86.7%** |
| 2008 oil shock benchmark | **12/12 checks passed** |
| COVID first-wave benchmark | **10/12 checks passed** |
| Blind identity test | **90% top-1** on a 5-way character ID task |
| Large harbor simulation | **300 agents**, **50 days**, **1,569 events fired**, **1,542 ripple events added** |
| Social activity in harbor run | **5,952 interactions**, **7,567 final relationship pairs** |

Grounded artifacts in this repo:

- Historical benchmark: [`examples/artifacts/historical_validation.json`](examples/artifacts/historical_validation.json)
- 50-day harbor run: [`artifacts/heatwave_harbor_50d_pass2_fixed_20260322.json`](artifacts/heatwave_harbor_50d_pass2_fixed_20260322.json)
- Blind identity test: [`artifacts/character_identity_blind_test_pass2_fixed_20260322.json`](artifacts/character_identity_blind_test_pass2_fixed_20260322.json)

## System Architecture

### 1. Heart Engine

Each agent maintains persistent internal state instead of being regenerated every turn.

Core state includes:

- emotional dynamics
- human profile and coping signature
- appraisal of events
- contradictory motives
- masking style
- interpreted memory
- ongoing personal story
- relationship pressure
- branchable likely futures

This is the layer that turns “same model, different names” into “different people with recurring distortions.”

### 2. Ripple Engine

The world tracks how local actions affect other people, organizations, and the wider economy.

It handles:

- dynamic event generation
- rumor spread and information diffusion
- coalition and bloc formation
- debt pressure and social obligations
- conflict loops
- mutual aid hubs
- boycott calls
- whistleblower leaks
- accountability hearings
- macro sector aggregation

This is where a private action becomes a public consequence.

### 3. LLM Agency

The repo already includes the hybrid scaffolding for selective LLM cognition:

- salience scoring for `llm_active` agents
- structured decision packet previews
- world snapshot exports
- what-if shock injection
- 3D viewer + inspector surface

The intended pattern is:

- deterministic substrate for most agents
- LLM-driven choice for the few agents who matter most at a given moment

That keeps the system scalable while preserving human richness where it matters.

## Historical Event Validation

The repo contains a historical validation harness that injects real-world shocks into the simulation and compares the resulting macro behavior against documented historical patterns.

Current benchmark set:

- 2008 Oil Price Surge
- 2008 Banking Panic / Lehman Brothers
- COVID-19 first-wave crisis
- Brand scandal / boycott pattern
- Military crisis / Cuban Missile pattern

Current benchmark summary:

- overall accuracy: **83.6%**
- direction accuracy: **86.7%**
- ranking accuracy: **71.4%**
- magnitude checks: **100%**

Per-event scores from the current artifact:

- 2008 Oil Price Surge: **100.0%**
- 2008 Banking Panic: **84.6%**
- COVID-19 Pandemic: **83.3%**
- Brand Scandal: **75.0%**
- Military Crisis: **75.0%**

This means the system is already useful for **scenario simulation and shock propagation testing**, while still needing more calibration before stronger real-world forecasting claims.

## Example What-If Questions

This repo is designed to answer questions like:

- What happens to households, transport, services, and trust if **oil prices surge 100%**?
- What happens to a chocolate company’s customer base if **child labor evidence leaks**?
- How do layoffs, debt pressure, and political response interact during a **banking panic**?
- How does a pandemic propagate through workers, hospitals, students, and neighborhoods?
- What do key leaders do differently when high-stakes events trigger selective LLM agency?

## Repo Capabilities Today

### Simulation

- multi-district social world
- persistent relationship graph
- dynamic event/ripple generation
- macro aggregation across sectors and population mood
- information propagation model
- external shock ingestion

### Interaction and inspection

- backend what-if API
- custom world snapshot generation
- 3D world viewer
- per-agent inspector panel
- event injection bar in the frontend

### Evaluation

- historical validation harness
- blind identity tests
- large-scenario diagnostics
- macro integration tests

## Quick Start

### 1. Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install fastapi uvicorn
cd frontend && npm install && cd ..
```

### 2. Run the backend + frontend

```bash
./start.sh
```

Default targets:

- backend: `http://localhost:8000`
- frontend: `http://localhost:3000`

### 3. Open the world viewer

Open:

- `http://localhost:3000/#/world`

### 4. Inject a shock

Example API call:

```bash
curl -X POST http://127.0.0.1:8000/api/world/what_if \
  -H "Content-Type: application/json" \
  -d '{
    "scenario": "heatwave_harbor",
    "days": 14,
    "information": "oil prices surge 100%",
    "llm_samples": 0
  }'
```

Other useful endpoints:

- `GET /api/world/snapshot`
- `POST /api/world/snapshot/custom`
- `POST /api/world/what_if`

## Key Paths

| Path | Purpose |
| --- | --- |
| `examples/learned_brain/world_sim/` | core world simulation engine |
| `examples/learned_brain/world_sim/world.py` | world loop and event execution |
| `examples/learned_brain/world_sim/world_agent.py` | per-agent internal state |
| `examples/learned_brain/world_sim/world_information.py` | external shock ingestion |
| `examples/learned_brain/world_sim/info_propagation.py` | information spread |
| `examples/learned_brain/world_sim/macro_aggregator.py` | macro metrics |
| `examples/learned_brain/world_sim/eval/historical_validation.py` | historical benchmark harness |
| `api_server.py` | FastAPI backend |
| `frontend/src/world-viewer/` | 3D world viewer and inspector UI |

## Product Direction

The ambition is not just “better NPCs.”

The ambition is a simulation substrate where:

- individually believable humans produce macro economic and social behavior
- world shocks can be injected and traced causally
- decision-makers can inspect both private state and population-level consequence
- gaming, finance, policy, and brand-risk scenarios can run on the same underlying engine

In short:

> **Humanoid Agent World Simulation is building a world where macro outcomes emerge from believable human behavior, not from abstract equations alone and not from prompt-only improvisation.**

## Current Status

This repo is already strong at:

- causal micro-to-macro simulation
- inspectable agent state
- historical pattern matching at the macro level
- low-cost scenario replay

It still needs further work on:

- deeper real-world calibration
- stronger domain-specific forecasting layers
- richer LLM execution for high-salience agents
- broader scale beyond current benchmark worlds

## License

MIT
