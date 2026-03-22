# 3D World Viewer

Interactive 3D visualization of the agent world simulation. Renders every simulated agent as a human avatar moving through a small town, with visible social behavior, interaction bubbles, thought clouds, and a deep inspector panel.

## Quick Start

```bash
# From the project root:
cd frontend
npm install
npm run dev

# Open in browser:
# http://localhost:3000#/world
```

The viewer loads `public/mock_snapshot.json` by default (pre-generated mock data for 30 agents over 15 simulated days / 360 ticks).

## Using Real Simulation Data

To generate a snapshot from the actual simulation (requires SBERT model):

```bash
# From the project root:
python -m examples.learned_brain.world_sim.export_snapshot --ticks 360
```

This writes `artifacts/world_snapshot.json`. Start the API server to serve it:

```bash
python api_server.py
```

The viewer will try `GET /api/world/snapshot` first, then fall back to the static mock file.

## Architecture

```
world-viewer/
├── WorldViewer.tsx          # Main page — Canvas, data loading, layout
├── store.ts                 # Zustand store — playback state, selection, data
├── types.ts                 # TypeScript types mirroring simulation data model
├── layout.ts                # Town 3D layout, agent positioning, behavior mapping
├── world-viewer.css         # All styles for the viewer UI
└── components/
    ├── TownEnvironment.tsx  # 3D scene — buildings, ground, trees, water, lighting
    ├── AgentLayer.tsx       # Positions all agents, draws interaction lines
    ├── AgentAvatar.tsx      # Individual avatar — body, animations, bubbles
    ├── InspectorPanel.tsx   # Right-side agent detail panel
    ├── PlaybackControls.tsx # Bottom timeline, play/pause, speed
    └── EventOverlay.tsx     # Top event banner + world pulse meters
```

## Controls

| Input | Action |
|-------|--------|
| Click agent | Open inspector panel |
| Click empty space | Close inspector |
| Hover agent | Show name, action, emotion tooltip |
| Space | Play / Pause |
| Arrow Left/Right | Step tick backward/forward |
| Arrow Up/Down | Increase/decrease playback speed |
| Mouse drag | Orbit camera |
| Scroll | Zoom in/out |

## Data Flow

1. Simulation runs → exports `world_snapshot.json` (per-tick state for all agents)
2. Frontend loads snapshot into Zustand store
3. Playback timer advances `currentTick`
4. Each tick: agents are grouped by location, positioned in 3D, rendered with emotion-driven appearance
5. Interactions draw connecting lines + speech bubbles
6. Distressed agents show thought clouds
7. Click opens inspector with full agent state: identity, heart bars, private mind, memories, relationships, future paths

## Agent Visual States

| Action | Visual |
|--------|--------|
| WORK/IDLE | Neutral posture, blue tint |
| CONFRONT/LASH_OUT | Forward lean, arms out, red glow, shake animation |
| COLLAPSE | Crouched, grey, no movement |
| FLEE | Fast movement toward home |
| WITHDRAW | Slight hunch, moves to home |
| RUMINATE | Subtle sway, thought bubble |
| SOCIALIZE/VENT | Arms out, grouped with others |
| HELP_OTHERS | Arms forward, green tint |
| CELEBRATE | Bounce animation, bright green |
| REST | Low posture, dim |

Colors encode emotional state:
- Red = angry (low valence + high arousal)
- Grey = depressed (low valence + low arousal)
- Amber = tense (mid valence + high arousal)
- Blue = neutral
- Green = positive (high valence)
