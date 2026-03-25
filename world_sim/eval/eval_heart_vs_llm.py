#!/usr/bin/env python3
"""Heart vs Pure-LLM comparison.

Runs the Small Town scenario, samples 8 agents at 10 key moments.
At each moment, generates paired outputs:
  - Heart system: state → deterministic action, then LLM generates speech from state
  - Pure LLM: full event history in prompt → LLM decides action + speech

Saves results for judging.

Usage:
    python -m world_sim.eval.eval_heart_vs_llm
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict

from openai import OpenAI

from ..scenarios import build_small_town, LAYOFF_TARGETS
from ..world import World
from ..world_agent import WorldAgent
from ..action_table import Action, get_action_description

# 8 agents spanning: laid-off, not-laid-off, management, community
EVAL_AGENTS = ["marcus", "rosa", "jake", "diana", "sarah", "richard", "tom", "greg"]

# 10 key moments across the 15-day arc
OBSERVATION_TICKS = [
    72,   # Day 3, noon — baseline, nothing happened
    82,   # Day 4, 10am — layoff rumors start
    105,  # Day 5, 9am — official announcement
    106,  # Day 5, 10am — individual terminations
    113,  # Day 5, 5pm — laid-off workers at bar
    128,  # Day 6, 8am — morning after
    158,  # Day 7, 14pm — community rally
    187,  # Day 8, 19pm — Greg confronts management at bar
    230,  # Day 10, 14pm — Rosa gets job lead
    286,  # Day 12, 22pm — Marcus breakdown
]


@dataclass
class ComparisonSample:
    tick: int
    time_str: str
    agent_id: str
    agent_name: str
    agent_background: str
    agent_temperament: str
    # Heart system outputs
    heart_action: str
    heart_state: dict
    heart_speech: str
    # Pure LLM outputs
    llm_action: str
    llm_reasoning: str
    llm_speech: str
    # Context for judge
    event_history: list[str]
    nearby_agents: list[str]
    location: str


def _tick_to_time(tick: int) -> str:
    day = tick // 24 + 1
    hour = tick % 24
    return f"Day {day}, {hour:02d}:00"


def _build_event_history(world: World, agent_id: str, up_to_tick: int) -> list[str]:
    """Reconstruct chronological event history visible to this agent."""
    history = []
    agent = world.agents[agent_id]

    for entry in agent.memory:
        if entry.tick <= up_to_tick:
            time_str = _tick_to_time(entry.tick)
            val_marker = "+" if entry.valence_at_time > 0.5 else "-" if entry.valence_at_time < 0.4 else " "
            history.append(f"[{time_str}] {val_marker} {entry.description}")

    return history


def _get_nearby_names(world: World, agent_id: str) -> list[str]:
    """Get names of agents at same location."""
    agent = world.agents[agent_id]
    return [
        a.personality.name
        for a in world.agents.values()
        if a.agent_id != agent_id and a.location == agent.location
    ]


def _get_relationship_summaries(world: World, agent_id: str) -> str:
    """Format relationship summaries for LLM prompt."""
    rels = world.relationships.get_agent_relationships(agent_id)
    if not rels:
        return "No established relationships."

    lines = []
    for other_id, rel in rels[:10]:
        other = world.agents.get(other_id)
        if not other:
            continue
        name = other.personality.name
        trust_desc = "high trust" if rel.trust > 0.5 else "some trust" if rel.trust > 0.1 else "low trust"
        warmth_desc = "warm" if rel.warmth > 0.4 else "neutral" if rel.warmth > -0.1 else "cold"
        res = world.relationships.get_resentment(agent_id, other_id)
        res_desc = f", resentful" if res > 0.2 else ""
        lines.append(f"- {name}: {trust_desc}, {warmth_desc} ({rel.familiarity} interactions){res_desc}")

    return "\n".join(lines)


def generate_heart_speech(client: OpenAI, world: World, agent_id: str) -> str:
    """Generate speech from heart state using gpt-4o-mini."""
    agent = world.agents[agent_id]
    s = agent.heart
    nearby = _get_nearby_names(world, agent_id)
    memories = _build_event_history(world, agent_id, world.tick_count)

    prompt = f"""You are {agent.personality.name}, {agent.personality.background}
Temperament: {agent.personality.temperament}

Current emotional state:
- Internal feeling: {s.internal_emotion} (arousal: {s.arousal:.2f}, valence: {s.valence:.2f})
- What you show outwardly: {s.surface_emotion} (divergence: {s.divergence:.2f})
- Tension: {s.tension:.2f}, Energy: {s.energy:.2f}, Vulnerability: {s.vulnerability:.2f}
- Impulse control: {s.impulse_control:.2f}
- Active emotional wounds: {len(s.wounds)}

Private human read:
{agent.render_subjective_brief()}

Likely futures:
{chr(10).join(f"- {b['label']}: {b['summary']}" for b in agent.get_future_branches())}

You are currently: {get_action_description(Action[agent.last_action], agent)}
Location: {agent.location}
Time: {world.time_str}

Recent events:
{chr(10).join(memories[-8:]) if memories else "Nothing notable."}

People nearby: {', '.join(nearby) if nearby else 'No one'}

Given your emotional state and what you are doing, write 1-2 sentences of what you say or do right now. Stay in character. If your surface emotion differs from your internal emotion, reflect that tension. Do not narrate — write only dialogue or brief action."""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100,
        temperature=0.7,
    )
    return resp.choices[0].message.content.strip()


def generate_llm_response(
    client: OpenAI, world: World, agent_id: str
) -> tuple[str, str, str]:
    """Pure LLM: full history → action + speech."""
    agent = world.agents[agent_id]
    history = _build_event_history(world, agent_id, world.tick_count)
    nearby = _get_nearby_names(world, agent_id)
    rels = _get_relationship_summaries(world, agent_id)

    prompt = f"""You are {agent.personality.name}, {agent.personality.background}
Temperament: {agent.personality.temperament}

Here is everything that has happened to you so far:
{chr(10).join(history) if history else "Nothing notable yet — normal routine."}

Your relationships:
{rels}

Current situation:
- Location: {agent.location}
- Time: {world.time_str}
- People nearby: {', '.join(nearby) if nearby else 'No one'}

What do you do right now? Choose one action from this list:
COLLAPSE, LASH_OUT, CONFRONT, FLEE, WITHDRAW, SEEK_COMFORT, RUMINATE, VENT,
SOCIALIZE, CELEBRATE, HELP_OTHERS, WORK, REST, IDLE

Respond in this exact format:
ACTION: [your chosen action]
REASONING: [1 sentence why]
SPEECH: [1-2 sentences of what you say or do, in character. Dialogue or brief action only.]"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
        temperature=0.7,
    )
    text = resp.choices[0].message.content.strip()

    # Parse response
    action = "WORK"
    reasoning = ""
    speech = ""
    for line in text.split("\n"):
        line = line.strip()
        if line.upper().startswith("ACTION:"):
            action = line.split(":", 1)[1].strip().upper()
        elif line.upper().startswith("REASONING:"):
            reasoning = line.split(":", 1)[1].strip()
        elif line.upper().startswith("SPEECH:"):
            speech = line.split(":", 1)[1].strip()

    return action, reasoning, speech


def run_evaluation(output_path: str = "eval_samples.json"):
    """Run the full evaluation."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set")
        return

    client = OpenAI(api_key=api_key)

    print("Building and running simulation...")
    world = build_small_town()
    world.initialize()

    max_tick = max(OBSERVATION_TICKS)
    samples: list[dict] = []
    total_calls = len(OBSERVATION_TICKS) * len(EVAL_AGENTS) * 2
    call_count = 0

    for tick in range(1, max_tick + 1):
        world.tick()

        if tick not in OBSERVATION_TICKS:
            continue

        print(f"\n{'─' * 60}")
        print(f"  Observation: {world.time_str} (tick {tick})")
        print(f"{'─' * 60}")

        for agent_id in EVAL_AGENTS:
            agent = world.agents[agent_id]
            s = agent.heart
            nearby = _get_nearby_names(world, agent_id)
            history = _build_event_history(world, agent_id, tick)

            # Heart system path
            call_count += 1
            print(f"  [{call_count}/{total_calls}] Heart speech for {agent.personality.name}...", end=" ", flush=True)
            heart_speech = generate_heart_speech(client, world, agent_id)
            print(f"done")

            # Pure LLM path
            call_count += 1
            print(f"  [{call_count}/{total_calls}] LLM response for {agent.personality.name}...", end=" ", flush=True)
            llm_action, llm_reasoning, llm_speech = generate_llm_response(client, world, agent_id)
            print(f"done ({llm_action})")

            sample = ComparisonSample(
                tick=tick,
                time_str=world.time_str,
                agent_id=agent_id,
                agent_name=agent.personality.name,
                agent_background=agent.personality.background,
                agent_temperament=agent.personality.temperament,
                heart_action=agent.last_action,
                heart_state={
                    "arousal": round(s.arousal, 3),
                    "valence": round(s.valence, 3),
                    "tension": round(s.tension, 3),
                    "impulse_control": round(s.impulse_control, 3),
                    "energy": round(s.energy, 3),
                    "vulnerability": round(s.vulnerability, 3),
                    "internal_emotion": s.internal_emotion,
                    "surface_emotion": s.surface_emotion,
                    "wounds": len(s.wounds),
                },
                heart_speech=heart_speech,
                llm_action=llm_action,
                llm_reasoning=llm_reasoning,
                llm_speech=llm_speech,
                event_history=history,
                nearby_agents=nearby,
                location=agent.location,
            )
            samples.append(asdict(sample))

    # Save
    with open(output_path, "w") as f:
        json.dump(samples, f, indent=2)

    print(f"\n{'═' * 60}")
    print(f"  Saved {len(samples)} comparison samples to {output_path}")
    print(f"  Total API calls: {call_count}")
    print(f"{'═' * 60}")


if __name__ == "__main__":
    run_evaluation()
