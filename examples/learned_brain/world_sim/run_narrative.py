#!/usr/bin/env python3
"""Run the world simulation with full LLM-generated narration.

Every interaction, event reaction, and solo emotional moment gets
dialogue/narration from gpt-4o-mini. Outputs a rich Markdown narrative.

Usage:
    python -m learned_brain.world_sim.run_narrative [--days 15] [--output narrative.md]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass

import openai

from .scenarios import build_small_town
from .world import World
from .world_agent import WorldAgent
from .action_table import Action, get_action_description


# ─── Narrator LLM ────────────────────────────────────────────────────────────

class NarratorLLM:
    """Async LLM client with concurrency control and cost tracking."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini", max_concurrent: int = 20):
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.model = model
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.total_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    async def generate(self, system: str, user: str, max_tokens: int = 200) -> str:
        async with self.semaphore:
            try:
                resp = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    max_tokens=max_tokens,
                    temperature=0.8,
                )
                self.total_calls += 1
                if resp.usage:
                    self.total_input_tokens += resp.usage.prompt_tokens
                    self.total_output_tokens += resp.usage.completion_tokens
                return resp.choices[0].message.content.strip()
            except Exception as e:
                self.total_calls += 1
                return f"[narration unavailable: {e}]"

    @property
    def estimated_cost(self) -> float:
        return (self.total_input_tokens * 0.15 / 1e6 +
                self.total_output_tokens * 0.60 / 1e6)


# ─── Prompt Builders ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are the narrative voice of a small-town drama simulation. Write vivid, concise prose. Use dialogue in quotes. Show don't tell. Match the emotional intensity to the character's state — don't be melodramatic when they're calm, don't be flat when they're devastated. Keep it grounded and human."""


def _agent_card(agent: WorldAgent) -> str:
    s = agent.heart
    wounds_str = f", carrying {len(s.wounds)} emotional wound(s)" if s.wounds else ""
    return (
        f"{agent.personality.name}: {agent.personality.background}\n"
        f"Temperament: {agent.personality.temperament}\n"
        f"Feeling: {s.internal_emotion} (showing: {s.surface_emotion}, divergence: {s.divergence:.1f})\n"
        f"Arousal: {s.arousal:.2f} | Valence: {s.valence:.2f} | Tension: {s.tension:.2f} | "
        f"Energy: {s.energy:.2f} | Vulnerability: {s.vulnerability:.2f} | "
        f"Impulse control: {s.impulse_control:.2f}{wounds_str}"
    )


def _recent_memories(agent: WorldAgent, n: int = 5) -> str:
    mems = agent.get_recent_memories(n)
    if not mems:
        return "No recent events."
    lines = []
    for m in mems:
        day = m.tick // 24 + 1
        hour = m.tick % 24
        lines.append(f"[Day {day}, {hour:02d}:00] {m.description}")
    return "\n".join(lines)


def _rel_summary(world: World, aid_a: str, aid_b: str) -> str:
    rel = world.relationships.get(aid_a, aid_b)
    if not rel:
        return "No prior relationship."
    res_ab = world.relationships.get_resentment(aid_a, aid_b)
    res_ba = world.relationships.get_resentment(aid_b, aid_a)
    parts = [f"trust: {rel.trust:+.2f}", f"warmth: {rel.warmth:+.2f}",
             f"{rel.familiarity} prior interactions"]
    if res_ab > 0.1:
        parts.append(f"resentment→: {res_ab:.2f}")
    if res_ba > 0.1:
        parts.append(f"resentment←: {res_ba:.2f}")
    return ", ".join(parts)


def build_interaction_prompt(
    world: World, aid_a: str, aid_b: str, interaction_type: str
) -> tuple[str, str]:
    agent_a = world.agents[aid_a]
    agent_b = world.agents[aid_b]
    action_a = agent_a.last_action
    action_b = agent_b.last_action

    user = f"""Two characters interact at {agent_a.location}, {world.time_str}.
Interaction type: {interaction_type}

CHARACTER A:
{_agent_card(agent_a)}
Action: {action_a}
Recent events:
{_recent_memories(agent_a)}
Last said: "{agent_a.last_speech or '(nothing recently)'}"

CHARACTER B:
{_agent_card(agent_b)}
Action: {action_b}
Recent events:
{_recent_memories(agent_b)}
Last said: "{agent_b.last_speech or '(nothing recently)'}"

Their relationship: {_rel_summary(world, aid_a, aid_b)}

Write 3-6 lines of dialogue between them, then a one-sentence narrator observation about the subtext or body language. Format:
{agent_a.personality.name}: "dialogue"
{agent_b.personality.name}: "dialogue"
[Narrator: observation]"""

    return SYSTEM_PROMPT, user


def build_event_reaction_prompt(
    world: World, agent_id: str, event_desc: str, is_targeted: bool
) -> tuple[str, str]:
    agent = world.agents[agent_id]
    targeted_str = "This event is directed specifically at you." if is_targeted else "You witness this event happening around you."

    user = f"""{world.time_str} at {agent.location}.

EVENT: {event_desc}
{targeted_str}

CHARACTER:
{_agent_card(agent)}
Recent events:
{_recent_memories(agent)}

Write 2-3 sentences: the character's immediate reaction — internal thoughts, body language, or spoken words. If their surface emotion differs from internal, show the gap."""

    return SYSTEM_PROMPT, user


def build_solo_moment_prompt(world: World, agent_id: str) -> tuple[str, str]:
    agent = world.agents[agent_id]
    action_desc = get_action_description(Action[agent.last_action], agent)

    user = f"""{world.time_str} at {agent.location}.

CHARACTER:
{_agent_card(agent)}
Currently: {action_desc}
Recent events:
{_recent_memories(agent)}

Write 1-2 sentences of narration — what the character is doing, thinking, or feeling in this moment. Be specific and grounded."""

    return SYSTEM_PROMPT, user


def build_daily_summary_prompt(world: World, day_events: list[str]) -> tuple[str, str]:
    summary = world.get_world_summary()
    most_distressed = summary["most_distressed"][:5]

    distressed_lines = []
    for d in most_distressed:
        distressed_lines.append(
            f"  {d['name']}: {d['internal']}/{d['surface']}, "
            f"valence={d['valence']:.2f}, vulnerability={d['vulnerability']:.2f}"
        )

    user = f"""End of Day {world.day}. Write a brief (3-4 sentence) omniscient narrator summary of the day.

Key stats: {summary['agent_count']} people, avg energy {summary['avg_energy']}, avg mood {summary['avg_valence']}

Notable events today:
{chr(10).join(day_events) if day_events else 'Routine day, nothing major.'}

Most affected people:
{chr(10).join(distressed_lines)}

Action breakdown: {dict(summary['action_counts'])}

Capture the emotional temperature of the town. What's the undercurrent?"""

    return SYSTEM_PROMPT, user


# ─── Narrative Runner ─────────────────────────────────────────────────────────

SOLO_NARRATE_ACTIONS = {"COLLAPSE", "LASH_OUT", "FLEE", "WITHDRAW", "RUMINATE",
                        "SEEK_COMFORT", "CONFRONT"}


async def run_narrative(
    days: int = 15,
    output_path: str = "narrative.md",
    budget: float = 10.0,
):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set")
        return

    narrator = NarratorLLM(api_key=api_key)

    print(f"\n{'═' * 80}")
    print(f"  NARRATIVE SIMULATION: Small Town, Bad News")
    print(f"  {days} days | 30 agents | LLM narration via gpt-4o-mini")
    print(f"{'═' * 80}")

    world = build_small_town()
    world.initialize()

    total_ticks = days * 24
    md_lines: list[str] = []
    md_lines.append("# Small Town, Bad News\n")
    md_lines.append("*A 15-day simulation of 30 lives in a small company town.*\n")
    md_lines.append("---\n")

    day_events: list[str] = []
    current_day = 0
    t0 = time.time()

    for tick in range(1, total_ticks + 1):
        summary = world.tick()
        hour = world.hour_of_day
        day = world.day

        # New day header
        if day != current_day:
            # Daily summary for previous day
            if current_day > 0 and day_events:
                sys_p, usr_p = build_daily_summary_prompt(world, day_events)
                day_summary = await narrator.generate(sys_p, usr_p, max_tokens=250)
                md_lines.append(f"\n> *{day_summary}*\n")
                md_lines.append("---\n")

            current_day = day
            day_events = []
            md_lines.append(f"\n## Day {day}\n")
            print(f"\n  Day {day} | LLM calls so far: {narrator.total_calls} | "
                  f"Cost: ${narrator.estimated_cost:.3f}")

        # Skip sleep ticks (nothing happens)
        if hour >= 22 or hour < 6:
            continue

        events = summary.get("events", [])
        interactions = summary.get("interactions", [])

        # Collect agents with solo emotional actions (non-interacting)
        interacting_ids = set()
        for ix in interactions:
            interacting_ids.add(ix["agent_a"])
            interacting_ids.add(ix["agent_b"])

        solo_agents = []
        for agent in world.agents.values():
            if (agent.last_action in SOLO_NARRATE_ACTIONS and
                    agent.agent_id not in interacting_ids and
                    agent.heart.vulnerability > 0.3):
                solo_agents.append(agent.agent_id)

        # Skip ticks with nothing to narrate
        if not events and not interactions and not solo_agents:
            continue

        # Tick header
        md_lines.append(f"\n### {world.time_str}\n")

        # Budget check
        if narrator.estimated_cost > budget:
            md_lines.append(f"\n*[Budget limit ${budget:.2f} reached. Stopping narration.]*\n")
            print(f"  Budget limit reached at tick {tick}")
            break

        tasks = []

        # Event reactions
        for event in events:
            desc = event["description"]
            targets = event.get("targets")
            day_events.append(f"⚡ {desc[:80]}")
            md_lines.append(f"\n> **EVENT at {event['location']}:** {desc}\n")

            # Generate reactions for affected agents
            affected = []
            for agent in world.agents.values():
                if agent.location != event["location"]:
                    continue
                if targets and agent.agent_id not in targets:
                    continue
                is_targeted = targets is not None and agent.agent_id in targets
                affected.append((agent.agent_id, is_targeted))

            # Limit to 8 reactions per event to control costs
            for aid, is_targeted in affected[:8]:
                sys_p, usr_p = build_event_reaction_prompt(world, aid, desc, is_targeted)
                tasks.append(("event", aid, narrator.generate(sys_p, usr_p, max_tokens=150)))

        # Interactions
        for ix in interactions:
            sys_p, usr_p = build_interaction_prompt(
                world, ix["agent_a"], ix["agent_b"], ix["type"]
            )
            tasks.append(("interaction", (ix["agent_a"], ix["agent_b"], ix["type"]),
                          narrator.generate(sys_p, usr_p, max_tokens=300)))

        # Solo emotional moments
        for aid in solo_agents[:5]:  # limit to 5 per tick
            sys_p, usr_p = build_solo_moment_prompt(world, aid)
            tasks.append(("solo", aid, narrator.generate(sys_p, usr_p, max_tokens=100)))

        # Fire all LLM calls concurrently
        if tasks:
            results = await asyncio.gather(*[t[2] for t in tasks])

            for (task_type, context, _), result in zip(tasks, results):
                if task_type == "event":
                    agent = world.agents[context]
                    s = agent.heart
                    md_lines.append(
                        f"\n**{agent.personality.name}** "
                        f"*({s.internal_emotion}, valence: {s.valence:.2f}, "
                        f"vulnerability: {s.vulnerability:.2f})*:\n"
                    )
                    md_lines.append(f"{result}\n")
                    agent.last_speech = result[:100]

                elif task_type == "interaction":
                    aid_a, aid_b, ix_type = context
                    a = world.agents[aid_a]
                    b = world.agents[aid_b]
                    md_lines.append(
                        f"\n**{a.personality.name} × {b.personality.name}** "
                        f"({ix_type} at {a.location}):\n"
                    )
                    md_lines.append(f"{result}\n")
                    # Store last speech for both
                    for line in result.split("\n"):
                        line = line.strip()
                        if line.startswith(f"{a.personality.name}:"):
                            a.last_speech = line.split(":", 1)[1].strip().strip('"')
                        elif line.startswith(f"{b.personality.name}:"):
                            b.last_speech = line.split(":", 1)[1].strip().strip('"')

                elif task_type == "solo":
                    agent = world.agents[context]
                    s = agent.heart
                    md_lines.append(
                        f"\n**{agent.personality.name}** "
                        f"*({s.internal_emotion}, energy: {s.energy:.2f}, "
                        f"vulnerability: {s.vulnerability:.2f})*:\n"
                    )
                    md_lines.append(f"*{result}*\n")

        # World state snapshot at key hours
        if hour in (12, 20) and (events or interactions):
            snapshot = world.get_world_summary()
            md_lines.append(f"\n---\n*Town pulse: avg mood {snapshot['avg_valence']}, "
                            f"avg energy {snapshot['avg_energy']}*\n")

    # Final daily summary
    if day_events:
        sys_p, usr_p = build_daily_summary_prompt(world, day_events)
        day_summary = await narrator.generate(sys_p, usr_p, max_tokens=250)
        md_lines.append(f"\n> *{day_summary}*\n")

    # Epilogue — final state of key characters
    md_lines.append("\n---\n## Epilogue: Where They Are Now\n")
    for agent_id in ["marcus", "rosa", "greg", "diana", "richard", "tom", "sarah", "jake"]:
        agent = world.agents[agent_id]
        s = agent.heart
        sys_p = SYSTEM_PROMPT
        usr_p = (
            f"Write a 2-sentence epilogue for this character after 15 days of turmoil.\n\n"
            f"{_agent_card(agent)}\n"
            f"Recent events:\n{_recent_memories(agent)}\n\n"
            f"Where are they now, emotionally and practically?"
        )
        epilogue = await narrator.generate(sys_p, usr_p, max_tokens=100)
        md_lines.append(f"\n**{agent.personality.name}**: {epilogue}\n")

    elapsed = time.time() - t0

    # Write output
    with open(output_path, "w") as f:
        f.write("\n".join(md_lines))

    print(f"\n{'═' * 80}")
    print(f"  NARRATIVE COMPLETE")
    print(f"  {narrator.total_calls} LLM calls in {elapsed:.1f}s")
    print(f"  Input tokens: {narrator.total_input_tokens:,}")
    print(f"  Output tokens: {narrator.total_output_tokens:,}")
    print(f"  Estimated cost: ${narrator.estimated_cost:.3f}")
    print(f"  Output: {output_path} ({len(md_lines)} lines)")
    print(f"{'═' * 80}")


def main():
    parser = argparse.ArgumentParser(description="Narrative World Simulation")
    parser.add_argument("--days", type=int, default=15)
    parser.add_argument("--output", type=str, default="narrative.md")
    parser.add_argument("--budget", type=float, default=10.0)
    args = parser.parse_args()

    asyncio.run(run_narrative(days=args.days, output_path=args.output, budget=args.budget))


if __name__ == "__main__":
    main()
