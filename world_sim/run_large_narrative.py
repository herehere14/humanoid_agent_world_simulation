#!/usr/bin/env python3
"""Run the large-scale 300-agent simulation with LLM narration.

Features beyond run_narrative.py:
  - 300 agents across 8 districts
  - Simultaneous multi-district event narration
  - Ripple mechanic: agent actions create new events
  - Selective narration: top interactions ranked by drama
  - District pulse tracking
  - Cross-district encounter highlights

Usage:
    python -m world_sim.run_large_narrative [--days 10] [--output city_narrative.md]
"""

from __future__ import annotations

import argparse
import asyncio
import os
import time
from collections import defaultdict

import openai

from .dynamic_events import DISTRICT_MAP, DynamicEventEngine, compute_district_stats
from .scenarios_large import build_large_town
from .world import World
from .world_agent import WorldAgent
from .action_table import Action, get_action_description


# ─── Narrator LLM ────────────────────────────────────────────────────────────

class NarratorLLM:
    """Async LLM client with concurrency control and cost tracking."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini", max_concurrent: int = 25):
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

SYSTEM_PROMPT = """You are the narrative voice of a city-wide crisis simulation with 300 characters across 8 districts. Write vivid, concise prose. Use dialogue in quotes. Show don't tell. Match emotional intensity to character state. Keep it grounded and human. Reference the specific district/location."""


def _agent_card(agent: WorldAgent) -> str:
    s = agent.heart
    wounds = f", carrying {len(s.wounds)} emotional wound(s)" if s.wounds else ""
    futures = "\n".join(
        f"- {branch['label']}: {branch['summary']}"
        for branch in agent.get_future_branches()
    )
    return (
        f"{agent.personality.name}: {agent.personality.background}\n"
        f"Temperament: {agent.personality.temperament}\n"
        f"Feeling: {s.internal_emotion} (showing: {s.surface_emotion}, "
        f"divergence: {s.divergence:.1f})\n"
        f"Arousal: {s.arousal:.2f} | Valence: {s.valence:.2f} | "
        f"Tension: {s.tension:.2f} | Energy: {s.energy:.2f} | "
        f"Vulnerability: {s.vulnerability:.2f}{wounds}\n"
        f"{agent.render_subjective_brief()}\n"
        f"Near futures:\n{futures}"
    )


def _recent_memories(agent: WorldAgent, n: int = 5) -> str:
    mems = agent.get_recent_memories(n)
    if not mems:
        return "No recent events."
    lines = []
    for m in mems:
        day = m.tick // 24 + 1
        hour = m.tick % 24
        lines.append(f"[Day {day}, {hour:02d}:00] {m.description[:80]}")
    return "\n".join(lines)


def _rel_summary(world: World, aid_a: str, aid_b: str) -> str:
    rel = world.relationships.get(aid_a, aid_b)
    if not rel:
        return "Strangers."
    parts = [f"trust: {rel.trust:+.2f}", f"warmth: {rel.warmth:+.2f}",
             f"{rel.familiarity} prior interactions"]
    res = world.relationships.get_resentment(aid_a, aid_b)
    if res > 0.1:
        parts.append(f"resentment: {res:.2f}")
    return ", ".join(parts)


def build_event_reaction_prompt(
    world: World, agent_id: str, event_desc: str, is_targeted: bool,
    district: str,
) -> tuple[str, str]:
    agent = world.agents[agent_id]
    targeted_str = ("This event is directed specifically at you."
                    if is_targeted else "You witness this event.")
    user = f"""{world.time_str} at {agent.location} ({district}).

EVENT: {event_desc}
{targeted_str}

CHARACTER:
{_agent_card(agent)}
Recent events:
{_recent_memories(agent)}

Write 2-3 sentences: immediate reaction — internal thoughts, body language, or spoken words. If surface emotion differs from internal, show the gap."""
    return SYSTEM_PROMPT, user


def build_interaction_prompt(
    world: World, aid_a: str, aid_b: str, interaction_type: str,
    district: str,
) -> tuple[str, str]:
    a = world.agents[aid_a]
    b = world.agents[aid_b]
    user = f"""Two characters interact at {a.location} ({district}), {world.time_str}.
Interaction type: {interaction_type}

CHARACTER A:
{_agent_card(a)}
Action: {a.last_action}
Recent: {_recent_memories(a, 3)}

CHARACTER B:
{_agent_card(b)}
Action: {b.last_action}
Recent: {_recent_memories(b, 3)}

Relationship: {_rel_summary(world, aid_a, aid_b)}

Write 3-5 lines of dialogue, then a one-sentence narrator observation.
{a.personality.name}: "dialogue"
{b.personality.name}: "dialogue"
[Narrator: observation]"""
    return SYSTEM_PROMPT, user


def build_solo_moment_prompt(world: World, agent_id: str, district: str) -> tuple[str, str]:
    agent = world.agents[agent_id]
    action_desc = get_action_description(Action[agent.last_action], agent)
    user = f"""{world.time_str} at {agent.location} ({district}).

CHARACTER:
{_agent_card(agent)}
Currently: {action_desc}
Recent: {_recent_memories(agent)}

Write 1-2 sentences of narration — what the character is doing, thinking, or feeling."""
    return SYSTEM_PROMPT, user


def build_crowd_scene_prompt(
    world: World, location: str, agents_at_loc: list[WorldAgent],
    event_desc: str | None, district: str,
) -> tuple[str, str]:
    """Narrate a crowd scene with many agents at one location."""
    n = len(agents_at_loc)
    avg_valence = sum(a.heart.valence for a in agents_at_loc) / n
    avg_vulnerability = sum(a.heart.vulnerability for a in agents_at_loc) / n

    # Pick 5 representative characters to name
    sorted_agents = sorted(agents_at_loc, key=lambda a: a.heart.vulnerability, reverse=True)
    featured = sorted_agents[:5]
    featured_lines = []
    for a in featured:
        s = a.heart
        featured_lines.append(
            f"- {a.personality.name} ({a.personality.background[:50]}...): "
            f"{s.internal_emotion}, valence={s.valence:.2f}, vulnerability={s.vulnerability:.2f}"
        )

    event_line = f"\nEVENT: {event_desc}" if event_desc else ""

    user = f"""{world.time_str} at {location} ({district}).
{n} people are gathered here. Average mood: {avg_valence:.2f}. Average vulnerability: {avg_vulnerability:.2f}.{event_line}

Featured characters:
{chr(10).join(featured_lines)}

Write 4-6 sentences capturing the crowd's energy — the mix of emotions, snippets of overheard dialogue, body language patterns. Name at least 3 of the featured characters. Show how the group dynamic differs from individual reactions."""
    return SYSTEM_PROMPT, user


def build_district_pulse_prompt(
    world: World, district_stats: dict[str, dict],
) -> tuple[str, str]:
    """Generate a cross-district mood summary."""
    lines = []
    for district, stats in sorted(district_stats.items()):
        lines.append(
            f"- {district}: {stats['count']} people, "
            f"avg mood {stats['avg_valence']:.2f}, "
            f"avg energy {stats['avg_energy']:.2f}, "
            f"most common emotion: {stats['top_emotion']}"
        )

    user = f"""{world.time_str} — City-wide pulse check.

District status:
{chr(10).join(lines)}

Write 2-3 sentences of omniscient narrator summary — the emotional temperature across the city. What's the undercurrent? Which districts are in crisis, which are calm, and how are they connected?"""
    return SYSTEM_PROMPT, user


def build_ripple_prompt(
    world: World, source_loc: str, source_district: str,
    ripple_desc: str, affected_district: str,
) -> tuple[str, str]:
    """Narrate how an event in one district ripples to another."""
    user = f"""{world.time_str}

Something happened at {source_loc} ({source_district}): {ripple_desc}

This news is now reaching {affected_district}. Write 2-3 sentences showing how people in {affected_district} react to hearing this — the telephone game of rumor, the growing worry, the way information distorts as it travels across town."""
    return SYSTEM_PROMPT, user


def build_daily_summary_prompt(world: World, day_events: list[str],
                                district_stats: dict[str, dict]) -> tuple[str, str]:
    lines = []
    for d, stats in sorted(district_stats.items()):
        lines.append(f"  {d}: mood {stats['avg_valence']:.2f}, energy {stats['avg_energy']:.2f}")

    user = f"""End of Day {world.day}. Write 3-4 sentences of omniscient narrator summary.

Events today:
{chr(10).join(day_events[:15]) if day_events else 'Quiet day.'}

District moods:
{chr(10).join(lines)}

Capture the day's arc. What shifted? What's building?"""
    return SYSTEM_PROMPT, user


# ─── Dynamic event hooks ──────────────────────────────────────────────────────

SOLO_NARRATE_ACTIONS = {"COLLAPSE", "LASH_OUT", "FLEE", "WITHDRAW", "RUMINATE",
                        "SEEK_COMFORT", "CONFRONT"}


# ─── Main narrative runner ────────────────────────────────────────────────────

async def run_large_narrative(
    days: int = 10,
    output_path: str = "city_narrative.md",
    budget: float = 15.0,
    max_interactions_per_tick: int = 12,
    max_event_reactions: int = 8,
    max_solo_per_tick: int = 5,
):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set")
        return

    narrator = NarratorLLM(api_key=api_key)

    print(f"\n{'═' * 80}")
    print(f"  CITY CRISIS SIMULATION: 300 Agents, 8 Districts")
    print(f"  {days} days | gpt-4o-mini narration | budget ${budget:.0f}")
    print(f"{'═' * 80}")

    world, agent_meta = build_large_town()
    world.initialize()
    event_engine = DynamicEventEngine()

    total_ticks = days * 24
    md: list[str] = []
    md.append("# Crossroads: A City in Crisis\n")
    md.append("*300 lives across 8 districts. When the factory explodes, everyone feels it.*\n")
    md.append("---\n")

    day_events: list[str] = []
    current_day = 0
    t0 = time.time()

    for tick in range(1, total_ticks + 1):
        summary = world.tick()
        hour = world.hour_of_day
        day = world.day

        # New day header
        if day != current_day:
            if current_day > 0 and day_events:
                district_stats = compute_district_stats(world)
                sys_p, usr_p = build_daily_summary_prompt(world, day_events, district_stats)
                day_summary = await narrator.generate(sys_p, usr_p, max_tokens=250)
                md.append(f"\n> *{day_summary}*\n")
                md.append("---\n")

            current_day = day
            day_events = []
            md.append(f"\n## Day {day}\n")
            print(f"\n  Day {day} | LLM calls: {narrator.total_calls} | "
                  f"Cost: ${narrator.estimated_cost:.3f}")

        # Skip sleep ticks
        if hour >= 22 or hour < 6:
            continue

        events = summary.get("events", [])
        interactions = summary.get("interactions", [])

        # Detect ripple events and inject for next tick
        ripples = event_engine.generate(world, summary, agent_meta)
        for ripple in ripples:
            world.schedule_event(ripple)

        # Collect agents with solo emotional actions
        interacting_ids = set()
        for ix in interactions:
            interacting_ids.add(ix["agent_a"])
            interacting_ids.add(ix["agent_b"])

        solo_agents = []
        for agent in world.agents.values():
            if (agent.last_action in SOLO_NARRATE_ACTIONS and
                    agent.agent_id not in interacting_ids and
                    agent.heart.vulnerability > 0.4):
                solo_agents.append(agent)
        # Rank by drama (vulnerability + 1-valence)
        solo_agents.sort(key=lambda a: a.heart.vulnerability + (1 - a.heart.valence), reverse=True)

        # Detect crowd scenes (20+ agents at one location)
        by_location: dict[str, list[WorldAgent]] = defaultdict(list)
        for agent in world.agents.values():
            by_location[agent.location].append(agent)
        crowd_locations = [(loc, agents) for loc, agents in by_location.items()
                           if len(agents) >= 20]

        # Skip ticks with nothing to narrate
        if not events and not interactions and not solo_agents and not crowd_locations:
            continue

        # Tick header
        md.append(f"\n### {world.time_str}\n")

        # Budget check
        if narrator.estimated_cost > budget:
            md.append(f"\n*[Budget limit ${budget:.2f} reached.]*\n")
            print(f"  Budget limit reached at tick {tick}")
            break

        tasks = []

        # ── Event reactions ──────────────────────────────────────────
        for event in events:
            desc = event["description"]
            targets = event.get("targets")
            loc = event["location"]
            district = DISTRICT_MAP.get(loc, "Unknown")
            day_events.append(f"[{district}] {desc[:80]}")
            md.append(f"\n> **{district} — {loc}:** {desc}\n")

            # Reactions from affected agents
            affected = []
            for agent in world.agents.values():
                if agent.location != loc:
                    continue
                if targets and agent.agent_id not in targets:
                    continue
                is_targeted = targets is not None and agent.agent_id in targets
                affected.append((agent, is_targeted))

            # Sort by vulnerability, take top N
            affected.sort(key=lambda x: x[0].heart.vulnerability, reverse=True)
            for agent, is_targeted in affected[:max_event_reactions]:
                sys_p, usr_p = build_event_reaction_prompt(
                    world, agent.agent_id, desc, is_targeted, district)
                tasks.append(("event", agent.agent_id,
                              narrator.generate(sys_p, usr_p, max_tokens=150)))

        # ── Crowd scenes ─────────────────────────────────────────────
        for loc, agents_at_loc in crowd_locations:
            district = DISTRICT_MAP.get(loc, "Unknown")
            # Find if there was an event at this location
            event_here = None
            for event in events:
                if event["location"] == loc:
                    event_here = event["description"]
                    break
            sys_p, usr_p = build_crowd_scene_prompt(
                world, loc, agents_at_loc, event_here, district)
            tasks.append(("crowd", loc,
                          narrator.generate(sys_p, usr_p, max_tokens=300)))

        # ── Interactions (top N by drama) ─────────────────────────────
        # Rank interactions by combined emotional intensity
        ranked_interactions = []
        for ix in interactions:
            a = world.agents[ix["agent_a"]]
            b = world.agents[ix["agent_b"]]
            drama_score = (a.heart.vulnerability + b.heart.vulnerability +
                           (1 - a.heart.valence) + (1 - b.heart.valence) +
                           a.heart.arousal + b.heart.arousal)
            # Bonus for cross-district encounters
            meta_a = agent_meta.get(ix["agent_a"], {})
            meta_b = agent_meta.get(ix["agent_b"], {})
            if meta_a.get("role") != meta_b.get("role"):
                drama_score += 1.0
            ranked_interactions.append((drama_score, ix))

        ranked_interactions.sort(key=lambda x: x[0], reverse=True)

        for _, ix in ranked_interactions[:max_interactions_per_tick]:
            district = DISTRICT_MAP.get(world.agents[ix["agent_a"]].location, "Unknown")
            sys_p, usr_p = build_interaction_prompt(
                world, ix["agent_a"], ix["agent_b"], ix["type"], district)
            tasks.append(("interaction", (ix["agent_a"], ix["agent_b"], ix["type"]),
                          narrator.generate(sys_p, usr_p, max_tokens=300)))

        # ── Solo emotional moments (top N) ───────────────────────────
        for agent in solo_agents[:max_solo_per_tick]:
            district = DISTRICT_MAP.get(agent.location, "Unknown")
            sys_p, usr_p = build_solo_moment_prompt(world, agent.agent_id, district)
            tasks.append(("solo", agent.agent_id,
                          narrator.generate(sys_p, usr_p, max_tokens=100)))

        # Fire all LLM calls concurrently
        if tasks:
            results = await asyncio.gather(*[t[2] for t in tasks])

            for (task_type, context, _), result in zip(tasks, results):
                if task_type == "event":
                    agent = world.agents[context]
                    s = agent.heart
                    meta = agent_meta.get(context, {})
                    role = meta.get("role", "unknown")
                    md.append(
                        f"\n**{agent.personality.name}** [{role}] "
                        f"*({s.internal_emotion}, valence: {s.valence:.2f}, "
                        f"vulnerability: {s.vulnerability:.2f})*:\n"
                    )
                    md.append(f"{result}\n")
                    agent.last_speech = result[:100]

                elif task_type == "crowd":
                    loc = context
                    district = DISTRICT_MAP.get(loc, "Unknown")
                    n_here = len(by_location.get(loc, []))
                    md.append(
                        f"\n**CROWD SCENE — {district} ({loc}, {n_here} people):**\n"
                    )
                    md.append(f"{result}\n")

                elif task_type == "interaction":
                    aid_a, aid_b, ix_type = context
                    a = world.agents[aid_a]
                    b = world.agents[aid_b]
                    meta_a = agent_meta.get(aid_a, {})
                    meta_b = agent_meta.get(aid_b, {})
                    cross = ""
                    if meta_a.get("role") != meta_b.get("role"):
                        cross = " [CROSS-ENCOUNTER]"
                    district = DISTRICT_MAP.get(a.location, "Unknown")
                    md.append(
                        f"\n**{a.personality.name} ({meta_a.get('role', '?')}) "
                        f"× {b.personality.name} ({meta_b.get('role', '?')})** "
                        f"({ix_type} at {a.location}, {district}){cross}:\n"
                    )
                    md.append(f"{result}\n")

                    # Store last speech
                    for line in result.split("\n"):
                        line = line.strip()
                        if line.startswith(f"{a.personality.name}:"):
                            a.last_speech = line.split(":", 1)[1].strip().strip('"')
                        elif line.startswith(f"{b.personality.name}:"):
                            b.last_speech = line.split(":", 1)[1].strip().strip('"')

                elif task_type == "solo":
                    agent = world.agents[context]
                    s = agent.heart
                    meta = agent_meta.get(context, {})
                    district = DISTRICT_MAP.get(agent.location, "Unknown")
                    md.append(
                        f"\n**{agent.personality.name}** [{meta.get('role', '?')}] "
                        f"*({s.internal_emotion}, energy: {s.energy:.2f}, "
                        f"vulnerability: {s.vulnerability:.2f})* "
                        f"at {district}:\n"
                    )
                    md.append(f"*{result}*\n")

        # District pulse every 6 hours (at 12:00, 18:00)
        if hour in (12, 18) and (events or interactions):
            district_stats = compute_district_stats(world)
            sys_p, usr_p = build_district_pulse_prompt(world, district_stats)
            pulse = await narrator.generate(sys_p, usr_p, max_tokens=200)
            md.append(f"\n---\n")
            # Compact district stats line
            stat_parts = []
            for d, s in sorted(district_stats.items()):
                stat_parts.append(f"{d}: {s['avg_valence']:.2f}")
            md.append(f"*District moods: {' | '.join(stat_parts)}*\n")
            md.append(f"\n*{pulse}*\n")

    # Final daily summary
    if day_events:
        district_stats = compute_district_stats(world)
        sys_p, usr_p = build_daily_summary_prompt(world, day_events, district_stats)
        day_summary = await narrator.generate(sys_p, usr_p, max_tokens=250)
        md.append(f"\n> *{day_summary}*\n")

    # ── Epilogue ──────────────────────────────────────────────────────────
    md.append("\n---\n## Epilogue: The City After\n")

    # Pick 12 representative characters across roles
    epilogue_agents = []
    roles_covered = set()
    # Sort all agents by vulnerability (most affected first)
    all_sorted = sorted(world.agents.values(),
                        key=lambda a: a.heart.vulnerability + len(a.heart.wounds),
                        reverse=True)
    for agent in all_sorted:
        meta = agent_meta.get(agent.agent_id, {})
        role = meta.get("role", "unknown")
        if role not in roles_covered:
            epilogue_agents.append(agent)
            roles_covered.add(role)
        if len(epilogue_agents) >= 10:
            break

    epilogue_tasks = []
    for agent in epilogue_agents:
        meta = agent_meta.get(agent.agent_id, {})
        sys_p = SYSTEM_PROMPT
        usr_p = (
            f"Write a 2-sentence epilogue for this character after {days} days of crisis.\n\n"
            f"Role: {meta.get('role', 'unknown')}\n"
            f"{_agent_card(agent)}\n"
            f"Recent events:\n{_recent_memories(agent)}\n\n"
            f"Where are they now, emotionally and practically?"
        )
        epilogue_tasks.append((agent, narrator.generate(sys_p, usr_p, max_tokens=120)))

    if epilogue_tasks:
        epilogue_results = await asyncio.gather(*[t[1] for t in epilogue_tasks])
        for (agent, _), result in zip(epilogue_tasks, epilogue_results):
            meta = agent_meta.get(agent.agent_id, {})
            md.append(
                f"\n**{agent.personality.name}** [{meta.get('role', '?')}]: {result}\n"
            )

    elapsed = time.time() - t0

    # Write output
    with open(output_path, "w") as f:
        f.write("\n".join(md))

    print(f"\n{'═' * 80}")
    print(f"  CITY NARRATIVE COMPLETE")
    print(f"  {narrator.total_calls} LLM calls in {elapsed:.1f}s")
    print(f"  Input tokens: {narrator.total_input_tokens:,}")
    print(f"  Output tokens: {narrator.total_output_tokens:,}")
    print(f"  Estimated cost: ${narrator.estimated_cost:.3f}")
    print(f"  Output: {output_path} ({len(md)} lines)")
    print(f"  Agents: {len(world.agents)} | Relationship pairs: {world.relationships.pair_count}")
    print(f"{'═' * 80}")


def main():
    parser = argparse.ArgumentParser(description="Large-Scale City Crisis Simulation")
    parser.add_argument("--days", type=int, default=10)
    parser.add_argument("--output", type=str, default="city_narrative.md")
    parser.add_argument("--budget", type=float, default=15.0)
    args = parser.parse_args()

    asyncio.run(run_large_narrative(
        days=args.days, output_path=args.output, budget=args.budget))


if __name__ == "__main__":
    main()
