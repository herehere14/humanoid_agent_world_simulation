"""Dashboard — terminal display for the world simulation.

Shows world state, most distressed agents, recent events, and
allows drilling into individual agents.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict

from .world import World
from .world_agent import WorldAgent


def _bar(value: float, width: int = 20, label: str = "") -> str:
    """Render a simple bar chart in terminal."""
    filled = int(value * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"{label}{bar} {value:.2f}"


def _emotion_color(valence: float, arousal: float) -> str:
    """Return a text indicator for emotional state."""
    if valence < 0.2:
        return "!!" if arousal > 0.5 else ".."
    elif valence < 0.4:
        return "- " if arousal < 0.3 else "-!"
    elif valence > 0.7:
        return "++" if arousal > 0.4 else "+ "
    return "  "


def print_tick_summary(world: World, verbose: bool = False):
    """Print a one-line summary of the current tick."""
    summary = world.get_world_summary()
    tick_data = world.tick_log[-1] if world.tick_log else {}

    # One-line summary
    print(f"\n{'─' * 80}")
    print(f"  {summary['time']}  |  Agents: {summary['agent_count']}  |  "
          f"Avg energy: {summary['avg_energy']}  |  Avg valence: {summary['avg_valence']}  |  "
          f"Avg arousal: {summary['avg_arousal']}")

    # Events this tick
    events = tick_data.get("events", [])
    if events:
        for e in events:
            targets = f" [→ {', '.join(e['targets'])}]" if e.get("targets") else ""
            print(f"  ⚡ EVENT at {e['location']}: {e['description'][:80]}...{targets}")

    # Interactions this tick
    interactions = tick_data.get("interactions", [])
    if interactions:
        for ix in interactions:
            a_name = world.agents[ix["agent_a"]].personality.name
            b_name = world.agents[ix["agent_b"]].personality.name
            print(f"  ↔ {ix['type'].upper()}: {a_name} × {b_name} at {ix['location']}")

    # Action distribution
    ac = summary["action_counts"]
    non_routine = {k: v for k, v in ac.items() if k not in ("WORK", "REST", "IDLE")}
    if non_routine:
        parts = [f"{k}:{v}" for k, v in sorted(non_routine.items(), key=lambda x: -x[1])]
        print(f"  Actions: {', '.join(parts)}")

    if verbose:
        # Show top 5 most distressed
        print(f"\n  Most distressed:")
        for d in summary["most_distressed"][:5]:
            print(f"    {d['name']:<12s} vuln={d['vulnerability']:.2f} "
                  f"energy={d['energy']:.2f} valence={d['valence']:.2f} "
                  f"arousal={d['arousal']:.2f} [{d['internal']}/{d['surface']}] "
                  f"→ {d['action']}")


def print_agent_detail(world: World, agent_id: str):
    """Print detailed dashboard for one agent."""
    dashboard = world.get_agent_dashboard(agent_id)
    if not dashboard:
        print(f"  Agent '{agent_id}' not found.")
        return

    print(f"\n{'═' * 60}")
    print(f"  {dashboard['name']} ({dashboard['id']})")
    print(f"  Location: {dashboard['location']}  |  Action: {dashboard['action']}")
    print(f"{'═' * 60}")

    # Heart state bars
    print(f"\n  Heart State:")
    print(f"    {_bar(dashboard['arousal'], label='Arousal:    ')}")
    print(f"    {_bar(dashboard['valence'], label='Valence:    ')}")
    print(f"    {_bar(dashboard['tension'], label='Tension:    ')}")
    print(f"    {_bar(dashboard['impulse_control'], label='Impulse:    ')}")
    print(f"    {_bar(dashboard['energy'], label='Energy:     ')}")
    print(f"    {_bar(dashboard['vulnerability'], label='Vulnerable: ')}")

    print(f"\n  Emotion: internal={dashboard['internal']}  surface={dashboard['surface']}  "
          f"divergence={dashboard['divergence']:.2f}")

    # Relationships
    rels = dashboard.get("relationships", [])
    if rels:
        print(f"\n  Relationships ({len(rels)}):")
        for r in rels[:8]:
            res_to = f" ⚠ resent:{r['resentment_toward']:.1f}" if r['resentment_toward'] > 0.1 else ""
            res_from = f" ← resent:{r['resentment_from']:.1f}" if r['resentment_from'] > 0.1 else ""
            print(f"    {r['other_name']:<12s} trust={r['trust']:+.2f} warmth={r['warmth']:+.2f} "
                  f"({r['interactions']} interactions){res_to}{res_from}")

    # Recent memories
    mems = dashboard.get("recent_memories", [])
    if mems:
        print(f"\n  Recent memories:")
        for m in mems[-8:]:
            val_icon = "+" if m["valence"] > 0.5 else "-" if m["valence"] < 0.4 else " "
            print(f"    [{m['time']}] {val_icon} {m['description'][:60]}")

    # Last speech
    if dashboard.get("last_speech"):
        print(f"\n  Last said: \"{dashboard['last_speech']}\"")

    print(f"{'═' * 60}")


def print_world_overview(world: World):
    """Print a full world overview — all agents sorted by state."""
    summary = world.get_world_summary()

    print(f"\n{'═' * 90}")
    print(f"  WORLD OVERVIEW — {summary['time']}")
    print(f"  {summary['agent_count']} agents | {summary['relationship_count']} relationships")
    print(f"{'═' * 90}")

    # All agents sorted by vulnerability
    agents_sorted = sorted(
        world.agents.values(),
        key=lambda a: a.heart.vulnerability,
        reverse=True,
    )

    print(f"\n  {'Name':<14s} {'Location':<10s} {'Action':<14s} "
          f"{'Arous':>5s} {'Val':>5s} {'Tens':>5s} {'Imp':>5s} {'Enrg':>5s} {'Vuln':>5s} "
          f"{'Internal':<12s} {'Surface':<12s}")
    print(f"  {'─'*14} {'─'*10} {'─'*14} {'─'*5} {'─'*5} {'─'*5} {'─'*5} {'─'*5} {'─'*5} {'─'*12} {'─'*12}")

    for agent in agents_sorted:
        s = agent.heart
        ec = _emotion_color(s.valence, s.arousal)
        print(f"  {agent.personality.name:<14s} {agent.location:<10s} {agent.last_action:<14s} "
              f"{s.arousal:>5.2f} {s.valence:>5.2f} {s.tension:>5.2f} "
              f"{s.impulse_control:>5.2f} {s.energy:>5.2f} {s.vulnerability:>5.2f} "
              f"{s.internal_emotion:<12s} {s.surface_emotion:<12s} {ec}")

    print(f"\n  Action summary: {dict(summary['action_counts'])}")
    print(f"{'═' * 90}")


def save_tick_log(world: World, path: str = "world_sim_log.jsonl"):
    """Save all tick data as JSONL for external visualization."""
    with open(path, "w") as f:
        for tick_data in world.tick_log:
            f.write(json.dumps(tick_data, default=str) + "\n")
    print(f"  Saved {len(world.tick_log)} ticks to {path}")
