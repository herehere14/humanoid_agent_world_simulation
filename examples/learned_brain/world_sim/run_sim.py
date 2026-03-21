#!/usr/bin/env python3
"""Run the world simulation — Small Town, Bad News.

Usage:
    python -m learned_brain.world_sim.run_sim [--days N] [--verbose] [--agent AGENT_ID]

Examples:
    python -m learned_brain.world_sim.run_sim --days 15
    python -m learned_brain.world_sim.run_sim --days 15 --verbose
    python -m learned_brain.world_sim.run_sim --days 15 --agent marcus
"""

from __future__ import annotations

import argparse
import time

from .scenarios import build_small_town
from .dashboard import (
    print_tick_summary,
    print_agent_detail,
    print_world_overview,
    save_tick_log,
)


def run(days: int = 15, verbose: bool = False, focus_agent: str | None = None):
    """Run the simulation for N days."""
    print(f"\n{'═' * 80}")
    print(f"  WORLD SIMULATION: Small Town, Bad News")
    print(f"  {days} days ({days * 24} ticks) | 30 agents | 6 locations")
    print(f"{'═' * 80}")

    world = build_small_town()
    world.initialize()

    total_ticks = days * 24
    t0 = time.time()

    # Key moments to show full overview
    overview_ticks = {
        3 * 24,      # End of Day 3 (before drama)
        5 * 24 + 12, # Day 5 noon (layoffs just happened)
        5 * 24 + 20, # Day 5 evening (bar gathering)
        7 * 24 + 15, # Day 7 afternoon (community rally)
        10 * 24,     # Day 10 (one week after)
        14 * 24 + 12, # Day 14 noon (company picnic)
        days * 24,   # Final tick
    }

    # Ticks to show detailed output
    detail_hours = {9, 10, 11, 12, 17, 19, 20}  # daytime + evening

    for tick in range(1, total_ticks + 1):
        summary = world.tick()
        hour = world.hour_of_day

        # Determine what to print
        has_events = bool(summary.get("events"))
        has_interactions = bool(summary.get("interactions"))
        is_daytime = hour in detail_hours

        if has_events or (verbose and is_daytime) or (has_interactions and is_daytime):
            print_tick_summary(world, verbose=verbose)

        if tick in overview_ticks:
            print_world_overview(world)
            if focus_agent:
                print_agent_detail(world, focus_agent)

    elapsed = time.time() - t0

    # Final summary
    print(f"\n\n{'═' * 80}")
    print(f"  SIMULATION COMPLETE")
    print(f"  {total_ticks} ticks in {elapsed:.1f}s ({total_ticks / elapsed:.0f} ticks/sec)")
    print(f"{'═' * 80}")

    print_world_overview(world)

    # Show detailed view of key characters
    for agent_id in ["marcus", "rosa", "diana", "greg", "richard", "tom", "sarah"]:
        print_agent_detail(world, agent_id)

    # Show most interesting relationship changes
    print(f"\n  Key relationship changes:")
    for agent_id in ["marcus", "rosa", "greg"]:
        agent = world.agents[agent_id]
        rels = world.relationships.get_agent_relationships(agent_id)
        for other_id, rel in rels[:5]:
            other_name = world.agents[other_id].personality.name if other_id in world.agents else other_id
            if rel.familiarity > 0 or abs(rel.trust) > 0.1:
                res = world.relationships.get_resentment(agent_id, other_id)
                print(f"    {agent.personality.name} → {other_name}: trust={rel.trust:+.2f} "
                      f"warmth={rel.warmth:+.2f} resentment={res:.2f} ({rel.familiarity} interactions)")

    save_tick_log(world, "world_sim_log.jsonl")


def main():
    parser = argparse.ArgumentParser(description="World Simulation: Small Town, Bad News")
    parser.add_argument("--days", type=int, default=15, help="Days to simulate")
    parser.add_argument("--verbose", action="store_true", help="Show detailed output each tick")
    parser.add_argument("--agent", type=str, default=None, help="Agent ID to focus on")
    args = parser.parse_args()

    run(days=args.days, verbose=args.verbose, focus_agent=args.agent)


if __name__ == "__main__":
    main()
