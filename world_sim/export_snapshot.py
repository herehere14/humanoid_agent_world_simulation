"""Export simulation snapshots as JSON for the 3D world viewer.

Runs the small-town scenario and writes per-tick state to a JSON file
that the frontend can load for playback.

Usage:
    python -m world_sim.export_snapshot [--ticks 360] [--out artifacts/world_snapshot.json]
"""

from __future__ import annotations

import json
import argparse
import time
from pathlib import Path

from .scenarios import build_small_town
from .action_table import Action


def export_world_snapshot(max_ticks: int = 360, out_path: str | None = None) -> dict:
    """Run simulation and export full snapshot data."""
    print("Building small town scenario...")
    world = build_small_town()
    world.initialize()

    # Collect location metadata
    locations_meta = {}
    for loc_id, loc in world.locations.items():
        locations_meta[loc_id] = {
            "id": loc_id,
            "name": loc.name,
            "default_activity": loc.default_activity,
        }

    # Collect agent metadata (static info)
    agents_meta = {}
    for aid, agent in world.agents.items():
        agents_meta[aid] = {
            "id": aid,
            "name": agent.personality.name,
            "background": agent.personality.background,
            "temperament": agent.personality.temperament,
            "identity_tags": list(agent.identity_tags),
            "coalitions": list(agent.coalitions),
            "rival_coalitions": list(agent.rival_coalitions),
            "private_burden": agent.private_burden,
        }

    # Run simulation and collect tick data
    ticks_data = []
    t0 = time.time()

    for i in range(max_ticks):
        tick_summary = world.tick()

        # Collect per-agent state this tick
        agent_states = {}
        for aid, agent in world.agents.items():
            dashboard = agent.get_dashboard_state()
            # Add relationship data
            rels = []
            for other_id, rel in world.relationships.get_agent_relationships(aid)[:8]:
                other_name = world.agents[other_id].personality.name if other_id in world.agents else other_id
                rels.append({
                    "other_id": other_id,
                    "other_name": other_name,
                    "trust": round(rel.trust, 2),
                    "warmth": round(rel.warmth, 2),
                    "resentment_toward": round(world.relationships.get_resentment(aid, other_id), 2),
                    "resentment_from": round(world.relationships.get_resentment(other_id, aid), 2),
                    "grievance_toward": round(world.relationships.get_grievance(aid, other_id), 2),
                    "grievance_from": round(world.relationships.get_grievance(other_id, aid), 2),
                    "debt_toward": round(world.relationships.get_debt(aid, other_id), 2),
                    "debt_from": round(world.relationships.get_debt(other_id, aid), 2),
                    "alliance_strength": round(rel.alliance_strength, 2),
                    "rivalry": round(rel.rivalry, 2),
                    "support_events": rel.support_events,
                    "conflict_events": rel.conflict_events,
                    "betrayal_events": rel.betrayal_events,
                })
            dashboard["relationships"] = rels

            # Recent memories
            dashboard["recent_memories"] = [
                {
                    "tick": m.tick,
                    "description": m.description,
                    "interpretation": m.interpretation,
                    "story_beat": m.story_beat,
                    "valence": round(m.valence_at_time, 2),
                    "other": m.other_agent_id,
                }
                for m in agent.get_recent_memories(10)
            ]

            agent_states[aid] = dashboard

        # Build tick record
        tick_record = {
            "tick": tick_summary["tick"],
            "time": tick_summary["time"],
            "events": tick_summary["events"],
            "interactions": tick_summary["interactions"],
            "llm_focus": tick_summary.get("llm_focus", []),
            "llm_packets": tick_summary.get("llm_packets", []),
            "agent_states": agent_states,
        }
        ticks_data.append(tick_record)

        if (i + 1) % 24 == 0:
            elapsed = time.time() - t0
            print(f"  Day {(i+1)//24} complete ({i+1} ticks, {elapsed:.1f}s)")

    elapsed = time.time() - t0
    print(f"Simulation complete: {max_ticks} ticks in {elapsed:.1f}s")

    snapshot = {
        "scenario": "small_town",
        "total_ticks": max_ticks,
        "locations": locations_meta,
        "agents": agents_meta,
        "ticks": ticks_data,
    }

    if out_path is None:
        out_path = str(Path(__file__).parent.parent / "artifacts" / "world_snapshot.json")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(snapshot, f, indent=None, separators=(",", ":"))

    size_mb = Path(out_path).stat().st_size / 1024 / 1024
    print(f"Snapshot written to {out_path} ({size_mb:.1f} MB)")
    return snapshot


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export world simulation snapshot for 3D viewer")
    parser.add_argument("--ticks", type=int, default=360, help="Number of ticks to simulate (default: 360 = 15 days)")
    parser.add_argument("--out", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()
    export_world_snapshot(max_ticks=args.ticks, out_path=args.out)
