#!/usr/bin/env python3
"""Scale benchmark — 1000 agents, performance + state variance + LLM spot-check.

Tests:
1. Wall clock time per tick at 1000 agents
2. State variance across agents (do personalities create divergent trajectories?)
3. Event sensitivity (do affected agents differ from unaffected?)
4. LLM spot-check: sample 30 agents at 10 moments, compare heart vs pure-LLM

Usage:
    python -m world_sim.eval.scale_benchmark [--agents 1000] [--skip-llm]
"""

from __future__ import annotations

import json
import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from openai import OpenAI

from ..world import World, Location, ScheduledEvent
from ..world_agent import WorldAgent, SharedBrain, HeartState, Personality
from ..action_table import Action, select_action, TickContext, get_action_description
from ..contagion import apply_contagion
from ..relationship import RelationshipStore
from ..scenarios import AGENTS, AGENT_ROLES, _make_schedule, LAYOFF_TARGETS

OBSERVATION_TICKS = [72, 82, 105, 106, 113, 128, 158, 187, 230, 286]
N_SAMPLE_AGENTS = 30


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def build_large_world(n_agents: int = 1000, seed: int = 42) -> tuple[World, list[str]]:
    """Build a scaled-up version of Small Town with n_agents.

    Returns (world, layoff_target_ids).
    """
    rng = random.Random(seed)
    world = World()

    # Same 6 locations
    for loc in [
        Location("office", "Meridian Corp Office", "Working at desk, meetings, routine office work"),
        Location("home", "Home", "At home, family time, relaxing"),
        Location("bar", "The Tap Bar & Grill", "Drinking, socializing, unwinding after work"),
        Location("park", "Riverside Park", "Walking, sitting on benches, children playing"),
        Location("church", "Community Church", "Quiet reflection, community gatherings"),
        Location("school", "Lincoln High School", "Teaching, studying, school activities"),
    ]:
        world.add_location(loc)

    # Role distribution (proportional to original)
    roles = ["office_worker"] * 60 + ["manager"] * 5 + ["bartender"] * 5 + \
            ["teacher"] * 10 + ["retiree"] * 20
    base_templates = AGENTS  # 30 personality templates

    agent_ids = []
    for i in range(n_agents):
        base = base_templates[i % len(base_templates)]
        role = roles[i % len(roles)]

        # Perturb personality parameters
        personality = Personality(
            name=f"{base.name}_{i:04d}",
            background=base.background,
            temperament=base.temperament,
            arousal_rise_rate=_clamp(base.arousal_rise_rate + rng.uniform(-0.1, 0.1), 0.3, 0.95),
            arousal_decay_rate=_clamp(base.arousal_decay_rate + rng.uniform(-0.05, 0.05), 0.75, 0.97),
            valence_momentum=_clamp(base.valence_momentum + rng.uniform(-0.1, 0.1), 0.2, 0.7),
            impulse_drain_rate=_clamp(base.impulse_drain_rate + rng.uniform(-0.05, 0.05), 0.05, 0.3),
            impulse_restore_rate=_clamp(base.impulse_restore_rate + rng.uniform(-0.003, 0.003), 0.002, 0.02),
            energy_drain_rate=_clamp(base.energy_drain_rate + rng.uniform(-0.02, 0.02), 0.02, 0.15),
            energy_regen_rate=_clamp(base.energy_regen_rate + rng.uniform(-0.005, 0.005), 0.003, 0.025),
            vulnerability_weight=_clamp(base.vulnerability_weight + rng.uniform(-0.3, 0.3), 0.3, 1.8),
        )

        aid = f"agent_{i:04d}"
        agent = WorldAgent(
            agent_id=aid,
            personality=personality,
            schedule=_make_schedule(role),
        )
        world.add_agent(agent)
        agent_ids.append(aid)

    # Set up some random relationships (sparse — ~5 per agent)
    for aid in agent_ids:
        n_rels = rng.randint(2, 8)
        for _ in range(n_rels):
            other = rng.choice(agent_ids)
            if other == aid:
                continue
            rel = world.relationships.get_or_create(aid, other)
            if rel.familiarity == 0:  # only set if new
                rel.trust = rng.uniform(-0.2, 0.8)
                rel.warmth = rng.uniform(-0.2, 0.7)
                rel.familiarity = rng.randint(1, 100)

    # Layoff targets: ~7% of office workers
    office_agents = [aid for aid in agent_ids
                     if roles[agent_ids.index(aid) % len(roles)] == "office_worker"]
    n_layoffs = max(1, int(len(office_agents) * 0.07))
    layoff_ids = rng.sample(office_agents, n_layoffs)

    # Schedule events (same structure as small town)
    def dt(day: int, hour: int) -> int:
        return (day - 1) * 24 + hour

    world.schedule_event(ScheduledEvent(
        tick=dt(4, 10), location="office",
        description="Layoff rumors circulating through the office.",
        emotional_text="I'm about to lose my job and I can't afford it. I'm panicking, my heart is racing.",
    ))
    world.schedule_event(ScheduledEvent(
        tick=dt(4, 15), location="office",
        description="Restructuring plan spotted. Fear intensifies.",
        emotional_text="I'm terrified, something terrible is happening. I might get fired.",
    ))
    world.schedule_event(ScheduledEvent(
        tick=dt(5, 9), location="office",
        description="Official layoff announcement — positions being eliminated.",
        emotional_text="I'm panicking, I can't breathe. I'm about to lose my job. This is outrageous.",
    ))

    # Individual layoff notifications
    for i, aid in enumerate(layoff_ids):
        name = world.agents[aid].personality.name
        world.schedule_event(ScheduledEvent(
            tick=dt(5, 10) + i // 10,
            location="office",
            description=f"{name} is told their position has been eliminated.",
            emotional_text="I just got fired and I don't know what to do. I feel worthless and hopeless.",
            target_agent_ids=[aid],
        ))

    world.schedule_event(ScheduledEvent(
        tick=dt(6, 8), location="home",
        description="Morning after layoffs. Weight of unemployment.",
        emotional_text="I barely slept. I feel worthless. Everything feels flat and empty.",
        target_agent_ids=layoff_ids,
    ))

    world.schedule_event(ScheduledEvent(
        tick=dt(8, 19), location="bar",
        description="Anger boils over at the bar. Workers confront each other.",
        emotional_text="I'm furious and I can't hold back. How dare they do this to me.",
    ))

    return world, layoff_ids


def run_scale_benchmark(
    n_agents: int = 1000,
    skip_llm: bool = False,
    output_path: str = "scale_results.json",
):
    """Run the full scale benchmark."""
    print(f"\n{'═' * 70}")
    print(f"  SCALE BENCHMARK: {n_agents} agents, {max(OBSERVATION_TICKS)} ticks")
    print(f"{'═' * 70}")

    # Build world
    print(f"\n  Building {n_agents}-agent world...")
    t0 = time.time()
    world, layoff_ids = build_large_world(n_agents)
    print(f"  Built in {time.time() - t0:.2f}s")

    print(f"  Layoff targets: {len(layoff_ids)} agents")
    print(f"  Initializing SharedBrain...")
    world.initialize()

    # Classify agents
    all_ids = list(world.agents.keys())
    office_ids = set(aid for aid in all_ids if world.agents[aid].schedule.get(10) == "office")
    layoff_set = set(layoff_ids)
    bystander_ids = office_ids - layoff_set
    remote_ids = set(all_ids) - office_ids

    # Run simulation with timing
    max_tick = max(OBSERVATION_TICKS)
    tick_times = []
    snapshots = {}

    print(f"\n  Running {max_tick} ticks...")
    sim_t0 = time.time()

    for tick in range(1, max_tick + 1):
        t_start = time.perf_counter()
        world.tick()
        t_elapsed = time.perf_counter() - t_start
        tick_times.append(t_elapsed)

        if tick in OBSERVATION_TICKS:
            # Capture state snapshot
            snapshot = {}
            for aid in all_ids:
                s = world.agents[aid].heart
                snapshot[aid] = {
                    "valence": s.valence,
                    "arousal": s.arousal,
                    "tension": s.tension,
                    "impulse_control": s.impulse_control,
                    "energy": s.energy,
                    "vulnerability": s.vulnerability,
                    "wounds": len(s.wounds),
                    "internal_emotion": s.internal_emotion,
                    "surface_emotion": s.surface_emotion,
                }
            snapshots[tick] = snapshot

        if tick % 50 == 0:
            print(f"    Tick {tick}/{max_tick} ({t_elapsed * 1000:.1f}ms)")

    sim_elapsed = time.time() - sim_t0

    # ─── Performance Report ───
    print(f"\n{'═' * 70}")
    print(f"  PERFORMANCE REPORT")
    print(f"{'═' * 70}")
    print(f"  Total simulation time: {sim_elapsed:.2f}s")
    print(f"  Ticks per second: {max_tick / sim_elapsed:.1f}")
    print(f"  Avg tick time: {np.mean(tick_times) * 1000:.2f}ms")
    print(f"  Max tick time: {np.max(tick_times) * 1000:.2f}ms")
    print(f"  Min tick time: {np.min(tick_times) * 1000:.2f}ms")
    print(f"  P95 tick time: {np.percentile(tick_times, 95) * 1000:.2f}ms")

    # Estimated LLM cost for equivalent simulation
    est_llm_calls = n_agents * max_tick
    est_llm_cost = est_llm_calls * 1000 * 0.15 / 1e6 + est_llm_calls * 100 * 0.60 / 1e6
    print(f"\n  Estimated pure-LLM cost for same simulation:")
    print(f"    {est_llm_calls:,} API calls × ~1100 tokens = ~${est_llm_cost:.2f}")

    # ─── State Variance Report ───
    print(f"\n{'═' * 70}")
    print(f"  STATE VARIANCE REPORT")
    print(f"{'═' * 70}")

    dims = ["valence", "arousal", "tension", "impulse_control", "energy", "vulnerability"]

    print(f"\n  Std dev across all {n_agents} agents at each observation tick:")
    print(f"  {'Tick':>6s} {'Time':<14s} {'Valence':>8s} {'Arousal':>8s} {'Tension':>8s} {'Impulse':>8s} {'Energy':>8s} {'Vuln':>8s}")
    print(f"  {'─' * 6} {'─' * 14} {'─' * 8} {'─' * 8} {'─' * 8} {'─' * 8} {'─' * 8} {'─' * 8}")

    for tick in OBSERVATION_TICKS:
        snap = snapshots[tick]
        stds = {}
        for dim in dims:
            values = [snap[aid][dim] for aid in all_ids]
            stds[dim] = np.std(values)

        time_str = f"Day {tick // 24 + 1}, {tick % 24:02d}:00"
        print(f"  {tick:>6d} {time_str:<14s} {stds['valence']:>8.4f} {stds['arousal']:>8.4f} "
              f"{stds['tension']:>8.4f} {stds['impulse_control']:>8.4f} "
              f"{stds['energy']:>8.4f} {stds['vulnerability']:>8.4f}")

    # ─── Event Sensitivity Report ───
    print(f"\n{'═' * 70}")
    print(f"  EVENT SENSITIVITY (Cohen's d between groups)")
    print(f"{'═' * 70}")

    print(f"\n  Groups: Laid-off ({len(layoff_set)}) | Bystanders ({len(bystander_ids)}) | Remote ({len(remote_ids)})")

    for tick in [105, 128, 230]:  # announcement, morning after, recovery
        snap = snapshots[tick]
        time_str = f"Day {tick // 24 + 1}, {tick % 24:02d}:00"
        print(f"\n  Tick {tick} ({time_str}):")

        for dim in ["valence", "energy", "vulnerability"]:
            laid_off_vals = [snap[aid][dim] for aid in layoff_set if aid in snap]
            bystander_vals = [snap[aid][dim] for aid in bystander_ids if aid in snap]
            remote_vals = [snap[aid][dim] for aid in remote_ids if aid in snap]

            # Cohen's d: (mean1 - mean2) / pooled_std
            def cohens_d(a, b):
                a, b = np.array(a), np.array(b)
                if len(a) < 2 or len(b) < 2:
                    return 0.0
                pooled_std = np.sqrt((np.var(a) + np.var(b)) / 2)
                if pooled_std < 1e-8:
                    return 0.0
                return (np.mean(a) - np.mean(b)) / pooled_std

            d_lo_by = cohens_d(laid_off_vals, bystander_vals)
            d_lo_re = cohens_d(laid_off_vals, remote_vals)

            print(f"    {dim:<14s}: laid-off vs bystander d={d_lo_by:+.3f}  |  laid-off vs remote d={d_lo_re:+.3f}")

    # ─── Action Distribution ───
    print(f"\n{'═' * 70}")
    print(f"  ACTION DISTRIBUTION AT KEY MOMENTS")
    print(f"{'═' * 70}")

    for tick in [72, 105, 128, 230]:
        snap = snapshots[tick]
        time_str = f"Day {tick // 24 + 1}, {tick % 24:02d}:00"

        # Count actions by group
        for group_name, group_ids in [("Laid-off", layoff_set), ("Bystanders", bystander_ids)]:
            action_counts = defaultdict(int)
            for aid in group_ids:
                if aid in world.agents:
                    action_counts[world.agents[aid].last_action] += 1

            total = sum(action_counts.values())
            if total == 0:
                continue
            non_routine = {k: v for k, v in action_counts.items() if k not in ("WORK", "REST", "IDLE")}
            routine = {k: v for k, v in action_counts.items() if k in ("WORK", "REST", "IDLE")}

            parts = [f"{k}:{v}" for k, v in sorted(non_routine.items(), key=lambda x: -x[1])]
            routine_parts = [f"{k}:{v}" for k, v in routine.items()]
            all_parts = parts + routine_parts
            print(f"  [{time_str}] {group_name:<12s}: {', '.join(all_parts[:8])}")

    # ─── Emotion Distribution ───
    print(f"\n{'═' * 70}")
    print(f"  EMOTION DISTRIBUTION AT KEY MOMENTS")
    print(f"{'═' * 70}")

    for tick in [72, 105, 128, 230]:
        snap = snapshots[tick]
        time_str = f"Day {tick // 24 + 1}, {tick % 24:02d}:00"

        for group_name, group_ids in [("Laid-off", layoff_set), ("Bystanders", bystander_ids), ("Remote", remote_ids)]:
            emotion_counts = defaultdict(int)
            for aid in group_ids:
                if aid in snap:
                    emotion_counts[snap[aid]["internal_emotion"]] += 1

            total = sum(emotion_counts.values())
            if total == 0:
                continue
            parts = [f"{k}:{v}" for k, v in sorted(emotion_counts.items(), key=lambda x: -x[1])]
            print(f"  [{time_str}] {group_name:<12s}: {', '.join(parts)}")

    # ─── LLM Spot-check ───
    if not skip_llm:
        print(f"\n{'═' * 70}")
        print(f"  LLM SPOT-CHECK ({N_SAMPLE_AGENTS} sampled agents)")
        print(f"{'═' * 70}")

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("  OPENAI_API_KEY not set — skipping LLM comparison")
        else:
            _run_llm_spotcheck(world, snapshots, layoff_set, bystander_ids, remote_ids, api_key)

    # Save results
    results = {
        "n_agents": n_agents,
        "n_ticks": max_tick,
        "sim_time_s": sim_elapsed,
        "ticks_per_sec": max_tick / sim_elapsed,
        "avg_tick_ms": np.mean(tick_times) * 1000,
        "p95_tick_ms": np.percentile(tick_times, 95) * 1000,
        "n_layoffs": len(layoff_ids),
        "est_llm_cost": est_llm_cost,
    }
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {output_path}")


def _run_llm_spotcheck(
    world: World,
    snapshots: dict,
    layoff_set: set[str],
    bystander_ids: set[str],
    remote_ids: set[str],
    api_key: str,
):
    """Sample 30 agents, compare heart action vs LLM action at key moments."""
    rng = random.Random(42)
    client = OpenAI(api_key=api_key)

    # Stratified sample: 10 laid-off, 10 bystanders, 10 remote
    sample_laid = rng.sample(sorted(layoff_set), min(10, len(layoff_set)))
    sample_bystander = rng.sample(sorted(bystander_ids), min(10, len(bystander_ids)))
    sample_remote = rng.sample(sorted(remote_ids), min(10, len(remote_ids)))
    sampled = sample_laid + sample_bystander + sample_remote

    # Only check 3 key moments to keep costs down
    check_ticks = [105, 128, 230]  # announcement, morning after, recovery

    matches = 0
    total = 0
    action_agreement = defaultdict(lambda: {"match": 0, "total": 0})

    for tick in check_ticks:
        snap = snapshots[tick]
        time_str = f"Day {tick // 24 + 1}, {tick % 24:02d}:00"

        for aid in sampled:
            agent = world.agents[aid]
            heart_action = agent.last_action

            # Build event history from agent memory
            history = []
            for mem in agent.memory:
                if mem.tick <= tick:
                    t_str = f"Day {mem.tick // 24 + 1}, {mem.tick % 24:02d}:00"
                    history.append(f"[{t_str}] {mem.description}")

            prompt = f"""You are {agent.personality.name}, {agent.personality.background}
Temperament: {agent.personality.temperament}

Events you've experienced:
{chr(10).join(history[-10:]) if history else "Nothing notable."}

Current: {time_str} at {agent.location}

What do you do? Choose ONE: COLLAPSE, LASH_OUT, CONFRONT, FLEE, WITHDRAW, SEEK_COMFORT, RUMINATE, VENT, SOCIALIZE, CELEBRATE, HELP_OTHERS, WORK, REST, IDLE

Reply with just the action name."""

            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0.0,
            )
            llm_action = resp.choices[0].message.content.strip().upper()

            match = heart_action == llm_action
            if match:
                matches += 1
            total += 1

            group = "laid-off" if aid in layoff_set else "bystander" if aid in bystander_ids else "remote"
            action_agreement[group]["total"] += 1
            if match:
                action_agreement[group]["match"] += 1

    print(f"\n  Action agreement (heart vs LLM): {matches}/{total} ({matches / total * 100:.1f}%)")
    for group, counts in action_agreement.items():
        pct = counts["match"] / counts["total"] * 100 if counts["total"] > 0 else 0
        print(f"    {group:<12s}: {counts['match']}/{counts['total']} ({pct:.0f}%)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents", type=int, default=1000)
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument("--output", default="scale_results.json")
    args = parser.parse_args()
    run_scale_benchmark(n_agents=args.agents, skip_llm=args.skip_llm, output_path=args.output)
