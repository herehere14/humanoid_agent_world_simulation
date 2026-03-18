#!/usr/bin/env python3
"""Side-by-side comparison of Agent Improvement vs Human Mode.

Shows the same task processed through both pipelines, highlighting
how the two modes produce structurally different outputs.

Run:
    python -m examples.two_mode_comparison
"""

from __future__ import annotations

import json

from prompt_forest.modes.orchestrator import ModeOrchestrator


def print_section(title: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def main() -> None:
    task = "Should we proceed with the risky but potentially high-reward project?"

    print_section("Two-Mode Comparison")
    print(f"  Task: \"{task}\"\n")

    # ── Agent Improvement Mode ────────────────────────────────────────────
    print("  ┌─────────────────────────────────────────────────────────────┐")
    print("  │  MODE 1: AGENT IMPROVEMENT                                 │")
    print("  └─────────────────────────────────────────────────────────────┘")

    agent_orch = ModeOrchestrator(mode="agent_improvement")
    agent_result = agent_orch.run_task(task)

    agent_state = agent_orch.get_state()
    print(f"  Branches: {agent_state['branch_count']}")
    print(f"  Top branches:")
    for b in sorted(agent_state["branches"], key=lambda x: x["weight"], reverse=True)[:5]:
        print(f"    {b['name']:30s} weight={b['weight']:.3f}")

    print(f"\n  No internal state (optimizes for task performance only)")
    print(f"  No conflict detection")
    print(f"  No experiential memory")

    # ── Human Mode ────────────────────────────────────────────────────────
    print("\n  ┌─────────────────────────────────────────────────────────────┐")
    print("  │  MODE 2: HUMAN MODE                                        │")
    print("  └─────────────────────────────────────────────────────────────┘")

    human_orch = ModeOrchestrator(
        mode="human_mode",
        initial_state={"curiosity": 0.7, "fear": 0.6, "ambition": 0.65},
    )
    human_result = human_orch.run_task(task)

    human_state = human_orch.get_state()
    print(f"  Branches: {human_state['branch_count']}")
    print(f"  Mood: {human_state['mood_valence']:+.3f}")
    print(f"  Arousal: {human_state['arousal']:.3f}")
    print(f"  Dominant drives: {human_state['dominant_drives']}")

    conflicts = human_state.get("active_conflicts", [])
    if conflicts:
        print(f"  Active conflicts:")
        for c in conflicts:
            print(f"    {c['drive_a']} vs {c['drive_b']} (intensity={c['intensity']:.2f})")

    print(f"  Memory: {human_state.get('experiential_memory_count', 0)} entries")

    # Routing comparison
    if "routing" in human_result:
        routing = human_result["routing"]
        print(f"\n  Cognitive context: {routing['task_type']}")
        print(f"  Activated branches: {routing['activated_branches']}")

    if "evaluation_signal" in human_result:
        ev = human_result["evaluation_signal"]
        print(f"\n  Coherence evaluation:")
        print(f"    Reward:     {ev['reward_score']:.3f}")
        print(f"    Confidence: {ev['confidence']:.3f}")
        if ev["coherence_details"]:
            d = ev["coherence_details"]
            print(f"    Coherence:  {d.get('coherence', 0):.3f}")
            print(f"    Believable: {d.get('believability', 0):.3f}")
            print(f"    Conflict:   {d.get('conflict_handling', 0):.3f}")

    # ── Summary ───────────────────────────────────────────────────────────
    print_section("Key Differences")
    print("  Agent Improvement Mode:")
    print("    • Optimizes for task correctness and coverage")
    print("    • Branch weights adapt based on performance")
    print("    • No internal state — stateless optimization")
    print()
    print("  Human Mode:")
    print("    • Models cognitive-behavioral processes")
    print("    • State-conditioned routing (mood, drives, conflicts)")
    print("    • Same task → different responses based on internal state")
    print("    • Competing drives create conflicts that must be resolved")
    print("    • Experiential memory creates approach/avoidance biases")
    print("    • Evaluation measures behavioral coherence, not just quality")


if __name__ == "__main__":
    main()
