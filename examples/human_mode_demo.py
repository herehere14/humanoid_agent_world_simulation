#!/usr/bin/env python3
"""Human Mode demonstration: same agent, different internal states.

This script shows how the same task produces fundamentally different
behavior depending on the agent's internal state — modeling how a human
would respond differently when confident vs anxious, curious vs fatigued.

Run:
    python -m examples.human_mode_demo
"""

from __future__ import annotations

import json

from prompt_forest.modes.orchestrator import ModeOrchestrator


def print_section(title: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def print_state_summary(state: dict) -> None:
    print(f"  Mood:     {state['mood_valence']:+.3f}")
    print(f"  Arousal:  {state['arousal']:.3f}")
    print(f"  Drives:   {', '.join(state['dominant_drives'])}")
    conflicts = state.get("active_conflicts", [])
    if conflicts:
        for c in conflicts:
            print(f"  Conflict: {c['drive_a']} vs {c['drive_b']} (intensity {c['intensity']:.2f})")


def run_scenario(
    title: str,
    task: str,
    initial_state: dict[str, float],
    events: list[tuple[str, float]] | None = None,
) -> dict:
    """Run a single scenario and print results."""
    print_section(title)

    orch = ModeOrchestrator(mode="human_mode", initial_state=initial_state)

    # Inject any pre-task events
    if events:
        for event_type, intensity in events:
            orch.inject_event(event_type, intensity)
            print(f"  [Event injected: {event_type} @ intensity {intensity}]")

    # Show pre-task state
    state_info = orch.get_state()
    print("  --- Internal State Before Task ---")
    print_state_summary(state_info)

    # Run task
    print(f"\n  Task: \"{task}\"\n")
    result = orch.run_task(task, task_type="auto")

    # Show routing
    routing = result["routing"]
    print(f"  Cognitive context: {routing['task_type']}")
    print(f"  Activated branches: {routing['activated_branches']}")

    # Show top 5 branch scores
    scores = sorted(routing["branch_scores"].items(), key=lambda x: x[1], reverse=True)
    print("\n  Top branch scores:")
    for name, score in scores[:5]:
        print(f"    {name:30s} {score:.4f}")

    # Show evaluation
    ev = result["evaluation_signal"]
    print(f"\n  Reward:     {ev['reward_score']:.3f}")
    print(f"  Confidence: {ev['confidence']:.3f}")
    print(f"  Selected:   {ev['selected_branch']}")
    if ev["failure_reason"]:
        print(f"  Issues:     {ev['failure_reason']}")
    if ev["coherence_details"]:
        details = ev["coherence_details"]
        print(f"  Coherence:  {details.get('coherence', 0):.3f}")
        print(f"  Consistency:{details.get('consistency', 0):.3f}")
        print(f"  Believable: {details.get('believability', 0):.3f}")
        print(f"  Conflict:   {details.get('conflict_handling', 0):.3f}")

    # Show state after
    state_after = result["human_state"]["after"]
    print("\n  --- Internal State After Task ---")
    print(f"  Mood:  {state_after['mood_valence']:+.3f}")
    print(f"  Drives: {state_after['dominant_drives']}")

    # Show conflicts resolved
    if result["conflicts"]:
        print("\n  Conflicts resolved:")
        for c in result["conflicts"]:
            print(f"    {c['drive_a']} vs {c['drive_b']}: {c['resolution']} "
                  f"(weight={c['resolution_weight']:.2f})")

    print(f"\n  Timings: {result['timings']}")
    return result


def main() -> None:
    task = "Should I invest significant resources into this new but unproven technology?"

    # ── Scenario 1: Confident Explorer ────────────────────────────────────
    run_scenario(
        title="Scenario 1: Confident Explorer",
        task=task,
        initial_state={
            "confidence": 0.85,
            "curiosity": 0.80,
            "ambition": 0.75,
            "fear": 0.15,
            "stress": 0.10,
            "trust": 0.70,
            "motivation": 0.80,
        },
    )

    # ── Scenario 2: Anxious Defender ──────────────────────────────────────
    run_scenario(
        title="Scenario 2: Anxious Defender",
        task=task,
        initial_state={
            "confidence": 0.20,
            "curiosity": 0.25,
            "ambition": 0.30,
            "fear": 0.85,
            "stress": 0.80,
            "trust": 0.20,
            "self_protection": 0.80,
            "caution": 0.75,
        },
    )

    # ── Scenario 3: Internally Conflicted ─────────────────────────────────
    run_scenario(
        title="Scenario 3: Internally Conflicted (high curiosity AND high fear)",
        task=task,
        initial_state={
            "curiosity": 0.80,
            "fear": 0.75,
            "ambition": 0.70,
            "caution": 0.65,
            "stress": 0.50,
            "confidence": 0.50,
        },
    )

    # ── Scenario 4: Fatigued and Under Pressure ──────────────────────────
    run_scenario(
        title="Scenario 4: Fatigued Under Deadline Pressure",
        task=task,
        initial_state={
            "fatigue": 0.80,
            "motivation": 0.30,
            "stress": 0.60,
            "impulse": 0.50,
            "reflection": 0.30,
        },
        events=[("deadline_pressure", 0.8)],
    )

    # ── Scenario 5: Post-Success Momentum ─────────────────────────────────
    print_section("Scenario 5: Momentum — Same agent after repeated success")
    orch = ModeOrchestrator(
        mode="human_mode",
        initial_state={"confidence": 0.5, "motivation": 0.5},
    )

    for i in range(3):
        # Simulate success by injecting positive events
        orch.inject_event("reward", intensity=0.7)
        orch.inject_event("social_praise", intensity=0.5)

    state_info = orch.get_state()
    print("  After 3 rounds of success + praise:")
    print_state_summary(state_info)
    print(f"  Memory entries: {state_info.get('experiential_memory_count', 0)}")

    result = orch.run_task(task, task_type="auto")
    print(f"\n  Routing: {result['routing']['activated_branches']}")
    print(f"  Reward:  {result['evaluation_signal']['reward_score']:.3f}")

    print_section("Demo Complete")
    print("  The same task produced different routing, branch activation,")
    print("  and evaluation depending on the agent's internal state.")
    print("  This is the core of Human Mode: cognitive-behavioral variation.")


if __name__ == "__main__":
    main()
