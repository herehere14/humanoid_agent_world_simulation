#!/usr/bin/env python3
"""Emotion Interaction Demo: shows exactly how each emotion affects the system.

Runs through concrete scenarios showing numerical state changes,
drive conflicts, routing shifts, and cross-variable cascades.
"""

from __future__ import annotations

from prompt_forest.modes.orchestrator import ModeOrchestrator
from prompt_forest.modes.human_mode.router import HumanModeRouter
from prompt_forest.modes.human_mode.branches import create_human_mode_forest
from prompt_forest.state.human_state import HumanState
from prompt_forest.types import TaskInput


def bar(val: float, width: int = 30) -> str:
    """ASCII bar chart."""
    filled = int(val * width)
    return f"{'█' * filled}{'░' * (width - filled)} {val:.3f}"


def show_state(state: HumanState, label: str = "", variables: list[str] | None = None) -> None:
    if label:
        print(f"\n  ── {label} ──")
    vars_to_show = variables or [
        "confidence", "stress", "frustration", "fear", "curiosity",
        "motivation", "trust", "fatigue", "impulse", "reflection",
        "ambition", "empathy", "self_protection", "caution",
    ]
    for v in vars_to_show:
        val = state.get(v)
        print(f"    {v:20s} {bar(val)}")
    print(f"    {'mood_valence':20s} {state.mood_valence():+.3f}")
    print(f"    {'arousal':20s} {state.arousal_level():.3f}")
    conflicts = state.active_conflicts
    if conflicts:
        print(f"    CONFLICTS:")
        for c in conflicts:
            print(f"      ⚡ {c.drive_a} vs {c.drive_b} (intensity={c.intensity:.3f})")


def show_routing(state: HumanState, task_text: str = "Should I take a risk?") -> None:
    forest = create_human_mode_forest()
    router = HumanModeRouter(top_k=5, noise_level=0.0)
    task = TaskInput(task_id="t1", text=task_text, task_type="auto")
    decision, conflicts = router.route(task, forest, state)
    print(f"\n    Cognitive context: {decision.task_type}")
    print(f"    Activated branches: {decision.activated_branches}")
    ranked = sorted(decision.branch_scores.items(), key=lambda x: x[1], reverse=True)
    print(f"    Top scores:")
    for name, score in ranked[:6]:
        marker = " ◀" if name in decision.activated_branches else ""
        print(f"      {name:30s} {score:.4f}{marker}")


def section(title: str) -> None:
    print(f"\n{'='*72}")
    print(f"  {title}")
    print(f"{'='*72}")


def main() -> None:
    # ══════════════════════════════════════════════════════════════════════
    # DEMO 1: Fear cascades
    # ══════════════════════════════════════════════════════════════════════
    section("DEMO 1: FEAR CASCADE — How threat events ripple through state")

    state = HumanState(noise_level=0.0)
    show_state(state, "Baseline state",
               ["fear", "stress", "caution", "curiosity", "confidence", "impulse", "reflection"])

    print("\n  >>> Injecting THREAT event (intensity=0.8)...")
    state.inject_event("threat", intensity=0.8)
    show_state(state, "After threat",
               ["fear", "stress", "caution", "curiosity", "confidence", "impulse", "reflection"])
    print("""
    What happened:
    • fear ↑ — direct threat response
    • stress ↑ — threat causes stress
    • caution ↑ — defensive posture
    • curiosity ↓ — threat suppresses exploration
    • impulse stayed stable — not enough stress to trigger cross-effect yet
    """)

    print("  >>> Injecting SECOND threat (intensity=0.9)...")
    state.inject_event("threat", intensity=0.9)
    show_state(state, "After second threat",
               ["fear", "stress", "caution", "curiosity", "confidence", "impulse", "reflection"])
    show_routing(state)
    print("""
    Now stress is high enough to trigger CROSS-EFFECTS:
    • reflection ↓ — high stress impairs deliberate thinking
    • impulse ↑ — fight-or-flight kicks in
    • fear_risk branch now dominates routing
    """)

    # ══════════════════════════════════════════════════════════════════════
    # DEMO 2: Confidence → Ambition feedback loop
    # ══════════════════════════════════════════════════════════════════════
    section("DEMO 2: SUCCESS SPIRAL — Confidence feeds ambition")

    state = HumanState(
        initial_values={"confidence": 0.5, "ambition": 0.4, "motivation": 0.5},
        noise_level=0.0,
    )
    print("\n  Starting: confidence=0.5, ambition=0.4, motivation=0.5")
    print(f"  {'Round':<8} {'confidence':>12} {'ambition':>12} {'motivation':>12} {'mood':>8}")
    print(f"  {'─'*52}")

    for i in range(8):
        state.apply_outcome(reward=0.85, task_type="general")
        print(f"  {i+1:<8} {state.get('confidence'):>12.3f} "
              f"{state.get('ambition'):>12.3f} {state.get('motivation'):>12.3f} "
              f"{state.mood_valence():>+8.3f}")

    print("""
    Pattern: success → confidence ↑ → ambition ↑ (cross-effect when conf > 0.7)
    → motivation ↑ → positive mood → even more confidence
    This is the "winner effect" — success breeds more success-seeking behavior.
    """)

    # ══════════════════════════════════════════════════════════════════════
    # DEMO 3: Failure spiral
    # ══════════════════════════════════════════════════════════════════════
    section("DEMO 3: FAILURE SPIRAL — Frustration amplifies stress")

    state = HumanState(
        initial_values={"confidence": 0.6, "stress": 0.2, "frustration": 0.1},
        noise_level=0.0,
    )
    print("\n  Starting: confidence=0.6, stress=0.2, frustration=0.1")
    print(f"  {'Round':<8} {'confidence':>12} {'stress':>12} {'frustration':>12} "
          f"{'fear':>8} {'motivation':>12} {'mood':>8}")
    print(f"  {'─'*72}")

    for i in range(10):
        state.apply_outcome(reward=0.15, task_type="general")
        print(f"  {i+1:<8} {state.get('confidence'):>12.3f} "
              f"{state.get('stress'):>12.3f} {state.get('frustration'):>12.3f} "
              f"{state.get('fear'):>8.3f} {state.get('motivation'):>12.3f} "
              f"{state.mood_valence():>+8.3f}")

    print("""
    Pattern: failure → frustration ↑ → frustration_amp = 1.0 + frustration
    → next failure's stress delta is MULTIPLIED by frustration_amp
    → stress ↑↑ → reflection ↓, impulse ↑ (cross-effects)
    → confidence ↓ → mood goes negative → emotional momentum goes negative
    This models "learned helplessness" — repeated failure compounds.
    """)

    # ══════════════════════════════════════════════════════════════════════
    # DEMO 4: Drive conflicts — curiosity vs fear
    # ══════════════════════════════════════════════════════════════════════
    section("DEMO 4: APPROACH-AVOIDANCE CONFLICT — Curiosity vs Fear")

    state = HumanState(
        initial_values={"curiosity": 0.75, "fear": 0.70},
        noise_level=0.0,
    )
    show_state(state, "High curiosity AND high fear",
               ["curiosity", "fear", "stress", "reflection", "impulse"])

    state.update({})  # trigger conflict detection
    show_state(state, "After conflict detection",
               ["curiosity", "fear", "stress", "reflection", "impulse"])

    print("\n    How conflict affects routing:")
    show_routing(state, "Should I explore this unknown but dangerous area?")

    print("""
    When curiosity ≈ fear (both > 0.4, gap < 0.25):
    • A DriveConflict fires
    • Router activates branches for BOTH sides (curiosity_exploration + fear_risk)
    • conflict_resolver gets a 1.5x score boost
    • The system doesn't just pick one — it processes the TENSION
    • Stress increases slightly from unresolved conflict (cognitive dissonance)
    """)

    # Now resolve with different strategies
    print("  Resolution strategies:")
    for strategy in ["dominant", "weighted_compromise", "noisy"]:
        s = HumanState(initial_values={"curiosity": 0.75, "fear": 0.70}, noise_level=0.0)
        s.update({})
        for c in s.active_conflicts:
            if {c.drive_a, c.drive_b} == {"curiosity", "fear"}:
                s.resolve_conflict(c, strategy=strategy)
                print(f"\n    Strategy: {strategy}")
                print(f"      Resolution: {c.resolution}")
                print(f"      Weight: {c.resolution_weight:.3f} (0=drive_a wins, 1=drive_b wins)")
                print(f"      curiosity after: {s.get('curiosity'):.3f}")
                print(f"      fear after:      {s.get('fear'):.3f}")

    # ══════════════════════════════════════════════════════════════════════
    # DEMO 5: Fatigue degrades everything
    # ══════════════════════════════════════════════════════════════════════
    section("DEMO 5: COGNITIVE FATIGUE — Progressive degradation")

    print(f"\n  {'fatigue':>10} {'motivation':>12} {'curiosity':>12} "
          f"{'drive_str':>12} {'arousal':>10} routing_top_branch")
    print(f"  {'─'*75}")

    for fatigue_level in [0.0, 0.2, 0.4, 0.6, 0.8, 0.95]:
        s = HumanState(
            initial_values={"fatigue": fatigue_level, "curiosity": 0.6, "motivation": 0.6},
            noise_level=0.0,
        )
        s.update({})  # apply cross-effects

        forest = create_human_mode_forest()
        router = HumanModeRouter(top_k=3, noise_level=0.0)
        task = TaskInput(task_id="t", text="Analyze this", task_type="auto")
        decision, _ = router.route(task, forest, s)
        top_branch = max(decision.branch_scores, key=decision.branch_scores.get)

        print(f"  {fatigue_level:>10.2f} {s.get('motivation'):>12.3f} "
              f"{s.get('curiosity'):>12.3f} {s.drive_strength('curiosity'):>12.3f} "
              f"{s.arousal_level():>10.3f} {top_branch}")

    print("""
    Pattern:
    • fatigue > 0.7 triggers cross-effects: motivation ↓, curiosity ↓
    • drive_strength() applies fatigue penalty to ALL drives
    • Router penalizes high cognitive-cost branches under fatigue
    • At extreme fatigue, fast/low-cost branches take over
    """)

    # ══════════════════════════════════════════════════════════════════════
    # DEMO 6: Social events shape personality
    # ══════════════════════════════════════════════════════════════════════
    section("DEMO 6: SOCIAL SHAPING — Praise vs Rejection over time")

    print("\n  --- Agent A: Receives repeated praise ---")
    state_praised = HumanState(noise_level=0.0)
    print(f"  {'Event':<20} {'confidence':>12} {'trust':>8} {'empathy':>8} "
          f"{'self_prot':>10} {'mood':>8}")
    print(f"  {'─'*66}")
    print(f"  {'baseline':<20} {state_praised.get('confidence'):>12.3f} "
          f"{state_praised.get('trust'):>8.3f} {state_praised.get('empathy'):>8.3f} "
          f"{state_praised.get('self_protection'):>10.3f} {state_praised.mood_valence():>+8.3f}")

    for i in range(6):
        state_praised.inject_event("social_praise", intensity=0.7)
        print(f"  {'praise #' + str(i+1):<20} {state_praised.get('confidence'):>12.3f} "
              f"{state_praised.get('trust'):>8.3f} {state_praised.get('empathy'):>8.3f} "
              f"{state_praised.get('self_protection'):>10.3f} {state_praised.mood_valence():>+8.3f}")

    print("\n  --- Agent B: Receives repeated rejection ---")
    state_rejected = HumanState(noise_level=0.0)
    print(f"  {'Event':<20} {'confidence':>12} {'trust':>8} {'empathy':>8} "
          f"{'self_prot':>10} {'mood':>8}")
    print(f"  {'─'*66}")
    print(f"  {'baseline':<20} {state_rejected.get('confidence'):>12.3f} "
          f"{state_rejected.get('trust'):>8.3f} {state_rejected.get('empathy'):>8.3f} "
          f"{state_rejected.get('self_protection'):>10.3f} {state_rejected.mood_valence():>+8.3f}")

    for i in range(6):
        state_rejected.inject_event("social_rejection", intensity=0.7)
        print(f"  {'rejection #' + str(i+1):<20} {state_rejected.get('confidence'):>12.3f} "
              f"{state_rejected.get('trust'):>8.3f} {state_rejected.get('empathy'):>8.3f} "
              f"{state_rejected.get('self_protection'):>10.3f} {state_rejected.mood_valence():>+8.3f}")

    print("\n  Now both agents process the SAME task:")
    show_routing(state_praised, "Should I trust this person's advice?")
    print("    ^ Praised agent: high trust → empathy_social prominent")
    show_routing(state_rejected, "Should I trust this person's advice?")
    print("    ^ Rejected agent: low trust → self_protection prominent")

    # ══════════════════════════════════════════════════════════════════════
    # DEMO 7: Deadline pressure shifts cognition
    # ══════════════════════════════════════════════════════════════════════
    section("DEMO 7: TIME PRESSURE — Impulse overtakes reflection")

    print(f"\n  {'pressure':>10} {'impulse':>10} {'reflection':>12} {'stress':>8} "
          f"{'fatigue':>10} top_routing")
    print(f"  {'─'*65}")

    for pressure in [0.0, 0.3, 0.5, 0.7, 0.9]:
        s = HumanState(noise_level=0.0)
        if pressure > 0:
            s.inject_event("deadline_pressure", intensity=pressure)

        forest = create_human_mode_forest()
        router = HumanModeRouter(top_k=3, noise_level=0.0)
        task = TaskInput(task_id="t", text="Make a decision", task_type="auto")
        decision, _ = router.route(task, forest, s)
        top = sorted(decision.branch_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        top_names = [n for n, _ in top]

        print(f"  {pressure:>10.1f} {s.get('impulse'):>10.3f} "
              f"{s.get('reflection'):>12.3f} {s.get('stress'):>8.3f} "
              f"{s.get('fatigue'):>10.3f} {', '.join(top_names)}")

    print("""
    Pattern: as pressure increases:
    • impulse ↑ (act now!)
    • reflection ↓ (no time to think)
    • stress ↑ → further impairs reflection (cross-effect)
    • fatigue ↑ (pressure is exhausting)
    • Routing shifts from reflective → impulsive branches
    """)

    # ══════════════════════════════════════════════════════════════════════
    # DEMO 8: Homeostatic recovery
    # ══════════════════════════════════════════════════════════════════════
    section("DEMO 8: HOMEOSTATIC RECOVERY — Everything heals with time")

    state = HumanState(
        initial_values={"stress": 0.95, "fear": 0.85, "confidence": 0.05, "motivation": 0.05},
        noise_level=0.0, decay_rate=0.08,
    )
    print(f"\n  Starting from extreme negative state:")
    print(f"  {'turn':>6} {'stress':>8} {'fear':>8} {'confidence':>12} "
          f"{'motivation':>12} {'mood':>8}")
    print(f"  {'─'*54}")
    print(f"  {'start':>6} {state.get('stress'):>8.3f} {state.get('fear'):>8.3f} "
          f"{state.get('confidence'):>12.3f} {state.get('motivation'):>12.3f} "
          f"{state.mood_valence():>+8.3f}")

    for i in range(20):
        state.update({})  # just let time pass
        if (i + 1) % 4 == 0 or i == 0:
            print(f"  {i+1:>6} {state.get('stress'):>8.3f} {state.get('fear'):>8.3f} "
                  f"{state.get('confidence'):>12.3f} {state.get('motivation'):>12.3f} "
                  f"{state.mood_valence():>+8.3f}")

    print("""
    All variables gradually return toward their homeostatic baselines:
    • stress: 0.95 → ~0.20 (baseline)
    • fear: 0.85 → ~0.15 (baseline)
    • confidence: 0.05 → ~0.55 (baseline)
    • motivation: 0.05 → ~0.60 (baseline)
    Rate is controlled by decay_rate (0.08 = 8% per turn toward baseline).
    """)

    # ══════════════════════════════════════════════════════════════════════
    # DEMO 9: Full pipeline — same task, 4 emotional profiles
    # ══════════════════════════════════════════════════════════════════════
    section("DEMO 9: SAME TASK, FOUR EMOTIONAL PROFILES — Full pipeline comparison")

    task = "Should I invest heavily in this risky but promising opportunity?"

    profiles = {
        "Confident Explorer": {"confidence": 0.9, "curiosity": 0.85, "ambition": 0.8,
                                "fear": 0.1, "stress": 0.1},
        "Anxious Defender":   {"confidence": 0.15, "fear": 0.85, "stress": 0.8,
                                "self_protection": 0.8, "caution": 0.75, "trust": 0.2},
        "Torn / Conflicted":  {"curiosity": 0.8, "fear": 0.75, "ambition": 0.7,
                                "caution": 0.65},
        "Exhausted Worker":   {"fatigue": 0.9, "motivation": 0.2, "stress": 0.6,
                                "impulse": 0.5, "reflection": 0.2},
    }

    for profile_name, initial in profiles.items():
        print(f"\n  ┌{'─'*68}┐")
        print(f"  │  {profile_name:<66}│")
        print(f"  └{'─'*68}┘")

        orch = ModeOrchestrator(mode="human_mode", initial_state=initial)
        result = orch.run_task(task, task_type="auto")

        routing = result["routing"]
        ev = result["evaluation_signal"]
        state_after = result["human_state"]["after"]

        print(f"    Cognitive context:  {routing['task_type']}")
        print(f"    Activated branches: {routing['activated_branches']}")

        top_scores = sorted(routing["branch_scores"].items(), key=lambda x: x[1], reverse=True)[:4]
        for name, score in top_scores:
            print(f"      {name:30s} {score:.4f}")

        print(f"    Reward:       {ev['reward_score']:.3f}")
        print(f"    Selected:     {ev['selected_branch']}")
        print(f"    Mood after:   {state_after['mood_valence']:+.3f}")
        if result["conflicts"]:
            for c in result["conflicts"]:
                print(f"    Conflict:     {c['drive_a']} vs {c['drive_b']} → {c['resolution']}")

    print("""
    The same question produces four fundamentally different cognitive paths:
    • Explorer: curiosity_exploration leads, context = "exploration"
    • Defender: fear_risk leads, context = "threat_response"
    • Conflicted: both sides activate + conflict_resolver
    • Exhausted: avoids high-cost branches, impulse takes over
    """)

    section("DEMO COMPLETE")


if __name__ == "__main__":
    main()
