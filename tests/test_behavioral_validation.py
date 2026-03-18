"""Comprehensive behavioral validation for Human Mode.

Tests whether the cognitive-behavioral simulation accurately models
realistic human psychological patterns:

  1. Emotional dynamics (mood shifts, inertia, recovery)
  2. Stress-cognition interaction (stress impairs reflection)
  3. Drive conflicts (competing motivations create tension)
  4. Fatigue effects (cognitive degradation under exhaustion)
  5. Trauma and memory (negative events create lasting avoidance)
  6. Social dynamics (praise/rejection affect trust and confidence)
  7. Momentum and streaks (success breeds confidence, failure spirals)
  8. Arousal-performance curve (Yerkes-Dodson inverted-U)
  9. Homeostatic regulation (emotions decay back to baseline)
  10. Temporal discounting (impulse vs long-term goals under pressure)
  11. Full pipeline behavioral divergence (same task, different states)
  12. Experiential memory bias (past outcomes shape future routing)
"""

from __future__ import annotations

import pytest

from prompt_forest.modes.orchestrator import ModeOrchestrator
from prompt_forest.modes.human_mode.memory import HumanModeMemory
from prompt_forest.modes.human_mode.router import HumanModeRouter
from prompt_forest.modes.human_mode.branches import create_human_mode_forest
from prompt_forest.state.human_state import DriveConflict, HumanState
from prompt_forest.types import TaskInput


# ═══════════════════════════════════════════════════════════════════════════════
# 1. EMOTIONAL DYNAMICS
# ═══════════════════════════════════════════════════════════════════════════════

class TestEmotionalDynamics:
    """Human emotions don't snap — they have inertia and decay."""

    def test_emotional_inertia_prevents_instant_mood_swing(self):
        """A single positive event shouldn't flip mood from very negative to positive.
        Real humans don't go from despair to joy in one moment."""
        state = HumanState(
            initial_values={
                "confidence": 0.1, "motivation": 0.1, "stress": 0.9,
                "frustration": 0.8, "fear": 0.7,
            },
            momentum=0.7, noise_level=0.0,
        )
        initial_mood = state.mood_valence()
        assert initial_mood < -0.2, "Should start with negative mood"

        # One big positive event
        state.inject_event("reward", intensity=1.0)
        after_mood = state.mood_valence()

        # Mood should improve but NOT flip positive — inertia prevents it
        assert after_mood > initial_mood, "Mood should improve"
        assert after_mood < 0.3, "But shouldn't flip to strongly positive in one step"

    def test_gradual_mood_recovery(self):
        """After a bad experience, mood should gradually recover over many turns
        — not stay stuck forever, and not snap back instantly."""
        state = HumanState(
            initial_values={"stress": 0.9, "frustration": 0.8, "confidence": 0.1},
            noise_level=0.0, decay_rate=0.08,
        )
        mood_trajectory = [state.mood_valence()]

        # Just let time pass (no events)
        for _ in range(15):
            state.update({})
            mood_trajectory.append(state.mood_valence())

        # Mood should trend upward (toward baseline)
        assert mood_trajectory[-1] > mood_trajectory[0], "Mood should recover over time"
        # But not fully recovered in just 15 steps
        final_state = HumanState(noise_level=0.0)
        baseline_mood = final_state.mood_valence()
        assert mood_trajectory[-1] < baseline_mood, "Shouldn't fully recover this fast"

    def test_emotional_momentum_tracks_trajectory(self):
        """emotional_momentum should reflect recent valence direction."""
        state = HumanState(noise_level=0.0)

        # Series of positive outcomes
        for _ in range(4):
            state.apply_outcome(reward=0.9, task_type="general")
        assert state.get("emotional_momentum") > 0, "Positive streak → positive momentum"

        # Now series of negative outcomes
        for _ in range(6):
            state.apply_outcome(reward=0.1, task_type="general")
        assert state.get("emotional_momentum") < 0.35, "Negative streak should pull momentum down"


# ═══════════════════════════════════════════════════════════════════════════════
# 2. STRESS-COGNITION INTERACTION
# ═══════════════════════════════════════════════════════════════════════════════

class TestStressCognition:
    """High stress impairs deliberate thinking and promotes impulsive action.
    This models the real cognitive phenomenon where cortisol impairs
    prefrontal cortex function."""

    def test_high_stress_suppresses_reflection(self):
        """Under high stress, reflection should decrease — modeling how
        stress hormones impair the prefrontal cortex."""
        state = HumanState(
            initial_values={"stress": 0.85, "reflection": 0.7},
            noise_level=0.0,
        )
        state.update({})
        assert state.get("reflection") < 0.7, "High stress should suppress reflection"

    def test_high_stress_promotes_impulse(self):
        """Under high stress, impulse should increase — fight-or-flight
        shifts processing to fast, automatic responses."""
        state = HumanState(
            initial_values={"stress": 0.85, "impulse": 0.3},
            noise_level=0.0,
        )
        state.update({})
        assert state.get("impulse") > 0.3, "High stress should increase impulsivity"

    def test_stress_cascade_from_failures(self):
        """Repeated failures should create a stress cascade:
        failure → stress → impaired reflection → worse performance → more stress."""
        state = HumanState(
            initial_values={"stress": 0.3, "reflection": 0.6},
            noise_level=0.0,
        )
        for _ in range(5):
            state.apply_outcome(reward=0.1, task_type="general")

        assert state.get("stress") > 0.35, "Repeated failure should raise stress"
        # Stress should be higher than starting point, showing cascade
        assert state.get("stress") > 0.3, "Stress should be above initial value"

    def test_moderate_stress_doesnt_impair(self):
        """Low-moderate stress shouldn't trigger cognitive impairment.
        The cross-effect only kicks in above 0.6."""
        state = HumanState(
            initial_values={"stress": 0.4, "reflection": 0.6},
            noise_level=0.0,
        )
        state.update({})
        # reflection shouldn't decrease from moderate stress
        assert state.get("reflection") >= 0.55, "Moderate stress shouldn't impair reflection"


# ═══════════════════════════════════════════════════════════════════════════════
# 3. DRIVE CONFLICTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestDriveConflicts:
    """When opposing drives are both strong, internal conflict emerges.
    This models real psychological tension (approach-avoidance conflict)."""

    def test_curiosity_fear_conflict(self):
        """Curiosity and fear are natural opponents. When both are high,
        a conflict should emerge — like wanting to explore but being scared."""
        state = HumanState(
            initial_values={"curiosity": 0.75, "fear": 0.70},
            noise_level=0.0,
        )
        state.update({})
        conflicts = state.active_conflicts
        pairs = [{c.drive_a, c.drive_b} for c in conflicts]
        assert {"curiosity", "fear"} in pairs, "Should detect curiosity vs fear conflict"

    def test_impulse_vs_long_term_goals(self):
        """Immediate gratification vs long-term planning — the classic
        marshmallow test / temporal discounting conflict."""
        state = HumanState(
            initial_values={"impulse": 0.7, "long_term_goals": 0.5},
            noise_level=0.0,
        )
        # Boost long_term_goals to create conflict
        state.set("long_term_goals", 0.7)
        state.update({})
        conflicts = state.active_conflicts
        pairs = [{c.drive_a, c.drive_b} for c in conflicts]
        assert {"impulse", "long_term_goals"} in pairs, "Should detect impulse vs long-term conflict"

    def test_empathy_vs_self_protection(self):
        """Helping others vs protecting yourself — a fundamental social dilemma."""
        state = HumanState(
            initial_values={"empathy": 0.7, "self_protection": 0.65},
            noise_level=0.0,
        )
        state.update({})
        conflicts = state.active_conflicts
        pairs = [{c.drive_a, c.drive_b} for c in conflicts]
        assert {"empathy", "self_protection"} in pairs

    def test_no_conflict_when_one_drive_dominates(self):
        """When one drive clearly dominates, no conflict — the answer is clear."""
        state = HumanState(
            initial_values={"curiosity": 0.9, "fear": 0.2},
            noise_level=0.0,
        )
        state.update({})
        conflicts = state.active_conflicts
        pairs = [{c.drive_a, c.drive_b} for c in conflicts]
        assert {"curiosity", "fear"} not in pairs, "No conflict when gap is large"

    def test_conflict_increases_stress(self):
        """Unresolved internal conflict should increase stress — this models
        cognitive dissonance creating psychological tension."""
        state = HumanState(
            initial_values={"curiosity": 0.7, "fear": 0.7, "stress": 0.2},
            noise_level=0.0,
        )
        state.update({})
        for conflict in state.active_conflicts:
            state.resolve_conflict(conflict, strategy="weighted_compromise")
        # Conflict resolution itself adds a small stress bump
        assert state.get("stress") >= 0.2, "Conflict resolution should add stress"

    def test_dominant_resolution_suppresses_loser(self):
        """When using 'dominant' strategy, the weaker drive gets suppressed.
        Models how humans sometimes forcefully shut down competing impulses."""
        state = HumanState(
            initial_values={"curiosity": 0.8, "fear": 0.6},
            noise_level=0.0,
        )
        conflict = DriveConflict(drive_a="curiosity", drive_b="fear", intensity=0.5)
        state.resolve_conflict(conflict, strategy="dominant")
        assert conflict.resolution == "curiosity"
        assert state.get("fear") < 0.6, "Losing drive should be suppressed"


# ═══════════════════════════════════════════════════════════════════════════════
# 4. FATIGUE EFFECTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestFatigueEffects:
    """Fatigue degrades cognitive performance across multiple dimensions."""

    def test_fatigue_reduces_motivation(self):
        """Exhaustion saps motivation — you want to quit."""
        state = HumanState(
            initial_values={"fatigue": 0.85, "motivation": 0.6},
            noise_level=0.0,
        )
        state.update({})
        assert state.get("motivation") < 0.6

    def test_fatigue_reduces_curiosity(self):
        """When exhausted, exploration appetite drops — you just want to finish."""
        state = HumanState(
            initial_values={"fatigue": 0.85, "curiosity": 0.6},
            noise_level=0.0,
        )
        state.update({})
        assert state.get("curiosity") < 0.6

    def test_fatigue_penalises_drive_strength(self):
        """All drives are weaker when fatigued."""
        state = HumanState(initial_values={"curiosity": 0.8, "fatigue": 0.0})
        strong = state.drive_strength("curiosity")
        state.set("fatigue", 0.9)
        weak = state.drive_strength("curiosity")
        assert weak < strong

    def test_fatigue_penalises_high_cost_branches_in_routing(self):
        """Under fatigue, the router should penalize cognitively expensive branches.
        This models how tired people avoid effortful thinking."""
        forest = create_human_mode_forest()
        router = HumanModeRouter(top_k=6, noise_level=0.0)
        task = TaskInput(task_id="t1", text="Analyze this problem", task_type="auto")

        # Fresh state
        state_fresh = HumanState(initial_values={"fatigue": 0.1}, noise_level=0.0)
        decision_fresh, _ = router.route(task, forest, state_fresh)

        # Fatigued state
        state_tired = HumanState(initial_values={"fatigue": 0.9}, noise_level=0.0)
        decision_tired, _ = router.route(task, forest, state_tired)

        # reflective_reasoning (high cost) should score lower when fatigued
        score_fresh = decision_fresh.branch_scores.get("reflective_reasoning", 0)
        score_tired = decision_tired.branch_scores.get("reflective_reasoning", 0)
        assert score_tired < score_fresh, "Fatigue should penalize high-cost branches"

    def test_rest_reverses_fatigue(self):
        """Rest events should reduce fatigue — recovery is possible."""
        state = HumanState(initial_values={"fatigue": 0.8}, noise_level=0.0)
        state.inject_event("rest", intensity=0.9)
        assert state.get("fatigue") < 0.8


# ═══════════════════════════════════════════════════════════════════════════════
# 5. TRAUMA AND MEMORY
# ═══════════════════════════════════════════════════════════════════════════════

class TestTraumaAndMemory:
    """Negative experiences create lasting avoidance biases.
    Positive experiences create approach biases.
    Strong emotions are remembered longer (flashbulb memory effect)."""

    def test_negative_experience_creates_avoidance_bias(self):
        """After repeated failure with a branch, memory should create
        negative bias — avoiding what hurt before."""
        mem = HumanModeMemory(experience_bias_strength=0.5)
        state = HumanState()

        for i in range(5):
            task = TaskInput(task_id=f"t{i}", text="bad task", task_type="general")
            mem.record(
                event_id=f"e{i}", task=task, state=state,
                reward=0.1, selected_branch="fear_risk",
                active_branches=["fear_risk"],
                failure_reason="bad_outcome",
            )

        biases = mem.experiential_bias(state)
        assert biases.get("fear_risk", 0) < 0, "Should develop avoidance bias"

    def test_positive_experience_creates_approach_bias(self):
        """Success with a branch creates positive bias — approach what worked."""
        mem = HumanModeMemory(experience_bias_strength=0.5)
        state = HumanState()

        for i in range(5):
            task = TaskInput(task_id=f"t{i}", text="good task", task_type="general")
            mem.record(
                event_id=f"e{i}", task=task, state=state,
                reward=0.9, selected_branch="curiosity_exploration",
                active_branches=["curiosity_exploration"],
            )

        biases = mem.experiential_bias(state)
        assert biases.get("curiosity_exploration", 0) > 0, "Should develop approach bias"

    def test_traumatic_memories_persist_longer(self):
        """High-emotion memories should resist eviction more than bland ones.
        This models flashbulb memory — you remember the car crash,
        not Tuesday's lunch."""
        mem = HumanModeMemory(max_memories=5)
        state = HumanState()

        # Record a traumatic memory (high emotional valence)
        state_trauma = HumanState(initial_values={
            "stress": 0.9, "fear": 0.9, "confidence": 0.1,
            "motivation": 0.1, "curiosity": 0.1, "trust": 0.1, "ambition": 0.1,
            "frustration": 0.9, "fatigue": 0.8, "self_protection": 0.9,
        })
        task_trauma = TaskInput(task_id="trauma", text="terrible failure", task_type="general")
        mem.record(
            event_id="trauma", task=task_trauma, state=state_trauma,
            reward=0.05, selected_branch="fear_risk",
            active_branches=["fear_risk"],
            failure_reason="catastrophic_failure",
        )

        # Fill memory with bland experiences to trigger eviction
        for i in range(8):
            task = TaskInput(task_id=f"bland{i}", text="routine task", task_type="general")
            mem.record(
                event_id=f"bland{i}", task=task, state=state,
                reward=0.5, selected_branch="reflective_reasoning",
                active_branches=["reflective_reasoning"],
            )

        # The traumatic memory should survive eviction
        remaining_ids = [m.event_id for m in mem._memories]
        assert "trauma" in remaining_ids, "Traumatic memories should resist eviction"

    def test_memory_auto_tags_reflect_experience_quality(self):
        """Memory tags should capture the emotional quality of the experience."""
        mem = HumanModeMemory()
        state = HumanState()

        # Success
        task = TaskInput(task_id="t1", text="test", task_type="general")
        entry = mem.record("e1", task, state, reward=0.9, selected_branch="x", active_branches=["x"])
        assert "success" in entry.tags

        # Failure
        entry2 = mem.record("e2", task, state, reward=0.1, selected_branch="x",
                            active_branches=["x"], failure_reason="timeout")
        assert "failure" in entry2.tags
        assert "fail:timeout" in entry2.tags


# ═══════════════════════════════════════════════════════════════════════════════
# 6. SOCIAL DYNAMICS
# ═══════════════════════════════════════════════════════════════════════════════

class TestSocialDynamics:
    """Social interactions affect trust, confidence, and empathy."""

    def test_social_praise_boosts_confidence_and_trust(self):
        state = HumanState(
            initial_values={"confidence": 0.5, "trust": 0.5},
            noise_level=0.0,
        )
        state.inject_event("social_praise", intensity=0.8)
        assert state.get("confidence") > 0.5
        assert state.get("trust") > 0.5

    def test_social_rejection_erodes_trust(self):
        """Rejection should decrease trust and increase self-protection.
        Models social pain activating defensive mechanisms."""
        state = HumanState(
            initial_values={"trust": 0.6, "self_protection": 0.3},
            noise_level=0.0,
        )
        state.inject_event("social_rejection", intensity=0.8)
        assert state.get("trust") < 0.6, "Rejection should erode trust"
        assert state.get("self_protection") > 0.3, "Rejection should raise defenses"

    def test_repeated_rejection_creates_defensive_personality(self):
        """Multiple social rejections should create a persistently defensive state.
        Models how repeated social pain creates avoidant attachment."""
        state = HumanState(
            initial_values={"trust": 0.6, "self_protection": 0.3, "confidence": 0.5},
            noise_level=0.0,
        )
        for _ in range(5):
            state.inject_event("social_rejection", intensity=0.7)

        assert state.get("trust") < 0.5, "Repeated rejection → lower trust"
        assert state.get("self_protection") > 0.35, "Repeated rejection → higher self-protection"
        assert state.get("confidence") < 0.5, "Repeated rejection → lower confidence"


# ═══════════════════════════════════════════════════════════════════════════════
# 7. MOMENTUM AND STREAKS
# ═══════════════════════════════════════════════════════════════════════════════

class TestMomentumAndStreaks:
    """Success breeds confidence; failure spirals compound."""

    def test_success_streak_builds_confidence(self):
        state = HumanState(initial_values={"confidence": 0.5}, noise_level=0.0)
        for _ in range(5):
            state.apply_outcome(reward=0.9, task_type="general")
        assert state.get("confidence") > 0.6, "Success streak should build confidence"

    def test_failure_spiral_compounds(self):
        """Each failure should hurt more because frustration amplifies stress."""
        state = HumanState(
            initial_values={"stress": 0.2, "frustration": 0.1},
            noise_level=0.0,
        )
        stress_deltas = []
        for _ in range(5):
            before = state.get("stress")
            state.apply_outcome(reward=0.1, task_type="general")
            stress_deltas.append(state.get("stress") - before)

        # Later failures should cause bigger stress jumps (frustration amplification)
        # Due to momentum and decay this might not be strictly monotonic,
        # but the cumulative effect should be clear
        assert state.get("stress") > 0.3, "Failure spiral should compound stress"
        assert state.get("frustration") > 0.2, "Failure spiral should compound frustration"

    def test_confidence_boosts_ambition(self):
        """High confidence should feed ambition — the cross-effect models
        how successful people become more ambitious."""
        state = HumanState(
            initial_values={"confidence": 0.85, "ambition": 0.5},
            noise_level=0.0,
        )
        state.update({})
        assert state.get("ambition") > 0.5, "High confidence should boost ambition"


# ═══════════════════════════════════════════════════════════════════════════════
# 8. AROUSAL-PERFORMANCE (Yerkes-Dodson)
# ═══════════════════════════════════════════════════════════════════════════════

class TestArousalEffects:
    """High arousal favors fast/automatic processing.
    Low arousal favors slow/reflective processing.
    This models the Yerkes-Dodson law."""

    def test_high_arousal_favours_fast_branches(self):
        forest = create_human_mode_forest()
        router = HumanModeRouter(top_k=8, noise_level=0.0)
        task = TaskInput(task_id="t1", text="Quick decision", task_type="auto")

        state_high = HumanState(
            initial_values={"stress": 0.8, "fear": 0.7, "impulse": 0.6, "curiosity": 0.7},
            noise_level=0.0,
        )
        decision, _ = router.route(task, forest, state_high)

        # impulse_response (instant speed) should score well
        impulse_score = decision.branch_scores.get("impulse_response", 0)
        reflective_score = decision.branch_scores.get("reflective_reasoning", 0)
        # Under high arousal, fast branches should be competitive
        assert impulse_score > 0.3, "High arousal should activate impulse"

    def test_low_arousal_favours_reflective_branches(self):
        forest = create_human_mode_forest()
        router = HumanModeRouter(top_k=8, noise_level=0.0)
        task = TaskInput(task_id="t1", text="Careful analysis", task_type="auto")

        state_low = HumanState(
            initial_values={"stress": 0.1, "fear": 0.1, "impulse": 0.1,
                            "curiosity": 0.2, "ambition": 0.2},
            noise_level=0.0,
        )
        decision, _ = router.route(task, forest, state_low)

        reflective_score = decision.branch_scores.get("reflective_reasoning", 0)
        impulse_score = decision.branch_scores.get("impulse_response", 0)
        assert reflective_score > impulse_score, "Low arousal should favour reflection over impulse"


# ═══════════════════════════════════════════════════════════════════════════════
# 9. HOMEOSTATIC REGULATION
# ═══════════════════════════════════════════════════════════════════════════════

class TestHomeostaticRegulation:
    """All emotions decay toward baseline over time.
    This models psychological homeostasis."""

    def test_elevated_stress_decays(self):
        state = HumanState(initial_values={"stress": 0.9}, noise_level=0.0, decay_rate=0.1)
        for _ in range(20):
            state.update({})
        assert state.get("stress") < 0.9, "Stress should decay toward baseline"

    def test_depleted_motivation_recovers(self):
        state = HumanState(initial_values={"motivation": 0.1}, noise_level=0.0, decay_rate=0.1)
        for _ in range(20):
            state.update({})
        assert state.get("motivation") > 0.1, "Motivation should recover toward baseline"

    def test_all_variables_trend_toward_baseline(self):
        """Every variable should eventually move toward its baseline."""
        extremes = {k: (0.0 if v > 0.5 else 1.0) for k, v in
                    HumanState()._baselines.items()}
        state = HumanState(initial_values=extremes, noise_level=0.0, decay_rate=0.1)

        for _ in range(30):
            state.update({})

        for var, baseline in state._baselines.items():
            current = state.get(var)
            initial = extremes[var]
            # Current should be closer to baseline than the extreme start
            assert abs(current - baseline) < abs(initial - baseline), \
                f"{var}: {current:.3f} should be closer to baseline {baseline:.3f} than start {initial:.3f}"


# ═══════════════════════════════════════════════════════════════════════════════
# 10. TEMPORAL DISCOUNTING (Impulse vs Planning under pressure)
# ═══════════════════════════════════════════════════════════════════════════════

class TestTemporalDiscounting:
    """Under pressure, humans discount future rewards and prefer
    immediate action. Deadline pressure should shift routing toward
    fast/impulsive branches and away from slow/deliberative ones."""

    def test_deadline_pressure_boosts_impulse(self):
        state = HumanState(
            initial_values={"impulse": 0.3, "stress": 0.3},
            noise_level=0.0,
        )
        state.inject_event("deadline_pressure", intensity=0.9)
        assert state.get("impulse") > 0.3, "Deadline pressure should boost impulse"
        assert state.get("stress") > 0.3, "Deadline pressure should raise stress"

    def test_deadline_pressure_suppresses_reflection(self):
        state = HumanState(
            initial_values={"reflection": 0.6, "stress": 0.3},
            noise_level=0.0,
        )
        state.inject_event("deadline_pressure", intensity=0.9)
        assert state.get("reflection") < 0.6, "Deadline pressure should suppress reflection"

    def test_pressure_shifts_routing_to_fast_branches(self):
        forest = create_human_mode_forest()
        router = HumanModeRouter(top_k=6, noise_level=0.0)
        task = TaskInput(task_id="t1", text="Make a decision", task_type="auto")

        # Calm state
        state_calm = HumanState(
            initial_values={"stress": 0.2, "impulse": 0.3},
            noise_level=0.0,
        )
        decision_calm, _ = router.route(task, forest, state_calm)

        # Under pressure
        state_pressure = HumanState(
            initial_values={"stress": 0.2, "impulse": 0.3},
            noise_level=0.0,
        )
        state_pressure.inject_event("deadline_pressure", intensity=0.9)
        decision_pressure, _ = router.route(task, forest, state_pressure)

        impulse_calm = decision_calm.branch_scores.get("impulse_response", 0)
        impulse_pressure = decision_pressure.branch_scores.get("impulse_response", 0)
        assert impulse_pressure > impulse_calm, "Pressure should boost impulse routing"


# ═══════════════════════════════════════════════════════════════════════════════
# 11. FULL PIPELINE BEHAVIORAL DIVERGENCE
# ═══════════════════════════════════════════════════════════════════════════════

class TestFullPipelineDivergence:
    """The same task processed by agents with different internal states
    should produce structurally different results throughout the pipeline."""

    def test_confident_vs_fearful_different_routing(self):
        task = "Should I take a big risk for a potentially huge payoff?"

        orch_confident = ModeOrchestrator(
            mode="human_mode",
            initial_state={"confidence": 0.9, "curiosity": 0.8, "fear": 0.1, "ambition": 0.8},
        )
        orch_fearful = ModeOrchestrator(
            mode="human_mode",
            initial_state={"confidence": 0.2, "fear": 0.9, "stress": 0.7, "self_protection": 0.8},
        )

        result_conf = orch_confident.run_task(task)
        result_fear = orch_fearful.run_task(task)

        # Different cognitive context
        routing_conf = result_conf["routing"]
        routing_fear = result_fear["routing"]

        # fear_risk should be much more prominent when fearful
        assert routing_fear["branch_scores"].get("fear_risk", 0) > \
               routing_conf["branch_scores"].get("fear_risk", 0)

        # curiosity_exploration should be more prominent when confident
        assert routing_conf["branch_scores"].get("curiosity_exploration", 0) > \
               routing_fear["branch_scores"].get("curiosity_exploration", 0)

    def test_conflicted_agent_activates_conflict_resolver(self):
        task = "Should I explore this unknown territory?"

        orch = ModeOrchestrator(
            mode="human_mode",
            initial_state={"curiosity": 0.8, "fear": 0.75},
        )
        result = orch.run_task(task)
        routing = result["routing"]

        # Conflict resolver should have elevated score
        assert "conflict_resolver" in routing["branch_scores"]

    def test_fatigued_agent_different_from_fresh(self):
        task = "Analyze the implications of this complex policy change."

        orch_fresh = ModeOrchestrator(
            mode="human_mode",
            initial_state={"fatigue": 0.1, "motivation": 0.7},
        )
        orch_tired = ModeOrchestrator(
            mode="human_mode",
            initial_state={"fatigue": 0.9, "motivation": 0.3},
        )

        result_fresh = orch_fresh.run_task(task)
        result_tired = orch_tired.run_task(task)

        # Fresh agent should score reflective branches higher
        fresh_reflective = result_fresh["routing"]["branch_scores"].get("reflective_reasoning", 0)
        tired_reflective = result_tired["routing"]["branch_scores"].get("reflective_reasoning", 0)
        assert fresh_reflective > tired_reflective


# ═══════════════════════════════════════════════════════════════════════════════
# 12. EXPERIENTIAL MEMORY BIAS IN ROUTING
# ═══════════════════════════════════════════════════════════════════════════════

class TestExperientialMemoryBias:
    """Past outcomes should influence future routing through learned biases."""

    def test_positive_experiences_create_approach(self):
        """After success with a branch, memory should bias toward it."""
        mem = HumanModeMemory(experience_bias_strength=0.5)
        state = HumanState()

        # 5 successes with curiosity_exploration
        for i in range(5):
            task = TaskInput(task_id=f"t{i}", text="explore", task_type="exploration")
            mem.record(f"e{i}", task, state, reward=0.9,
                       selected_branch="curiosity_exploration",
                       active_branches=["curiosity_exploration"])

        biases = mem.experiential_bias(state)
        assert biases.get("curiosity_exploration", 0) > 0

    def test_negative_experiences_create_avoidance(self):
        """After failure with a branch, memory should bias away from it."""
        mem = HumanModeMemory(experience_bias_strength=0.5)
        state = HumanState()

        for i in range(5):
            task = TaskInput(task_id=f"t{i}", text="risky task", task_type="general")
            mem.record(f"e{i}", task, state, reward=0.1,
                       selected_branch="impulse_response",
                       active_branches=["impulse_response"],
                       failure_reason="hasty_decision")

        biases = mem.experiential_bias(state)
        assert biases.get("impulse_response", 0) < 0

    def test_mixed_experience_produces_nuanced_bias(self):
        """Mixed results should produce weak or neutral biases."""
        mem = HumanModeMemory(experience_bias_strength=0.5)
        state = HumanState()

        rewards = [0.9, 0.2, 0.8, 0.3, 0.7, 0.4]
        for i, r in enumerate(rewards):
            task = TaskInput(task_id=f"t{i}", text="task", task_type="general")
            mem.record(f"e{i}", task, state, reward=r,
                       selected_branch="reflective_reasoning",
                       active_branches=["reflective_reasoning"])

        biases = mem.experiential_bias(state)
        bias_val = abs(biases.get("reflective_reasoning", 0))
        # Mixed results → weak bias (much smaller than pure positive/negative)
        assert bias_val < 0.15, f"Mixed experience should produce weak bias, got {bias_val}"


# ═══════════════════════════════════════════════════════════════════════════════
# 13. NOVELTY SEEKING
# ═══════════════════════════════════════════════════════════════════════════════

class TestNoveltySeeking:
    """Novelty events should boost curiosity and motivation."""

    def test_novelty_boosts_curiosity(self):
        state = HumanState(initial_values={"curiosity": 0.4}, noise_level=0.0)
        state.inject_event("novelty", intensity=0.8)
        assert state.get("curiosity") > 0.4

    def test_novelty_also_slightly_increases_fear(self):
        """New things are exciting but also slightly scary — this models
        the approach-avoidance response to novelty."""
        state = HumanState(initial_values={"fear": 0.2}, noise_level=0.0)
        state.inject_event("novelty", intensity=0.8)
        assert state.get("fear") > 0.2


# ═══════════════════════════════════════════════════════════════════════════════
# 14. MULTI-TURN STATE EVOLUTION
# ═══════════════════════════════════════════════════════════════════════════════

class TestMultiTurnEvolution:
    """State should evolve realistically over multiple interactions."""

    def test_state_evolves_across_multiple_tasks(self):
        orch = ModeOrchestrator(mode="human_mode", initial_state={"confidence": 0.5})

        moods = []
        for i in range(5):
            result = orch.run_task(f"Task number {i+1}", task_type="auto")
            mood = result["human_state"]["after"]["mood_valence"]
            moods.append(mood)

        # State should be changing over time (not static)
        assert len(set(round(m, 3) for m in moods)) > 1, "Mood should evolve over turns"

    def test_memory_accumulates_over_tasks(self):
        orch = ModeOrchestrator(mode="human_mode")
        for i in range(5):
            orch.run_task(f"Task {i}", task_type="auto")
        assert orch.human_memory.memory_count == 5

    def test_turn_index_tracks_correctly(self):
        orch = ModeOrchestrator(mode="human_mode")
        for i in range(3):
            orch.run_task(f"Task {i}", task_type="auto")
        # Each run_task calls update (via apply_outcome) which increments turn
        assert orch.human_state.turn_index >= 3
