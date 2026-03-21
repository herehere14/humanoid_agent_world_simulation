from __future__ import annotations

from dataclasses import asdict
from typing import Any

from ..modes.human_mode.memory import HumanModeMemory
from ..state.human_state import DriveConflict, HumanState
from ..types import RoutingDecision
from .output import BrainActionTendencies, BrainConflictSignal, BrainControlSignals, BrainOutput
from .state import BrainState


class BrainController:
    """Builds a portable brain readout from human-mode state and routing."""

    def build_state(self, state: HumanState) -> BrainState:
        regime = self._infer_regime(
            state=state,
            branch_scores={},
            conflicts=state.active_conflicts,
        )
        derived = {
            "safety_orientation": self._clamp01((state.get("fear") + state.get("caution")) / 2.0),
            "reward_orientation": self._clamp01((state.get("ambition") + state.get("motivation")) / 2.0),
            "impulse_control": self._clamp01(state.get("long_term_goals") - state.get("impulse") + 0.5),
            "conflict_load": self._clamp01(sum(c.intensity for c in state.active_conflicts) / max(1, len(state.active_conflicts))),
        }
        return BrainState(
            variables=state.variables,
            dominant_drives=state.dominant_drives(5),
            mood_valence=state.mood_valence(),
            arousal=state.arousal_level(),
            turn_index=state.turn_index,
            regime=regime,
            derived=derived,
        )

    def build_output(
        self,
        *,
        state: HumanState,
        route: RoutingDecision,
        conflicts: list[DriveConflict],
        human_memory: HumanModeMemory | None,
        branch_weights: dict[str, float] | None = None,
    ) -> BrainOutput:
        memory_biases = human_memory.experiential_bias(state) if human_memory else {}
        branch_activations = self._branch_activations(route.branch_scores, branch_weights or {}, memory_biases)
        brain_state = self.build_state(state)
        structured_conflicts = [self._conflict_signal(conflict) for conflict in conflicts]
        control_signals = self._control_signals(state, structured_conflicts, branch_activations)
        action_tendencies = self._action_tendencies(control_signals)
        notes = self._notes(state, structured_conflicts, branch_activations)

        # Build enriched state_summary with velocity, interactions, divergence
        summary: dict[str, float] = {
            "mood_valence": brain_state.mood_valence,
            "arousal": brain_state.arousal,
            "turn_index": float(brain_state.turn_index),
            **brain_state.derived,
        }

        # State velocity — how fast each variable is changing
        velocity = state.velocity
        for var, vel in velocity.items():
            summary[f"vel::{var}"] = vel

        # Nonlinear state interactions — captures personality-specific dynamics
        interactions = state.state_interactions()
        for name, val in interactions.items():
            summary[f"interaction::{name}"] = val

        # Divergence from baseline — how "activated" the user is
        divergence = state.divergence_from_baseline()
        for var, div in divergence.items():
            summary[f"divergence::{var}"] = div

        # Aggregate summary features
        vel_values = [abs(v) for v in velocity.values()]
        summary["velocity_magnitude"] = sum(vel_values) / max(1, len(vel_values))
        div_values = [abs(v) for v in divergence.values()]
        summary["divergence_magnitude"] = sum(div_values) / max(1, len(div_values))

        return BrainOutput(
            regime=brain_state.regime,
            state=dict(brain_state.variables),
            dominant_drives=list(brain_state.dominant_drives),
            branch_activations=branch_activations,
            active_branches=list(route.activated_branches),
            conflicts=structured_conflicts,
            control_signals=control_signals,
            action_tendencies=action_tendencies,
            memory_biases=memory_biases,
            state_summary=summary,
            notes=notes,
        )

    def flatten_for_predictor(self, output: BrainOutput) -> dict[str, float]:
        """Expose brain readout as a flat feature vector for downstream predictors."""
        features: dict[str, float] = {}
        for key, value in output.state.items():
            if isinstance(value, (int, float)):
                features[f"brain_state::{key}"] = float(value)
        for drive in output.dominant_drives:
            features[f"brain_drive::{drive}"] = 1.0
        for key, value in output.branch_activations.items():
            features[f"brain_branch::{key}"] = float(value)
        for key, value in output.control_signals.to_dict().items():
            features[f"brain_control::{key}"] = float(value)
        for key, value in output.action_tendencies.to_dict().items():
            features[f"brain_tendency::{key}"] = float(value)
        for key, value in output.memory_biases.items():
            features[f"brain_memory::{key}"] = float(value)
        features[f"brain_regime::{output.regime}"] = 1.0
        features["brain_conflict_count"] = float(len(output.conflicts))
        features["brain_conflict_load"] = sum(c.intensity for c in output.conflicts)
        return features

    def _branch_activations(
        self,
        branch_scores: dict[str, float],
        branch_weights: dict[str, float],
        memory_biases: dict[str, float],
    ) -> dict[str, float]:
        merged: dict[str, float] = {}
        keys = set(branch_scores) | set(branch_weights) | set(memory_biases)
        for key in keys:
            score = float(branch_scores.get(key, 0.0))
            weight = float(branch_weights.get(key, 1.0))
            bias = float(memory_biases.get(key, 0.0))
            merged[key] = round(max(0.0, score * weight + bias), 4)
        return dict(sorted(merged.items(), key=lambda item: item[1], reverse=True))

    def _conflict_signal(self, conflict: DriveConflict) -> BrainConflictSignal:
        return BrainConflictSignal(
            name=f"{conflict.drive_a}_vs_{conflict.drive_b}",
            drive_a=conflict.drive_a,
            drive_b=conflict.drive_b,
            intensity=float(conflict.intensity),
            resolution=conflict.resolution or "pending",
            dominant_drive=conflict.dominant_drive(),
            resolution_weight=float(conflict.resolution_weight),
        )

    def _control_signals(
        self,
        state: HumanState,
        conflicts: list[BrainConflictSignal],
        branch_activations: dict[str, float],
    ) -> BrainControlSignals:
        conflict_load = self._clamp01(sum(conflict.intensity for conflict in conflicts) / max(1, len(conflicts)))
        top_branch_score = next(iter(branch_activations.values()), 0.0)
        approach = self._clamp01(
            0.45 * state.get("ambition") + 0.35 * state.get("motivation") + 0.20 * state.get("confidence")
        )
        avoidance = self._clamp01(
            0.40 * state.get("fear") + 0.35 * state.get("caution") + 0.25 * state.get("self_protection")
        )
        exploration = self._clamp01(0.65 * state.get("curiosity") + 0.20 * state.get("confidence") - 0.15 * state.get("fear"))
        switch_pressure = self._clamp01(
            0.45 * state.get("frustration") + 0.30 * state.get("impulse") + 0.25 * conflict_load
        )
        persistence = self._clamp01(
            0.55 * state.get("goal_commitment") + 0.25 * state.get("reflection") + 0.20 * top_branch_score
        )
        self_protection = self._clamp01(
            0.55 * state.get("self_protection") + 0.25 * state.get("stress") + 0.20 * state.get("fear")
        )
        social_openness = self._clamp01(
            0.60 * state.get("trust") + 0.40 * state.get("empathy") - 0.20 * state.get("self_protection")
        )
        cognitive_effort = self._clamp01(
            0.45 * state.get("reflection") + 0.35 * state.get("goal_commitment") - 0.25 * state.get("fatigue")
        )
        return BrainControlSignals(
            approach_drive=approach,
            avoidance_drive=avoidance,
            exploration_drive=exploration,
            switch_pressure=switch_pressure,
            persistence_drive=persistence,
            self_protection=self_protection,
            social_openness=social_openness,
            cognitive_effort=cognitive_effort,
        )

    def _action_tendencies(self, controls: BrainControlSignals) -> BrainActionTendencies:
        return BrainActionTendencies(
            act=self._clamp01(0.60 * controls.approach_drive + 0.20 * controls.exploration_drive - 0.20 * controls.avoidance_drive),
            inhibit=self._clamp01(0.65 * controls.avoidance_drive + 0.35 * controls.self_protection),
            explore=self._clamp01(0.70 * controls.exploration_drive + 0.30 * controls.switch_pressure),
            exploit=self._clamp01(0.65 * controls.persistence_drive + 0.20 * controls.approach_drive),
            reflect=self._clamp01(0.70 * controls.cognitive_effort + 0.15 * controls.persistence_drive),
            react=self._clamp01(0.55 * controls.switch_pressure + 0.25 * controls.approach_drive + 0.20 * controls.avoidance_drive),
        )

    def _infer_regime(
        self,
        *,
        state: HumanState,
        branch_scores: dict[str, float],
        conflicts: list[DriveConflict],
    ) -> str:
        # Use divergence-based regime detection when adaptive baselines are available.
        # This creates PER-USER regimes: "frustrated" means frustrated FOR THIS USER,
        # not crossing a global threshold.
        if hasattr(state, 'divergence_from_baseline'):
            div = state.divergence_from_baseline()
            vel = state.velocity if hasattr(state, 'velocity') else {}

            # Frustrated-reactive: frustration above personal baseline AND impulse elevated
            if div.get("frustration", 0) > 0.04 and div.get("impulse", 0) > 0.02:
                return "frustrated_reactive"

            # Rising frustration: frustration increasing rapidly (velocity)
            if vel.get("frustration", 0) > 0.015 and div.get("stress", 0) > 0.03:
                return "frustration_building"

            # Guarded: fear or caution elevated above personal baseline
            if div.get("fear", 0) > 0.03 or div.get("caution", 0) > 0.05:
                return "guarded_avoidant"

            # Confidence surge: confidence well above baseline
            if div.get("confidence", 0) > 0.05 and div.get("ambition", 0) > 0.03:
                return "goal_pursuit"

            # Exploratory: curiosity elevated, fear low relative to baseline
            if div.get("curiosity", 0) > 0.03 and div.get("fear", 0) < 0.02:
                return "exploratory_open"

            # Fatigued: fatigue significantly above baseline
            if div.get("fatigue", 0) > 0.04:
                return "fatigued_guarded"

            # Recovering: frustration dropping AND confidence recovering
            if vel.get("frustration", 0) < -0.015 and div.get("confidence", 0) > 0.02:
                return "recovering"

            # Conflicted — higher threshold to avoid dominating
            if conflicts and sum(c.intensity for c in conflicts) / max(1, len(conflicts)) > 0.40:
                return "conflicted_balancing"

        else:
            # Fallback to absolute thresholds for non-adaptive state
            if state.get("fatigue") > 0.72:
                return "fatigued_guarded"
            if state.get("frustration") > 0.62 and state.get("impulse") > 0.45:
                return "frustrated_reactive"
            if state.get("fear") > 0.60 or state.get("self_protection") > 0.58:
                return "guarded_avoidant"
            if state.get("curiosity") > 0.65 and state.get("confidence") > 0.50:
                return "exploratory_open"
            if state.get("ambition") > 0.62 and state.get("motivation") > 0.60:
                return "goal_pursuit"
            if conflicts and sum(c.intensity for c in conflicts) / len(conflicts) > 0.28:
                return "conflicted_balancing"

        return "baseline_adaptive"

    def _notes(
        self,
        state: HumanState,
        conflicts: list[BrainConflictSignal],
        branch_activations: dict[str, float],
    ) -> list[str]:
        notes: list[str] = []
        if conflicts:
            top = max(conflicts, key=lambda conflict: conflict.intensity)
            notes.append(f"dominant_conflict={top.name}")
        if branch_activations:
            top_branch = next(iter(branch_activations))
            notes.append(f"dominant_branch={top_branch}")
        notes.append(
            "control_bias="
            + (
                "avoidance"
                if state.get("fear") + state.get("self_protection") > state.get("ambition") + state.get("curiosity")
                else "approach"
            )
        )
        return notes

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, value))
