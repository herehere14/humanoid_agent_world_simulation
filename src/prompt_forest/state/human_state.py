"""Persistent internal state model for Human Mode.

Models structured, updateable psychological-state variables that influence
routing, evaluation, and branch selection.  This is NOT a claim of
consciousness -- it is a modular cognitive-behavioral state architecture.

Key design principles:
  - State variables have bounded ranges [0, 1].
  - Variables exhibit *momentum* (emotional inertia) so changes propagate
    gradually rather than snapping to new values.
  - Competing drives can create *conflicts* that the routing layer must
    resolve, rather than simply picking the single best branch.
  - State decays toward a homeostatic baseline over time (fatigue fades,
    stress dissipates) unless reinforced.

The state is fully serialisable so it can persist across sessions.
"""

from __future__ import annotations

import math
import random
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any


# ──────────────────────────────────────────────────────────────────────────────
# Drive definitions
# ──────────────────────────────────────────────────────────────────────────────

# Each drive has an opponent.  When two opposing drives are both high, a
# conflict emerges that routing must resolve.
DRIVE_OPPOSITIONS: dict[str, str] = {
    "curiosity": "fear",
    "fear": "curiosity",
    "impulse": "long_term_goals",
    "long_term_goals": "impulse",
    "empathy": "self_protection",
    "self_protection": "empathy",
    "ambition": "caution",
    "caution": "ambition",
    "honesty": "self_justification",
    "self_justification": "honesty",
    "reflection": "impulse",
}

# Homeostatic baselines -- the resting value each variable decays toward.
DEFAULT_BASELINES: dict[str, float] = {
    "confidence": 0.55,
    "stress": 0.20,
    "frustration": 0.10,
    "trust": 0.50,
    "fatigue": 0.15,
    "curiosity": 0.60,
    "fear": 0.15,
    "motivation": 0.60,
    "emotional_momentum": 0.0,
    "goal_commitment": 0.65,
    "empathy": 0.50,
    "ambition": 0.55,
    "caution": 0.35,
    "honesty": 0.60,
    "self_protection": 0.30,
    "self_justification": 0.25,
    "impulse": 0.30,
    "reflection": 0.50,
}


@dataclass
class DriveConflict:
    """Represents an active conflict between two competing drives."""

    drive_a: str
    drive_b: str
    intensity: float  # 0-1, how strong the tension is
    resolution: str = ""  # which drive won, or "compromise"
    resolution_weight: float = 0.5  # 0 = drive_a wins, 1 = drive_b wins

    def dominant_drive(self) -> str:
        return self.drive_b if self.resolution_weight > 0.5 else self.drive_a

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StateSnapshot:
    """Immutable snapshot of the internal state at a point in time."""

    variables: dict[str, float]
    active_conflicts: list[DriveConflict]
    dominant_drives: list[str]
    turn_index: int
    mood_valence: float  # -1 (negative) to +1 (positive)

    def to_dict(self) -> dict[str, Any]:
        return {
            "variables": dict(self.variables),
            "active_conflicts": [c.to_dict() for c in self.active_conflicts],
            "dominant_drives": list(self.dominant_drives),
            "turn_index": self.turn_index,
            "mood_valence": self.mood_valence,
        }


class HumanState:
    """Persistent, mutable internal state for Human Mode.

    This object is designed to live across multiple turns/episodes and be
    mutated by the routing, evaluation, and memory systems.

    Parameters
    ----------
    initial_values:
        Override specific state variables at construction time.
    decay_rate:
        Per-turn decay toward homeostatic baselines (default 0.05).
    momentum:
        Inertia coefficient -- high values mean state changes slowly.
    noise_level:
        Simulated bounded rationality; adds small perturbations.
    """

    def __init__(
        self,
        initial_values: dict[str, float] | None = None,
        decay_rate: float = 0.05,
        momentum: float = 0.7,
        noise_level: float = 0.08,
    ) -> None:
        self._baselines = dict(DEFAULT_BASELINES)
        self._variables: dict[str, float] = dict(self._baselines)
        if initial_values:
            for k, v in initial_values.items():
                self._variables[k] = self._clamp(v)
        self._decay_rate = decay_rate
        self._momentum = momentum
        self._noise_level = noise_level
        self._turn_index = 0
        self._history: list[StateSnapshot] = []
        self._active_conflicts: list[DriveConflict] = []

    # ── Public API ────────────────────────────────────────────────────────

    def get(self, variable: str) -> float:
        return self._variables.get(variable, 0.5)

    def set(self, variable: str, value: float) -> None:
        self._variables[variable] = self._clamp(value)

    @property
    def variables(self) -> dict[str, float]:
        return dict(self._variables)

    @property
    def turn_index(self) -> int:
        return self._turn_index

    @property
    def active_conflicts(self) -> list[DriveConflict]:
        return list(self._active_conflicts)

    def snapshot(self) -> StateSnapshot:
        return StateSnapshot(
            variables=dict(self._variables),
            active_conflicts=list(self._active_conflicts),
            dominant_drives=self.dominant_drives(top_k=3),
            turn_index=self._turn_index,
            mood_valence=self.mood_valence(),
        )

    # ── State mutation ────────────────────────────────────────────────────

    def update(self, deltas: dict[str, float]) -> StateSnapshot:
        """Apply deltas with momentum and noise, then detect conflicts.

        This is the primary mutation method called after each turn.  Deltas
        are blended with the current value using the momentum coefficient
        so that state changes are gradual.

        Returns a snapshot *after* the update.
        """
        self._turn_index += 1

        for var, delta in deltas.items():
            old = self._variables.get(var, self._baselines.get(var, 0.5))
            # Momentum: blend old value with target
            target = old + delta
            new = old * self._momentum + target * (1.0 - self._momentum)
            # Add bounded-rationality noise
            if self._noise_level > 0:
                new += random.gauss(0, self._noise_level)
            self._variables[var] = self._clamp(new)

        # Cross-variable interactions
        self._apply_cross_effects()
        # Decay toward baselines
        self._decay_toward_baseline()
        # Detect competing-drive conflicts
        self._detect_conflicts()
        # Record history
        snap = self.snapshot()
        self._history.append(snap)
        return snap

    def apply_outcome(
        self,
        reward: float,
        task_type: str = "",
        failure_reason: str = "",
    ) -> StateSnapshot:
        """Compute state deltas from a task outcome and apply them.

        High reward → +confidence, +motivation, -stress, -frustration.
        Low reward → +stress, +frustration, -confidence, +fatigue.
        Repeated failure → compounding frustration and fear.
        """
        deltas: dict[str, float] = {}

        if reward >= 0.75:
            deltas["confidence"] = 0.08
            deltas["motivation"] = 0.06
            deltas["stress"] = -0.05
            deltas["frustration"] = -0.08
            deltas["ambition"] = 0.04
            deltas["trust"] = 0.03
        elif reward >= 0.5:
            deltas["confidence"] = 0.02
            deltas["motivation"] = 0.01
            deltas["frustration"] = -0.02
        else:
            frustration_amp = 1.0 + self.get("frustration")
            deltas["confidence"] = -0.10
            deltas["stress"] = 0.12 * frustration_amp
            deltas["frustration"] = 0.15
            deltas["motivation"] = -0.08
            deltas["fatigue"] = 0.05
            deltas["fear"] = 0.06

        # Task-type specific effects
        if "social" in task_type or "empathy" in task_type:
            deltas["empathy"] = deltas.get("empathy", 0) + 0.05
        if failure_reason and "constraint" in failure_reason.lower():
            deltas["caution"] = deltas.get("caution", 0) + 0.08

        # Emotional momentum tracks recent trajectory
        recent_valence = self._recent_valence_trend()
        deltas["emotional_momentum"] = recent_valence * 0.3

        return self.update(deltas)

    def inject_event(self, event_type: str, intensity: float = 0.5) -> StateSnapshot:
        """Inject an external event that perturbs state.

        Example event types: 'threat', 'reward', 'social_praise',
        'social_rejection', 'novelty', 'deadline_pressure', 'rest'.
        """
        intensity = self._clamp(intensity)
        event_map: dict[str, dict[str, float]] = {
            "threat": {"fear": 0.2, "stress": 0.15, "caution": 0.12, "curiosity": -0.08},
            "reward": {"motivation": 0.15, "confidence": 0.10, "ambition": 0.08, "stress": -0.06},
            "social_praise": {"confidence": 0.12, "trust": 0.10, "empathy": 0.06, "motivation": 0.08},
            "social_rejection": {"trust": -0.15, "self_protection": 0.12, "stress": 0.10, "confidence": -0.08},
            "novelty": {"curiosity": 0.18, "motivation": 0.06, "fear": 0.04, "reflection": 0.05},
            "deadline_pressure": {"stress": 0.20, "impulse": 0.12, "fatigue": 0.08, "reflection": -0.06},
            "rest": {"fatigue": -0.20, "stress": -0.15, "motivation": 0.05, "reflection": 0.08},
        }
        deltas = {}
        base_deltas = event_map.get(event_type, {})
        for var, base_delta in base_deltas.items():
            deltas[var] = base_delta * intensity
        return self.update(deltas)

    # ── Conflict detection ────────────────────────────────────────────────

    def _detect_conflicts(self, threshold: float = 0.15) -> None:
        """Detect active conflicts between opposing drives."""
        self._active_conflicts = []
        seen: set[tuple[str, str]] = set()

        for drive_a, drive_b in DRIVE_OPPOSITIONS.items():
            if drive_b not in self._variables or drive_a not in self._variables:
                continue
            pair = tuple(sorted([drive_a, drive_b]))
            if pair in seen:
                continue
            seen.add(pair)

            val_a = self._variables[drive_a]
            val_b = self._variables[drive_b]

            # Conflict when both drives are above threshold and close in value
            if val_a > 0.4 and val_b > 0.4:
                gap = abs(val_a - val_b)
                if gap < threshold + 0.1:
                    intensity = min(1.0, (val_a + val_b) / 2.0 - 0.3)
                    if intensity > threshold:
                        # Resolution leans toward the stronger drive
                        weight = val_b / (val_a + val_b + 1e-9)
                        self._active_conflicts.append(
                            DriveConflict(
                                drive_a=drive_a,
                                drive_b=drive_b,
                                intensity=round(intensity, 4),
                                resolution="pending",
                                resolution_weight=round(weight, 4),
                            )
                        )

        # Keep only top conflicts
        self._active_conflicts.sort(key=lambda c: c.intensity, reverse=True)
        self._active_conflicts = self._active_conflicts[:5]

    def resolve_conflict(self, conflict: DriveConflict, strategy: str = "weighted_compromise") -> DriveConflict:
        """Resolve a drive conflict and update state accordingly.

        Strategies:
          - dominant: stronger drive wins, weaker is suppressed.
          - weighted_compromise: proportional blend based on current values.
          - noisy: add randomness to resolution (simulates impulsivity).
        """
        val_a = self.get(conflict.drive_a)
        val_b = self.get(conflict.drive_b)

        if strategy == "dominant":
            if val_a >= val_b:
                conflict.resolution = conflict.drive_a
                conflict.resolution_weight = 0.2
                self._variables[conflict.drive_b] *= 0.7
            else:
                conflict.resolution = conflict.drive_b
                conflict.resolution_weight = 0.8
                self._variables[conflict.drive_a] *= 0.7
        elif strategy == "noisy":
            noise = random.gauss(0, 0.15)
            weight = val_b / (val_a + val_b + 1e-9) + noise
            weight = self._clamp(weight)
            conflict.resolution_weight = weight
            conflict.resolution = "compromise_noisy"
        else:  # weighted_compromise
            weight = val_b / (val_a + val_b + 1e-9)
            conflict.resolution_weight = weight
            conflict.resolution = "compromise"

        # Stress increases from unresolved internal conflict
        self._variables["stress"] = self._clamp(
            self.get("stress") + conflict.intensity * 0.05
        )
        return conflict

    # ── Queries ───────────────────────────────────────────────────────────

    def mood_valence(self) -> float:
        """Compute overall mood from -1 (negative) to +1 (positive)."""
        positive = (
            self.get("confidence")
            + self.get("motivation")
            + self.get("curiosity")
            + self.get("trust")
            + self.get("ambition")
        )
        negative = (
            self.get("stress")
            + self.get("frustration")
            + self.get("fear")
            + self.get("fatigue")
            + self.get("self_protection")
        )
        total = positive + negative + 1e-9
        return round((positive - negative) / total, 4)

    def dominant_drives(self, top_k: int = 3) -> list[str]:
        """Return the top-K drives by current magnitude."""
        drive_vars = [
            k for k in self._variables
            if k not in {"emotional_momentum"}
        ]
        ranked = sorted(drive_vars, key=lambda v: self._variables[v], reverse=True)
        return ranked[:top_k]

    def arousal_level(self) -> float:
        """Overall activation level: high when strong emotions are present."""
        excitatory = [
            self.get("stress"),
            self.get("curiosity"),
            self.get("fear"),
            self.get("ambition"),
            self.get("impulse"),
        ]
        return round(sum(excitatory) / len(excitatory), 4)

    def drive_strength(self, drive: str) -> float:
        """Effective drive strength accounting for fatigue and stress."""
        base = self.get(drive)
        fatigue_penalty = self.get("fatigue") * 0.2
        return self._clamp(base - fatigue_penalty)

    def routing_bias(self) -> dict[str, float]:
        """Compute state-conditioned routing biases for each branch category.

        Maps drive variables to branch influence weights that the router can
        use to modulate branch scoring.
        """
        biases: dict[str, float] = {}
        mapping = {
            "reflective_reasoning": ["reflection", "confidence"],
            "impulse_response": ["impulse"],
            "fear_risk": ["fear", "caution"],
            "curiosity_exploration": ["curiosity"],
            "empathy_social": ["empathy", "trust"],
            "self_justification": ["self_justification", "self_protection"],
            "long_term_goals": ["long_term_goals", "goal_commitment", "ambition"],
            "emotional_regulation": ["reflection", "emotional_momentum"],
            "moral_evaluation": ["honesty", "empathy"],
            "ambition_reward": ["ambition", "motivation"],
        }
        for branch_key, drives in mapping.items():
            bias = 0.0
            for d in drives:
                bias += self.drive_strength(d)
            biases[branch_key] = round(bias / len(drives), 4)
        return biases

    # ── Internal helpers ──────────────────────────────────────────────────

    def _apply_cross_effects(self) -> None:
        """Model cross-variable interactions (e.g., high stress → less reflection)."""
        stress = self.get("stress")
        if stress > 0.6:
            self._variables["reflection"] = self._clamp(
                self.get("reflection") - (stress - 0.6) * 0.15
            )
            self._variables["impulse"] = self._clamp(
                self.get("impulse") + (stress - 0.6) * 0.10
            )

        fatigue = self.get("fatigue")
        if fatigue > 0.7:
            self._variables["motivation"] = self._clamp(
                self.get("motivation") - (fatigue - 0.7) * 0.2
            )
            self._variables["curiosity"] = self._clamp(
                self.get("curiosity") - (fatigue - 0.7) * 0.15
            )

        # High confidence boosts ambition
        conf = self.get("confidence")
        if conf > 0.7:
            self._variables["ambition"] = self._clamp(
                self.get("ambition") + (conf - 0.7) * 0.08
            )

    def _decay_toward_baseline(self) -> None:
        """Gradually move all variables toward homeostatic baselines."""
        for var, baseline in self._baselines.items():
            current = self._variables.get(var, baseline)
            diff = baseline - current
            self._variables[var] = current + diff * self._decay_rate

    def _recent_valence_trend(self, window: int = 3) -> float:
        """Average mood valence over recent history."""
        if not self._history:
            return 0.0
        recent = self._history[-window:]
        return sum(s.mood_valence for s in recent) / len(recent)

    @staticmethod
    def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
        return max(lo, min(hi, value))

    # ── Serialisation ─────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return {
            "variables": dict(self._variables),
            "baselines": dict(self._baselines),
            "turn_index": self._turn_index,
            "active_conflicts": [c.to_dict() for c in self._active_conflicts],
            "history_length": len(self._history),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HumanState":
        state = cls(initial_values=data.get("variables"))
        state._turn_index = data.get("turn_index", 0)
        state._baselines = data.get("baselines", dict(DEFAULT_BASELINES))
        return state

    def __repr__(self) -> str:
        mood = self.mood_valence()
        top = self.dominant_drives(3)
        conflicts = len(self._active_conflicts)
        return (
            f"HumanState(turn={self._turn_index}, mood={mood:+.2f}, "
            f"top={top}, conflicts={conflicts})"
        )
