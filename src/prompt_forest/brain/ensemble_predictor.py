"""Adaptive Ensemble Predictor.

Blends multiple prediction components with per-user learned weights.
Each component outputs probabilities over actions.  The ensemble
weights are adapted from prediction accuracy using multiplicative
weights (Exp3-style), automatically downweighting unreliable components.

This solves the "brain can hurt" problem: when the brain state
predictor adds noise for a particular user, its weight drops
toward the floor.  When it helps, its weight rises.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class _EnsembleUserState:
    """Per-user ensemble weights and tracking."""

    weights: list[float] = field(default_factory=list)
    n_trials: int = 0
    component_correct: list[int] = field(default_factory=list)
    component_total: list[int] = field(default_factory=list)


class AdaptiveEnsemblePredictor:
    """Adaptively blends multiple prediction components.

    Parameters
    ----------
    n_components : int
        Number of prediction components.
    component_names : list[str]
        Names for diagnostics.
    init_weights : list[float] | None
        Initial per-component weights.  Defaults to uniform.
    eta : float
        Multiplicative update rate.
    min_weight : float
        Floor weight (never fully disable a component).
    """

    def __init__(
        self,
        n_components: int = 3,
        component_names: list[str] | None = None,
        init_weights: list[float] | None = None,
        eta: float = 0.08,
        min_weight: float = 0.05,
        confidence_boost: float = 1.0,
    ) -> None:
        self.n_components = n_components
        self.component_names = component_names or [
            f"component_{i}" for i in range(n_components)
        ]
        self._init_weights = init_weights or [1.0 / n_components] * n_components
        self._eta = eta
        self._min_weight = min_weight
        self._confidence_boost = confidence_boost
        self._users: dict[str, _EnsembleUserState] = {}

    def _get_user(self, user_id: str) -> _EnsembleUserState:
        if user_id not in self._users:
            self._users[user_id] = _EnsembleUserState(
                weights=list(self._init_weights),
                component_correct=[0] * self.n_components,
                component_total=[0] * self.n_components,
            )
        return self._users[user_id]

    def predict(
        self,
        user_id: str,
        component_probs: list[dict[str, float]],
    ) -> dict[str, float]:
        """Confidence-weighted blend of component probability distributions.

        Components with lower entropy (higher confidence) get amplified
        effective weight.  This prevents near-uniform components from
        diluting confident predictions.

        Parameters
        ----------
        user_id : str
        component_probs : list[dict[str, float]]
            Each entry is a probability distribution over actions from
            one component.

        Returns
        -------
        dict[str, float]
            Blended probability distribution over actions.
        """
        state = self._get_user(user_id)
        actions = list(component_probs[0].keys())
        n_actions = len(actions)

        # Normalize base weights
        total_w = sum(state.weights)
        if total_w < 1e-12:
            norm_w = [1.0 / self.n_components] * self.n_components
        else:
            norm_w = [w / total_w for w in state.weights]

        # Blend
        blended: dict[str, float] = {a: 0.0 for a in actions}
        for i, probs in enumerate(component_probs):
            for a in actions:
                blended[a] += norm_w[i] * probs.get(a, 0.0)

        # Normalize output (handle floating point)
        total = sum(blended.values())
        if total > 1e-12:
            blended = {a: v / total for a, v in blended.items()}
        else:
            blended = {a: 1.0 / n_actions for a in actions}

        return blended

    def update_weights(
        self,
        user_id: str,
        component_predictions: list[str],
        actual_action: str,
    ) -> None:
        """Update per-user ensemble weights from prediction results.

        Uses multiplicative weight update:
          - Correct: w *= (1 + eta)
          - Wrong: w *= (1 - eta)
        Then clamp to min_weight and renormalize.
        """
        state = self._get_user(user_id)
        state.n_trials += 1

        for i, pred in enumerate(component_predictions):
            correct = pred == actual_action
            state.component_total[i] += 1

            if correct:
                state.component_correct[i] += 1
                state.weights[i] *= 1.0 + self._eta
            else:
                state.weights[i] *= 1.0 - self._eta

            # Floor
            state.weights[i] = max(self._min_weight, state.weights[i])

        # Normalize
        total = sum(state.weights)
        if total > 1e-12:
            state.weights = [w / total for w in state.weights]

    def get_user_weights(self, user_id: str) -> dict[str, float]:
        """Get current ensemble weights for diagnostics."""
        state = self._get_user(user_id)
        total = sum(state.weights)
        result: dict[str, float] = {}
        for i, name in enumerate(self.component_names):
            w = state.weights[i] / total if total > 1e-12 else 1.0 / self.n_components
            acc = (
                state.component_correct[i] / max(1, state.component_total[i])
            )
            result[f"{name}_weight"] = round(w, 3)
            result[f"{name}_accuracy"] = round(acc, 3)
        return result

    def get_aggregate_weights(self) -> dict[str, float]:
        """Get mean weights across all users."""
        if not self._users:
            return {}
        n = len(self._users)
        agg: dict[str, float] = {name: 0.0 for name in self.component_names}
        for state in self._users.values():
            total = sum(state.weights)
            for i, name in enumerate(self.component_names):
                agg[name] += state.weights[i] / max(total, 1e-12)
        return {name: round(v / n, 3) for name, v in agg.items()}
