"""Learned per-user state transition model.

Replaces hardcoded apply_outcome() deltas with per-user learned
transition sensitivities.  The brain learns HOW each individual
responds to outcomes — not just what they do.

This is the personalization core:
  same outcome → different state change for different users

For one user, a loss triggers fear + withdrawal.
For another, it triggers frustration + impulsive switching.
The transition model learns which pattern applies.

RL updates the sensitivity parameters based on whether the
downstream predictor correctly anticipated the user's behavior.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TransitionParameters:
    """Per-user learned transition sensitivities.

    Each parameter controls how strongly a category of outcome
    affects the internal state.  All start at 1.0 (neutral) and
    are adapted by the RL layer.
    """

    loss_sensitivity: float = 1.0
    reward_sensitivity: float = 1.0
    perseveration: float = 0.5
    switch_tendency: float = 0.5
    frustration_buildup: float = 1.0
    recovery_rate: float = 0.05
    exploration_decay: float = 0.98
    curiosity_recovery: float = 0.3
    recency_weight: float = 0.7


class LearnedTransitionModel:
    """Per-user learned state transition model.

    Maps ``(current_state, outcome, action, context)`` → state deltas
    with learnable sensitivity parameters.

    The model also tracks per-action expected-value (EV) estimates and
    a recent outcome buffer so that downstream components can reason
    about the user's reward history.

    Parameters
    ----------
    lr : float
        Base learning rate for RL parameter updates.
    """

    def __init__(self, lr: float = 0.02) -> None:
        self._users: dict[str, TransitionParameters] = {}
        self._user_ev: dict[str, dict[str, float]] = {}
        self._user_counts: dict[str, dict[str, int]] = {}
        self._user_outcomes: dict[str, deque] = {}
        self._user_action_outcomes: dict[str, dict[str, deque]] = {}
        self._lr = lr

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_params(self, user_id: str) -> TransitionParameters:
        if user_id not in self._users:
            self._users[user_id] = TransitionParameters()
            self._user_ev[user_id] = {}
            self._user_counts[user_id] = {}
            self._user_outcomes[user_id] = deque(maxlen=30)
            self._user_action_outcomes[user_id] = {}
        return self._users[user_id]

    # ------------------------------------------------------------------
    # Core transition
    # ------------------------------------------------------------------

    def compute_deltas(
        self,
        user_id: str,
        state_vars: dict[str, float],
        outcome: float,
        action: str,
        context: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """Compute state deltas from an outcome using learned parameters.

        The deltas are meant to be fed directly into
        ``HumanState.update(deltas)``.
        """
        params = self.get_params(user_id)

        # ---- Update EV estimates ----
        ev_dict = self._user_ev[user_id]
        counts = self._user_counts[user_id]
        old_ev = ev_dict.get(action, 0.0)
        n = counts.get(action, 0) + 1
        alpha = max(0.05, 1.0 / n)
        ev_dict[action] = old_ev + alpha * (outcome - old_ev)
        counts[action] = n

        # Per-action outcome tracking
        ao = self._user_action_outcomes[user_id]
        if action not in ao:
            ao[action] = deque(maxlen=20)
        ao[action].append(outcome)

        # Global outcome tracking
        self._user_outcomes[user_id].append(outcome)

        # ---- Compute deltas ----
        deltas: dict[str, float] = {}
        mag = min(1.0, abs(outcome) / (abs(outcome) + 100.0))

        if outcome > 0:
            deltas["confidence"] = 0.08 * mag * params.reward_sensitivity
            deltas["motivation"] = 0.06 * mag * params.reward_sensitivity
            deltas["stress"] = -0.05 * mag * params.reward_sensitivity
            deltas["frustration"] = -0.08 * mag * params.reward_sensitivity
            deltas["ambition"] = 0.04 * mag * params.reward_sensitivity
            deltas["goal_commitment"] = 0.03 * mag * params.perseveration
        elif outcome < 0:
            deltas["confidence"] = -0.10 * mag * params.loss_sensitivity
            deltas["stress"] = (
                0.12 * mag * params.loss_sensitivity * params.frustration_buildup
            )
            deltas["frustration"] = 0.15 * mag * params.loss_sensitivity
            deltas["fear"] = 0.06 * mag * params.loss_sensitivity
            deltas["motivation"] = -0.08 * mag * params.loss_sensitivity
            deltas["fatigue"] = 0.03 * mag
            deltas["impulse"] = 0.05 * mag * params.switch_tendency
            deltas["caution"] = 0.04 * mag * params.loss_sensitivity

        # ---- Recent-trajectory effects ----
        recent = list(self._user_outcomes[user_id])
        if len(recent) >= 3:
            recent_3 = recent[-3:]
            neg_count = sum(1 for o in recent_3 if o < 0)
            pos_count = sum(1 for o in recent_3 if o > 0)

            if neg_count >= 2:
                # Losing streak → switch pressure
                deltas["frustration"] = (
                    deltas.get("frustration", 0.0)
                    + 0.06 * params.loss_sensitivity
                )
                deltas["impulse"] = (
                    deltas.get("impulse", 0.0) + 0.05 * params.switch_tendency
                )
                deltas["fear"] = (
                    deltas.get("fear", 0.0) + 0.04 * params.loss_sensitivity
                )

            if pos_count >= 2:
                # Winning streak → perseveration
                deltas["confidence"] = (
                    deltas.get("confidence", 0.0)
                    + 0.04 * params.reward_sensitivity
                )
                deltas["goal_commitment"] = (
                    deltas.get("goal_commitment", 0.0)
                    + 0.05 * params.perseveration
                )
                deltas["ambition"] = (
                    deltas.get("ambition", 0.0)
                    + 0.03 * params.reward_sensitivity
                )

        if len(recent) >= 5:
            recent_5 = recent[-5:]
            neg_5 = sum(1 for o in recent_5 if o < 0)
            if neg_5 >= 4:
                # Extended losing streak → strong switch + exploration
                deltas["curiosity"] = deltas.get("curiosity", 0.0) + 0.06
                deltas["impulse"] = (
                    deltas.get("impulse", 0.0)
                    + 0.08 * params.switch_tendency
                )

        return deltas

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def get_ev_features(self, user_id: str) -> dict[str, float]:
        """Get learned EV estimates as features for prediction."""
        ev_dict = self._user_ev.get(user_id, {})
        counts = self._user_counts.get(user_id, {})
        features: dict[str, float] = {}
        for action, ev in ev_dict.items():
            features[f"ev::{action}"] = ev / (abs(ev) + 100.0)
            n = counts.get(action, 0)
            features[f"ev_confidence::{action}"] = min(1.0, n / 20.0)
        # EV differential — strong signal for learned preferences
        evs = list(ev_dict.values())
        if len(evs) >= 2:
            spread = max(evs) - min(evs)
            features["ev_spread"] = spread / (abs(spread) + 50.0)
            # Which action has better EV?
            best_action = max(ev_dict, key=ev_dict.get)  # type: ignore[arg-type]
            features[f"ev_best_is_{best_action}"] = min(
                1.0, spread / (abs(spread) + 30.0)
            )
        return features

    def get_outcome_features(self, user_id: str) -> dict[str, float]:
        """Get recent outcome trajectory features."""
        outcomes = list(self._user_outcomes.get(user_id, []))
        if not outcomes:
            return {}

        features: dict[str, float] = {}
        for window in [3, 5, 10]:
            recent = outcomes[-min(window, len(outcomes)) :]
            if recent:
                m = sum(recent) / len(recent)
                features[f"outcome_mean_{window}"] = m / (abs(m) + 100.0)
                features[f"outcome_neg_rate_{window}"] = sum(
                    1 for o in recent if o < 0
                ) / len(recent)

        features["last_outcome_sign"] = 1.0 if outcomes[-1] > 0 else -1.0
        features["last_outcome_mag"] = min(
            1.0, abs(outcomes[-1]) / (abs(outcomes[-1]) + 100.0)
        )

        # Volatility
        if len(outcomes) >= 4:
            diffs = [
                abs(outcomes[i] - outcomes[i - 1])
                for i in range(1, len(outcomes))
            ]
            features["outcome_volatility"] = min(
                1.0,
                (sum(diffs[-3:]) / len(diffs[-3:]))
                / ((sum(diffs[-3:]) / len(diffs[-3:])) + 100.0),
            )

        return features

    def get_per_action_features(self, user_id: str) -> dict[str, float]:
        """Get per-action outcome features."""
        ao = self._user_action_outcomes.get(user_id, {})
        features: dict[str, float] = {}
        for action, outcomes in ao.items():
            recent = list(outcomes)[-5:]
            if recent:
                m = sum(recent) / len(recent)
                features[f"action_outcome_mean::{action}"] = m / (abs(m) + 100.0)
                features[f"action_neg_rate::{action}"] = sum(
                    1 for o in recent if o < 0
                ) / len(recent)
        return features

    # ------------------------------------------------------------------
    # RL adaptation
    # ------------------------------------------------------------------

    def rl_update(
        self,
        user_id: str,
        prediction_correct: bool,
        predicted_action: str,
        actual_action: str,
        outcome: float,
    ) -> None:
        """Adjust transition sensitivities based on prediction accuracy.

        If the predictor failed, the transition model needs to be more
        responsive to the kinds of outcomes that led to the surprise.
        """
        params = self.get_params(user_id)
        lr = self._lr

        if not prediction_correct:
            if outcome < 0:
                params.loss_sensitivity *= 1.0 + lr * 2.0
                params.switch_tendency *= 1.0 + lr * 1.5
                params.frustration_buildup *= 1.0 + lr * 1.0
            else:
                params.reward_sensitivity *= 1.0 + lr * 2.0
                params.perseveration *= 1.0 + lr * 1.5
        else:
            # Gentle regression toward defaults when correct
            params.loss_sensitivity += lr * 0.2 * (1.0 - params.loss_sensitivity)
            params.reward_sensitivity += lr * 0.2 * (1.0 - params.reward_sensitivity)
            params.switch_tendency += lr * 0.1 * (0.5 - params.switch_tendency)
            params.perseveration += lr * 0.1 * (0.5 - params.perseveration)
            params.frustration_buildup += lr * 0.1 * (1.0 - params.frustration_buildup)

        # Clamp with wider range for real personalization
        params.loss_sensitivity = max(0.1, min(5.0, params.loss_sensitivity))
        params.reward_sensitivity = max(0.1, min(5.0, params.reward_sensitivity))
        params.switch_tendency = max(0.05, min(4.0, params.switch_tendency))
        params.perseveration = max(0.05, min(4.0, params.perseveration))
        params.frustration_buildup = max(0.3, min(4.0, params.frustration_buildup))

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_params_dict(self, user_id: str) -> dict[str, float]:
        p = self.get_params(user_id)
        return {
            "loss_sensitivity": round(p.loss_sensitivity, 3),
            "reward_sensitivity": round(p.reward_sensitivity, 3),
            "perseveration": round(p.perseveration, 3),
            "switch_tendency": round(p.switch_tendency, 3),
            "frustration_buildup": round(p.frustration_buildup, 3),
            "recovery_rate": round(p.recovery_rate, 3),
        }
