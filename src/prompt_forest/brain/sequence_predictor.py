"""Bayesian Sequence Predictor for action sequences.

Learns conditional action probabilities from observed sequences:
  P(action_t | action_{t-1}, outcome_sign_{t-1})      — bigram
  P(action_t | action_{t-2}, action_{t-1}, outcome)    — trigram
  P(action_t | streak_length, streak_outcome)          — streak patterns

Uses Dirichlet smoothing (pseudo-counts) so predictions are well-calibrated
even with very few observations.  Exponential recency weighting ensures
the model tracks behavioral changes over time.

Domain-general: any sequential decision task has action-outcome sequences.
This captures win-stay/lose-shift, perseveration, exploration bursts,
and outcome-conditional switching patterns.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class _SeqUserState:
    """Per-user sequence statistics."""

    # Bigram: P(next | prev_action, outcome_sign)
    # Key: (prev_action, outcome_sign) → counts per action
    bigram_counts: dict[tuple[str, str], dict[str, float]] = field(default_factory=dict)

    # Trigram: P(next | prev2, prev1, outcome_sign)
    trigram_counts: dict[tuple[str, str, str], dict[str, float]] = field(default_factory=dict)

    # Streak: P(next | streak_length_bucket, streak_outcome_sign)
    streak_counts: dict[tuple[int, str], dict[str, float]] = field(default_factory=dict)

    # Unigram (recency-weighted)
    unigram_counts: dict[str, float] = field(default_factory=dict)

    # Action-only bigram: P(next | prev_action) — no outcome conditioning
    action_bigram_counts: dict[str, dict[str, float]] = field(default_factory=dict)

    # Per-user adaptive component weights (multiplicative update)
    # [bigram, trigram, streak, unigram, action_bigram]
    component_weights: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0, 1.0, 1.0])
    component_correct: list[int] = field(default_factory=lambda: [0, 0, 0, 0, 0])
    component_total: list[int] = field(default_factory=lambda: [0, 0, 0, 0, 0])

    # History
    action_history: list[str] = field(default_factory=list)
    outcome_history: list[float] = field(default_factory=list)

    n_trials: int = 0
    n_correct: int = 0


def _outcome_sign(outcome: float) -> str:
    if outcome > 0:
        return "pos"
    elif outcome < 0:
        return "neg"
    return "zero"


def _streak_bucket(length: int) -> int:
    """Bucket streak lengths: 1, 2, 3+."""
    return min(length, 3)


class SequencePredictor:
    """Bayesian n-gram sequence predictor with outcome conditioning.

    Parameters
    ----------
    actions : list[str]
        Available actions.
    alpha : float
        Dirichlet smoothing pseudo-count per action.
    recency_decay : float
        Exponential decay for recency weighting (0.9 = recent counts ~2x older).
    bigram_weight : float
        Weight of bigram component in final blend.
    trigram_weight : float
        Weight of trigram component.
    streak_weight : float
        Weight of streak component.
    unigram_weight : float
        Weight of unigram (base rate) component.
    """

    def __init__(
        self,
        actions: list[str],
        alpha: float = 0.5,
        recency_decay: float = 0.95,
        bigram_weight: float = 0.35,
        trigram_weight: float = 0.15,
        streak_weight: float = 0.10,
        unigram_weight: float = 0.15,
        action_bigram_weight: float = 0.25,
        pop_transfer_weight: float = 0.3,
    ) -> None:
        self.actions = actions
        self.n_actions = len(actions)
        self._alpha = alpha
        self._recency_decay = recency_decay
        self._w_bigram = bigram_weight
        self._w_trigram = trigram_weight
        self._w_streak = streak_weight
        self._w_unigram = unigram_weight
        self._w_action_bigram = action_bigram_weight
        self._pop_transfer_weight = pop_transfer_weight
        self._users: dict[str, _SeqUserState] = {}
        # Population-level accumulated bigram counts for transfer
        self._pop_bigram: dict[tuple[str, str], dict[str, float]] = {}
        self._pop_unigram: dict[str, float] = {}
        self._n_completed_users: int = 0

    def _get_user(self, user_id: str) -> _SeqUserState:
        if user_id not in self._users:
            state = _SeqUserState()
            # Bootstrap from population prior if available
            if self._n_completed_users >= 5:
                tw = self._pop_transfer_weight
                for key, counts in self._pop_bigram.items():
                    state.bigram_counts[key] = {
                        a: tw * c for a, c in counts.items()
                    }
                state.unigram_counts = {
                    a: tw * c for a, c in self._pop_unigram.items()
                }
            self._users[user_id] = state
        return self._users[user_id]

    def _dirichlet_probs(self, counts: dict[str, float] | None) -> dict[str, float]:
        """Convert counts to probabilities with Dirichlet smoothing."""
        alpha = self._alpha
        probs = {}
        total = 0.0
        for a in self.actions:
            c = (counts or {}).get(a, 0.0) + alpha
            probs[a] = c
            total += c
        if total > 0:
            probs = {a: v / total for a, v in probs.items()}
        return probs

    def _get_streak(self, state: _SeqUserState) -> tuple[int, str]:
        """Get current streak length and outcome sign."""
        if not state.action_history:
            return 0, "zero"
        last_action = state.action_history[-1]
        streak = 1
        for i in range(len(state.action_history) - 2, -1, -1):
            if state.action_history[i] == last_action:
                streak += 1
            else:
                break
        # Last outcome sign
        if state.outcome_history:
            osign = _outcome_sign(state.outcome_history[-1])
        else:
            osign = "zero"
        return streak, osign

    def _compute_components(self, state: _SeqUserState) -> list[dict[str, float]]:
        """Compute probability distributions from each sub-component."""
        if not state.action_history:
            uniform = {a: 1.0 / self.n_actions for a in self.actions}
            return [uniform, uniform, uniform, uniform, uniform]

        last_a = state.action_history[-1]
        last_o = _outcome_sign(state.outcome_history[-1]) if state.outcome_history else "zero"

        # 1. Bigram (outcome-conditioned)
        bigram_key = (last_a, last_o)
        bigram_probs = self._dirichlet_probs(state.bigram_counts.get(bigram_key))

        # 2. Trigram
        if len(state.action_history) >= 2:
            prev2 = state.action_history[-2]
            trigram_key = (prev2, last_a, last_o)
            trigram_probs = self._dirichlet_probs(state.trigram_counts.get(trigram_key))
        else:
            trigram_probs = bigram_probs

        # 3. Streak
        streak_len, streak_osign = self._get_streak(state)
        streak_key = (_streak_bucket(streak_len), streak_osign)
        streak_probs = self._dirichlet_probs(state.streak_counts.get(streak_key))

        # 4. Unigram
        unigram_probs = self._dirichlet_probs(state.unigram_counts)

        # 5. Action-only bigram (no outcome conditioning — more data per cell)
        action_bigram_probs = self._dirichlet_probs(
            state.action_bigram_counts.get(last_a)
        )

        return [bigram_probs, trigram_probs, streak_probs, unigram_probs, action_bigram_probs]

    def predict(self, user_id: str) -> dict[str, float]:
        """Return probability distribution over actions."""
        state = self._get_user(user_id)

        if not state.action_history:
            return {a: 1.0 / self.n_actions for a in self.actions}

        component_dists = self._compute_components(state)
        base_weights = [
            self._w_bigram, self._w_trigram, self._w_streak,
            self._w_unigram, self._w_action_bigram,
        ]

        # Effective weights = base_weight * adaptive_weight
        eff_weights = [
            base_weights[i] * state.component_weights[i]
            for i in range(5)
        ]
        total_w = sum(eff_weights)
        if total_w < 1e-12:
            total_w = 1.0
            eff_weights = base_weights[:]

        # Weighted blend
        blended: dict[str, float] = {a: 0.0 for a in self.actions}
        for i, probs in enumerate(component_dists):
            w = eff_weights[i] / total_w
            for a in self.actions:
                blended[a] += w * probs[a]

        # Normalize
        total = sum(blended.values())
        if total > 1e-12:
            blended = {a: v / total for a, v in blended.items()}

        return blended

    def update(
        self,
        user_id: str,
        actual_action: str,
        outcome: float,
    ) -> None:
        """Update sequence statistics from observed action + outcome."""
        state = self._get_user(user_id)

        # Apply recency decay to all existing counts
        decay = self._recency_decay
        for key in state.bigram_counts:
            for a in state.bigram_counts[key]:
                state.bigram_counts[key][a] *= decay
        for key in state.trigram_counts:
            for a in state.trigram_counts[key]:
                state.trigram_counts[key][a] *= decay
        for key in state.streak_counts:
            for a in state.streak_counts[key]:
                state.streak_counts[key][a] *= decay
        for a in state.unigram_counts:
            state.unigram_counts[a] *= decay
        for key in state.action_bigram_counts:
            for a in state.action_bigram_counts[key]:
                state.action_bigram_counts[key][a] *= decay

        # Update bigram counts
        if state.action_history:
            last_a = state.action_history[-1]
            last_o = _outcome_sign(state.outcome_history[-1]) if state.outcome_history else "zero"
            bigram_key = (last_a, last_o)
            if bigram_key not in state.bigram_counts:
                state.bigram_counts[bigram_key] = {}
            state.bigram_counts[bigram_key][actual_action] = (
                state.bigram_counts[bigram_key].get(actual_action, 0.0) + 1.0
            )

        # Update trigram counts
        if len(state.action_history) >= 2:
            prev2 = state.action_history[-2]
            last_a = state.action_history[-1]
            last_o = _outcome_sign(state.outcome_history[-1]) if state.outcome_history else "zero"
            trigram_key = (prev2, last_a, last_o)
            if trigram_key not in state.trigram_counts:
                state.trigram_counts[trigram_key] = {}
            state.trigram_counts[trigram_key][actual_action] = (
                state.trigram_counts[trigram_key].get(actual_action, 0.0) + 1.0
            )

        # Update streak counts
        streak_len, streak_osign = self._get_streak(state)
        streak_key = (_streak_bucket(streak_len), streak_osign)
        if streak_key not in state.streak_counts:
            state.streak_counts[streak_key] = {}
        state.streak_counts[streak_key][actual_action] = (
            state.streak_counts[streak_key].get(actual_action, 0.0) + 1.0
        )

        # Update unigram counts
        state.unigram_counts[actual_action] = (
            state.unigram_counts.get(actual_action, 0.0) + 1.0
        )

        # Update action-only bigram counts
        if state.action_history:
            last_a = state.action_history[-1]
            if last_a not in state.action_bigram_counts:
                state.action_bigram_counts[last_a] = {}
            state.action_bigram_counts[last_a][actual_action] = (
                state.action_bigram_counts[last_a].get(actual_action, 0.0) + 1.0
            )

        # Track per-component accuracy and adapt weights
        state.n_trials += 1
        component_dists = self._compute_components(state)
        eta = 0.12  # adaptation rate for component weights
        for i, probs in enumerate(component_dists):
            pred_action = max(probs, key=probs.get)
            state.component_total[i] += 1
            if pred_action == actual_action:
                state.component_correct[i] += 1
                state.component_weights[i] *= 1.0 + eta
            else:
                state.component_weights[i] *= 1.0 - eta
            state.component_weights[i] = max(0.1, state.component_weights[i])

        # Normalize component weights
        total_cw = sum(state.component_weights)
        if total_cw > 1e-12:
            state.component_weights = [w / total_cw for w in state.component_weights]

        # Track overall accuracy
        overall_probs = self.predict(user_id)
        if max(overall_probs, key=overall_probs.get) == actual_action:
            state.n_correct += 1

        # Record history
        state.action_history.append(actual_action)
        state.outcome_history.append(outcome)

    def finalize_user(self, user_id: str) -> None:
        """Accumulate this user's statistics into the population prior.

        Call after a user's data has been fully processed (after holdout).
        Uses running mean so each user contributes equally regardless
        of how many trials they had.
        """
        state = self._users.get(user_id)
        if state is None or state.n_trials < 10:
            return

        self._n_completed_users += 1
        n = self._n_completed_users
        decay = (n - 1) / n  # running mean weight for old data

        # Accumulate bigram counts (normalized per user to prevent
        # users with more trials from dominating)
        for key, counts in state.bigram_counts.items():
            total = sum(counts.values())
            if total < 1e-12:
                continue
            if key not in self._pop_bigram:
                self._pop_bigram[key] = {}
            for a in self.actions:
                norm_c = counts.get(a, 0.0) / total
                old = self._pop_bigram[key].get(a, 0.0)
                self._pop_bigram[key][a] = decay * old + (1.0 / n) * norm_c

        # Accumulate unigram
        total_uni = sum(state.unigram_counts.values())
        if total_uni > 1e-12:
            for a in self.actions:
                norm_c = state.unigram_counts.get(a, 0.0) / total_uni
                old = self._pop_unigram.get(a, 0.0)
                self._pop_unigram[a] = decay * old + (1.0 / n) * norm_c

    def get_user_info(self, user_id: str) -> dict[str, Any]:
        state = self._get_user(user_id)
        return {
            "n_trials": state.n_trials,
            "n_correct": state.n_correct,
            "accuracy": round(state.n_correct / max(1, state.n_trials), 3),
        }

    @property
    def user_count(self) -> int:
        return len(self._users)
