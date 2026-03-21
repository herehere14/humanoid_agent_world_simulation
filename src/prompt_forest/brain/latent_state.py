"""Compact Learned Latent State Predictor.

A D-dimensional learned state per user that captures behavioral patterns
through online learning.  No hand-designed emotions or fixed labels —
the latent dimensions emerge as whatever internal states best predict
the user's next action.

Architecture:
  z ∈ R^D — latent state vector (D=4 by default)
  W_transition: maps (z_prev, action_onehot, outcome_features) → z_new
  W_readout: maps z → action logits

Online learning via simple gradient updates on prediction error.

Domain-general: the latent dimensions are unnamed and learned.  On IGT
they might correspond to "risk tolerance" and "exploration level".
On a trust game, different dimensions would emerge.  The architecture
discovers what matters rather than assuming it.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any


@dataclass
class _LatentUserState:
    """Per-user latent state and learned weights."""

    z: list[float]                          # latent state vector (D,)
    W_trans: list[list[float]]              # transition weights (D x input_dim)
    W_read: list[list[float]]              # readout weights (n_actions x D)
    b_read: list[float]                     # readout bias (n_actions,)
    n_trials: int = 0
    n_correct: int = 0
    lr: float = 0.025                       # current learning rate


def _tanh(x: float) -> float:
    return math.tanh(max(-10.0, min(10.0, x)))


def _softmax(logits: list[float]) -> list[float]:
    max_v = max(logits) if logits else 0.0
    exps = [math.exp(v - max_v) for v in logits]
    total = sum(exps)
    if total < 1e-12:
        n = len(logits)
        return [1.0 / n] * n
    return [e / total for e in exps]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(ai * bi for ai, bi in zip(a, b))


class LatentStatePredictor:
    """Compact learned latent state predictor.

    Parameters
    ----------
    actions : list[str]
        Available actions.
    D : int
        Latent state dimensionality.
    lr : float
        Initial learning rate for weight updates.
    anneal_rate : float
        Per-trial learning rate decay.
    mix : float
        How fast state changes (0 = static, 1 = fully replace).
    """

    def __init__(
        self,
        actions: list[str],
        D: int = 4,
        lr: float = 0.025,
        anneal_rate: float = 0.997,
        mix: float = 0.3,
    ) -> None:
        self.actions = actions
        self.n_actions = len(actions)
        self.D = D
        self._lr = lr
        self._anneal_rate = anneal_rate
        self._mix = mix
        # Input: z_prev (D) + action_onehot (n_actions) + outcome_sign (1) + outcome_mag (1) + ev_features (n_actions)
        self._input_dim = D + self.n_actions + 2 + self.n_actions
        self._users: dict[str, _LatentUserState] = {}
        self._rng = random.Random(42)

    def _init_user(self) -> _LatentUserState:
        D = self.D
        inp = self._input_dim
        scale_trans = 0.1 / math.sqrt(inp)
        scale_read = 0.1 / math.sqrt(D)

        W_trans = [
            [self._rng.gauss(0, scale_trans) for _ in range(inp)]
            for _ in range(D)
        ]
        W_read = [
            [self._rng.gauss(0, scale_read) for _ in range(D)]
            for _ in range(self.n_actions)
        ]
        return _LatentUserState(
            z=[0.0] * D,
            W_trans=W_trans,
            W_read=W_read,
            b_read=[0.0] * self.n_actions,
            lr=self._lr,
        )

    def _get_user(self, user_id: str) -> _LatentUserState:
        if user_id not in self._users:
            self._users[user_id] = self._init_user()
        return self._users[user_id]

    def _build_input(
        self,
        state: _LatentUserState,
        last_action: str | None,
        last_outcome: float | None,
        ev_estimates: dict[str, float] | None,
    ) -> list[float]:
        """Build input feature vector for transition."""
        inp: list[float] = []
        # z_prev
        inp.extend(state.z)
        # action one-hot
        for a in self.actions:
            inp.append(1.0 if a == last_action else 0.0)
        # outcome features
        if last_outcome is not None:
            inp.append(1.0 if last_outcome > 0 else -1.0)
            inp.append(min(1.0, abs(last_outcome) / (abs(last_outcome) + 100.0)))
        else:
            inp.extend([0.0, 0.0])
        # EV features (normalized)
        ev = ev_estimates or {}
        for a in self.actions:
            v = ev.get(a, 0.0)
            inp.append(v / (abs(v) + 100.0))
        return inp

    def _transition(self, state: _LatentUserState, inp: list[float]) -> list[float]:
        """Compute new latent state from input."""
        new_z = []
        for d in range(self.D):
            raw = _dot(state.W_trans[d], inp)
            activated = _tanh(raw)
            new_z.append(
                (1.0 - self._mix) * state.z[d] + self._mix * activated
            )
        return new_z

    def _readout(self, state: _LatentUserState) -> list[float]:
        """Compute action logits from latent state."""
        logits = []
        for a_idx in range(self.n_actions):
            logits.append(_dot(state.W_read[a_idx], state.z) + state.b_read[a_idx])
        return logits

    def predict(
        self,
        user_id: str,
        last_action: str | None = None,
        last_outcome: float | None = None,
        ev_estimates: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """Return softmax probabilities over actions.

        Call this BEFORE update() for each trial.
        """
        state = self._get_user(user_id)

        # Transition state based on last observation
        if last_action is not None:
            inp = self._build_input(state, last_action, last_outcome, ev_estimates)
            state.z = self._transition(state, inp)

        # Readout
        logits = self._readout(state)
        probs = _softmax(logits)
        return {a: p for a, p in zip(self.actions, probs)}

    def update(
        self,
        user_id: str,
        actual_action: str,
        last_action: str | None = None,
        last_outcome: float | None = None,
        ev_estimates: dict[str, float] | None = None,
    ) -> None:
        """Online weight update from prediction error.

        Uses a simplified perceptron-style update on the readout layer,
        with one-step backprop to the transition layer.
        """
        state = self._get_user(user_id)
        state.n_trials += 1

        # Current prediction
        logits = self._readout(state)
        probs = _softmax(logits)

        actual_idx = self.actions.index(actual_action)
        predicted_idx = max(range(self.n_actions), key=lambda i: probs[i])

        if predicted_idx == actual_idx:
            state.n_correct += 1

        lr = state.lr

        # ---- Readout update ----
        # Gradient of cross-entropy loss: dL/d_logit_i = prob_i - target_i
        for a_idx in range(self.n_actions):
            target = 1.0 if a_idx == actual_idx else 0.0
            grad = probs[a_idx] - target  # gradient of CE loss

            # Update readout weights
            for d in range(self.D):
                state.W_read[a_idx][d] -= lr * grad * state.z[d]
            state.b_read[a_idx] -= lr * grad

        # ---- Transition update (one-step backprop) ----
        # Backprop through readout to get dL/dz
        dL_dz = [0.0] * self.D
        for a_idx in range(self.n_actions):
            target = 1.0 if a_idx == actual_idx else 0.0
            grad = probs[a_idx] - target
            for d in range(self.D):
                dL_dz[d] += grad * state.W_read[a_idx][d]

        # Backprop through transition: z = (1-mix)*z_prev + mix*tanh(W@inp)
        # dL/dW[d][j] = dL/dz[d] * mix * (1 - tanh^2(W@inp)) * inp[j]
        if last_action is not None:
            inp = self._build_input(state, last_action, last_outcome, ev_estimates)
            trans_lr = lr * 0.5  # slower for stability
            for d in range(self.D):
                raw = _dot(state.W_trans[d], inp)
                tanh_val = _tanh(raw)
                dtanh = 1.0 - tanh_val * tanh_val  # tanh derivative
                scale = dL_dz[d] * self._mix * dtanh
                for j in range(self._input_dim):
                    state.W_trans[d][j] -= trans_lr * scale * inp[j]

        # Anneal
        state.lr = max(0.003, state.lr * self._anneal_rate)

    def get_user_info(self, user_id: str) -> dict[str, Any]:
        """Get diagnostics for a user."""
        state = self._get_user(user_id)
        return {
            "z": [round(v, 3) for v in state.z],
            "n_trials": state.n_trials,
            "n_correct": state.n_correct,
            "accuracy": round(state.n_correct / max(1, state.n_trials), 3),
            "lr": round(state.lr, 4),
        }

    @property
    def user_count(self) -> int:
        return len(self._users)
