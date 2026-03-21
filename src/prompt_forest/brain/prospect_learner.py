"""Online Prospect Theory Learner (VPP-style).

Implements Value-Plus-Perseverance (Worthy et al., 2013) with online
parameter adaptation instead of grid search.  This is the cognitive
science foundation for human decision prediction:

  - Prospect theory value function (non-linear gain/loss processing)
  - Per-action expected utility with recency-weighted learning
  - Perseveration (tendency to repeat actions regardless of value)
  - Softmax action selection with learned inverse temperature

Parameters are adapted online from prediction errors using finite-
difference gradient approximation.  This replaces the grid search
over all trials that SOTA cognitive models use.

Domain-general: prospect theory applies to any task with outcomes
(gambling, investing, trust games, social decisions).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class _ProspectParams:
    """Per-user prospect theory parameters."""

    alpha: float = 0.50       # outcome sensitivity [0.01, 1.5]
    lambda_: float = 2.0      # loss aversion [0.5, 5.0]
    A: float = 0.15           # EV learning rate [0.01, 0.8]
    theta: float = 2.0        # inverse temperature [0.1, 10.0]
    K: float = 0.30           # perseveration decay [0.05, 0.99]
    ep_p: float = 0.50        # perseveration gain on win [0.0, 3.0]
    ep_n: float = 0.30        # perseveration gain on loss [0.0, 3.0]
    w_rl: float = 0.65        # RL vs perseveration weight [0.0, 1.0]


# Parameter bounds for clamping
_BOUNDS: dict[str, tuple[float, float]] = {
    "alpha": (0.01, 1.5),
    "lambda_": (0.5, 5.0),
    "A": (0.01, 0.8),
    "theta": (0.1, 10.0),
    "K": (0.05, 0.99),
    "ep_p": (0.0, 3.0),
    "ep_n": (0.0, 3.0),
    "w_rl": (0.0, 1.0),
}

# Population priors (cognitive science literature defaults)
_POP_PRIORS = _ProspectParams()

# Perturbation sizes for finite-difference gradient
_PERTURB: dict[str, float] = {
    "alpha": 0.05,
    "lambda_": 0.2,
    "A": 0.03,
    "theta": 0.3,
    "K": 0.05,
    "ep_p": 0.1,
    "ep_n": 0.1,
    "w_rl": 0.05,
}


@dataclass
class _ProspectUserState:
    """Per-user prospect theory state."""

    params: _ProspectParams = field(default_factory=_ProspectParams)
    E: dict[str, float] = field(default_factory=dict)   # expected utilities
    P: dict[str, float] = field(default_factory=dict)   # perseveration scores
    n_trials: int = 0
    n_correct: int = 0
    last_action: str | None = None
    last_outcome: float | None = None
    adapt_lr: float = 0.04    # adaptation learning rate (anneals)


def _prospect_value(x: float, alpha: float, lambda_: float) -> float:
    """Prospect theory value function."""
    if x >= 0:
        return x ** alpha if x > 0 else 0.0
    else:
        return -lambda_ * (abs(x) ** alpha)


def _softmax(values: dict[str, float], theta: float) -> dict[str, float]:
    """Softmax over action values with inverse temperature theta."""
    max_v = max(values.values()) if values else 0.0
    exp_vals = {}
    for a, v in values.items():
        exp_vals[a] = math.exp(theta * (v - max_v))
    total = sum(exp_vals.values())
    if total < 1e-12:
        n = len(values)
        return {a: 1.0 / n for a in values}
    return {a: e / total for a, e in exp_vals.items()}


def _log_prob(
    actions: list[str],
    E: dict[str, float],
    P: dict[str, float],
    params: _ProspectParams,
    target_action: str,
) -> float:
    """Compute log P(target_action) under current parameters."""
    V = {
        a: params.w_rl * E.get(a, 0.0) + (1.0 - params.w_rl) * P.get(a, 0.0)
        for a in actions
    }
    probs = _softmax(V, params.theta)
    p = probs.get(target_action, 1e-8)
    return math.log(max(p, 1e-8))


class ProspectTheoryLearner:
    """Online prospect theory learner with per-user parameter adaptation.

    Implements VPP (Value-Plus-Perseverance) with online learning
    instead of grid search.

    Parameters
    ----------
    actions : list[str]
        Available actions (e.g., ["A", "B", "C", "D"]).
    adapt_lr : float
        Initial learning rate for parameter adaptation.
    anneal_rate : float
        Per-trial annealing factor for adaptation LR.
    """

    def __init__(
        self,
        actions: list[str],
        adapt_lr: float = 0.04,
        anneal_rate: float = 0.995,
    ) -> None:
        self.actions = actions
        self._users: dict[str, _ProspectUserState] = {}
        self._adapt_lr = adapt_lr
        self._anneal_rate = anneal_rate
        # Population transfer
        self._pop_params_sum: dict[str, float] = {}
        self._n_completed: int = 0

    def _get_user(self, user_id: str) -> _ProspectUserState:
        if user_id not in self._users:
            # Initialize from population centroid if available
            if self._n_completed >= 10:
                params = _ProspectParams()
                for pname in ["alpha", "lambda_", "A", "theta", "K", "ep_p", "ep_n", "w_rl"]:
                    pop_mean = self._pop_params_sum[pname] / self._n_completed
                    default = getattr(_POP_PRIORS, pname)
                    # Blend: 60% population mean, 40% default prior
                    blended = 0.6 * pop_mean + 0.4 * default
                    lo, hi = _BOUNDS[pname]
                    setattr(params, pname, max(lo, min(hi, blended)))
            else:
                params = _ProspectParams()
            state = _ProspectUserState(
                params=params,
                E={a: 0.0 for a in self.actions},
                P={a: 0.0 for a in self.actions},
                adapt_lr=self._adapt_lr,
            )
            self._users[user_id] = state
        return self._users[user_id]

    def finalize_user(self, user_id: str) -> None:
        """Accumulate learned parameters into population centroid."""
        state = self._users.get(user_id)
        if state is None or state.n_trials < 20:
            return
        self._n_completed += 1
        for pname in ["alpha", "lambda_", "A", "theta", "K", "ep_p", "ep_n", "w_rl"]:
            val = getattr(state.params, pname)
            self._pop_params_sum[pname] = (
                self._pop_params_sum.get(pname, 0.0) + val
            )

    def predict(self, user_id: str) -> dict[str, float]:
        """Return softmax probabilities over actions."""
        state = self._get_user(user_id)
        V = {
            a: state.params.w_rl * state.E[a]
            + (1.0 - state.params.w_rl) * state.P[a]
            for a in self.actions
        }
        return _softmax(V, state.params.theta)

    def update(
        self, user_id: str, actual_action: str, outcome: float
    ) -> None:
        """Update EV and perseveration from observed action + outcome."""
        state = self._get_user(user_id)
        p = state.params

        # Prospect value
        u = _prospect_value(outcome, p.alpha, p.lambda_)

        # Update expected utility for chosen action
        state.E[actual_action] = (
            state.E[actual_action] + p.A * (u - state.E[actual_action])
        )

        # Update perseveration
        for a in self.actions:
            if a == actual_action:
                ep = p.ep_p if outcome >= 0 else p.ep_n
                state.P[a] = p.K * state.P[a] + ep
            else:
                state.P[a] = p.K * state.P[a]

        state.last_action = actual_action
        state.last_outcome = outcome
        state.n_trials += 1

    def adapt_parameters(
        self,
        user_id: str,
        predicted_action: str,
        actual_action: str,
    ) -> None:
        """Online parameter adaptation from prediction error.

        Uses finite-difference gradient approximation: for each parameter,
        compute how a small perturbation would change log P(actual_action),
        then nudge the parameter in that direction.
        """
        state = self._get_user(user_id)
        correct = predicted_action == actual_action

        if correct:
            state.n_correct += 1
            # Gentle regression toward population priors
            lr = state.adapt_lr * 0.1
            pop = _POP_PRIORS
            p = state.params
            p.alpha += lr * (pop.alpha - p.alpha)
            p.lambda_ += lr * (pop.lambda_ - p.lambda_)
            p.A += lr * (pop.A - p.A)
            p.theta += lr * (pop.theta - p.theta)
            p.K += lr * (pop.K - p.K)
            p.ep_p += lr * (pop.ep_p - p.ep_p)
            p.ep_n += lr * (pop.ep_n - p.ep_n)
            p.w_rl += lr * (pop.w_rl - p.w_rl)
        else:
            # Finite-difference gradient ascent on log P(actual_action)
            lr = state.adapt_lr
            base_lp = _log_prob(
                self.actions, state.E, state.P, state.params, actual_action
            )

            param_names = ["alpha", "lambda_", "A", "theta", "K", "ep_p", "ep_n", "w_rl"]
            for pname in param_names:
                perturb = _PERTURB[pname]
                lo, hi = _BOUNDS[pname]

                # Save original
                orig = getattr(state.params, pname)

                # Compute perturbed log prob
                setattr(state.params, pname, min(hi, orig + perturb))
                lp_plus = _log_prob(
                    self.actions, state.E, state.P, state.params, actual_action
                )

                setattr(state.params, pname, max(lo, orig - perturb))
                lp_minus = _log_prob(
                    self.actions, state.E, state.P, state.params, actual_action
                )

                # Restore original
                setattr(state.params, pname, orig)

                # Finite-difference gradient
                grad = (lp_plus - lp_minus) / (2.0 * perturb)

                # Update
                new_val = orig + lr * grad
                new_val = max(lo, min(hi, new_val))
                setattr(state.params, pname, new_val)

        # Anneal adaptation rate
        state.adapt_lr = max(0.005, state.adapt_lr * self._anneal_rate)

        # Clamp all parameters
        for pname, (lo, hi) in _BOUNDS.items():
            cur = getattr(state.params, pname)
            setattr(state.params, pname, max(lo, min(hi, cur)))

    def get_user_params(self, user_id: str) -> dict[str, float]:
        """Get current parameters for diagnostics."""
        state = self._get_user(user_id)
        p = state.params
        return {
            "alpha": round(p.alpha, 3),
            "lambda": round(p.lambda_, 3),
            "A": round(p.A, 3),
            "theta": round(p.theta, 3),
            "K": round(p.K, 3),
            "ep_p": round(p.ep_p, 3),
            "ep_n": round(p.ep_n, 3),
            "w_rl": round(p.w_rl, 3),
            "n_trials": state.n_trials,
            "n_correct": state.n_correct,
            "adapt_lr": round(state.adapt_lr, 4),
        }

    @property
    def user_count(self) -> int:
        return len(self._users)
