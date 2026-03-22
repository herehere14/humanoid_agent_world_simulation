"""Behavioral Predictor: learns per-user branch→action mappings from observed behavior.

The core PromptForestEngine optimizes *output quality* — branch weights adapt so
the best branch is selected for a given task.  This module adds a second learning
objective: **behavioral prediction**.  Given the engine's branch activation scores
for a task, the predictor learns which action a specific human would take, and
updates its internal model from observed ground-truth actions.

Architecture
------------
For each (user_id, action) pair the predictor maintains:

1. **Action prior** — base-rate probability the user selects this action,
   learned via exponential moving average from observed frequencies.

2. **Branch–action association weights** — per-branch coefficients that
   capture how strongly a branch's activation correlates with a given action
   for this user.  Updated via gradient-like rule after each observation.

3. **Contextual features** — optional context vector (e.g. trial number,
   cumulative score) that the predictor learns linear weights for.

4. **Outcome signals** — tracks whether previous actions led to positive or
   negative outcomes, learning outcome-conditioned action tendencies
   (e.g. win-stay, lose-shift patterns).

5. **Sequence window** — maintains a sliding window of recent actions and
   outcomes, extracting temporal features like streaks, recent action
   frequencies, and transition patterns.

Prediction
----------
For a new trial the predictor computes an action score:

    score(action) = prior(action)
                  + sum_b( assoc[action][b] * branch_score[b] )
                  + sum_f( context_weight[action][f] * context[f] )
                  + outcome_bias[action][last_outcome_valence]
                  + sequence_features (streaks, transitions, recency)

The action with the highest score is predicted.

Learning
--------
After observing the ground-truth action, the predictor:
- Updates the prior toward the observed action (EMA).
- Strengthens associations between active branches and the correct action.
- Weakens associations between active branches and incorrect actions.
- Updates context weights similarly.
- Updates outcome-conditioned biases based on whether outcomes predict action shifts.
- Updates sequence transition weights from observed action patterns.

Integration with PromptForestEngine
------------------------------------
The predictor sits *alongside* the engine, not inside it.  The validation loop:
1. Runs the engine to get branch scores and selected branch.
2. Calls ``predictor.predict()`` using branch scores.
3. Compares prediction to ground truth.
4. Calls ``predictor.update()`` to adapt the mapping.
5. Calls ``engine.apply_feedback()`` with prediction-accuracy reward
   to drive branch weight changes toward predictive branches.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PredictionResult:
    """Result of a behavioral prediction."""
    predicted_action: str
    action_scores: dict[str, float]
    confidence: float  # max_score - second_max_score (margin)
    prior_contribution: dict[str, float]
    branch_contribution: dict[str, float]


@dataclass
class _OutcomeRecord:
    """A single observation with outcome."""
    action: str
    outcome: float  # positive = good outcome, negative = bad


@dataclass
class _BayesianWeight:
    """A weight with uncertainty tracking (online Bayesian estimation).

    Maintains mean and precision (inverse variance) of the weight.
    More observations → higher precision → more confident predictions.
    """
    mean: float = 0.0
    precision: float = 1.0  # starts uncertain (low precision = high variance)

    @property
    def variance(self) -> float:
        return 1.0 / max(self.precision, 1e-8)

    @property
    def confidence(self) -> float:
        """Confidence in [0, 1] based on precision. Ramps faster to avoid
        suppressing branch signals early on. Reaches 0.5 at precision=5."""
        return 1.0 - 1.0 / (1.0 + self.precision * 0.2)

    def update(self, target: float, observation_weight: float = 1.0) -> float:
        """Bayesian update: shift mean toward target, increase precision.

        Returns the delta applied to the mean.
        """
        # Kalman-like update: gain decreases as precision increases
        gain = observation_weight / (self.precision + observation_weight)
        old_mean = self.mean
        self.mean += gain * (target - self.mean)
        self.precision += observation_weight
        return self.mean - old_mean


@dataclass
class _UserModel:
    """Internal per-user learned model."""
    # Action priors: P(action) base rate
    priors: dict[str, float] = field(default_factory=dict)
    # Group priors: P(group) base rate for hierarchical action heads
    group_priors: dict[str, float] = field(default_factory=dict)
    # Branch-action association weights: assoc[action][branch] = _BayesianWeight
    associations: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Branch-group association weights
    group_associations: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Context feature weights: ctx_weights[action][feature] = weight
    context_weights: dict[str, dict[str, float]] = field(default_factory=dict)
    # Context feature weights for action groups
    group_context_weights: dict[str, dict[str, float]] = field(default_factory=dict)
    # Observation count for learning rate annealing
    n_observations: int = 0
    # Running action counts for base-rate tracking
    action_counts: dict[str, int] = field(default_factory=dict)
    # Running action-group counts for hierarchical prior tracking
    group_action_counts: dict[str, int] = field(default_factory=dict)
    # --- Outcome signals ---
    # outcome_bias[action][valence_bin] = weight
    # valence_bin: "positive", "negative", "neutral"
    outcome_bias: dict[str, dict[str, float]] = field(default_factory=dict)
    # Last outcome for conditioning next prediction
    last_outcome: float = 0.0
    last_action: str = ""
    # --- Sequence window ---
    # Recent history: deque of (action, outcome) pairs
    history: deque = field(default_factory=lambda: deque(maxlen=5))
    # Transition weights: P(next_action | last_action, outcome_valence)
    # transition[last_action][valence_bin][next_action] = weight
    transitions: dict[str, dict[str, dict[str, float]]] = field(default_factory=dict)
    # --- Branch predictiveness tracking ---
    # How well each branch predicts actions (for optimizer feedback)
    # branch_predictiveness[branch] = (correct_count, total_count)
    branch_predictiveness: dict[str, list] = field(default_factory=dict)


class BehavioralPredictor:
    """Learns per-user mappings from branch activations to behavioral actions.

    Parameters
    ----------
    actions:
        The set of possible actions (e.g. ["safe", "risky"] for IGT).
    learning_rate:
        Base learning rate for association updates.
    prior_learning_rate:
        Learning rate for action prior updates.
    prior_smoothing:
        Laplace smoothing for initial priors.
    association_decay:
        L2 regularization on association weights (prevents unbounded growth).
    context_learning_rate:
        Learning rate for context feature weight updates.
    min_learning_rate:
        Floor for annealed learning rate.
    anneal_rate:
        How quickly learning rate decays with observations.
        Effective LR = max(min_lr, base_lr / (1 + anneal_rate * n_obs)).
    """

    def __init__(
        self,
        actions: list[str],
        action_groups: dict[str, str] | None = None,
        learning_rate: float = 0.15,
        prior_learning_rate: float = 0.08,
        prior_smoothing: float = 0.01,
        association_decay: float = 0.005,
        context_learning_rate: float = 0.05,
        min_learning_rate: float = 0.02,
        anneal_rate: float = 0.01,
        outcome_learning_rate: float = 0.12,
        sequence_window: int = 5,
        transition_learning_rate: float = 0.10,
    ) -> None:
        self.actions = list(actions)
        self.n_actions = len(actions)
        if action_groups is None:
            self.action_groups = {action: action for action in self.actions}
        else:
            self.action_groups = {action: str(action_groups.get(action, action)) for action in self.actions}
        self.groups = list(dict.fromkeys(self.action_groups[action] for action in self.actions))
        self.group_to_actions = {
            group: [action for action in self.actions if self.action_groups[action] == group]
            for group in self.groups
        }
        self.use_hierarchy = len(self.groups) < self.n_actions
        self.lr = learning_rate
        self.prior_lr = prior_learning_rate
        self.prior_smoothing = prior_smoothing
        self.assoc_decay = association_decay
        self.ctx_lr = context_learning_rate
        self.min_lr = min_learning_rate
        self.anneal_rate = anneal_rate
        self.outcome_lr = outcome_learning_rate
        self.sequence_window = sequence_window
        self.transition_lr = transition_learning_rate
        self.user_adapter_k = 18.0
        self.global_lr_scale = 0.35
        self.use_global_adapter = False

        self._users: dict[str, _UserModel] = {}
        self._global_model = self._new_model()

    def _new_model(self) -> _UserModel:
        uniform = 1.0 / self.n_actions
        group_uniform = 1.0 / max(1, len(self.groups))
        valence_bins = ["positive", "negative", "neutral"]
        return _UserModel(
            priors={a: uniform for a in self.actions},
            group_priors={g: group_uniform for g in self.groups},
            associations={a: {} for a in self.actions},
            group_associations={g: {} for g in self.groups},
            context_weights={a: {} for a in self.actions},
            group_context_weights={g: {} for g in self.groups},
            action_counts={a: 0 for a in self.actions},
            group_action_counts={g: 0 for g in self.groups},
            outcome_bias={a: {v: 0.0 for v in valence_bins} for a in self.actions},
            transitions={
                a: {v: {a2: 0.0 for a2 in self.actions} for v in valence_bins}
                for a in self.actions
            },
            history=deque(maxlen=self.sequence_window),
        )

    def _get_or_create_user(self, user_id: str) -> _UserModel:
        if user_id not in self._users:
            self._users[user_id] = self._new_model()
        return self._users[user_id]

    @staticmethod
    def _valence_bin(outcome: float) -> str:
        if outcome > 0:
            return "positive"
        elif outcome < 0:
            return "negative"
        return "neutral"

    def _effective_lr(self, model: _UserModel) -> float:
        return max(self.min_lr, self.lr / (1.0 + self.anneal_rate * model.n_observations))

    def _branch_signal_scale(self, model: _UserModel, branch_scores: dict[str, float]) -> float:
        if not branch_scores or model.n_observations < 8:
            return 0.0

        predictiveness: list[float] = []
        for branch_name in branch_scores:
            stats = model.branch_predictiveness.get(branch_name)
            if not stats or stats[1] < 3:
                continue
            accuracy = stats[0] / max(1, stats[1])
            shrunk = (accuracy * stats[1] + 0.5 * 5) / (stats[1] + 5)
            predictiveness.append(shrunk)

        if not predictiveness:
            return 0.25

        mean_pred = sum(predictiveness) / len(predictiveness)
        return max(0.0, min(1.0, (mean_pred - 0.5) / 0.1))

    @staticmethod
    def _head_score(
        labels: list[str],
        priors: dict[str, float],
        associations: dict[str, dict[str, float]],
        context_weights: dict[str, dict[str, float]],
        branch_scores: dict[str, float],
        context: dict[str, float],
    ) -> dict[str, float]:
        scores: dict[str, float] = {}
        for label in labels:
            score = priors.get(label, 0.0)
            for branch, activation in branch_scores.items():
                score += associations.get(label, {}).get(branch, 0.0) * activation
            for feat, val in context.items():
                score += context_weights.get(label, {}).get(feat, 0.0) * val
            scores[label] = score
        return scores

    def predict(
        self,
        user_id: str,
        branch_scores: dict[str, float],
        context: dict[str, float] | None = None,
    ) -> PredictionResult:
        """Predict the most likely action for this user given branch activation scores.

        Parameters
        ----------
        user_id:
            Unique identifier for the human participant.
        branch_scores:
            Dict of branch_name → activation score from the engine's routing.
            These are the raw scores before selection (all branches, not just top-k).
        context:
            Optional contextual features (e.g. {"trial_progress": 0.5, "cumulative": 100}).

        Returns
        -------
        PredictionResult with the predicted action and score breakdown.
        """
        model = self._get_or_create_user(user_id)
        context = context or {}

        action_scores: dict[str, float] = {}
        prior_contrib: dict[str, float] = {}
        branch_contrib: dict[str, float] = {}

        # Compute sequence features from history window
        seq_features = self._compute_sequence_features(model)
        group_context = dict(context)
        if seq_features and model.n_observations >= 5:
            group_context.update({f"group::{feat}": val for feat, val in seq_features.items()})

        global_model = self._global_model
        use_global = self.use_global_adapter and global_model.n_observations >= 8
        user_mix = model.n_observations / (model.n_observations + self.user_adapter_k)
        global_mix = (1.0 - user_mix) if use_global else 0.0

        group_scores: dict[str, float] = {}
        if self.use_hierarchy:
            user_group_scores = self._head_score(
                labels=self.groups,
                priors=model.group_priors,
                associations=model.group_associations,
                context_weights=model.group_context_weights,
                branch_scores=branch_scores,
                context=group_context,
            )
            if use_global:
                global_group_scores = self._head_score(
                    labels=self.groups,
                    priors=global_model.group_priors,
                    associations=global_model.group_associations,
                    context_weights=global_model.group_context_weights,
                    branch_scores=branch_scores,
                    context=group_context,
                )
                group_scores = {
                    group: (user_mix * user_group_scores.get(group, 0.0))
                    + (global_mix * global_group_scores.get(group, 0.0))
                    for group in self.groups
                }
            else:
                group_scores = user_group_scores

        branch_scale = self._branch_signal_scale(model, branch_scores)

        for action in self.actions:
            # Prior contribution
            user_prior = model.priors.get(action, 1.0 / self.n_actions)
            global_prior = global_model.priors.get(action, 1.0 / self.n_actions) if use_global else 0.0
            prior_score = (user_mix * user_prior) + (global_mix * global_prior)
            prior_contrib[action] = prior_score

            # Branch association contribution (gradient-based point estimates)
            assocs = model.associations.get(action, {})
            global_assocs = global_model.associations.get(action, {}) if use_global else {}
            b_score = 0.0
            for branch, activation in branch_scores.items():
                user_weight = assocs.get(branch, 0.0)
                global_weight = global_assocs.get(branch, 0.0)
                weight = (user_mix * user_weight) + (global_mix * global_weight)
                b_score += weight * activation
            b_score *= branch_scale
            branch_contrib[action] = b_score

            # Context contribution
            ctx_score = 0.0
            ctx_w = model.context_weights.get(action, {})
            global_ctx_w = global_model.context_weights.get(action, {}) if use_global else {}
            for feat, val in context.items():
                weight = (user_mix * ctx_w.get(feat, 0.0)) + (global_mix * global_ctx_w.get(feat, 0.0))
                ctx_score += weight * val

            # Outcome-conditioned bias: how does last outcome affect this action?
            # Scale down to prevent overwhelming priors with few observations
            outcome_score = 0.0
            if model.last_action and model.n_observations >= 5:
                valence = self._valence_bin(model.last_outcome)
                raw = model.outcome_bias.get(action, {}).get(valence, 0.0)
                # Ramp up contribution as we see more data
                ramp = min(1.0, (model.n_observations - 5) / 20.0)
                outcome_score = raw * 0.3 * ramp

            # Transition score: P(this action | last_action, outcome_valence)
            transition_score = 0.0
            if model.last_action and model.n_observations >= 10:
                valence = self._valence_bin(model.last_outcome)
                trans = model.transitions.get(model.last_action, {}).get(valence, {})
                raw = trans.get(action, 0.0)
                ramp = min(1.0, (model.n_observations - 10) / 20.0)
                transition_score = raw * 0.25 * ramp

            # Sequence feature score (streak, recency — only core features)
            seq_score = 0.0
            if seq_features and model.n_observations >= 5:
                ctx_w_action = model.context_weights.get(action, {})
                for feat, val in seq_features.items():
                    seq_score += ctx_w_action.get(feat, 0.0) * val
                seq_score *= 0.3  # dampen to prevent overfitting

            hierarchy_score = 0.0
            if self.use_hierarchy:
                hierarchy_score = 0.6 * group_scores.get(self.action_groups[action], 0.0)

            action_scores[action] = (
                prior_score + b_score + ctx_score
                + outcome_score + transition_score + seq_score + hierarchy_score
            )

        # Confidence: margin between top two scores
        sorted_scores = sorted(action_scores.values(), reverse=True)
        confidence = (sorted_scores[0] - sorted_scores[1]) if len(sorted_scores) >= 2 else 1.0

        predicted = max(action_scores, key=action_scores.get)

        return PredictionResult(
            predicted_action=predicted,
            action_scores=action_scores,
            confidence=confidence,
            prior_contribution=prior_contrib,
            branch_contribution=branch_contrib,
        )

    def _compute_sequence_features(self, model: _UserModel) -> dict[str, float]:
        """Extract temporal features from the history window.

        Uses multiple horizons so the predictor can represent regime shifts
        without needing a full recurrent model.
        """
        if len(model.history) < 2:
            return {}

        history = list(model.history)
        features: dict[str, float] = {}

        def _window(n: int) -> list[_OutcomeRecord]:
            return history[-min(n, len(history)) :]

        short = _window(4)
        medium = _window(8)
        long = _window(12)

        def _action_share(window: list[_OutcomeRecord], action: str) -> float:
            return sum(1 for item in window if item.action == action) / max(1, len(window))

        def _switch_rate(window: list[_OutcomeRecord]) -> float:
            if len(window) < 2:
                return 0.0
            switches = sum(1 for prev, cur in zip(window, window[1:]) if prev.action != cur.action)
            return switches / max(1, len(window) - 1)

        def _mean_outcome(window: list[_OutcomeRecord]) -> float:
            if not window:
                return 0.0
            mean_out = sum(item.outcome for item in window) / len(window)
            return max(-1.0, min(1.0, mean_out / (abs(mean_out) + 50.0)))

        action_counts: dict[str, int] = {}
        for item in long:
            action_counts[item.action] = action_counts.get(item.action, 0) + 1
        dominant = max(action_counts, key=action_counts.get)
        dominant_share = action_counts[dominant] / max(1, len(long))

        streak_action = history[-1].action
        streak_len = 1
        for item in reversed(history[:-1]):
            if item.action != streak_action:
                break
            streak_len += 1

        for action in action_counts:
            features[f"share_short::{action}"] = _action_share(short, action)
            features[f"share_medium::{action}"] = _action_share(medium, action)
            features[f"share_long::{action}"] = _action_share(long, action)

        features["recent_dominant_is_" + dominant] = dominant_share
        features["switch_rate_short"] = _switch_rate(short)
        features["switch_rate_long"] = _switch_rate(long)
        features["current_streak_norm"] = min(1.0, streak_len / 6.0)
        features["current_streak_is_" + streak_action] = 1.0
        features["last_outcome_sign"] = 1.0 if history[-1].outcome > 0 else -1.0
        features["recent_outcome_mean_short"] = _mean_outcome(short)
        features["recent_outcome_mean_long"] = _mean_outcome(long)
        features["regime_stable"] = dominant_share * (1.0 - features["switch_rate_long"])
        features["regime_volatile"] = (1.0 - dominant_share) * features["switch_rate_long"]
        features["regime_reversal_pressure"] = max(0.0, -features["recent_outcome_mean_long"]) * dominant_share

        if history[-1].outcome > 0:
            features["after_win"] = 1.0
        else:
            features["after_loss"] = 1.0

        return features

    def update(
        self,
        user_id: str,
        branch_scores: dict[str, float],
        actual_action: str,
        context: dict[str, float] | None = None,
        outcome: float | None = None,
    ) -> dict[str, Any]:
        """Update the user model based on observed ground-truth action.

        Parameters
        ----------
        user_id:
            The human participant.
        branch_scores:
            Branch activation scores from this trial.
        actual_action:
            The action the human actually took.
        context:
            Optional contextual features.
        outcome:
            Numeric outcome of this action (e.g. net payoff). Positive = good,
            negative = bad. Used to learn outcome-conditioned biases and
            win-stay/lose-shift patterns.

        Returns
        -------
        Dict with update statistics (delta magnitudes, new priors, etc.).
        """
        model = self._get_or_create_user(user_id)
        context = context or {}
        if outcome is None:
            outcome = 0.0
        model.n_observations += 1
        model.action_counts[actual_action] = model.action_counts.get(actual_action, 0) + 1
        actual_group = self.action_groups.get(actual_action, actual_action)
        model.group_action_counts[actual_group] = model.group_action_counts.get(actual_group, 0) + 1

        lr = self._effective_lr(model)
        stats: dict[str, Any] = {"lr": round(lr, 4), "n_obs": model.n_observations}

        # --- Update priors ---
        total_obs = max(1, model.n_observations)
        prior_deltas: dict[str, float] = {}
        for action in self.actions:
            observed_freq = model.action_counts.get(action, 0) / total_obs
            old_prior = model.priors[action]
            new_prior = (1.0 - self.prior_lr) * old_prior + self.prior_lr * observed_freq
            new_prior = max(self.prior_smoothing / self.n_actions, new_prior)
            model.priors[action] = new_prior
            prior_deltas[action] = new_prior - old_prior

        total_prior = sum(model.priors.values())
        if total_prior > 0:
            for action in self.actions:
                model.priors[action] /= total_prior

        stats["priors"] = {a: round(v, 4) for a, v in model.priors.items()}
        stats["prior_deltas"] = {a: round(v, 4) for a, v in prior_deltas.items()}

        group_prior_deltas: dict[str, float] = {}
        if self.use_hierarchy:
            for group in self.groups:
                observed_freq = model.group_action_counts.get(group, 0) / total_obs
                old_prior = model.group_priors[group]
                new_prior = (1.0 - self.prior_lr) * old_prior + self.prior_lr * observed_freq
                new_prior = max(self.prior_smoothing / max(1, len(self.groups)), new_prior)
                model.group_priors[group] = new_prior
                group_prior_deltas[group] = new_prior - old_prior

            total_group_prior = sum(model.group_priors.values())
            if total_group_prior > 0:
                for group in self.groups:
                    model.group_priors[group] /= total_group_prior

            stats["group_priors"] = {g: round(v, 4) for g, v in model.group_priors.items()}
            stats["group_prior_deltas"] = {g: round(v, 4) for g, v in group_prior_deltas.items()}

        # --- Update branch-action associations (gradient-based) ---
        max_bs = max(abs(v) for v in branch_scores.values()) if branch_scores else 1.0
        max_bs = max(max_bs, 1e-8)

        assoc_delta_sum = 0.0
        group_assoc_delta_sum = 0.0
        for branch, raw_activation in branch_scores.items():
            activation = raw_activation / max_bs
            if abs(activation) < 0.01:
                continue

            for action in self.actions:
                assocs = model.associations[action]
                current = assocs.get(branch, 0.0)

                if action == actual_action:
                    delta = lr * activation * (1.0 - current)
                else:
                    delta = -lr * activation * (current + 0.1) * 0.5

                delta -= self.assoc_decay * current

                new_val = max(-2.0, min(2.0, current + delta))
                assocs[branch] = new_val
                assoc_delta_sum += abs(delta)

            if self.use_hierarchy:
                for group in self.groups:
                    assocs = model.group_associations[group]
                    current = assocs.get(branch, 0.0)

                    if group == actual_group:
                        delta = lr * activation * (1.0 - current)
                    else:
                        delta = -lr * activation * (current + 0.1) * 0.5

                    delta -= self.assoc_decay * current
                    new_val = max(-2.0, min(2.0, current + delta))
                    assocs[branch] = new_val
                    group_assoc_delta_sum += abs(delta)

            # --- Track branch predictiveness (for optimizer connection) ---
            if branch not in model.branch_predictiveness:
                model.branch_predictiveness[branch] = [0, 0]  # [correct, total]
            bp = model.branch_predictiveness[branch]
            # Check if this branch's weight for the actual action was the highest
            action_weights = {}
            for a in self.actions:
                action_weights[a] = model.associations[a].get(branch, 0.0)
            if action_weights:
                predicted_by_branch = max(action_weights, key=action_weights.get)
                bp[1] += 1
                if predicted_by_branch == actual_action:
                    bp[0] += 1

        stats["assoc_delta_sum"] = round(assoc_delta_sum, 4)
        if self.use_hierarchy:
            stats["group_assoc_delta_sum"] = round(group_assoc_delta_sum, 4)

        # --- Update context weights (includes sequence features) ---
        # Merge sequence features into context for learning
        seq_features = self._compute_sequence_features(model)
        all_context = {**context, **seq_features}
        group_context = {**context, **{f"group::{feat}": val for feat, val in seq_features.items()}}

        ctx_delta_sum = 0.0
        for feat, val in all_context.items():
            if abs(val) < 1e-8:
                continue
            for action in self.actions:
                ctx_w = model.context_weights[action]
                current = ctx_w.get(feat, 0.0)

                if action == actual_action:
                    delta = self.ctx_lr * val * (1.0 - current)
                else:
                    delta = -self.ctx_lr * val * current * 0.3

                new_val = max(-1.0, min(1.0, current + delta))
                ctx_w[feat] = new_val
                ctx_delta_sum += abs(delta)

        stats["ctx_delta_sum"] = round(ctx_delta_sum, 4)

        group_ctx_delta_sum = 0.0
        if self.use_hierarchy:
            for feat, val in group_context.items():
                if abs(val) < 1e-8:
                    continue
                for group in self.groups:
                    ctx_w = model.group_context_weights[group]
                    current = ctx_w.get(feat, 0.0)

                    if group == actual_group:
                        delta = self.ctx_lr * val * (1.0 - current)
                    else:
                        delta = -self.ctx_lr * val * current * 0.3

                    new_val = max(-1.0, min(1.0, current + delta))
                    ctx_w[feat] = new_val
                    group_ctx_delta_sum += abs(delta)

            stats["group_ctx_delta_sum"] = round(group_ctx_delta_sum, 4)

        # --- Update outcome-conditioned biases ---
        # Learn: after a positive/negative outcome, which action does this user pick?
        if model.last_action:
            valence = self._valence_bin(model.last_outcome)
            for action in self.actions:
                bias = model.outcome_bias[action]
                current = bias.get(valence, 0.0)
                if action == actual_action:
                    delta = self.outcome_lr * (1.0 - current)
                else:
                    delta = -self.outcome_lr * current * 0.5
                bias[valence] = max(-1.0, min(1.0, current + delta))

        # --- Update transition weights ---
        # Learn: P(actual_action | last_action, last_outcome_valence)
        if model.last_action:
            valence = self._valence_bin(model.last_outcome)
            trans = model.transitions.get(model.last_action, {}).get(valence, {})
            for action in self.actions:
                current = trans.get(action, 0.0)
                if action == actual_action:
                    delta = self.transition_lr * (1.0 - current)
                else:
                    delta = -self.transition_lr * current * 0.3
                trans[action] = max(-1.0, min(1.0, current + delta))

        # --- Update history and last-outcome state ---
        model.history.append(_OutcomeRecord(action=actual_action, outcome=outcome))
        model.last_action = actual_action
        model.last_outcome = outcome

        global_model = self._global_model
        global_model.n_observations += 1
        global_model.action_counts[actual_action] = global_model.action_counts.get(actual_action, 0) + 1
        global_model.group_action_counts[actual_group] = global_model.group_action_counts.get(actual_group, 0) + 1

        global_prior_lr = self.prior_lr * self.global_lr_scale
        global_total_obs = max(1, global_model.n_observations)
        for action in self.actions:
            observed_freq = global_model.action_counts.get(action, 0) / global_total_obs
            old_prior = global_model.priors[action]
            new_prior = (1.0 - global_prior_lr) * old_prior + global_prior_lr * observed_freq
            global_model.priors[action] = max(self.prior_smoothing / self.n_actions, new_prior)
        total_prior = sum(global_model.priors.values())
        if total_prior > 0:
            for action in self.actions:
                global_model.priors[action] /= total_prior

        if self.use_hierarchy:
            for group in self.groups:
                observed_freq = global_model.group_action_counts.get(group, 0) / global_total_obs
                old_prior = global_model.group_priors[group]
                new_prior = (1.0 - global_prior_lr) * old_prior + global_prior_lr * observed_freq
                global_model.group_priors[group] = max(
                    self.prior_smoothing / max(1, len(self.groups)), new_prior
                )
            total_group_prior = sum(global_model.group_priors.values())
            if total_group_prior > 0:
                for group in self.groups:
                    global_model.group_priors[group] /= total_group_prior

        global_lr = lr * self.global_lr_scale
        max_bs = max(abs(v) for v in branch_scores.values()) if branch_scores else 1.0
        max_bs = max(max_bs, 1e-8)
        for branch, raw_activation in branch_scores.items():
            activation = raw_activation / max_bs
            if abs(activation) < 0.01:
                continue

            for action in self.actions:
                assocs = global_model.associations[action]
                current = assocs.get(branch, 0.0)
                if action == actual_action:
                    delta = global_lr * activation * (1.0 - current)
                else:
                    delta = -global_lr * activation * (current + 0.1) * 0.5
                delta -= self.assoc_decay * current
                assocs[branch] = max(-2.0, min(2.0, current + delta))

            if self.use_hierarchy:
                for group in self.groups:
                    assocs = global_model.group_associations[group]
                    current = assocs.get(branch, 0.0)
                    if group == actual_group:
                        delta = global_lr * activation * (1.0 - current)
                    else:
                        delta = -global_lr * activation * (current + 0.1) * 0.5
                    delta -= self.assoc_decay * current
                    assocs[branch] = max(-2.0, min(2.0, current + delta))

        for feat, val in all_context.items():
            if abs(val) < 1e-8:
                continue
            for action in self.actions:
                ctx_w = global_model.context_weights[action]
                current = ctx_w.get(feat, 0.0)
                if action == actual_action:
                    delta = (self.ctx_lr * self.global_lr_scale) * val * (1.0 - current)
                else:
                    delta = -(self.ctx_lr * self.global_lr_scale) * val * current * 0.3
                ctx_w[feat] = max(-1.0, min(1.0, current + delta))

        if self.use_hierarchy:
            for feat, val in group_context.items():
                if abs(val) < 1e-8:
                    continue
                for group in self.groups:
                    ctx_w = global_model.group_context_weights[group]
                    current = ctx_w.get(feat, 0.0)
                    if group == actual_group:
                        delta = (self.ctx_lr * self.global_lr_scale) * val * (1.0 - current)
                    else:
                        delta = -(self.ctx_lr * self.global_lr_scale) * val * current * 0.3
                    ctx_w[feat] = max(-1.0, min(1.0, current + delta))

        return stats

    def get_user_model_summary(self, user_id: str) -> dict[str, Any]:
        """Get a summary of the learned model for a user."""
        model = self._users.get(user_id)
        if model is None:
            return {"user_id": user_id, "status": "no_model"}

        # Top associations per action
        top_assocs: dict[str, list[tuple[str, float]]] = {}
        for action in self.actions:
            assocs = model.associations.get(action, {})
            sorted_a = sorted(assocs.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
            top_assocs[action] = [(b, round(w, 4)) for b, w in sorted_a]

        return {
            "user_id": user_id,
            "n_observations": model.n_observations,
            "priors": {a: round(v, 4) for a, v in model.priors.items()},
            "group_priors": {g: round(v, 4) for g, v in model.group_priors.items()},
            "action_counts": dict(model.action_counts),
            "group_action_counts": dict(model.group_action_counts),
            "top_associations": top_assocs,
        }

    def prediction_accuracy_reward(
        self,
        predicted: str,
        actual: str,
        confidence: float,
    ) -> float:
        """Compute a reward signal suitable for engine.apply_feedback().

        Maps prediction accuracy into a reward that works with the engine's
        weight update mechanism (apply_feedback uses delta = 0.5 * lr * (reward - 0.5)).

        - Correct prediction: reward in [0.7, 1.0] scaled by confidence
        - Wrong prediction: reward in [0.0, 0.3] scaled by inverse confidence

        This ensures correct predictions always push weights up and wrong
        predictions always push weights down, unlike binary 0/1 which
        gets diluted by the blending in apply_feedback.
        """
        predicted_group = self.action_groups.get(predicted, predicted)
        actual_group = self.action_groups.get(actual, actual)

        if predicted == actual:
            # High reward, boosted by confidence
            return 0.72 + 0.28 * min(1.0, confidence)
        elif predicted_group == actual_group:
            # Partial credit for getting the coarse safe/risky decision right.
            return 0.56
        else:
            # Low reward, lowered further by confidence (confident wrong = worse)
            return 0.28 - 0.18 * min(1.0, confidence)

    # ------------------------------------------------------------------
    # Predictor→Optimizer connection (improvement #5)
    # ------------------------------------------------------------------

    def get_branch_predictiveness(self, user_id: str) -> dict[str, float]:
        """Get how predictive each branch is for this user's behavior.

        Returns a dict of branch_name → predictiveness score in [0, 1].
        This can be fed back to the engine to boost routing toward
        branches that are informative about user behavior.
        """
        model = self._users.get(user_id)
        if model is None:
            return {}

        scores: dict[str, float] = {}
        for branch, (correct, total) in model.branch_predictiveness.items():
            if total < 3:
                continue  # not enough data
            accuracy = correct / total
            # Shrinkage toward 0.5 with low counts
            shrunk = (accuracy * total + 0.5 * 5) / (total + 5)
            scores[branch] = shrunk
        return scores

    def predictiveness_weight_bonus(
        self, user_id: str, branch_scores: dict[str, float]
    ) -> dict[str, float]:
        """Compute weight bonuses for branches based on their predictiveness.

        Returns adjusted branch scores where more predictive branches
        get a boost. This is the bridge from predictor → router.
        """
        pred_scores = self.get_branch_predictiveness(user_id)
        if not pred_scores:
            return branch_scores

        adjusted = dict(branch_scores)
        for branch, score in adjusted.items():
            predictiveness = pred_scores.get(branch, 0.5)
            # Bonus: predictive branches get up to 20% boost
            bonus = (predictiveness - 0.5) * 0.4  # range: [-0.2, +0.2]
            adjusted[branch] = score + bonus
        return adjusted

    # ------------------------------------------------------------------
    # Transfer learning across users
    # ------------------------------------------------------------------

    def _user_signature(self, model: _UserModel) -> dict[str, float]:
        """Compute a lightweight behavioral signature for similarity matching.

        Uses action priors, outcome biases, and transition tendencies —
        all things that characterize a user's behavioral type.
        """
        sig: dict[str, float] = {}
        # Action priors (most important signal)
        for a, v in model.priors.items():
            sig[f"prior_{a}"] = v
        # Win-stay / lose-shift tendencies
        for action in self.actions:
            ob = model.outcome_bias.get(action, {})
            sig[f"ob_{action}_pos"] = ob.get("positive", 0.0)
            sig[f"ob_{action}_neg"] = ob.get("negative", 0.0)
        # Dominant transition patterns
        for src_action in self.actions:
            for valence in ["positive", "negative"]:
                trans = model.transitions.get(src_action, {}).get(valence, {})
                for dst_action in self.actions:
                    sig[f"tr_{src_action}_{valence}_{dst_action}"] = trans.get(dst_action, 0.0)
        return sig

    def _signature_similarity(
        self, sig_a: dict[str, float], sig_b: dict[str, float]
    ) -> float:
        """Cosine similarity between two user signatures."""
        keys = set(sig_a) | set(sig_b)
        if not keys:
            return 0.0
        dot = sum(sig_a.get(k, 0.0) * sig_b.get(k, 0.0) for k in keys)
        mag_a = math.sqrt(sum(sig_a.get(k, 0.0) ** 2 for k in keys))
        mag_b = math.sqrt(sum(sig_b.get(k, 0.0) ** 2 for k in keys))
        if mag_a < 1e-8 or mag_b < 1e-8:
            return 0.0
        return dot / (mag_a * mag_b)

    def find_similar_users(
        self,
        user_id: str,
        top_k: int = 3,
        min_observations: int = 20,
    ) -> list[tuple[str, float]]:
        """Find the most similar completed users to a given user.

        Parameters
        ----------
        user_id:
            The target user to find matches for.
        top_k:
            Number of similar users to return.
        min_observations:
            Only consider donor users with at least this many observations.

        Returns
        -------
        List of (user_id, similarity_score) tuples, sorted by similarity desc.
        """
        target = self._users.get(user_id)
        if target is None or target.n_observations < 5:
            return []

        target_sig = self._user_signature(target)
        similarities: list[tuple[str, float]] = []

        for uid, model in self._users.items():
            if uid == user_id:
                continue
            if model.n_observations < min_observations:
                continue
            donor_sig = self._user_signature(model)
            sim = self._signature_similarity(target_sig, donor_sig)
            similarities.append((uid, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def warm_start_from_similar(
        self,
        user_id: str,
        top_k: int = 3,
        min_donor_observations: int = 20,
        transfer_weight: float = 0.3,
        min_similarity: float = 0.7,
    ) -> dict[str, Any]:
        """Warm-start a user's model from similar completed users.

        Blends the target user's current weights with a weighted average
        of similar users' learned weights. Only transfers if similarity
        exceeds min_similarity threshold.

        Parameters
        ----------
        user_id:
            The user to warm-start.
        top_k:
            Number of similar donors to average over.
        min_donor_observations:
            Minimum observations a donor must have.
        transfer_weight:
            How much of the donor's weights to blend in (0=none, 1=full).
        min_similarity:
            Minimum similarity threshold for transfer.

        Returns
        -------
        Dict with transfer statistics.
        """
        similar = self.find_similar_users(
            user_id, top_k=top_k, min_observations=min_donor_observations
        )
        # Filter by minimum similarity
        similar = [(uid, sim) for uid, sim in similar if sim >= min_similarity]

        if not similar:
            return {"transferred": False, "reason": "no_similar_users"}

        target = self._users[user_id]
        stats: dict[str, Any] = {
            "transferred": True,
            "donors": [(uid, round(sim, 3)) for uid, sim in similar],
        }

        # Compute similarity-weighted average of donor models
        total_sim = sum(sim for _, sim in similar)

        # Transfer outcome biases
        for action in self.actions:
            for valence in ["positive", "negative", "neutral"]:
                donor_avg = 0.0
                for uid, sim in similar:
                    donor = self._users[uid]
                    donor_avg += (sim / total_sim) * donor.outcome_bias.get(
                        action, {}
                    ).get(valence, 0.0)
                current = target.outcome_bias[action].get(valence, 0.0)
                target.outcome_bias[action][valence] = (
                    (1.0 - transfer_weight) * current + transfer_weight * donor_avg
                )

        # Transfer transition weights
        for src_action in self.actions:
            for valence in ["positive", "negative", "neutral"]:
                for dst_action in self.actions:
                    donor_avg = 0.0
                    for uid, sim in similar:
                        donor = self._users[uid]
                        donor_avg += (sim / total_sim) * donor.transitions.get(
                            src_action, {}
                        ).get(valence, {}).get(dst_action, 0.0)
                    current = target.transitions[src_action][valence].get(dst_action, 0.0)
                    target.transitions[src_action][valence][dst_action] = (
                        (1.0 - transfer_weight) * current + transfer_weight * donor_avg
                    )

        # Transfer context weights (only for features that donors have learned)
        for action in self.actions:
            donor_ctx: dict[str, float] = {}
            donor_ctx_count: dict[str, float] = {}
            for uid, sim in similar:
                donor = self._users[uid]
                for feat, w in donor.context_weights.get(action, {}).items():
                    donor_ctx[feat] = donor_ctx.get(feat, 0.0) + (sim / total_sim) * w
                    donor_ctx_count[feat] = donor_ctx_count.get(feat, 0.0) + 1

            for feat, avg_w in donor_ctx.items():
                current = target.context_weights[action].get(feat, 0.0)
                target.context_weights[action][feat] = (
                    (1.0 - transfer_weight) * current + transfer_weight * avg_w
                )

        return stats

    @property
    def user_count(self) -> int:
        return len(self._users)

    def reset_user(self, user_id: str) -> None:
        """Reset a user's learned model."""
        self._users.pop(user_id, None)

    @staticmethod
    def brain_output_features(brain_output: dict[str, Any] | None) -> dict[str, float]:
        """Flatten a Prompt Forest brain readout into predictor features.

        This is the preferred integration path when Prompt Forest is acting as
        the external cognitive layer for another agent. Predictors should learn
        from the brain's latent state and control signals rather than manually
        reconstructing those fields from routing internals.
        """
        if not isinstance(brain_output, dict):
            return {}

        features: dict[str, float] = {}
        for key, value in (brain_output.get("state", {}) or {}).items():
            if isinstance(value, (int, float)):
                features[f"brain_state::{key}"] = float(value)
        for drive in brain_output.get("dominant_drives", []) or []:
            features[f"brain_drive::{drive}"] = 1.0
        for key, value in (brain_output.get("branch_activations", {}) or {}).items():
            if isinstance(value, (int, float)):
                features[f"brain_branch::{key}"] = float(value)
        for key, value in (brain_output.get("control_signals", {}) or {}).items():
            if isinstance(value, (int, float)):
                features[f"brain_control::{key}"] = float(value)
        for key, value in (brain_output.get("action_tendencies", {}) or {}).items():
            if isinstance(value, (int, float)):
                features[f"brain_tendency::{key}"] = float(value)
        for key, value in (brain_output.get("memory_biases", {}) or {}).items():
            if isinstance(value, (int, float)):
                features[f"brain_memory::{key}"] = float(value)
        for key, value in (brain_output.get("state_summary", {}) or {}).items():
            if isinstance(value, (int, float)):
                features[f"brain_summary::{key}"] = float(value)
        regime = str(brain_output.get("regime", "") or "").strip()
        if regime:
            features[f"brain_regime::{regime}"] = 1.0
        conflicts = brain_output.get("conflicts", []) or []
        features["brain_conflict_count"] = float(len(conflicts))
        max_intensity = 0.0
        for conflict in conflicts:
            if not isinstance(conflict, dict):
                continue
            name = str(conflict.get("name", "") or "").strip()
            intensity = conflict.get("intensity")
            if name:
                features[f"brain_conflict::{name}"] = 1.0
            if isinstance(intensity, (int, float)):
                max_intensity = max(max_intensity, float(intensity))
        features["brain_conflict_max_intensity"] = max_intensity
        return features
