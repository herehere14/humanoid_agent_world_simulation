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
    # Branch-action association weights: assoc[action][branch] = _BayesianWeight
    associations: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Context feature weights: ctx_weights[action][feature] = weight
    context_weights: dict[str, dict[str, float]] = field(default_factory=dict)
    # Observation count for learning rate annealing
    n_observations: int = 0
    # Running action counts for base-rate tracking
    action_counts: dict[str, int] = field(default_factory=dict)
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

        self._users: dict[str, _UserModel] = {}

    def _get_or_create_user(self, user_id: str) -> _UserModel:
        if user_id not in self._users:
            uniform = 1.0 / self.n_actions
            valence_bins = ["positive", "negative", "neutral"]
            model = _UserModel(
                priors={a: uniform for a in self.actions},
                associations={a: {} for a in self.actions},
                context_weights={a: {} for a in self.actions},
                action_counts={a: 0 for a in self.actions},
                outcome_bias={a: {v: 0.0 for v in valence_bins} for a in self.actions},
                transitions={
                    a: {v: {a2: 0.0 for a2 in self.actions} for v in valence_bins}
                    for a in self.actions
                },
                history=deque(maxlen=self.sequence_window),
            )
            self._users[user_id] = model
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

        for action in self.actions:
            # Prior contribution
            prior_score = model.priors.get(action, 1.0 / self.n_actions)
            prior_contrib[action] = prior_score

            # Branch association contribution (gradient-based point estimates)
            assocs = model.associations.get(action, {})
            b_score = 0.0
            for branch, activation in branch_scores.items():
                weight = assocs.get(branch, 0.0)
                b_score += weight * activation
            branch_contrib[action] = b_score

            # Context contribution
            ctx_score = 0.0
            ctx_w = model.context_weights.get(action, {})
            for feat, val in context.items():
                ctx_score += ctx_w.get(feat, 0.0) * val

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

            action_scores[action] = (
                prior_score + b_score + ctx_score
                + outcome_score + transition_score + seq_score
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

        Kept minimal (4 features) to avoid overfitting with limited observations.
        """
        if len(model.history) < 2:
            return {}

        history = list(model.history)
        features: dict[str, float] = {}

        # 1. Last action was same as most-common recent action (momentum)
        action_counts: dict[str, int] = {}
        for h in history:
            action_counts[h.action] = action_counts.get(h.action, 0) + 1
        dominant = max(action_counts, key=action_counts.get)
        features["recent_dominant_is_" + dominant] = action_counts[dominant] / len(history)

        # 2. Last outcome valence (most important single feature)
        features["last_outcome_sign"] = 1.0 if history[-1].outcome > 0 else -1.0

        # 3. Win-stay signal: did the user repeat after a win?
        #    (Encoded as: last outcome was positive AND we're predicting same action)
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

        # --- Update branch-action associations (gradient-based) ---
        max_bs = max(abs(v) for v in branch_scores.values()) if branch_scores else 1.0
        max_bs = max(max_bs, 1e-8)

        assoc_delta_sum = 0.0
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

        # --- Update context weights (includes sequence features) ---
        # Merge sequence features into context for learning
        seq_features = self._compute_sequence_features(model)
        all_context = {**context, **seq_features}

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
            "action_counts": dict(model.action_counts),
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
        if predicted == actual:
            # High reward, boosted by confidence
            return 0.7 + 0.3 * min(1.0, confidence)
        else:
            # Low reward, lowered further by confidence (confident wrong = worse)
            return 0.3 - 0.2 * min(1.0, confidence)

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
