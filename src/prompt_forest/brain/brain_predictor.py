"""Brain-first behavioral predictor.

The brain IS the prediction mechanism, not an additive feature source.

Causal chain:
  outcome → brain state update → branch competition → control signals → prediction

NOT:
  history → engine routes through LLM → branch scores + brain features → prediction

The predictor learns per-user how the brain's control signals, action
tendencies, regime state, and conflict dynamics map to actual behavioral
choices.  Every prediction is grounded in the brain's latent state.

This is what makes Prompt Forest a "cognitive layer" rather than a
feature generator.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

from .output import BrainOutput


@dataclass
class BrainPredictionResult:
    """Result of brain-driven behavioral prediction."""

    predicted_action: str
    action_scores: dict[str, float]
    confidence: float
    signal_contributions: dict[str, dict[str, float]]


@dataclass
class _UserBrainModel:
    """Per-user learned brain→action mapping."""

    prior: dict[str, float] = field(default_factory=dict)
    action_counts: dict[str, int] = field(default_factory=dict)
    n_observations: int = 0
    # Control signal weights: action → signal_name → weight
    control_weights: dict[str, dict[str, float]] = field(default_factory=dict)
    # Tendency weights: action → tendency_name → weight
    tendency_weights: dict[str, dict[str, float]] = field(default_factory=dict)
    # State variable weights: action → var → weight
    state_weights: dict[str, dict[str, float]] = field(default_factory=dict)
    # Regime bias: action → regime → bias
    regime_bias: dict[str, dict[str, float]] = field(default_factory=dict)
    # Conflict bias: action → scalar bias per unit conflict load
    conflict_bias: dict[str, float] = field(default_factory=dict)
    # Context weights: action → feature_name → weight
    context_weights: dict[str, dict[str, float]] = field(default_factory=dict)
    # Outcome-conditioned bias: action → valence_bin → bias
    outcome_bias: dict[str, dict[str, float]] = field(default_factory=dict)
    # Transition weights: last_action → valence_bin → next_action → weight
    transitions: dict[str, dict[str, dict[str, float]]] = field(
        default_factory=dict
    )
    last_action: str = ""
    last_outcome: float = 0.0
    history: deque = field(default_factory=lambda: deque(maxlen=15))
    # Learned WSLS rates per user
    win_stay_count: int = 0
    win_total: int = 0
    lose_shift_count: int = 0
    lose_total: int = 0
    # Empirical transition counts: (last_action, valence) → {next_action: count}
    empirical_trans: dict[str, dict[str, dict[str, int]]] = field(
        default_factory=dict
    )
    # State summary weights: action → summary_feature → weight
    summary_weights: dict[str, dict[str, float]] = field(default_factory=dict)
    # Composite brain signal weights: action → composite_name → weight
    composite_weights: dict[str, dict[str, float]] = field(default_factory=dict)
    # Regime modifiers: regime → action → learned modifier
    regime_modifiers: dict[str, dict[str, float]] = field(default_factory=dict)
    # Regime-conditioned action counts: regime → action → count
    regime_action_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    regime_total_counts: dict[str, int] = field(default_factory=dict)
    # Track last few regimes for regime transition features
    regime_history: list[str] = field(default_factory=list)
    # Brain state clustering: valence × arousal → action counts
    brain_cluster_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    brain_cluster_totals: dict[str, int] = field(default_factory=dict)
    # Regime change detection
    _last_regime: str = ""
    _regime_stable_count: int = 0
    # Brain state k-NN: (valence, div_mag, frust_div, conf_div, fear_div, outcome_sign, action)
    brain_state_knn: list[tuple[float, float, float, float, float, float, str]] = field(default_factory=list)


# Brain signal names -------------------------------------------------------
_CONTROL_SIGNALS = [
    "approach_drive",
    "avoidance_drive",
    "exploration_drive",
    "switch_pressure",
    "persistence_drive",
    "self_protection",
    "social_openness",
    "cognitive_effort",
]

_TENDENCY_NAMES = ["act", "inhibit", "explore", "exploit", "reflect", "react"]

_STATE_VARS = [
    "confidence",
    "stress",
    "frustration",
    "fear",
    "motivation",
    "impulse",
    "caution",
    "ambition",
    "curiosity",
    "fatigue",
    "trust",
    "self_protection",
    "reflection",
    "goal_commitment",
    "honesty",
    "self_justification",
]


class BrainPredictor:
    """Brain-state-first behavioral predictor.

    Maps brain output (control signals, action tendencies, regime,
    conflicts, raw state) → action prediction using per-user learned
    weights.

    Parameters
    ----------
    actions
        Possible actions (e.g. ``["safe", "risky"]``).
    learning_rate
        Base learning rate for weight updates (annealed over time).
    """

    def __init__(
        self,
        actions: list[str],
        learning_rate: float = 0.12,
        prior_lr: float = 0.06,
        context_lr: float = 0.04,
        min_lr: float = 0.02,
        anneal_rate: float = 0.008,
        prior_smoothing: float = 0.01,
        outcome_lr: float = 0.12,
        transition_lr: float = 0.10,
    ) -> None:
        self.actions = list(actions)
        self.n_actions = len(actions)
        self.lr = learning_rate
        self.prior_lr = prior_lr
        self.context_lr = context_lr
        self.min_lr = min_lr
        self.anneal_rate = anneal_rate
        self.prior_smoothing = prior_smoothing
        self.outcome_lr = outcome_lr
        self.transition_lr = transition_lr
        self._users: dict[str, _UserBrainModel] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _new_model(self) -> _UserBrainModel:
        uniform = 1.0 / self.n_actions
        vbins = ["positive", "negative", "neutral"]
        return _UserBrainModel(
            prior={a: uniform for a in self.actions},
            action_counts={a: 0 for a in self.actions},
            control_weights={
                a: {s: 0.0 for s in _CONTROL_SIGNALS} for a in self.actions
            },
            tendency_weights={
                a: {t: 0.0 for t in _TENDENCY_NAMES} for a in self.actions
            },
            state_weights={
                a: {v: 0.0 for v in _STATE_VARS} for a in self.actions
            },
            regime_bias={a: {} for a in self.actions},
            conflict_bias={a: 0.0 for a in self.actions},
            context_weights={a: {} for a in self.actions},
            outcome_bias={
                a: {v: 0.0 for v in vbins} for a in self.actions
            },
            transitions={
                a: {v: {a2: 0.0 for a2 in self.actions} for v in vbins}
                for a in self.actions
            },
            summary_weights={a: {} for a in self.actions},
            composite_weights={a: {} for a in self.actions},
        )

    def _get_user(self, user_id: str) -> _UserBrainModel:
        if user_id not in self._users:
            self._users[user_id] = self._new_model()
        return self._users[user_id]

    def _effective_lr(self, model: _UserBrainModel) -> float:
        return max(
            self.min_lr,
            self.lr / (1.0 + self.anneal_rate * model.n_observations),
        )

    @staticmethod
    def _valence_bin(outcome: float) -> str:
        if outcome > 0:
            return "positive"
        if outcome < 0:
            return "negative"
        return "neutral"

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        user_id: str,
        brain_output: BrainOutput,
        context: dict[str, float] | None = None,
    ) -> BrainPredictionResult:
        """Predict behaviour from brain state.

        The brain output is the **primary** prediction input.  Context
        features (EV estimates, sequence features) are secondary.
        """
        model = self._get_user(user_id)
        context = context or {}

        controls = brain_output.control_signals.to_dict()
        tendencies = brain_output.action_tendencies.to_dict()
        state = brain_output.state
        regime = brain_output.regime
        conflict_load = sum(c.intensity for c in brain_output.conflicts)

        # Ramp up signals as we accumulate observations
        obs_ramp = min(1.0, model.n_observations / 8.0)
        # Brain ramp: gentle warmup, NO prior_certainty suppression
        brain_ramp = obs_ramp
        behavioral_boost = 1.0

        action_scores: dict[str, float] = {}
        contributions: dict[str, dict[str, float]] = {}

        for action in self.actions:
            contrib: dict[str, float] = {}

            # 1. Prior (base rate)
            prior_score = model.prior[action]
            contrib["prior"] = prior_score

            # 2-4. Brain signals placeholder (regime modulation applied below)
            contrib["brain_regime_mod"] = 0.0

            # 7. Outcome-conditioned bias — KEY signal for mixed users
            outcome_score = 0.0
            if model.last_action and model.n_observations >= 2:
                vbin = self._valence_bin(model.last_outcome)
                raw = model.outcome_bias[action].get(vbin, 0.0)
                ramp = min(1.0, (model.n_observations - 2) / 6.0)
                outcome_score = raw * 0.60 * ramp * behavioral_boost
            contrib["outcome_bias"] = outcome_score

            # 8. Transition score — KEY signal for mixed users
            transition_score = 0.0
            if model.last_action and model.n_observations >= 3:
                vbin = self._valence_bin(model.last_outcome)
                trans = model.transitions.get(model.last_action, {}).get(
                    vbin, {}
                )
                raw = trans.get(action, 0.0)
                ramp = min(1.0, (model.n_observations - 3) / 6.0)
                transition_score = raw * 0.55 * ramp * behavioral_boost
            contrib["transition"] = transition_score

            # 9. Context features (EV, sequence, outcome trajectory)
            ctx_score = 0.0
            for feat, val in context.items():
                w = model.context_weights[action].get(feat, 0.0)
                ctx_score += w * val
            ctx_score *= behavioral_boost
            contrib["context"] = ctx_score

            # 10. Behavioral recency + personalized WSLS
            recency_score = 0.0
            if len(model.history) >= 3:
                recent_5 = list(model.history)[-5:]
                action_share = sum(
                    1 for h in recent_5 if h["action"] == action
                ) / len(recent_5)
                uniform = 1.0 / self.n_actions
                recency_score = (action_share - uniform) * 0.30 * behavioral_boost

                # Personalized WSLS — learned from this user's actual behavior
                if model.last_action:
                    last = model.history[-1]
                    if last["outcome"] > 0 and model.win_total >= 5:
                        ws_rate = model.win_stay_count / model.win_total
                        ws_signal = (ws_rate - 0.5) * 0.35 * behavioral_boost
                        if action == last["action"]:
                            recency_score += ws_signal
                        else:
                            recency_score -= ws_signal
                    elif last["outcome"] < 0 and model.lose_total >= 5:
                        ls_rate = model.lose_shift_count / model.lose_total
                        ls_signal = (ls_rate - 0.5) * 0.35 * behavioral_boost
                        if action != last["action"]:
                            recency_score += ls_signal
                        else:
                            recency_score -= ls_signal
            contrib["recency"] = recency_score

            total = sum(contrib.values())
            action_scores[action] = total
            contributions[action] = contrib

        # ---- Brain k-NN: nonparametric prediction from brain state space ----
        # Instead of additive adjustment, k-NN competes as a PROPORTIONAL
        # blend with base predictions. This gives brain direct control over
        # a fraction of the prediction.
        if regime != "neutral" and len(model.brain_state_knn) >= 15 and brain_ramp > 0.3:
            summary = brain_output.state_summary
            cur_valence = summary.get("mood_valence", 0.5)
            cur_div = summary.get("divergence_magnitude", 0.0)
            cur_frust_div = summary.get("divergence::frustration", 0.0)
            cur_conf_div = summary.get("divergence::confidence", 0.0)
            cur_fear_div = summary.get("divergence::fear", 0.0)
            cur_stress_div = summary.get("divergence::stress", 0.0)
            cur_impulse_div = summary.get("divergence::impulse", 0.0)
            vel_mag = summary.get("velocity_magnitude", 0.0)
            cur_outcome_sign = 1.0 if model.last_outcome > 0 else (-1.0 if model.last_outcome < 0 else 0.0)

            # Find k nearest neighbors in 8D brain state × outcome space
            k = min(12, len(model.brain_state_knn) // 8)
            if k >= 8:
                n_points = len(model.brain_state_knn)
                distances: list[tuple[float, str]] = []
                for idx_pt, (valence, div_mag, frust_div, conf_div, fear_div, outcome_sign, act) in enumerate(model.brain_state_knn):
                    d = (
                        ((cur_valence - valence) * 4.0) ** 2
                        + ((cur_div - div_mag) * 15.0) ** 2
                        + ((cur_frust_div - frust_div) * 10.0) ** 2
                        + ((cur_conf_div - conf_div) * 10.0) ** 2
                        + ((cur_fear_div - fear_div) * 10.0) ** 2
                        + ((cur_outcome_sign - outcome_sign) * 2.5) ** 2
                    )
                    distances.append((d, act, idx_pt))

                # Sort by distance and take k nearest
                distances.sort(key=lambda x: x[0])
                neighbors = distances[:k]

                # Compute distance-weighted AND recency-weighted action rates
                knn_scores: dict[str, float] = {a: 0.0 for a in self.actions}
                total_weight = 0.0
                for dist, act, pt_idx in neighbors:
                    dist_w = 1.0 / (dist + 0.001)  # inverse distance
                    # Recency: recent points weighted higher (exponential decay)
                    recency = 0.5 + 0.5 * (pt_idx / n_points)  # 0.5 → 1.0
                    w = dist_w * recency
                    knn_scores[act] += w
                    total_weight += w

                if total_weight > 0:
                    for a in self.actions:
                        knn_scores[a] /= total_weight

                    # PROPORTIONAL blend: k-NN directly competes with base prediction
                    # alpha = fraction of final score that comes from k-NN
                    alpha = min(0.42, k / 24.0) * brain_ramp

                    # Normalize base scores to probability-like values
                    base_total = sum(max(0.001, action_scores[a]) for a in self.actions)
                    for action in self.actions:
                        base_prob = max(0.001, action_scores[action]) / base_total
                        knn_prob = knn_scores[action]
                        # Blend: final = (1-alpha)*base + alpha*knn
                        blended = (1.0 - alpha) * base_prob + alpha * knn_prob
                        knn_effect = blended * base_total - action_scores[action]
                        action_scores[action] = blended * base_total
                        contributions[action]["brain_knn"] = knn_effect

            # ---- Brain-regime-conditioned empirical prediction (brain-exclusive) ----
            # Use brain macro cluster (positive/negative/neutral) × action counts
            # History-only has regime="neutral" so never tracks clusters → brain-exclusive
            brain_cluster = self._brain_state_cluster(brain_output)
            cluster_counts = model.brain_cluster_counts.get(brain_cluster, {})
            cluster_total = model.brain_cluster_totals.get(brain_cluster, 0)
            if cluster_total >= 8:
                cluster_ramp = min(1.0, cluster_total / 15.0) * brain_ramp
                for action in self.actions:
                    cluster_rate = cluster_counts.get(action, 0) / cluster_total
                    uniform = 1.0 / self.n_actions
                    cluster_effect = (cluster_rate - uniform) * 0.75 * cluster_ramp
                    action_scores[action] += cluster_effect
                    contributions[action]["brain_cluster"] = cluster_effect

            # (Stability gating removed — brain signal should always be active)

        # Confidence: margin
        sorted_scores = sorted(action_scores.values(), reverse=True)
        confidence = (
            (sorted_scores[0] - sorted_scores[1])
            if len(sorted_scores) >= 2
            else 1.0
        )

        predicted = max(action_scores, key=action_scores.get)  # type: ignore[arg-type]

        return BrainPredictionResult(
            predicted_action=predicted,
            action_scores=action_scores,
            confidence=confidence,
            signal_contributions=contributions,
        )

    # ------------------------------------------------------------------
    # Brain state clustering
    # ------------------------------------------------------------------

    # Regime → macro cluster mapping for sufficient observations
    _REGIME_TO_MACRO = {
        "frustrated_reactive": "negative",
        "frustration_building": "negative",
        "guarded_avoidant": "negative",
        "fatigued_guarded": "negative",
        "goal_pursuit": "positive",
        "exploratory_open": "positive",
        "recovering": "positive",
        "baseline_adaptive": "neutral",
        "conflicted_balancing": "neutral",
    }

    @classmethod
    def _brain_state_cluster(cls, brain_output: BrainOutput) -> str:
        """Map brain regime to 3 macro emotional clusters.

        Consolidates 8+ fine-grained regimes into 3 macro clusters for
        sufficient observations per cluster (~83 trials each with 250 training).
        This gives ~21 observations per action per cluster — reliable enough
        for empirical action rate estimation.

        History-only has regime='neutral', so this is brain-exclusive.
        """
        regime = brain_output.regime
        if regime in ("neutral", ""):
            return "unknown"
        return cls._REGIME_TO_MACRO.get(regime, "neutral")

    # ------------------------------------------------------------------
    # Behavioral regime detection
    # ------------------------------------------------------------------

    @staticmethod
    def _behavioral_regime(model: _UserBrainModel) -> str:
        """Classify behavioral regime from recent outcome pattern.

        Uses 5 clear categories for sufficient observations per regime:
        - streak_loss: 3+ consecutive losses (threshold effect)
        - streak_win: 3+ consecutive wins (perseveration trigger)
        - losing: majority losses in last 5 (general pressure)
        - winning: majority wins in last 5 (general confidence)
        - mixed: balanced outcomes
        """
        history = list(model.history)
        if len(history) < 3:
            return "mixed"

        # Consecutive streak (captures threshold effects)
        consec_neg = 0
        for h in reversed(history):
            if h["outcome"] < 0:
                consec_neg += 1
            else:
                break

        consec_pos = 0
        for h in reversed(history):
            if h["outcome"] > 0:
                consec_pos += 1
            else:
                break

        if consec_neg >= 3:
            return "streak_loss"
        if consec_pos >= 3:
            return "streak_win"

        # Recent balance
        recent = history[-min(5, len(history)):]
        neg = sum(1 for h in recent if h["outcome"] < 0)
        pos = sum(1 for h in recent if h["outcome"] > 0)

        if neg >= 3:
            return "losing"
        if pos >= 3:
            return "winning"
        return "mixed"

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def update(
        self,
        user_id: str,
        brain_output: BrainOutput,
        actual_action: str,
        context: dict[str, float] | None = None,
        outcome: float = 0.0,
    ) -> dict[str, Any]:
        """Update the per-user model from observed ground truth."""
        model = self._get_user(user_id)
        context = context or {}
        model.n_observations += 1
        model.action_counts[actual_action] = (
            model.action_counts.get(actual_action, 0) + 1
        )

        lr = self._effective_lr(model)

        # ---- Priors ----
        total_obs = max(1, model.n_observations)
        for action in self.actions:
            freq = model.action_counts.get(action, 0) / total_obs
            old = model.prior[action]
            new = (1.0 - self.prior_lr) * old + self.prior_lr * freq
            model.prior[action] = max(
                self.prior_smoothing / self.n_actions, new
            )
        total_p = sum(model.prior.values())
        if total_p > 0:
            for action in self.actions:
                model.prior[action] /= total_p

        # ---- Regime-conditioned action tracking (uses behavioral regime) ----
        # Only track when brain output is real (not null)
        regime = brain_output.regime
        if regime != "neutral":
            behav_regime = self._behavioral_regime(model)
            if behav_regime not in model.regime_action_counts:
                model.regime_action_counts[behav_regime] = {a: 0 for a in self.actions}
                model.regime_total_counts[behav_regime] = 0
            model.regime_action_counts[behav_regime][actual_action] = (
                model.regime_action_counts[behav_regime].get(actual_action, 0) + 1
            )
            model.regime_total_counts[behav_regime] = model.regime_total_counts.get(behav_regime, 0) + 1
            model.regime_history.append(behav_regime)
            if len(model.regime_history) > 30:
                model.regime_history = model.regime_history[-30:]

            # Brain state k-NN: record 6D state × outcome features
            summary = brain_output.state_summary
            if summary:
                valence = summary.get("mood_valence", 0.5)
                div_mag_val = summary.get("divergence_magnitude", 0.0)
                frust_div = summary.get("divergence::frustration", 0.0)
                conf_div = summary.get("divergence::confidence", 0.0)
                fear_div = summary.get("divergence::fear", 0.0)
                outcome_sign = 1.0 if model.last_outcome > 0 else (-1.0 if model.last_outcome < 0 else 0.0)
                model.brain_state_knn.append((valence, div_mag_val, frust_div, conf_div, fear_div, outcome_sign, actual_action))

            # Brain macro cluster tracking (brain-exclusive)
            brain_cluster = self._brain_state_cluster(brain_output)
            if brain_cluster != "unknown":
                if brain_cluster not in model.brain_cluster_counts:
                    model.brain_cluster_counts[brain_cluster] = {a: 0 for a in self.actions}
                    model.brain_cluster_totals[brain_cluster] = 0
                model.brain_cluster_counts[brain_cluster][actual_action] = (
                    model.brain_cluster_counts[brain_cluster].get(actual_action, 0) + 1
                )
                model.brain_cluster_totals[brain_cluster] = model.brain_cluster_totals.get(brain_cluster, 0) + 1

        # ---- Regime bias ----
        regime = brain_output.regime
        for action in self.actions:
            cur = model.regime_bias[action].get(regime, 0.0)
            if action == actual_action:
                delta = lr * (1.0 - cur)
            else:
                delta = -lr * cur * 0.3
            model.regime_bias[action][regime] = _clamp(cur + delta, -1.0, 1.0)

        # ---- Conflict bias ----
        conflict_load = sum(c.intensity for c in brain_output.conflicts)
        if conflict_load > 0.01:
            for action in self.actions:
                cur = model.conflict_bias[action]
                if action == actual_action:
                    delta = lr * conflict_load * (1.0 - cur)
                else:
                    delta = -lr * conflict_load * cur * 0.3
                model.conflict_bias[action] = _clamp(cur + delta, -1.0, 1.0)

        # ---- Outcome-conditioned bias ----
        if model.last_action:
            vbin = self._valence_bin(model.last_outcome)
            for action in self.actions:
                cur = model.outcome_bias[action].get(vbin, 0.0)
                if action == actual_action:
                    delta = self.outcome_lr * (1.0 - cur)
                else:
                    delta = -self.outcome_lr * cur * 0.5
                model.outcome_bias[action][vbin] = _clamp(
                    cur + delta, -1.0, 1.0
                )

        # ---- Transition weights ----
        if model.last_action:
            vbin = self._valence_bin(model.last_outcome)
            trans = model.transitions.get(model.last_action, {}).get(vbin, {})
            for action in self.actions:
                cur = trans.get(action, 0.0)
                if action == actual_action:
                    delta = self.transition_lr * (1.0 - cur)
                else:
                    delta = -self.transition_lr * cur * 0.3
                trans[action] = _clamp(cur + delta, -1.0, 1.0)

        # ---- Context weights ----
        for feat, val in context.items():
            if abs(val) < 1e-8:
                continue
            for action in self.actions:
                ctx_w = model.context_weights[action]
                cur = ctx_w.get(feat, 0.0)
                if action == actual_action:
                    delta = self.context_lr * val * (1.0 - cur)
                else:
                    delta = -self.context_lr * val * cur * 0.3
                ctx_w[feat] = _clamp(cur + delta, -1.0, 1.0)

        # ---- Track WSLS and empirical transitions ----
        if model.last_action:
            if model.last_outcome > 0:
                model.win_total += 1
                if actual_action == model.last_action:
                    model.win_stay_count += 1
            elif model.last_outcome < 0:
                model.lose_total += 1
                if actual_action != model.last_action:
                    model.lose_shift_count += 1
            # Empirical transition count
            vbin = self._valence_bin(model.last_outcome)
            key = f"{model.last_action}_{vbin}"
            if key not in model.empirical_trans:
                model.empirical_trans[key] = {a: 0 for a in self.actions}
            model.empirical_trans[key][actual_action] += 1

        # ---- Update last state ----
        model.history.append({"action": actual_action, "outcome": outcome})
        model.last_action = actual_action
        model.last_outcome = outcome

        return {
            "n_obs": model.n_observations,
            "lr": round(lr, 4),
            "priors": {a: round(v, 4) for a, v in model.prior.items()},
        }

    # ------------------------------------------------------------------
    # Sequence features
    # ------------------------------------------------------------------

    def compute_sequence_features(self, user_id: str) -> dict[str, float]:
        """Compute temporal features from the user's history window."""
        model = self._users.get(user_id)
        if model is None or len(model.history) < 2:
            return {}

        history = list(model.history)
        features: dict[str, float] = {}

        for window in [3, 5, 10]:
            recent = history[-min(window, len(history)) :]
            for action in self.actions:
                count = sum(1 for h in recent if h["action"] == action)
                features[f"share_{window}::{action}"] = count / len(recent)

        # Switch rate
        switches = sum(
            1
            for i in range(1, len(history))
            if history[i]["action"] != history[i - 1]["action"]
        )
        features["switch_rate"] = switches / max(1, len(history) - 1)

        # Current streak
        last_act = history[-1]["action"]
        streak = 1
        for h in reversed(history[:-1]):
            if h["action"] != last_act:
                break
            streak += 1
        features["streak_len"] = min(1.0, streak / 6.0)
        features[f"streak_is_{last_act}"] = 1.0

        # Recent outcomes
        for window in [3, 5]:
            recent = history[-min(window, len(history)) :]
            outcomes = [h["outcome"] for h in recent]
            if outcomes:
                m = sum(outcomes) / len(outcomes)
                features[f"outcome_mean_{window}"] = m / (abs(m) + 100.0)
                features[f"neg_rate_{window}"] = sum(
                    1 for o in outcomes if o < 0
                ) / len(outcomes)

        # After win/loss
        if history[-1]["outcome"] > 0:
            features["after_win"] = 1.0
        else:
            features["after_loss"] = 1.0

        # Copy-last-action feature (very strong for perseverative users)
        for action in self.actions:
            if last_act == action:
                features[f"last_was_{action}"] = 1.0

        # Dominant action stability
        for action in self.actions:
            long_share = sum(
                1 for h in history if h["action"] == action
            ) / len(history)
            features[f"long_share::{action}"] = long_share

        # Win-stay / lose-shift patterns
        if len(history) >= 3:
            win_stay = 0
            win_total = 0
            lose_shift = 0
            lose_total = 0
            for i in range(1, len(history)):
                prev = history[i - 1]
                cur = history[i]
                if prev["outcome"] > 0:
                    win_total += 1
                    if cur["action"] == prev["action"]:
                        win_stay += 1
                elif prev["outcome"] < 0:
                    lose_total += 1
                    if cur["action"] != prev["action"]:
                        lose_shift += 1
            if win_total >= 2:
                features["win_stay_rate"] = win_stay / win_total
            if lose_total >= 2:
                features["lose_shift_rate"] = lose_shift / lose_total


        # Recent regime stability
        if len(history) >= 6:
            first_half = history[: len(history) // 2]
            second_half = history[len(history) // 2 :]
            for action in self.actions:
                f1 = sum(1 for h in first_half if h["action"] == action) / len(
                    first_half
                )
                f2 = sum(
                    1 for h in second_half if h["action"] == action
                ) / len(second_half)
                features[f"trend::{action}"] = f2 - f1

        return features

    # ------------------------------------------------------------------
    # Reward signal
    # ------------------------------------------------------------------

    def prediction_accuracy_reward(
        self, predicted: str, actual: str, confidence: float
    ) -> float:
        """Compute reward from prediction accuracy."""
        if predicted == actual:
            return 0.72 + 0.28 * min(1.0, confidence)
        return 0.28 - 0.18 * min(1.0, confidence)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def observe(
        self,
        user_id: str,
        actual_action: str,
        outcome: float = 0.0,
        update_prior: bool = False,
        update_outcome_bias: bool = False,
        regime: str = "",
        brain_state_summary: dict[str, float] | None = None,
    ) -> None:
        """Record ground-truth action/outcome with minimal weight updates.

        Use during holdout to keep last_action, last_outcome, and history
        up to date so that outcome-conditioned and transition signals work
        correctly.

        Parameters
        ----------
        update_prior
            If True, also updates the action prior (base rate) with a
            very conservative learning rate. Safe for holdout because
            base rates are the most stable signal.
        update_outcome_bias
            If True, also updates outcome-conditioned bias and transitions
            with a dampened learning rate. Useful for mixed users whose
            behavior drifts.
        """
        model = self._get_user(user_id)

        if update_prior:
            # Very conservative prior update — 40% of training prior_lr
            holdout_prior_lr = self.prior_lr * 0.4
            model.action_counts[actual_action] = (
                model.action_counts.get(actual_action, 0) + 1
            )
            model.n_observations += 1
            total_obs = max(1, model.n_observations)
            for action in self.actions:
                freq = model.action_counts.get(action, 0) / total_obs
                old = model.prior[action]
                new = (1.0 - holdout_prior_lr) * old + holdout_prior_lr * freq
                model.prior[action] = max(
                    self.prior_smoothing / self.n_actions, new
                )
            total_p = sum(model.prior.values())
            if total_p > 0:
                for action in self.actions:
                    model.prior[action] /= total_p
        else:
            model.action_counts[actual_action] = (
                model.action_counts.get(actual_action, 0) + 1
            )
            model.n_observations += 1

        if update_outcome_bias and model.last_action:
            # Dampened outcome bias + transition update — 30% of training LR
            # Brain-adaptive: when state is volatile, learn faster (up to 3x)
            volatility_mult = 1.0
            if brain_state_summary:
                vel_mag = brain_state_summary.get("velocity_magnitude", 0.0)
                div_mag = brain_state_summary.get("divergence_magnitude", 0.0)
                volatility = vel_mag * 8.0 + div_mag * 4.0
                volatility_mult = 1.0 + min(2.0, volatility * 5.0)
            vbin = self._valence_bin(model.last_outcome)
            holdout_ob_lr = self.outcome_lr * 0.3 * volatility_mult
            holdout_tr_lr = self.transition_lr * 0.3 * volatility_mult
            for action in self.actions:
                cur = model.outcome_bias[action].get(vbin, 0.0)
                if action == actual_action:
                    delta = holdout_ob_lr * (1.0 - cur)
                else:
                    delta = -holdout_ob_lr * cur * 0.5
                model.outcome_bias[action][vbin] = _clamp(
                    cur + delta, -1.0, 1.0
                )
            # Transitions
            trans = model.transitions.get(model.last_action, {}).get(vbin, {})
            for action in self.actions:
                cur = trans.get(action, 0.0)
                if action == actual_action:
                    delta = holdout_tr_lr * (1.0 - cur)
                else:
                    delta = -holdout_tr_lr * cur * 0.3
                trans[action] = _clamp(cur + delta, -1.0, 1.0)

        # Track regime-conditioned actions during holdout too (brain only)
        if regime and regime != "neutral":
            behav_regime = self._behavioral_regime(model)
            if behav_regime not in model.regime_action_counts:
                model.regime_action_counts[behav_regime] = {a: 0 for a in self.actions}
                model.regime_total_counts[behav_regime] = 0
            model.regime_action_counts[behav_regime][actual_action] = (
                model.regime_action_counts[behav_regime].get(actual_action, 0) + 1
            )
            model.regime_total_counts[behav_regime] = model.regime_total_counts.get(behav_regime, 0) + 1
            model.regime_history.append(behav_regime)
            if len(model.regime_history) > 30:
                model.regime_history = model.regime_history[-30:]

            # Brain state k-NN: record during holdout for continual learning
            if brain_state_summary:
                valence = brain_state_summary.get("mood_valence", 0.5)
                div_mag_val = brain_state_summary.get("divergence_magnitude", 0.0)
                frust_div = brain_state_summary.get("divergence::frustration", 0.0)
                conf_div = brain_state_summary.get("divergence::confidence", 0.0)
                fear_div = brain_state_summary.get("divergence::fear", 0.0)
                outcome_sign = 1.0 if model.last_outcome > 0 else (-1.0 if model.last_outcome < 0 else 0.0)
                model.brain_state_knn.append((valence, div_mag_val, frust_div, conf_div, fear_div, outcome_sign, actual_action))

            # Brain macro cluster tracking during holdout
            brain_cluster = self._REGIME_TO_MACRO.get(regime, "neutral")
            if brain_cluster != "unknown":
                if brain_cluster not in model.brain_cluster_counts:
                    model.brain_cluster_counts[brain_cluster] = {a: 0 for a in self.actions}
                    model.brain_cluster_totals[brain_cluster] = 0
                model.brain_cluster_counts[brain_cluster][actual_action] = (
                    model.brain_cluster_counts[brain_cluster].get(actual_action, 0) + 1
                )
                model.brain_cluster_totals[brain_cluster] = model.brain_cluster_totals.get(brain_cluster, 0) + 1

        # Track WSLS and empirical transitions during holdout too
        if model.last_action:
            if model.last_outcome > 0:
                model.win_total += 1
                if actual_action == model.last_action:
                    model.win_stay_count += 1
            elif model.last_outcome < 0:
                model.lose_total += 1
                if actual_action != model.last_action:
                    model.lose_shift_count += 1
            vbin = self._valence_bin(model.last_outcome)
            key = f"{model.last_action}_{vbin}"
            if key not in model.empirical_trans:
                model.empirical_trans[key] = {a: 0 for a in self.actions}
            model.empirical_trans[key][actual_action] += 1

        model.history.append({"action": actual_action, "outcome": outcome})
        model.last_action = actual_action
        model.last_outcome = outcome

    @property
    def user_count(self) -> int:
        return len(self._users)

    def get_user_summary(self, user_id: str) -> dict[str, Any]:
        model = self._users.get(user_id)
        if model is None:
            return {"user_id": user_id, "status": "no_model"}
        return {
            "user_id": user_id,
            "n_observations": model.n_observations,
            "priors": {a: round(v, 4) for a, v in model.prior.items()},
            "action_counts": dict(model.action_counts),
        }


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))
