"""Unified Cognitive Predictor — Integrated Latent State Brain.

Embeds prospect theory and sequence statistics INTO a learned latent state,
rather than treating them as competing predictors.  The brain learns how
outcomes, actions, and cognitive signals combine to produce behavior.

Architecture:
  Each trial:
    1. Prospect theory computes EV estimates and perseveration scores
    2. Sequence stats compute bigram/unigram/WSLS transition probs
    3. All signals feed into a learned latent state transition
    4. Readout layer maps latent state → action probabilities
    5. Online learning adjusts weights from prediction error

The latent dimensions are unnamed and learned — they emerge as whatever
internal states best predict the user's behavior.  On a gambling task
they might capture risk tolerance and frustration.  On a trust game,
trust level and reciprocity expectation.

Domain-general: prospect theory and sequence statistics apply to any
sequential decision task with discrete actions and scalar outcomes.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Prospect theory helpers
# ---------------------------------------------------------------------------

@dataclass
class ProspectParams:
    """Per-user prospect theory parameters."""

    alpha: float = 0.50       # outcome sensitivity [0.01, 1.5]
    lambda_: float = 2.0      # loss aversion [0.5, 5.0]
    A: float = 0.15           # EV learning rate [0.01, 0.8]
    theta: float = 2.0        # inverse temperature [0.1, 10.0]
    K: float = 0.30           # perseveration decay [0.05, 0.99]
    ep_p: float = 0.50        # perseveration gain on win [0.0, 3.0]
    ep_n: float = 0.30        # perseveration gain on loss [0.0, 3.0]
    w_rl: float = 0.65        # RL vs perseveration weight [0.0, 1.0]


_PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "alpha": (0.01, 1.5),
    "lambda_": (0.5, 5.0),
    "A": (0.01, 0.8),
    "theta": (0.1, 10.0),
    "K": (0.05, 0.99),
    "ep_p": (0.0, 3.0),
    "ep_n": (0.0, 3.0),
    "w_rl": (0.0, 1.0),
}

_PARAM_PERTURB: dict[str, float] = {
    "alpha": 0.05, "lambda_": 0.2, "A": 0.03, "theta": 0.3,
    "K": 0.05, "ep_p": 0.1, "ep_n": 0.1, "w_rl": 0.05,
}

_POP_PRIORS = ProspectParams()


def _prospect_value(x: float, alpha: float, lambda_: float) -> float:
    """Prospect theory value function."""
    if x >= 0:
        return x ** alpha if x > 0 else 0.0
    return -lambda_ * (abs(x) ** alpha)


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def _softmax_dict(values: dict[str, float], theta: float) -> dict[str, float]:
    max_v = max(values.values()) if values else 0.0
    exp_vals = {a: math.exp(min(30.0, theta * (v - max_v)))
                for a, v in values.items()}
    total = sum(exp_vals.values())
    if total < 1e-12:
        return {a: 1.0 / len(values) for a in values}
    return {a: e / total for a, e in exp_vals.items()}


def _softmax_list(logits: list[float]) -> list[float]:
    max_v = max(logits) if logits else 0.0
    exps = [math.exp(min(30.0, v - max_v)) for v in logits]
    total = sum(exps)
    if total < 1e-12:
        return [1.0 / len(logits)] * len(logits)
    return [e / total for e in exps]


def _tanh(x: float) -> float:
    return math.tanh(max(-10.0, min(10.0, x)))


def _dot(a: list[float], b: list[float]) -> float:
    return sum(ai * bi for ai, bi in zip(a, b))


def _outcome_sign(outcome: float) -> str:
    if outcome > 0:
        return "pos"
    if outcome < 0:
        return "neg"
    return "zero"


# ---------------------------------------------------------------------------
# Per-user state
# ---------------------------------------------------------------------------

@dataclass
class CognitiveState:
    """Per-user integrated cognitive state."""

    # Prospect theory
    params: ProspectParams = field(default_factory=ProspectParams)
    ev: dict[str, float] = field(default_factory=dict)
    persev: dict[str, float] = field(default_factory=dict)

    # Learned latent state
    z: list[float] = field(default_factory=list)
    W_trans: list[list[float]] = field(default_factory=list)
    W_read: list[list[float]] = field(default_factory=list)
    b_read: list[float] = field(default_factory=list)
    lr: float = 0.025

    # Sequence statistics (Dirichlet-smoothed, recency-weighted)
    bigram: dict[tuple[str, str], dict[str, float]] = field(default_factory=dict)
    unigram: dict[str, float] = field(default_factory=dict)
    action_bigram: dict[str, dict[str, float]] = field(default_factory=dict)

    # Additional sequence stats
    trigram: dict[tuple[str, str, str], dict[str, float]] = field(default_factory=dict)
    streak_counts: dict[tuple[int, str], dict[str, float]] = field(default_factory=dict)
    fast_bigram: dict[tuple[str, str], dict[str, float]] = field(default_factory=dict)
    nodecay_unigram: dict[str, float] = field(default_factory=dict)
    action_trigram: dict[tuple[str, str], dict[str, float]] = field(default_factory=dict)

    # WSLS
    wsls_stay_win: float = 0.5
    wsls_total_win: float = 1.0
    wsls_shift_loss: float = 0.5
    wsls_total_loss: float = 1.0

    # History
    action_history: list[str] = field(default_factory=list)
    outcome_history: list[float] = field(default_factory=list)

    # Adaptive sequence component weights (10 components)
    seq_weights: list[float] = field(
        default_factory=lambda: [0.18, 0.14, 0.09, 0.09, 0.07, 0.05, 0.09, 0.07, 0.11, 0.11]
    )
    # Components: bigram, act_bigram, unigram, wsls, trigram, streak,
    #             fast_bigram, nodecay_uni, window_bigram, action_trigram

    # Tracking
    prev_probs: dict[str, float] | None = None
    n_trials: int = 0
    n_correct: int = 0
    adapt_lr: float = 0.03  # for PT parameter adaptation


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class UnifiedCognitivePredictor:
    """Integrated cognitive model with embedded prospect theory.

    Prospect theory and sequence statistics feed INTO a learned latent
    state.  The latent state IS the prediction mechanism — action
    probabilities come from a learned readout of the internal state.

    Parameters
    ----------
    actions : list[str]
        Available actions.
    D : int
        Latent state dimensionality.
    lr : float
        Initial learning rate for latent state weights.
    anneal_rate : float
        Per-trial LR decay.
    mix : float
        How fast latent state changes (0=static, 1=fully replace).
    bigram_alpha : float
        Dirichlet smoothing for sequence statistics.
    bigram_decay : float
        Recency decay for sequence statistics.
    adapt_pt : bool
        Whether to adapt prospect theory parameters online.
    """

    def __init__(
        self,
        actions: list[str],
        D: int = 6,
        lr: float = 0.025,
        anneal_rate: float = 0.997,
        mix: float = 0.3,
        bigram_alpha: float = 0.3,
        bigram_decay: float = 0.97,
        adapt_pt: bool = True,
        pop_seq_transfer_weight: float = 0.3,
    ) -> None:
        self.actions = actions
        self.n_actions = len(actions)
        self.D = D
        self._lr = lr
        self._anneal_rate = anneal_rate
        self._mix = mix
        self._bigram_alpha = bigram_alpha
        self._bigram_decay = bigram_decay
        self._adapt_pt = adapt_pt
        self._pop_seq_transfer = pop_seq_transfer_weight
        self._users: dict[str, CognitiveState] = {}

        # Input dimensionality:
        #   z_prev (D) + prospect_ev (n_actions) + prospect_persev (n_actions)
        #   + prospect_value (1) + bigram_probs (n_actions)
        #   + action_bigram_probs (n_actions) + unigram_probs (n_actions)
        #   + wsls_probs (n_actions) + outcome_sign (1) + outcome_mag (1)
        #   + trial_progress (1)
        self._input_dim = D + 5 * self.n_actions + 4
        self._n_seq_components = 10

        self._rng = random.Random(42)

        # Population transfer
        self._pop_W_read: list[list[float]] | None = None
        self._pop_W_trans: list[list[float]] | None = None
        self._pop_b_read: list[float] | None = None
        self._pop_bigram: dict[tuple[str, str], dict[str, float]] = {}
        self._pop_unigram: dict[str, float] = {}
        self._pop_action_bigram: dict[str, dict[str, float]] = {}
        self._pop_pt_params_sum: dict[str, float] = {}
        self._n_completed: int = 0

    # ----------------------------------------------------------------
    # User initialization
    # ----------------------------------------------------------------

    def _init_weights(self):
        """Initialize transition and readout weight matrices."""
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
        b_read = [0.0] * self.n_actions

        return W_trans, W_read, b_read

    def _get_user(self, user_id: str) -> CognitiveState:
        if user_id not in self._users:
            W_trans, W_read, b_read = self._init_weights()

            # Population transfer for weights
            if self._n_completed >= 10 and self._pop_W_read is not None:
                blend = 0.5  # 50% population, 50% random init
                for d in range(self.D):
                    for j in range(self._input_dim):
                        W_trans[d][j] = (
                            blend * self._pop_W_trans[d][j]
                            + (1 - blend) * W_trans[d][j]
                        )
                for a in range(self.n_actions):
                    for d in range(self.D):
                        W_read[a][d] = (
                            blend * self._pop_W_read[a][d]
                            + (1 - blend) * W_read[a][d]
                        )
                    b_read[a] = blend * self._pop_b_read[a] + (1 - blend) * b_read[a]

            # PT params from population centroid
            pt_params = ProspectParams()
            if self._n_completed >= 10:
                for pname in _PARAM_BOUNDS:
                    pop_mean = self._pop_pt_params_sum.get(pname, 0.0) / self._n_completed
                    default = getattr(_POP_PRIORS, pname)
                    blended = 0.6 * pop_mean + 0.4 * default
                    lo, hi = _PARAM_BOUNDS[pname]
                    setattr(pt_params, pname, max(lo, min(hi, blended)))

            state = CognitiveState(
                params=pt_params,
                ev={a: 0.0 for a in self.actions},
                persev={a: 0.0 for a in self.actions},
                z=[0.0] * self.D,
                W_trans=W_trans,
                W_read=W_read,
                b_read=b_read,
                lr=self._lr,
            )

            # Population transfer for sequence stats
            if self._n_completed >= 5:
                tw = self._pop_seq_transfer
                for key, counts in self._pop_bigram.items():
                    state.bigram[key] = {a: tw * c for a, c in counts.items()}
                for a, c in self._pop_unigram.items():
                    state.unigram[a] = tw * c
                for key, counts in self._pop_action_bigram.items():
                    state.action_bigram[key] = {a: tw * c for a, c in counts.items()}

            self._users[user_id] = state
        return self._users[user_id]

    # ----------------------------------------------------------------
    # Sequence statistics (as input features)
    # ----------------------------------------------------------------

    def _dirichlet_probs(self, counts: dict[str, float] | None) -> list[float]:
        """Convert counts to probability vector with Dirichlet smoothing."""
        alpha = self._bigram_alpha
        probs = []
        total = 0.0
        for a in self.actions:
            c = (counts or {}).get(a, 0.0) + alpha
            probs.append(c)
            total += c
        return [p / total for p in probs]

    def _bigram_probs(self, state: CognitiveState) -> list[float]:
        """P(next | prev_action, outcome_sign)."""
        if not state.action_history or not state.outcome_history:
            return [1.0 / self.n_actions] * self.n_actions
        last_a = state.action_history[-1]
        last_o = _outcome_sign(state.outcome_history[-1])
        return self._dirichlet_probs(state.bigram.get((last_a, last_o)))

    def _action_bigram_probs(self, state: CognitiveState) -> list[float]:
        """P(next | prev_action) — no outcome conditioning."""
        if not state.action_history:
            return [1.0 / self.n_actions] * self.n_actions
        return self._dirichlet_probs(state.action_bigram.get(state.action_history[-1]))

    def _unigram_probs(self, state: CognitiveState) -> list[float]:
        """Recency-weighted action frequencies."""
        return self._dirichlet_probs(state.unigram if state.unigram else None)

    def _trigram_probs(self, state: CognitiveState) -> list[float]:
        """P(next | prev2_action, prev_action, outcome_sign)."""
        if len(state.action_history) < 2:
            return self._bigram_probs(state)
        prev2 = state.action_history[-2]
        last_a = state.action_history[-1]
        last_o = _outcome_sign(state.outcome_history[-1]) if state.outcome_history else "zero"
        return self._dirichlet_probs(state.trigram.get((prev2, last_a, last_o)))

    def _streak_probs(self, state: CognitiveState) -> list[float]:
        """P(next | streak_length, last_outcome_sign)."""
        if not state.action_history:
            return [1.0 / self.n_actions] * self.n_actions
        last = state.action_history[-1]
        streak = 1
        for i in range(len(state.action_history) - 2, -1, -1):
            if state.action_history[i] == last:
                streak += 1
            else:
                break
        osign = _outcome_sign(state.outcome_history[-1]) if state.outcome_history else "zero"
        return self._dirichlet_probs(state.streak_counts.get((min(streak, 3), osign)))

    def _fast_bigram_probs(self, state: CognitiveState) -> list[float]:
        """Fast-decay bigram (captures very recent transitions)."""
        if not state.action_history:
            return [1.0 / self.n_actions] * self.n_actions
        last_a = state.action_history[-1]
        last_o = _outcome_sign(state.outcome_history[-1]) if state.outcome_history else "zero"
        return self._dirichlet_probs(state.fast_bigram.get((last_a, last_o)))

    def _nodecay_unigram_probs(self, state: CognitiveState) -> list[float]:
        """Overall action frequencies (no decay)."""
        return self._dirichlet_probs(state.nodecay_unigram if state.nodecay_unigram else None)

    def _window_bigram_probs(self, state: CognitiveState) -> list[float]:
        """Window bigram: last 10 trials only (fast adaptation)."""
        n = len(state.action_history)
        if n < 2:
            return [1.0 / self.n_actions] * self.n_actions
        window = min(10, n)
        recent_a = state.action_history[-window:]
        recent_o = state.outcome_history[-window:]
        last_a = recent_a[-1]
        last_o = _outcome_sign(recent_o[-1]) if recent_o else "zero"
        counts: dict[str, float] = {}
        for i in range(1, len(recent_a)):
            if (recent_a[i - 1] == last_a and
                    _outcome_sign(recent_o[i - 1]) == last_o):
                nxt = recent_a[i]
                counts[nxt] = counts.get(nxt, 0.0) + 1.0
        return self._dirichlet_probs(counts)

    def _action_trigram_probs(self, state: CognitiveState) -> list[float]:
        """Pure action trigram: P(next | prev2, prev1) — no outcome."""
        if len(state.action_history) < 2:
            return self._action_bigram_probs(state)
        prev2 = state.action_history[-2]
        prev1 = state.action_history[-1]
        return self._dirichlet_probs(state.action_trigram.get((prev2, prev1)))

    def _wsls_probs(self, state: CognitiveState) -> list[float]:
        """Win-stay/lose-shift predicted probabilities."""
        if not state.action_history or not state.outcome_history:
            return [1.0 / self.n_actions] * self.n_actions
        last_a = state.action_history[-1]
        last_o = state.outcome_history[-1]
        probs = [0.0] * self.n_actions
        if last_o >= 0:
            stay_rate = state.wsls_stay_win / max(1.0, state.wsls_total_win)
            for i, a in enumerate(self.actions):
                if a == last_a:
                    probs[i] = stay_rate
                else:
                    probs[i] = (1.0 - stay_rate) / max(1, self.n_actions - 1)
        else:
            shift_rate = state.wsls_shift_loss / max(1.0, state.wsls_total_loss)
            for i, a in enumerate(self.actions):
                if a == last_a:
                    probs[i] = 1.0 - shift_rate
                else:
                    probs[i] = shift_rate / max(1, self.n_actions - 1)
        total = sum(probs)
        return [p / total for p in probs] if total > 0 else [1.0 / self.n_actions] * self.n_actions

    # ----------------------------------------------------------------
    # Input vector construction
    # ----------------------------------------------------------------

    def _build_input(self, state: CognitiveState) -> list[float]:
        """Build the input feature vector for latent state transition.

        Combines all cognitive signals into a single vector.
        """
        inp: list[float] = []

        # 1. Previous latent state (D dims)
        inp.extend(state.z)

        # 2. Prospect theory EV estimates (n_actions, normalized)
        for a in self.actions:
            v = state.ev.get(a, 0.0)
            inp.append(v / (abs(v) + 100.0))

        # 3. Prospect theory perseveration scores (n_actions, normalized)
        for a in self.actions:
            p = state.persev.get(a, 0.0)
            inp.append(p / (abs(p) + 5.0))

        # 4. Prospect value of last outcome (1 dim)
        if state.outcome_history:
            pv = _prospect_value(
                state.outcome_history[-1],
                state.params.alpha,
                state.params.lambda_,
            )
            inp.append(pv / (abs(pv) + 100.0))
        else:
            inp.append(0.0)

        # 5. Outcome-conditioned bigram probs (n_actions)
        inp.extend(self._bigram_probs(state))

        # 6. Action-only bigram probs (n_actions)
        inp.extend(self._action_bigram_probs(state))

        # 7. Unigram probs (n_actions)
        inp.extend(self._unigram_probs(state))

        # 8. WSLS probs (n_actions)
        inp.extend(self._wsls_probs(state))

        # 9. Outcome sign (1)
        if state.outcome_history:
            o = state.outcome_history[-1]
            inp.append(1.0 if o > 0 else (-1.0 if o < 0 else 0.0))
        else:
            inp.append(0.0)

        # 10. Outcome magnitude (1)
        if state.outcome_history:
            o = state.outcome_history[-1]
            inp.append(min(1.0, abs(o) / (abs(o) + 100.0)))
        else:
            inp.append(0.0)

        # 11. Trial progress (1)
        inp.append(min(1.0, state.n_trials / 100.0))

        return inp

    def _seq_features(self, state: CognitiveState) -> list[float]:
        """Build sequence feature vector — 10 components × n_actions."""
        feats: list[float] = []
        feats.extend(self._bigram_probs(state))            # outcome-conditioned
        feats.extend(self._action_bigram_probs(state))     # action-only
        feats.extend(self._unigram_probs(state))           # recency-weighted freq
        feats.extend(self._wsls_probs(state))              # win-stay/lose-shift
        feats.extend(self._trigram_probs(state))           # 3-gram
        feats.extend(self._streak_probs(state))            # streak pattern
        feats.extend(self._fast_bigram_probs(state))       # fast-decay
        feats.extend(self._nodecay_unigram_probs(state))   # overall freq
        feats.extend(self._window_bigram_probs(state))     # window bigram
        feats.extend(self._action_trigram_probs(state))    # action trigram
        return feats

    # ----------------------------------------------------------------
    # Latent state transition + readout
    # ----------------------------------------------------------------

    def _transition(self, state: CognitiveState, inp: list[float]) -> list[float]:
        """Compute new latent state from input features."""
        new_z = []
        for d in range(self.D):
            raw = _dot(state.W_trans[d], inp)
            activated = _tanh(raw)
            new_z.append(
                (1.0 - self._mix) * state.z[d] + self._mix * activated
            )
        return new_z

    def _readout(self, state: CognitiveState, seq_feats: list[float]) -> list[float]:
        """Compute action logits: sequence baseline + latent adjustment.

        Residual architecture:
          output = log(seq_baseline_probs) + latent_adjustment

        The sequence baseline (average of bigram/unigram/wsls/action_bigram)
        requires NO learning and provides strong initial predictions.
        The latent state learns a cognitive adjustment on top — capturing
        what the raw sequence stats miss (e.g., prospect theory effects,
        internal state dynamics).
        """
        # Sequence baseline: adaptively weighted blend of components
        n_a = self.n_actions
        n_comp = self._n_seq_components
        w = state.seq_weights
        total_w = sum(w)
        base_probs = [0.0] * n_a
        for i in range(n_a):
            for c in range(n_comp):
                base_probs[i] += (w[c] / total_w) * seq_feats[c * n_a + i]

        # Convert to log-space for additive adjustment
        logits = []
        for a_idx in range(n_a):
            base_logit = math.log(max(base_probs[a_idx], 1e-8))
            latent_adj = _dot(state.W_read[a_idx], state.z) + state.b_read[a_idx]
            logits.append(base_logit + latent_adj)
        return logits

    # ----------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------

    def predict(self, user_id: str, **kwargs: Any) -> dict[str, float]:
        """Return action probabilities from integrated cognitive state."""
        state = self._get_user(user_id)

        # Build input from all cognitive signals
        inp = self._build_input(state)

        # Transition latent state
        state.z = self._transition(state, inp)

        # Readout: latent state + skip connection → action probabilities
        seq_feats = self._seq_features(state)
        logits = self._readout(state, seq_feats)
        probs = _softmax_list(logits)

        result = {a: p for a, p in zip(self.actions, probs)}
        state.prev_probs = result
        return result

    def update(
        self,
        user_id: str,
        actual_action: str,
        outcome: float,
        **kwargs: Any,
    ) -> None:
        """Update all cognitive components from observed action + outcome."""
        state = self._get_user(user_id)
        actual_idx = self.actions.index(actual_action)

        # ---- Track accuracy ----
        state.n_trials += 1
        if state.prev_probs is not None:
            pred = max(state.prev_probs, key=state.prev_probs.get)
            if pred == actual_action:
                state.n_correct += 1

        # ---- 1. Online weight update (residual latent state) ----
        seq_feats = self._seq_features(state)
        logits = self._readout(state, seq_feats)
        probs = _softmax_list(logits)
        lr = state.lr

        # Readout update: cross-entropy gradient on latent adjustment
        for a_idx in range(self.n_actions):
            target = 1.0 if a_idx == actual_idx else 0.0
            grad = probs[a_idx] - target
            for d in range(self.D):
                state.W_read[a_idx][d] -= lr * grad * state.z[d]
            state.b_read[a_idx] -= lr * grad

        # Transition update: one-step backprop
        dL_dz = [0.0] * self.D
        for a_idx in range(self.n_actions):
            target = 1.0 if a_idx == actual_idx else 0.0
            grad = probs[a_idx] - target
            for d in range(self.D):
                dL_dz[d] += grad * state.W_read[a_idx][d]

        inp = self._build_input(state)
        trans_lr = lr * 0.5  # slower for stability
        for d in range(self.D):
            raw = _dot(state.W_trans[d], inp)
            tanh_val = _tanh(raw)
            dtanh = 1.0 - tanh_val * tanh_val
            scale = dL_dz[d] * self._mix * dtanh
            for j in range(self._input_dim):
                state.W_trans[d][j] -= trans_lr * scale * inp[j]

        # Anneal learning rate
        state.lr = max(0.003, state.lr * self._anneal_rate)

        # ---- 1b. Adapt sequence component weights ----
        if state.action_history:
            eta = 0.08
            for c in range(self._n_seq_components):
                # Probability this component assigned to actual action
                p_actual = max(seq_feats[c * self.n_actions + actual_idx], 0.01)
                state.seq_weights[c] *= p_actual ** eta
                state.seq_weights[c] = max(0.01, state.seq_weights[c])
            total_sw = sum(state.seq_weights)
            state.seq_weights = [w / total_sw for w in state.seq_weights]

        # ---- 2. Update prospect theory state ----
        p = state.params
        u = _prospect_value(outcome, p.alpha, p.lambda_)
        state.ev[actual_action] += p.A * (u - state.ev[actual_action])

        ep = p.ep_p if outcome >= 0 else p.ep_n
        for a in self.actions:
            state.persev[a] *= p.K
        state.persev[actual_action] += ep

        # ---- 3. Adapt prospect theory parameters (optional) ----
        if self._adapt_pt and state.n_trials >= 5:
            self._adapt_pt_params(state, actual_action)

        # ---- 4. Update sequence statistics ----
        decay = self._bigram_decay
        fast_decay = 0.90

        # Decay existing counts
        for key in state.bigram:
            for a in state.bigram[key]:
                state.bigram[key][a] *= decay
        for a in state.unigram:
            state.unigram[a] *= decay
        for key in state.action_bigram:
            for a in state.action_bigram[key]:
                state.action_bigram[key][a] *= decay
        for key in state.trigram:
            for a in state.trigram[key]:
                state.trigram[key][a] *= decay
        for key in state.streak_counts:
            for a in state.streak_counts[key]:
                state.streak_counts[key][a] *= decay
        for key in state.fast_bigram:
            for a in state.fast_bigram[key]:
                state.fast_bigram[key][a] *= fast_decay
        for key in state.action_trigram:
            for a in state.action_trigram[key]:
                state.action_trigram[key][a] *= decay

        # Update counts
        if state.action_history:
            last_a = state.action_history[-1]
            last_o = _outcome_sign(
                state.outcome_history[-1]
            ) if state.outcome_history else "zero"

            # Outcome-conditioned bigram
            bkey = (last_a, last_o)
            if bkey not in state.bigram:
                state.bigram[bkey] = {}
            state.bigram[bkey][actual_action] = (
                state.bigram[bkey].get(actual_action, 0.0) + 1.0
            )

            # Fast-decay bigram
            if bkey not in state.fast_bigram:
                state.fast_bigram[bkey] = {}
            state.fast_bigram[bkey][actual_action] = (
                state.fast_bigram[bkey].get(actual_action, 0.0) + 1.0
            )

            # Action-only bigram
            if last_a not in state.action_bigram:
                state.action_bigram[last_a] = {}
            state.action_bigram[last_a][actual_action] = (
                state.action_bigram[last_a].get(actual_action, 0.0) + 1.0
            )

            # Trigram
            if len(state.action_history) >= 2:
                prev2 = state.action_history[-2]
                tkey = (prev2, last_a, last_o)
                if tkey not in state.trigram:
                    state.trigram[tkey] = {}
                state.trigram[tkey][actual_action] = (
                    state.trigram[tkey].get(actual_action, 0.0) + 1.0
                )

            # Action trigram
            if len(state.action_history) >= 2:
                prev2 = state.action_history[-2]
                atkey = (prev2, last_a)
                if atkey not in state.action_trigram:
                    state.action_trigram[atkey] = {}
                state.action_trigram[atkey][actual_action] = (
                    state.action_trigram[atkey].get(actual_action, 0.0) + 1.0
                )

            # WSLS
            last_outcome = state.outcome_history[-1] if state.outcome_history else 0
            if last_outcome >= 0:
                state.wsls_total_win += 1.0
                if actual_action == last_a:
                    state.wsls_stay_win += 1.0
            else:
                state.wsls_total_loss += 1.0
                if actual_action != last_a:
                    state.wsls_shift_loss += 1.0

        # Streak
        if state.action_history:
            last = state.action_history[-1]
            streak = 1
            for i in range(len(state.action_history) - 2, -1, -1):
                if state.action_history[i] == last:
                    streak += 1
                else:
                    break
            osign = _outcome_sign(state.outcome_history[-1]) if state.outcome_history else "zero"
            skey = (min(streak, 3), osign)
            if skey not in state.streak_counts:
                state.streak_counts[skey] = {}
            state.streak_counts[skey][actual_action] = (
                state.streak_counts[skey].get(actual_action, 0.0) + 1.0
            )

        # Unigram (recency-weighted)
        state.unigram[actual_action] = (
            state.unigram.get(actual_action, 0.0) + 1.0
        )

        # No-decay unigram
        state.nodecay_unigram[actual_action] = (
            state.nodecay_unigram.get(actual_action, 0.0) + 1.0
        )

        # Record history
        state.action_history.append(actual_action)
        state.outcome_history.append(outcome)

    # ----------------------------------------------------------------
    # Prospect theory parameter adaptation
    # ----------------------------------------------------------------

    def _adapt_pt_params(self, state: CognitiveState, actual_action: str) -> None:
        """Adapt prospect theory parameters via finite-difference gradient.

        For alpha/lambda/A, we simulate one-step-ahead EV updates to get
        a gradient signal — changing alpha/lambda changes how the NEXT
        outcome is processed, not past EVs.
        """
        lr = state.adapt_lr
        last_outcome = state.outcome_history[-1] if state.outcome_history else 0.0

        def _compute_lp(params: ProspectParams) -> float:
            """Log-prob of actual_action under PT model with given params.

            For params that affect EV computation (alpha, lambda_, A),
            simulate a one-step EV update from the last outcome to capture
            how the parameter change affects future behavior.
            """
            # Simulate one-step EV update with these params
            u = _prospect_value(last_outcome, params.alpha, params.lambda_)
            sim_ev = {}
            for a in self.actions:
                base = state.ev.get(a, 0.0)
                if a == state.action_history[-1] if state.action_history else False:
                    sim_ev[a] = base + params.A * (u - base)
                else:
                    sim_ev[a] = base

            V = {
                a: params.w_rl * sim_ev.get(a, 0.0)
                + (1.0 - params.w_rl) * state.persev.get(a, 0.0)
                for a in self.actions
            }
            pt_probs = _softmax_dict(V, params.theta)
            return math.log(max(pt_probs.get(actual_action, 1e-8), 1e-8))

        # For each parameter, compute gradient via finite difference
        for pname in ["alpha", "lambda_", "A", "theta", "K", "ep_p", "ep_n", "w_rl"]:
            perturb = _PARAM_PERTURB[pname]
            lo, hi = _PARAM_BOUNDS[pname]
            orig = getattr(state.params, pname)

            setattr(state.params, pname, min(hi, orig + perturb))
            lp_plus = _compute_lp(state.params)

            setattr(state.params, pname, max(lo, orig - perturb))
            lp_minus = _compute_lp(state.params)

            setattr(state.params, pname, orig)

            grad = (lp_plus - lp_minus) / (2.0 * perturb)
            new_val = max(lo, min(hi, orig + lr * grad))
            setattr(state.params, pname, new_val)

        # Anneal PT adaptation rate
        state.adapt_lr = max(0.005, state.adapt_lr * 0.995)

    # ----------------------------------------------------------------
    # Calibrate / adapt stubs (for eval script compatibility)
    # ----------------------------------------------------------------

    def calibrate(self, user_id: str) -> None:
        pass

    def adapt_parameters(self, user_id: str, predicted_action: str, actual_action: str) -> None:
        pass

    # ----------------------------------------------------------------
    # Population transfer
    # ----------------------------------------------------------------

    def finalize_user(self, user_id: str) -> None:
        """Accumulate this user's learned weights into population prior."""
        state = self._users.get(user_id)
        if state is None or state.n_trials < 20:
            return

        self._n_completed += 1
        n = self._n_completed
        decay = (n - 1) / n

        # Accumulate W_read and W_trans (running mean)
        if self._pop_W_read is None:
            self._pop_W_read = [row[:] for row in state.W_read]
            self._pop_W_trans = [row[:] for row in state.W_trans]
            self._pop_b_read = state.b_read[:]
        else:
            for a in range(self.n_actions):
                for d in range(self.D):
                    self._pop_W_read[a][d] = (
                        decay * self._pop_W_read[a][d]
                        + (1.0 / n) * state.W_read[a][d]
                    )
                self._pop_b_read[a] = (
                    decay * self._pop_b_read[a]
                    + (1.0 / n) * state.b_read[a]
                )
            for d in range(self.D):
                for j in range(self._input_dim):
                    self._pop_W_trans[d][j] = (
                        decay * self._pop_W_trans[d][j]
                        + (1.0 / n) * state.W_trans[d][j]
                    )

        # Accumulate PT params
        for pname in _PARAM_BOUNDS:
            val = getattr(state.params, pname)
            self._pop_pt_params_sum[pname] = (
                self._pop_pt_params_sum.get(pname, 0.0) + val
            )

        # Accumulate sequence stats
        for key, counts in state.bigram.items():
            total = sum(counts.values())
            if total < 1e-12:
                continue
            if key not in self._pop_bigram:
                self._pop_bigram[key] = {}
            for a in self.actions:
                norm_c = counts.get(a, 0.0) / total
                old = self._pop_bigram[key].get(a, 0.0)
                self._pop_bigram[key][a] = decay * old + (1.0 / n) * norm_c

        total_uni = sum(state.unigram.values())
        if total_uni > 1e-12:
            for a in self.actions:
                norm_c = state.unigram.get(a, 0.0) / total_uni
                old = self._pop_unigram.get(a, 0.0)
                self._pop_unigram[a] = decay * old + (1.0 / n) * norm_c

        for key, counts in state.action_bigram.items():
            total = sum(counts.values())
            if total < 1e-12:
                continue
            if key not in self._pop_action_bigram:
                self._pop_action_bigram[key] = {}
            for a in self.actions:
                norm_c = counts.get(a, 0.0) / total
                old = self._pop_action_bigram[key].get(a, 0.0)
                self._pop_action_bigram[key][a] = decay * old + (1.0 / n) * norm_c

    # ----------------------------------------------------------------
    # Diagnostics
    # ----------------------------------------------------------------

    def get_user_info(self, user_id: str) -> dict[str, Any]:
        state = self._get_user(user_id)
        p = state.params
        return {
            "n_trials": state.n_trials,
            "n_correct": state.n_correct,
            "accuracy": round(state.n_correct / max(1, state.n_trials), 3),
            "alpha": round(p.alpha, 3),
            "lambda_": round(p.lambda_, 3),
            "theta": round(p.theta, 3),
            "A": round(p.A, 3),
            "K": round(p.K, 3),
            "w_rl": round(p.w_rl, 3),
            "z": [round(v, 3) for v in state.z],
            "lr": round(state.lr, 4),
            "adapt_lr": round(state.adapt_lr, 4),
            "calibrated": False,
        }

    @property
    def user_count(self) -> int:
        return len(self._users)
