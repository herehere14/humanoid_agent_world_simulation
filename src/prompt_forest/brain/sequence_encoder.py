"""Stateless Sequence Feature Encoder.

Extracts a fixed-size feature vector from action-outcome history.
Pure function — no per-user state.  Used by the unified predictor
to feed sequence information into the learned transition function.

Domain-general: works for any sequential decision task with discrete
actions and scalar outcomes.
"""

from __future__ import annotations


def _outcome_sign(outcome: float) -> str:
    if outcome > 0:
        return "pos"
    elif outcome < 0:
        return "neg"
    return "zero"


class SequenceEncoder:
    """Encodes action-outcome history into a fixed-size feature vector.

    Parameters
    ----------
    actions : list[str]
        Available actions.
    """

    def __init__(self, actions: list[str]) -> None:
        self.actions = actions
        self.n_actions = len(actions)
        # Feature dim: see encode() breakdown
        self.feature_dim = (
            self.n_actions       # bigram P(next|last_action) per action
            + self.n_actions     # outcome-conditioned bigram P(next|last_a, last_o)
            + self.n_actions     # unigram action rates
            + self.n_actions     # per-action mean outcome (normalized)
            + 1                  # streak length (normalized)
            + 1                  # last-action-was-same indicator
            + 1                  # WSLS: win-stay rate
            + 1                  # WSLS: lose-shift rate
            + 2                  # recent win rate (window 3, 5)
            + 1                  # outcome volatility
            + 1                  # trial progress
            + 1                  # cumulative score normalized
        )

    def encode(
        self,
        action_history: list[str],
        outcome_history: list[float],
        trial_progress: float = 0.0,
        cumulative_score: float = 0.0,
    ) -> list[float]:
        """Encode history into feature vector.

        Parameters
        ----------
        action_history : list[str]
            Sequence of past actions (earliest first).
        outcome_history : list[float]
            Corresponding outcomes.
        trial_progress : float
            Fraction of trials completed (0 to 1).
        cumulative_score : float
            Running total score.

        Returns
        -------
        list[float]
            Fixed-size feature vector.
        """
        n = len(action_history)
        features: list[float] = []

        # --- Action-only bigram: P(next | last_action) ---
        if n >= 2:
            last_a = action_history[-1]
            bigram_counts = {a: 0.5 for a in self.actions}  # Dirichlet smoothing
            total = self.n_actions * 0.5
            for i in range(1, n):
                if action_history[i - 1] == last_a:
                    bigram_counts[action_history[i]] += 1.0
                    total += 1.0
            features.extend(bigram_counts[a] / total for a in self.actions)
        else:
            features.extend([1.0 / self.n_actions] * self.n_actions)

        # --- Outcome-conditioned bigram: P(next | last_action, last_outcome_sign) ---
        if n >= 2 and outcome_history:
            last_a = action_history[-1]
            last_o = _outcome_sign(outcome_history[-1])
            cond_counts = {a: 0.3 for a in self.actions}
            total = self.n_actions * 0.3
            for i in range(1, n):
                if i - 1 < len(outcome_history):
                    if (action_history[i - 1] == last_a and
                            _outcome_sign(outcome_history[i - 1]) == last_o):
                        cond_counts[action_history[i]] += 1.0
                        total += 1.0
            features.extend(cond_counts[a] / total for a in self.actions)
        else:
            features.extend([1.0 / self.n_actions] * self.n_actions)

        # --- Unigram action rates ---
        if n > 0:
            action_counts = {a: 0 for a in self.actions}
            for a in action_history:
                action_counts[a] += 1
            features.extend(action_counts[a] / n for a in self.actions)
        else:
            features.extend([1.0 / self.n_actions] * self.n_actions)

        # --- Per-action mean outcome (normalized) ---
        action_outcomes: dict[str, list[float]] = {a: [] for a in self.actions}
        for i in range(min(n, len(outcome_history))):
            action_outcomes[action_history[i]].append(outcome_history[i])
        for a in self.actions:
            outs = action_outcomes[a]
            if outs:
                m = sum(outs) / len(outs)
                features.append(m / (abs(m) + 100.0))
            else:
                features.append(0.0)

        # --- Streak length ---
        if n >= 2:
            streak = 1
            for i in range(n - 2, -1, -1):
                if action_history[i] == action_history[-1]:
                    streak += 1
                else:
                    break
            features.append(min(1.0, streak / 5.0))
        else:
            features.append(0.0)

        # --- Last-action-was-same ---
        if n >= 2:
            features.append(1.0 if action_history[-1] == action_history[-2] else 0.0)
        else:
            features.append(0.0)

        # --- WSLS rates ---
        win_stay_count = 0
        win_count = 0
        lose_shift_count = 0
        lose_count = 0
        for i in range(1, min(n, len(outcome_history))):
            if outcome_history[i - 1] > 0:
                win_count += 1
                if action_history[i] == action_history[i - 1]:
                    win_stay_count += 1
            elif outcome_history[i - 1] < 0:
                lose_count += 1
                if action_history[i] != action_history[i - 1]:
                    lose_shift_count += 1
        features.append(win_stay_count / max(1, win_count))
        features.append(lose_shift_count / max(1, lose_count))

        # --- Recent win rate (windows 3, 5) ---
        for window in [3, 5]:
            recent = outcome_history[-min(window, len(outcome_history)):]
            if recent:
                features.append(sum(1 for o in recent if o > 0) / len(recent))
            else:
                features.append(0.5)

        # --- Outcome volatility ---
        if len(outcome_history) >= 4:
            diffs = [abs(outcome_history[i] - outcome_history[i - 1])
                     for i in range(max(1, len(outcome_history) - 3), len(outcome_history))]
            avg_diff = sum(diffs) / len(diffs) if diffs else 0.0
            features.append(min(1.0, avg_diff / (avg_diff + 100.0)))
        else:
            features.append(0.0)

        # --- Trial progress ---
        features.append(trial_progress)

        # --- Cumulative score ---
        features.append(cumulative_score / (abs(cumulative_score) + 500.0))

        return features
