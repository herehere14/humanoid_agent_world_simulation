#!/usr/bin/env python3
"""Evaluate brain on synthetic extended-IGT dataset.

300 trials per participant — tests whether more data helps the
cognitive layer (prospect theory + latent state) contribute beyond
pure sequence statistics.

Runs full model vs seq-only ablation at multiple train/holdout splits.
"""

from __future__ import annotations

import os
import sys
import random
from statistics import mean, stdev

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.prompt_forest.brain.unified_predictor import UnifiedCognitivePredictor
from generate_extended_igt import load_dataset, SyntheticParticipant

ACTIONS = ["A", "B", "C", "D"]
SAFE = {"C", "D"}


def run_participant(
    participant: SyntheticParticipant,
    predictor: UnifiedCognitivePredictor,
    train_frac: float,
) -> dict:
    """Run one participant: train then holdout."""
    n_train = int(participant.n_trials * train_frac)
    uid = participant.participant_id

    # Training
    for t in range(n_train):
        action = participant.choices[t]
        outcome = participant.wins[t] + participant.losses[t]
        predictor.predict(uid)
        predictor.update(uid, actual_action=action, outcome=outcome)

    # Holdout
    n_holdout = participant.n_trials - n_train
    correct = 0
    for t in range(n_train, participant.n_trials):
        action = participant.choices[t]
        outcome = participant.wins[t] + participant.losses[t]
        probs = predictor.predict(uid)
        predicted = max(probs, key=probs.get)
        if predicted == action:
            correct += 1
        predictor.update(uid, actual_action=action, outcome=outcome)

    # History-only baseline (recency-weighted frequency + WSLS)
    hist_correct = _history_baseline(participant, train_frac)

    return {
        "id": uid,
        "model": participant.model,
        "safe_frac": participant.safe_fraction,
        "accuracy": correct / max(1, n_holdout),
        "history_acc": hist_correct / max(1, n_holdout),
        "n_holdout": n_holdout,
    }


def _history_baseline(p: SyntheticParticipant, train_frac: float) -> int:
    """Simple history-only baseline: recency-weighted prior + WSLS."""
    n_train = int(p.n_trials * train_frac)
    counts = {a: 0.5 for a in ACTIONS}
    decay = 0.98
    prev_a = None
    prev_o = None

    wsls_stay_win = {a: 0.5 for a in ACTIONS}
    wsls_total_win = {a: 0.5 for a in ACTIONS}
    wsls_shift_loss = {a: 0.5 for a in ACTIONS}
    wsls_total_loss = {a: 0.5 for a in ACTIONS}

    for t in range(n_train):
        action = p.choices[t]
        outcome = p.wins[t] + p.losses[t]
        for a in counts:
            counts[a] *= decay
        counts[action] += 1.0
        if prev_a is not None and prev_o is not None:
            if prev_o >= 0:
                wsls_total_win[prev_a] += 1
                if action == prev_a:
                    wsls_stay_win[prev_a] += 1
            else:
                wsls_total_loss[prev_a] += 1
                if action != prev_a:
                    wsls_shift_loss[prev_a] += 1
        prev_a = action
        prev_o = outcome

    correct = 0
    for t in range(n_train, p.n_trials):
        action = p.choices[t]
        outcome = p.wins[t] + p.losses[t]
        probs = {a: counts[a] for a in ACTIONS}
        if prev_a is not None and prev_o is not None:
            if prev_o >= 0:
                sr = wsls_stay_win[prev_a] / max(1, wsls_total_win[prev_a])
                probs[prev_a] *= (1.0 + sr)
            else:
                shr = wsls_shift_loss[prev_a] / max(1, wsls_total_loss[prev_a])
                probs[prev_a] *= max(0.1, 1.0 - shr)
        total = sum(probs.values())
        probs = {a: v / total for a, v in probs.items()}
        if max(probs, key=probs.get) == action:
            correct += 1
        for a in counts:
            counts[a] *= decay
        counts[action] += 1.0
        if prev_a is not None and prev_o is not None:
            if prev_o >= 0:
                wsls_total_win[prev_a] += 1
                if action == prev_a:
                    wsls_stay_win[prev_a] += 1
            else:
                wsls_total_loss[prev_a] += 1
                if action != prev_a:
                    wsls_shift_loss[prev_a] += 1
        prev_a = action
        prev_o = outcome

    return correct


def run_evaluation(
    participants: list[SyntheticParticipant],
    train_frac: float,
    label: str,
) -> None:
    """Run full model and seq-only ablation."""
    configs = {
        "Full (PT+latent+seq)": dict(D=6, lr=0.025, mix=0.3, adapt_pt=True),
        "Seq-only (no brain)":  dict(D=6, lr=0.0, mix=0.0, adapt_pt=False),
    }

    n_train = int(participants[0].n_trials * train_frac)
    n_hold = participants[0].n_trials - n_train

    print(f"\n  --- {label}: Train={n_train}, Holdout={n_hold} ---")
    print(f"  {'Config':<25s}  {'Overall':>8s}  {'vs Hist':>8s}  ", end="")
    print(f"{'Learner':>8s}  {'Mixed':>8s}  {'Risk':>8s}  ", end="")

    # Per-model breakdown header
    models = sorted(set(p.model for p in participants))
    for m in models:
        print(f"  {m:>10s}", end="")
    print()

    print(f"  {'-' * 25}  {'-' * 8}  {'-' * 8}  {'-' * 8}  {'-' * 8}  {'-' * 8}", end="")
    for _ in models:
        print(f"  {'-' * 10}", end="")
    print()

    hist_acc = None

    for config_label, kwargs in configs.items():
        pred = UnifiedCognitivePredictor(actions=ACTIONS, **kwargs)
        results = []
        for p in participants:
            r = run_participant(p, pred, train_frac)
            results.append(r)
            pred.finalize_user(p.participant_id)

        overall = mean(r["accuracy"] for r in results)
        if hist_acc is None:
            hist_acc = mean(r["history_acc"] for r in results)

        contrib = overall - hist_acc

        # Group breakdowns
        learners = [r for r in results if r["safe_frac"] > 0.65]
        mixed = [r for r in results if 0.35 <= r["safe_frac"] <= 0.65]
        risk = [r for r in results if r["safe_frac"] < 0.35]

        l_acc = mean(r["accuracy"] for r in learners) if learners else 0
        m_acc = mean(r["accuracy"] for r in mixed) if mixed else 0
        r_acc = mean(r["accuracy"] for r in risk) if risk else 0

        print(f"  {config_label:<25s}  {overall:>7.1%}  {contrib:>+7.1%}  ", end="")
        print(f"{l_acc:>7.1%}  {m_acc:>7.1%}  {r_acc:>7.1%}  ", end="")

        # Per-model breakdown
        for m in models:
            model_results = [r for r in results if r["model"] == m]
            if model_results:
                print(f"  {mean(r['accuracy'] for r in model_results):>9.1%}", end="")
            else:
                print(f"  {'n/a':>9s}", end="")
        print()

    print(f"  {'History-only baseline':<25s}  {hist_acc:>7.1%}")


def main():
    print("=" * 90)
    print("  SYNTHETIC EXTENDED-IGT EVALUATION")
    print("  200 participants × 300 trials — Does more data help the cognitive layer?")
    print("=" * 90)

    data_path = os.path.join(os.path.dirname(__file__), "synthetic_extended_igt.json")
    participants = load_dataset(data_path)

    from collections import Counter
    model_counts = Counter(p.model for p in participants)
    print(f"\n  Dataset: {len(participants)} participants × {participants[0].n_trials} trials")
    print(f"  Models: {dict(model_counts)}")

    safe_fracs = [p.safe_fraction for p in participants]
    n_l = sum(1 for f in safe_fracs if f > 0.65)
    n_m = sum(1 for f in safe_fracs if 0.35 <= f <= 0.65)
    n_r = sum(1 for f in safe_fracs if f < 0.35)
    print(f"  Profiles: learners={n_l}, mixed={n_m}, risk-seekers={n_r}")

    # Run at multiple splits
    for frac, label in [(0.70, "70/30 split"), (0.85, "85/15 split"), (0.90, "90/10 split")]:
        run_evaluation(participants, frac, label)

    print()
    print("=" * 90)


if __name__ == "__main__":
    main()
