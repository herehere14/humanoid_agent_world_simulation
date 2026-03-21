#!/usr/bin/env python3
"""Unified Cognitive Predictor — 4-Deck IGT Validation.

Tests the unified end-to-end learned cognitive predictor against
the old ensemble and history-only baselines.

Dataset: 504 real humans × 100 trials, ALL participants,
         train on first 70 trials, predict last 30.
"""

from __future__ import annotations

import os
import sys
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any

# Path setup
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.prompt_forest.brain.unified_predictor import UnifiedCognitivePredictor

# Reuse data loader
import run_adaptive_learning_validation as base_val

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DECK_MAP = {1: "A", 2: "B", 3: "C", 4: "D"}
ACTIONS = ["A", "B", "C", "D"]
SAFE_DECKS = {3, 4}
WINDOW_SIZE = 10
TRAIN_FRAC = float(os.environ.get("TRAIN_FRAC", "0.7"))


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class UnifiedResult:
    participant_id: str
    study: str
    safe_fraction: float
    holdout_accuracy: float
    random_accuracy: float
    majority_accuracy: float
    window_accuracies: list[float]
    user_info: dict[str, Any]


# ---------------------------------------------------------------------------
# Run one participant — Unified Predictor
# ---------------------------------------------------------------------------

def run_unified_participant(
    participant: base_val.RealIGTParticipant,
    predictor: UnifiedCognitivePredictor,
    do_calibrate: bool = True,
) -> UnifiedResult:
    """Run one participant through the unified cognitive predictor."""
    n_train = int(participant.n_trials * TRAIN_FRAC)
    n_holdout = participant.n_trials - n_train
    train_choices = participant.choices[:n_train]
    holdout_choices = participant.choices[n_train:]
    uid = participant.participant_id

    correct_per_window: dict[int, list[bool]] = {}
    cumulative_score = 0.0

    # ------- Training phase -------
    for idx, deck_choice in enumerate(train_choices):
        actual = DECK_MAP[deck_choice]
        net_outcome = participant.wins[idx] + participant.losses[idx]
        cumulative_score += net_outcome

        # Predict BEFORE seeing action
        probs = predictor.predict(uid)
        predicted = max(probs, key=probs.get)
        is_correct = predicted == actual
        correct_per_window.setdefault(idx // WINDOW_SIZE, []).append(is_correct)

        # Update cognitive state from observed action + outcome
        predictor.update(uid, actual_action=actual, outcome=net_outcome)

    # ------- Batch calibration after training -------
    if do_calibrate:
        predictor.calibrate(uid)

    # ------- Holdout phase -------
    holdout_correct = 0
    for idx, deck_choice in enumerate(holdout_choices):
        actual = DECK_MAP[deck_choice]
        net_outcome = participant.wins[n_train + idx] + participant.losses[n_train + idx]

        # Predict
        probs = predictor.predict(uid)
        predicted = max(probs, key=probs.get)
        if predicted == actual:
            holdout_correct += 1

        # Update (no adaptation during holdout — calibrated params are fixed)
        predictor.update(uid, actual_action=actual, outcome=net_outcome)

        cumulative_score += net_outcome

    # Random baseline
    rng = random.Random(hash(uid) % 100000 + 7777)
    random_correct = sum(
        1 for c in holdout_choices if rng.choice(ACTIONS) == DECK_MAP[c]
    )

    # Majority-class baseline
    train_deck_counts = {d: 0 for d in ACTIONS}
    for c in train_choices:
        train_deck_counts[DECK_MAP[c]] += 1
    majority_deck = max(train_deck_counts, key=train_deck_counts.get)
    majority_correct = sum(1 for c in holdout_choices if DECK_MAP[c] == majority_deck)

    window_accs = [
        sum(items) / len(items) if items else 0.0
        for _, items in sorted(correct_per_window.items())
    ]

    return UnifiedResult(
        participant_id=uid,
        study=participant.study,
        safe_fraction=participant.safe_fraction,
        holdout_accuracy=holdout_correct / max(1, n_holdout),
        random_accuracy=random_correct / max(1, n_holdout),
        majority_accuracy=majority_correct / max(1, n_holdout),
        window_accuracies=window_accs,
        user_info=predictor.get_user_info(uid),
    )


# ---------------------------------------------------------------------------
# History-only baseline (simple prior + WSLS)
# ---------------------------------------------------------------------------

def run_history_only_participant(
    participant: base_val.RealIGTParticipant,
) -> float:
    """Simple history-only baseline: recency-weighted prior + WSLS."""
    n_train = int(participant.n_trials * TRAIN_FRAC)
    n_holdout = participant.n_trials - n_train
    train_choices = participant.choices[:n_train]
    holdout_choices = participant.choices[n_train:]

    # Build recency-weighted action counts from training
    counts = {d: 0.5 for d in ACTIONS}  # Dirichlet smoothing
    decay = 0.98
    prev_action = None
    prev_outcome = None

    # Track transitions for WSLS
    wsls_stay_after_win = {d: 0.5 for d in ACTIONS}
    wsls_total_win = {d: 0.5 for d in ACTIONS}
    wsls_shift_after_loss = {d: 0.5 for d in ACTIONS}
    wsls_total_loss = {d: 0.5 for d in ACTIONS}

    for idx, deck_choice in enumerate(train_choices):
        actual = DECK_MAP[deck_choice]
        net_outcome = participant.wins[idx] + participant.losses[idx]

        # Decay and add
        for d in counts:
            counts[d] *= decay
        counts[actual] += 1.0

        # WSLS tracking
        if prev_action is not None and prev_outcome is not None:
            if prev_outcome >= 0:
                wsls_total_win[prev_action] += 1
                if actual == prev_action:
                    wsls_stay_after_win[prev_action] += 1
            else:
                wsls_total_loss[prev_action] += 1
                if actual != prev_action:
                    wsls_shift_after_loss[prev_action] += 1

        prev_action = actual
        prev_outcome = net_outcome

    # Holdout
    holdout_correct = 0
    for idx, deck_choice in enumerate(holdout_choices):
        actual = DECK_MAP[deck_choice]
        net_outcome = participant.wins[n_train + idx] + participant.losses[n_train + idx]

        # Predict using prior + WSLS
        probs = {d: counts[d] for d in ACTIONS}

        if prev_action is not None and prev_outcome is not None:
            if prev_outcome >= 0:
                # Win-stay: boost prev_action
                stay_rate = wsls_stay_after_win[prev_action] / max(1, wsls_total_win[prev_action])
                probs[prev_action] *= (1.0 + stay_rate)
            else:
                # Lose-shift: reduce prev_action
                shift_rate = wsls_shift_after_loss[prev_action] / max(1, wsls_total_loss[prev_action])
                probs[prev_action] *= max(0.1, 1.0 - shift_rate)

        total = sum(probs.values())
        probs = {d: v / total for d, v in probs.items()}
        predicted = max(probs, key=probs.get)

        if predicted == actual:
            holdout_correct += 1

        # Update
        for d in counts:
            counts[d] *= decay
        counts[actual] += 1.0

        if prev_action is not None and prev_outcome is not None:
            if prev_outcome >= 0:
                wsls_total_win[prev_action] += 1
                if actual == prev_action:
                    wsls_stay_after_win[prev_action] += 1
            else:
                wsls_total_loss[prev_action] += 1
                if actual != prev_action:
                    wsls_shift_after_loss[prev_action] += 1

        prev_action = actual
        prev_outcome = net_outcome

    return holdout_correct / max(1, n_holdout)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _print_report(
    results: list[UnifiedResult],
    history_accs: list[float],
) -> None:
    n = len(results)
    mean_acc = mean(r.holdout_accuracy for r in results)
    mean_rand = mean(r.random_accuracy for r in results)
    mean_maj = mean(r.majority_accuracy for r in results)
    mean_hist = mean(history_accs)
    std_acc = stdev(r.holdout_accuracy for r in results) if n > 1 else 0.0
    std_hist = stdev(history_accs) if n > 1 else 0.0

    learners = [i for i, r in enumerate(results) if r.safe_fraction > 0.65]
    mixed = [i for i, r in enumerate(results) if 0.35 <= r.safe_fraction <= 0.65]
    risk_seekers = [i for i, r in enumerate(results) if r.safe_fraction < 0.35]

    brain_contrib = mean_acc - mean_hist

    print()
    print("=" * 90)
    print("  UNIFIED COGNITIVE PREDICTOR — 4-DECK IGT VALIDATION")
    print("  Steingroever et al. (2015) — All 504 participants, 4 actions (A/B/C/D)")
    print("=" * 90)
    print()
    print(f"  Participants:     {n}")
    print(f"  Actions:          4 decks (A, B, C, D)")
    n_train = int(100 * TRAIN_FRAC)
    print(f"  Train/Holdout:    {n_train}/{100 - n_train} trials per participant (TRAIN_FRAC={TRAIN_FRAC})")
    print(f"  Random baseline:  25% (1/4 decks)")
    print()
    print("  --- Overall Results ---")
    print()
    print(f"  Unified Brain:    {mean_acc:.1%} (+/- {std_acc:.1%})")
    print(f"  History-Only:     {mean_hist:.1%} (+/- {std_hist:.1%})")
    print(f"  Majority-class:   {mean_maj:.1%}")
    print(f"  Random:           {mean_rand:.1%}")
    print()
    print(f"  Brain contribution: {brain_contrib:+.1%}")
    print(f"  Lift over random:   {mean_acc - mean_rand:+.1%}")
    print(f"  Lift over majority: {mean_acc - mean_maj:+.1%}")
    print()

    # Per-group breakdown
    print("  --- Per-Group Results ---")
    print(f"  {'Group':>30s}    {'Brain':>8s}  {'History':>8s}  {'Contrib':>8s}  {'Random':>8s}")
    print("  " + "-" * 76)
    for label, idxs in [
        ("ALL", list(range(n))),
        ("Learners (>65% safe)", learners),
        ("Mixed (35-65% safe)", mixed),
        ("Risk-seekers (<35%)", risk_seekers),
    ]:
        if not idxs:
            continue
        g_brain = mean(results[i].holdout_accuracy for i in idxs)
        g_hist = mean(history_accs[i] for i in idxs)
        g_rand = mean(results[i].random_accuracy for i in idxs)
        g_contrib = g_brain - g_hist
        print(
            f"  {label:>30s} (n={len(idxs):>3d})  "
            f"{g_brain:>7.1%}  {g_hist:>7.1%}  {g_contrib:>+7.1%}  {g_rand:>7.1%}"
        )
    print()

    # Learning curve
    max_windows = max(len(r.window_accuracies) for r in results)
    print("  --- Learning Curve (training windows) ---")
    for w in range(min(max_windows, 7)):
        accs_at_w = [
            r.window_accuracies[w]
            for r in results
            if w < len(r.window_accuracies)
        ]
        if accs_at_w:
            print(f"    Window {w+1}: {mean(accs_at_w):.1%}")
    print()

    # Top helps and hurts
    contribs = [(results[i], history_accs[i]) for i in range(n)]
    contribs.sort(key=lambda x: x[0].holdout_accuracy - x[1], reverse=True)
    print("  --- Top 10 brain helps ---")
    print(f"  {'ID':>10s}  {'safe%':>5s}  {'Brain':>6s}  {'Hist':>6s}  {'Contrib':>7s}")
    for r, h in contribs[:10]:
        c = r.holdout_accuracy - h
        print(f"  {r.participant_id:>10s}  {r.safe_fraction:>5.0%}  {r.holdout_accuracy:>5.0%}  {h:>5.0%}  {c:>+6.0%}")
    print()

    print("  --- Top 10 brain hurts ---")
    for r, h in contribs[-10:]:
        c = r.holdout_accuracy - h
        print(f"  {r.participant_id:>10s}  {r.safe_fraction:>5.0%}  {r.holdout_accuracy:>5.0%}  {h:>5.0%}  {c:>+6.0%}")
    print()

    n_helps = sum(1 for r, h in contribs if r.holdout_accuracy > h)
    n_hurts = sum(1 for r, h in contribs if r.holdout_accuracy < h)
    n_ties = n - n_helps - n_hurts
    print(f"  Brain helps: {n_helps}/{n}  |  Hurts: {n_hurts}/{n}  |  Ties: {n_ties}/{n}")
    print()

    # Verdicts
    print("  --- Verdicts ---")
    verdicts = [
        ("Brain > random (25%)", mean_acc > 0.25),
        ("Brain > majority-class", mean_acc > mean_maj),
        ("Brain > history-only", mean_acc > mean_hist),
        ("Brain contribution >= 10%", brain_contrib >= 0.10),
        ("Brain accuracy >= 60%", mean_acc >= 0.60),
        ("Brain accuracy >= 70%", mean_acc >= 0.70),
        ("Brain accuracy >= 80%", mean_acc >= 0.80),
    ]
    for label, passed in verdicts:
        marker = "[+]" if passed else "[-]"
        print(f"    {marker} {label}: {'PASS' if passed else 'FAIL'}")
    print()

    # Sample user diagnostics
    print("  --- Sample User Diagnostics (first 5) ---")
    for r in results[:5]:
        info = r.user_info
        z = info.get('z', [])
        z_str = ",".join(f"{v:.2f}" for v in z) if z else "?"
        print(f"    {r.participant_id}: acc={r.holdout_accuracy:.0%}  "
              f"a={info.get('alpha', '?')}  "
              f"lam={info.get('lambda_', '?')}  "
              f"A={info.get('A', '?')}  "
              f"th={info.get('theta', '?')}  "
              f"w_rl={info.get('w_rl', '?')}  "
              f"z=[{z_str}]")
    print()
    print("=" * 90)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 90)
    print("  UNIFIED COGNITIVE PREDICTOR — 4-DECK IGT")
    print("  Loading Steingroever et al. (2015) — ALL 504 participants, 4 decks")
    print("=" * 90)
    print()

    participants = base_val.load_steingroever_data(max_participants=9999, seed=0)

    safe_fracs = [p.safe_fraction for p in participants]
    n_learn = sum(1 for f in safe_fracs if f > 0.65)
    n_mixed = sum(1 for f in safe_fracs if 0.35 <= f <= 0.65)
    n_risk = sum(1 for f in safe_fracs if f < 0.35)
    print(
        f"  Loaded {len(participants)} participants "
        f"(learners={n_learn}, mixed={n_mixed}, risk-seekers={n_risk})"
    )
    print()

    # ===== UNIFIED PREDICTOR =====
    print("  === Running UNIFIED COGNITIVE PREDICTOR — Integrated Latent State ===")
    predictor = UnifiedCognitivePredictor(
        actions=ACTIONS,
        D=6,
        lr=0.025,
        mix=0.3,
        bigram_alpha=0.3,
        bigram_decay=0.97,
        adapt_pt=True,
        pop_seq_transfer_weight=0.3,
    )

    results: list[UnifiedResult] = []
    for i, participant in enumerate(participants):
        if (i + 1) % 50 == 0 or i == 0 or i == len(participants) - 1:
            print(
                f"    [{i+1:>3d}/{len(participants)}] {participant.participant_id:>10s} "
                f"({participant.study:>18s}, safe={participant.safe_fraction:.0%})",
                flush=True,
            )
        result = run_unified_participant(participant, predictor, do_calibrate=False)
        results.append(result)
        predictor.finalize_user(participant.participant_id)
    print()

    # ===== HISTORY-ONLY =====
    print("  === Running HISTORY-ONLY baseline ===")
    history_accs: list[float] = []
    for i, participant in enumerate(participants):
        if (i + 1) % 50 == 0 or i == 0 or i == len(participants) - 1:
            print(f"    [{i+1:>3d}/{len(participants)}] {participant.participant_id:>10s}", flush=True)
        acc = run_history_only_participant(participant)
        history_accs.append(acc)
    print()

    # Report
    _print_report(results, history_accs)


if __name__ == "__main__":
    main()
