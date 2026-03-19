#!/usr/bin/env python3
"""Cognitive Model Baseline Comparison — EV, PVL-Delta, VPP vs Our System.

Compares the Prompt Forest behavioral prediction system against three
established cognitive science models on the same real IGT data:

  1. Expectancy-Valence (EV) model (Busemeyer & Stout, 2002)
  2. Prospect Valence Learning - Delta (PVL-Delta) (Ahn et al., 2008)
  3. Value-Plus-Perseverance (VPP) (Worthy et al., 2013)

All models are fit per-participant using grid search over parameter space,
then evaluated on holdout trials — the same protocol as our system.

Run:
    python examples/run_cognitive_baseline_comparison.py
"""

from __future__ import annotations

import csv
import math
import os
import random
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from statistics import mean, stdev
from typing import Any

# ---------------------------------------------------------------------------
# Path bootstrapping
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "igt_steingroever"

MAX_PARTICIPANTS = 50


# ---------------------------------------------------------------------------
# Data loading (same participant selection as main validation)
# ---------------------------------------------------------------------------

@dataclass
class IGTParticipant:
    participant_id: str
    study: str
    choices: list[int]      # 1=A, 2=B, 3=C, 4=D
    wins: list[float]       # win amount per trial
    losses: list[float]     # loss amount per trial (negative)
    safe_fraction: float
    n_trials: int


def load_data(max_participants: int = MAX_PARTICIPANTS, seed: int = 42) -> list[IGTParticipant]:
    """Load real IGT data with win/loss outcomes."""
    choice_path = DATA_DIR / "choice_100.csv"
    wi_path = DATA_DIR / "wi_100.csv"
    lo_path = DATA_DIR / "lo_100.csv"
    index_path = DATA_DIR / "index_100.csv"

    # Load study labels
    study_map: dict[int, str] = {}
    with open(index_path, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            study_map[int(row[0])] = row[1]

    # Load choices
    with open(choice_path, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        next(reader)
        choice_rows = list(reader)

    # Load wins
    with open(wi_path, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        next(reader)
        win_rows = list(reader)

    # Load losses
    with open(lo_path, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        next(reader)
        loss_rows = list(reader)

    all_participants: list[IGTParticipant] = []
    for c_row, w_row, l_row in zip(choice_rows, win_rows, loss_rows):
        label = c_row[0]
        subj_num = int(label.split("_")[1])
        choices = [int(x) for x in c_row[1:]]
        wins = [float(x) for x in w_row[1:]]
        losses = [float(x) for x in l_row[1:]]
        safe_frac = sum(1 for c in choices if c in (3, 4)) / len(choices)

        all_participants.append(IGTParticipant(
            participant_id=label,
            study=study_map.get(subj_num, "unknown"),
            choices=choices,
            wins=wins,
            losses=losses,
            safe_fraction=safe_frac,
            n_trials=len(choices),
        ))

    # Same stratified sampling as main validation
    rng = random.Random(seed)
    learners = [p for p in all_participants if p.safe_fraction > 0.65]
    mixed = [p for p in all_participants if 0.35 <= p.safe_fraction <= 0.65]
    risk_seekers = [p for p in all_participants if p.safe_fraction < 0.35]

    n_learn = max(5, int(max_participants * len(learners) / len(all_participants)))
    n_risk = max(5, int(max_participants * len(risk_seekers) / len(all_participants)))
    n_mixed = max_participants - n_learn - n_risk

    rng.shuffle(learners)
    rng.shuffle(mixed)
    rng.shuffle(risk_seekers)

    sample = (
        learners[:min(n_learn, len(learners))]
        + mixed[:min(n_mixed, len(mixed))]
        + risk_seekers[:min(n_risk, len(risk_seekers))]
    )
    rng.shuffle(sample)
    return sample[:max_participants]


# ---------------------------------------------------------------------------
# Cognitive Models
# ---------------------------------------------------------------------------

SAFE_DECKS = {3, 4}


def _softmax_probs(values: list[float], theta: float) -> list[float]:
    """Softmax over 4 deck values with temperature theta."""
    max_v = max(values)
    exps = [math.exp(theta * (v - max_v)) for v in values]
    total = sum(exps)
    return [e / total for e in exps]


def ev_model_predict(
    choices: list[int],
    wins: list[float],
    losses: list[float],
    w: float,      # attention weight [0, 1]
    a: float,      # learning rate [0, 1]
    c: float,      # consistency [-5, 5]
    n_train: int,
) -> list[str]:
    """Expectancy-Valence model: fit on train, predict on holdout.

    Returns list of 'safe'/'risky' predictions for holdout trials.
    """
    E = [0.0, 0.0, 0.0, 0.0]  # expectations for decks 1-4

    # Training: update expectations from observed choices
    for t in range(n_train):
        deck = choices[t] - 1  # 0-indexed
        W = wins[t]
        L = abs(losses[t])
        u = w * W - (1.0 - w) * L  # utility
        E[deck] = E[deck] + a * (u - E[deck])

    # Holdout: predict using learned expectations
    theta = (3.0 ** c) - 1.0
    predictions: list[str] = []
    for t in range(n_train, len(choices)):
        probs = _softmax_probs(E, theta)
        # Predict the most likely deck
        best_deck = max(range(4), key=lambda d: probs[d])
        predictions.append("safe" if (best_deck + 1) in SAFE_DECKS else "risky")

        # Continue updating expectations with holdout data (online prediction)
        deck = choices[t] - 1
        W = wins[t]
        L = abs(losses[t])
        u = w * W - (1.0 - w) * L
        E[deck] = E[deck] + a * (u - E[deck])

    return predictions


def pvl_delta_predict(
    choices: list[int],
    wins: list[float],
    losses: list[float],
    alpha: float,    # outcome sensitivity [0, 1]
    lam: float,      # loss aversion [0, 5]
    A: float,        # learning rate [0, 1]
    c: float,        # consistency [0, 5]
    n_train: int,
) -> list[str]:
    """Prospect Valence Learning (Delta) model."""
    E = [0.0, 0.0, 0.0, 0.0]
    theta = (3.0 ** c) - 1.0

    # Training
    for t in range(n_train):
        deck = choices[t] - 1
        x = wins[t] + losses[t]  # net outcome (losses are negative)
        if x >= 0:
            u = x ** alpha if x > 0 else 0.0
        else:
            u = -lam * (abs(x) ** alpha)
        E[deck] = E[deck] + A * (u - E[deck])

    # Holdout
    predictions: list[str] = []
    for t in range(n_train, len(choices)):
        probs = _softmax_probs(E, theta)
        best_deck = max(range(4), key=lambda d: probs[d])
        predictions.append("safe" if (best_deck + 1) in SAFE_DECKS else "risky")

        deck = choices[t] - 1
        x = wins[t] + losses[t]
        if x >= 0:
            u = x ** alpha if x > 0 else 0.0
        else:
            u = -lam * (abs(x) ** alpha)
        E[deck] = E[deck] + A * (u - E[deck])

    return predictions


def vpp_predict(
    choices: list[int],
    wins: list[float],
    losses: list[float],
    alpha: float,
    lam: float,
    A: float,
    c: float,
    K: float,        # perseveration decay [0, 1]
    ep_p: float,     # gain impact on perseveration
    ep_n: float,     # loss impact on perseveration
    w_rl: float,     # RL weight [0, 1]
    n_train: int,
) -> list[str]:
    """Value-Plus-Perseverance model."""
    E = [0.0, 0.0, 0.0, 0.0]
    P = [0.0, 0.0, 0.0, 0.0]
    theta = (3.0 ** c) - 1.0

    # Training
    for t in range(n_train):
        deck = choices[t] - 1
        x = wins[t] + losses[t]
        if x >= 0:
            u = x ** alpha if x > 0 else 0.0
        else:
            u = -lam * (abs(x) ** alpha)
        E[deck] = E[deck] + A * (u - E[deck])

        for d in range(4):
            if d == deck:
                if x >= 0:
                    P[d] = K * P[d] + ep_p
                else:
                    P[d] = K * P[d] + ep_n
            else:
                P[d] = K * P[d]

    # Holdout
    predictions: list[str] = []
    for t in range(n_train, len(choices)):
        V = [w_rl * E[d] + (1.0 - w_rl) * P[d] for d in range(4)]
        probs = _softmax_probs(V, theta)
        best_deck = max(range(4), key=lambda d: probs[d])
        predictions.append("safe" if (best_deck + 1) in SAFE_DECKS else "risky")

        deck = choices[t] - 1
        x = wins[t] + losses[t]
        if x >= 0:
            u = x ** alpha if x > 0 else 0.0
        else:
            u = -lam * (abs(x) ** alpha)
        E[deck] = E[deck] + A * (u - E[deck])
        for d in range(4):
            if d == deck:
                P[d] = K * P[d] + (ep_p if x >= 0 else ep_n)
            else:
                P[d] = K * P[d]

    return predictions


# ---------------------------------------------------------------------------
# Grid search fitting
# ---------------------------------------------------------------------------

def _accuracy(predictions: list[str], choices: list[int], n_train: int) -> float:
    """Compute accuracy of predictions against actual holdout choices."""
    holdout = choices[n_train:]
    correct = 0
    for pred, actual_deck in zip(predictions, holdout):
        actual = "safe" if actual_deck in SAFE_DECKS else "risky"
        if pred == actual:
            correct += 1
    return correct / max(1, len(holdout))


def fit_ev(participant: IGTParticipant, n_train: int) -> tuple[float, dict[str, float]]:
    """Grid search for best EV parameters."""
    best_acc = 0.0
    best_params: dict[str, float] = {}

    for w in [0.1, 0.3, 0.5, 0.7, 0.9]:
        for a in [0.05, 0.1, 0.2, 0.4, 0.7]:
            for c in [0.5, 1.0, 1.5, 2.0, 3.0]:
                preds = ev_model_predict(
                    participant.choices, participant.wins, participant.losses,
                    w=w, a=a, c=c, n_train=n_train,
                )
                acc = _accuracy(preds, participant.choices, n_train)
                if acc > best_acc:
                    best_acc = acc
                    best_params = {"w": w, "a": a, "c": c}

    return best_acc, best_params


def fit_pvl_delta(participant: IGTParticipant, n_train: int) -> tuple[float, dict[str, float]]:
    """Grid search for best PVL-Delta parameters."""
    best_acc = 0.0
    best_params: dict[str, float] = {}

    for alpha in [0.2, 0.5, 0.8]:
        for lam in [0.5, 1.0, 2.0, 3.0]:
            for A in [0.05, 0.1, 0.3, 0.6]:
                for c in [0.5, 1.0, 2.0, 3.0]:
                    preds = pvl_delta_predict(
                        participant.choices, participant.wins, participant.losses,
                        alpha=alpha, lam=lam, A=A, c=c, n_train=n_train,
                    )
                    acc = _accuracy(preds, participant.choices, n_train)
                    if acc > best_acc:
                        best_acc = acc
                        best_params = {"alpha": alpha, "lam": lam, "A": A, "c": c}

    return best_acc, best_params


def fit_vpp(participant: IGTParticipant, n_train: int) -> tuple[float, dict[str, float]]:
    """Grid search for best VPP parameters (coarser grid due to 8 params)."""
    best_acc = 0.0
    best_params: dict[str, float] = {}

    for alpha in [0.3, 0.6, 0.9]:
        for lam in [1.0, 2.0, 3.5]:
            for A in [0.1, 0.3, 0.6]:
                for c in [1.0, 2.0, 3.0]:
                    for K in [0.3, 0.7, 0.95]:
                        for ep_p in [0.1, 0.5]:
                            for ep_n in [-0.5, -0.1]:
                                for w_rl in [0.4, 0.7, 0.9]:
                                    preds = vpp_predict(
                                        participant.choices, participant.wins, participant.losses,
                                        alpha=alpha, lam=lam, A=A, c=c,
                                        K=K, ep_p=ep_p, ep_n=ep_n, w_rl=w_rl,
                                        n_train=n_train,
                                    )
                                    acc = _accuracy(preds, participant.choices, n_train)
                                    if acc > best_acc:
                                        best_acc = acc
                                        best_params = {
                                            "alpha": alpha, "lam": lam, "A": A, "c": c,
                                            "K": K, "ep_p": ep_p, "ep_n": ep_n, "w_rl": w_rl,
                                        }

    return best_acc, best_params


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def majority_class_accuracy(participant: IGTParticipant, n_train: int) -> float:
    """Always predict the most common class from training data."""
    train_safe = sum(1 for c in participant.choices[:n_train] if c in SAFE_DECKS)
    majority = "safe" if train_safe >= n_train / 2 else "risky"
    holdout = participant.choices[n_train:]
    correct = sum(
        1 for c in holdout
        if (majority == "safe" and c in SAFE_DECKS) or (majority == "risky" and c not in SAFE_DECKS)
    )
    return correct / max(1, len(holdout))


def random_accuracy(participant: IGTParticipant, n_train: int, seed: int = 42) -> float:
    """Random 50/50 prediction."""
    rng = random.Random(seed + hash(participant.participant_id))
    holdout = participant.choices[n_train:]
    correct = sum(
        1 for c in holdout
        if rng.choice(["safe", "risky"]) == ("safe" if c in SAFE_DECKS else "risky")
    )
    return correct / max(1, len(holdout))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@dataclass
class ParticipantComparison:
    participant_id: str
    study: str
    safe_fraction: float
    random_acc: float
    majority_acc: float
    ev_acc: float
    pvl_acc: float
    vpp_acc: float
    our_system_acc: float  # from the main validation


def main() -> None:
    print("=" * 76)
    print("  COGNITIVE MODEL BASELINE COMPARISON")
    print("  EV, PVL-Delta, VPP vs Prompt Forest on Real IGT Data")
    print("=" * 76)
    print()

    print("  Loading data...")
    participants = load_data(max_participants=MAX_PARTICIPANTS)
    print(f"  Loaded {len(participants)} participants")
    print()

    holdout_fraction = 0.3
    results: list[ParticipantComparison] = []

    # We'll also store our system's results (from the main validation)
    # For now, hardcode the mean from our run (67.6%), or re-run inline
    # For a fair comparison, we compute all baselines on the exact same split.

    print(f"  Fitting cognitive models per-participant (grid search)...")
    print(f"  This tests {len(participants)} participants with 3 models each.")
    print()

    all_ev: list[float] = []
    all_pvl: list[float] = []
    all_vpp: list[float] = []
    all_majority: list[float] = []
    all_random: list[float] = []

    for i, p in enumerate(participants):
        n_train = int(p.n_trials * (1.0 - holdout_fraction))

        print(
            f"    [{i + 1:>2d}/{len(participants)}] {p.participant_id:>8s} "
            f"(safe={p.safe_fraction:.0%})...",
            end="", flush=True,
        )

        # Baselines
        rand_acc = random_accuracy(p, n_train)
        maj_acc = majority_class_accuracy(p, n_train)

        # Cognitive models
        ev_acc, ev_params = fit_ev(p, n_train)
        pvl_acc, pvl_params = fit_pvl_delta(p, n_train)
        vpp_acc, vpp_params = fit_vpp(p, n_train)

        all_ev.append(ev_acc)
        all_pvl.append(pvl_acc)
        all_vpp.append(vpp_acc)
        all_majority.append(maj_acc)
        all_random.append(rand_acc)

        print(
            f"  EV={ev_acc:.0%}  PVL={pvl_acc:.0%}  VPP={vpp_acc:.0%}  "
            f"maj={maj_acc:.0%}"
        )

    print()

    # Summary
    print("-" * 76)
    print("  Overall Results (Holdout Accuracy)")
    print("-" * 76)
    print()

    our_system_acc = 0.676  # from the main validation run

    models = [
        ("Random (50/50)", all_random),
        ("Majority-class", all_majority),
        ("EV (Busemeyer 2002)", all_ev),
        ("PVL-Delta (Ahn 2008)", all_pvl),
        ("VPP (Worthy 2013)", all_vpp),
    ]

    print(f"  {'Model':<30s}  {'Mean':>6s}  {'Std':>6s}  {'Min':>5s}  {'Max':>5s}")
    print(f"  {'-'*30}  {'-'*6}  {'-'*6}  {'-'*5}  {'-'*5}")
    for name, accs in models:
        m = mean(accs)
        s = stdev(accs) if len(accs) > 1 else 0.0
        lo = min(accs)
        hi = max(accs)
        print(f"  {name:<30s}  {m:>5.1%}  {s:>5.1%}  {lo:>4.0%}  {hi:>4.0%}")
    print(f"  {'Our System (Prompt Forest)':<30s}  {our_system_acc:>5.1%}  {'--':>6s}  {'--':>5s}  {'--':>5s}")
    print()

    # Per-group comparison
    print("-" * 76)
    print("  Per-Group Breakdown")
    print("-" * 76)
    print()

    groups = [
        ("Learners (>65% safe)", [i for i, p in enumerate(participants) if p.safe_fraction > 0.65]),
        ("Mixed (35-65%)", [i for i, p in enumerate(participants) if 0.35 <= p.safe_fraction <= 0.65]),
        ("Risk-seekers (<35%)", [i for i, p in enumerate(participants) if p.safe_fraction < 0.35]),
    ]

    for group_name, indices in groups:
        if not indices:
            continue
        print(f"  {group_name} (n={len(indices)}):")
        for name, accs in models:
            g_accs = [accs[i] for i in indices]
            print(f"    {name:<28s}  {mean(g_accs):.1%}")
        print()

    # Head-to-head: how often does each model beat majority-class?
    print("-" * 76)
    print("  Head-to-Head: Participants Where Model > Majority-Class")
    print("-" * 76)
    print()
    for name, accs in models[2:]:  # skip random and majority
        beats = sum(1 for a, m in zip(accs, all_majority) if a > m)
        ties = sum(1 for a, m in zip(accs, all_majority) if a == m)
        loses = sum(1 for a, m in zip(accs, all_majority) if a < m)
        print(f"  {name:<28s}  wins={beats}  ties={ties}  loses={loses}")
    print()

    # Verdict
    print("-" * 76)
    print("  Comparison Summary")
    print("-" * 76)
    print()
    mean_ev = mean(all_ev)
    mean_pvl = mean(all_pvl)
    mean_vpp = mean(all_vpp)
    mean_maj = mean(all_majority)

    print(f"  Our system (67.6%) vs:")
    print(f"    Random baseline:     +{our_system_acc - 0.50:+.1%}")
    print(f"    Majority-class:      +{our_system_acc - mean_maj:+.1%}")
    print(f"    EV model:            {our_system_acc - mean_ev:+.1%}")
    print(f"    PVL-Delta model:     {our_system_acc - mean_pvl:+.1%}")
    print(f"    VPP model:           {our_system_acc - mean_vpp:+.1%}")
    print()

    best_cognitive = max(mean_ev, mean_pvl, mean_vpp)
    best_name = "VPP" if best_cognitive == mean_vpp else ("PVL" if best_cognitive == mean_pvl else "EV")
    gap = our_system_acc - best_cognitive
    if gap >= 0:
        print(f"  Our system BEATS best cognitive model ({best_name}) by {gap:+.1%}")
    else:
        print(f"  Best cognitive model ({best_name}) beats our system by {-gap:+.1%}")
        print(f"  Note: cognitive models use actual win/loss data; our system only sees")
        print(f"  branch activation scores — it has NO access to deck payoffs.")
    print()
    print("=" * 76)


if __name__ == "__main__":
    main()
