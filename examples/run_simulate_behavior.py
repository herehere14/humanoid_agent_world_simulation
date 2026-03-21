#!/usr/bin/env python3
"""Behavior Simulation — Does the brain act like a human?

Trains the brain on a real participant's history, then has the brain
GENERATE its own sequence of actions by sampling from its predicted
probabilities.  Compares the generated behavioral signatures against
the real participant's actual behavior.

This is NOT prediction accuracy — this is: "if we put the brain in
the participant's shoes, does it behave the same way?"
"""

from __future__ import annotations

import os
import sys
import random
from collections import Counter, defaultdict
from statistics import mean

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.prompt_forest.brain.unified_predictor import UnifiedCognitivePredictor
import run_adaptive_learning_validation as base_val

DECK_MAP = {1: "A", 2: "B", 3: "C", 4: "D"}
ACTIONS = ["A", "B", "C", "D"]
SAFE = {"C", "D"}


# ---------------------------------------------------------------------------
# IGT payoff model (empirical from dataset)
# ---------------------------------------------------------------------------

def build_payoff_model(
    participants: list[base_val.RealIGTParticipant],
) -> dict[str, list[float]]:
    """Build empirical outcome distributions per deck."""
    outcomes: dict[str, list[float]] = {a: [] for a in ACTIONS}
    for p in participants:
        for i, c in enumerate(p.choices):
            outcomes[DECK_MAP[c]].append(p.wins[i] + p.losses[i])
    return outcomes


def sample_outcome(
    payoff_model: dict[str, list[float]], action: str, rng: random.Random
) -> float:
    return rng.choice(payoff_model[action])


# ---------------------------------------------------------------------------
# Behavioral signature extraction
# ---------------------------------------------------------------------------

def behavioral_signature(actions: list[str], outcomes: list[float]) -> dict:
    """Extract behavioral fingerprint from an action-outcome sequence."""
    n = len(actions)
    if n == 0:
        return {}

    # Deck preferences
    counts = Counter(actions)
    freqs = {a: counts.get(a, 0) / n for a in ACTIONS}

    # Safe deck rate
    safe_rate = sum(1 for a in actions if a in SAFE) / n

    # Switch rate
    switches = sum(1 for i in range(1, n) if actions[i] != actions[i - 1])
    switch_rate = switches / max(1, n - 1)

    # WSLS
    win_stay = 0
    win_count = 0
    lose_shift = 0
    lose_count = 0
    for i in range(1, min(n, len(outcomes))):
        if outcomes[i - 1] > 0:
            win_count += 1
            if actions[i] == actions[i - 1]:
                win_stay += 1
        elif outcomes[i - 1] < 0:
            lose_count += 1
            if actions[i] != actions[i - 1]:
                lose_shift += 1

    # Streaks
    streaks = []
    cur = 1
    for i in range(1, n):
        if actions[i] == actions[i - 1]:
            cur += 1
        else:
            streaks.append(cur)
            cur = 1
    streaks.append(cur)
    avg_streak = mean(streaks) if streaks else 1.0
    max_streak = max(streaks) if streaks else 1

    # Learning: safe rate in first half vs second half
    mid = n // 2
    safe_first = sum(1 for a in actions[:mid] if a in SAFE) / max(1, mid)
    safe_second = sum(1 for a in actions[mid:] if a in SAFE) / max(1, n - mid)

    # Entropy
    import math
    entropy = 0.0
    for a in ACTIONS:
        p = freqs.get(a, 0)
        if p > 0:
            entropy -= p * math.log2(p)

    return {
        "freqs": freqs,
        "safe_rate": safe_rate,
        "switch_rate": switch_rate,
        "win_stay": win_stay / max(1, win_count),
        "lose_shift": lose_shift / max(1, lose_count),
        "avg_streak": avg_streak,
        "max_streak": max_streak,
        "safe_first_half": safe_first,
        "safe_second_half": safe_second,
        "learning_delta": safe_second - safe_first,
        "entropy": entropy,
    }


# ---------------------------------------------------------------------------
# Simulate behavior
# ---------------------------------------------------------------------------

def simulate_participant(
    predictor: UnifiedCognitivePredictor,
    user_id: str,
    n_trials: int,
    payoff_model: dict[str, list[float]],
    rng: random.Random,
) -> tuple[list[str], list[float]]:
    """Generate a sequence of actions by sampling from the brain."""
    sim_actions: list[str] = []
    sim_outcomes: list[float] = []

    for t in range(n_trials):
        # Brain predicts probability distribution
        probs = predictor.predict(user_id)

        # SAMPLE from the distribution (not argmax — mimicking, not predicting)
        r = rng.random()
        cumulative = 0.0
        chosen = ACTIONS[-1]
        for a in ACTIONS:
            cumulative += probs.get(a, 0.0)
            if r <= cumulative:
                chosen = a
                break

        # Get outcome from payoff model
        outcome = sample_outcome(payoff_model, chosen, rng)

        # Update brain state
        predictor.update(user_id, actual_action=chosen, outcome=outcome)

        sim_actions.append(chosen)
        sim_outcomes.append(outcome)

    return sim_actions, sim_outcomes


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_signature_comparison(
    label: str,
    real_sig: dict,
    sim_sigs: list[dict],
    real_actions: list[str],
    sim_actions_example: list[str],
):
    """Compare real vs simulated behavioral signatures."""
    print(f"\n  {'=' * 70}")
    print(f"  {label}")
    print(f"  {'=' * 70}")

    # Show action sequences (first 50 chars)
    def seq_str(actions, n=60):
        return "".join(actions[:n])

    print(f"\n  Real sequence:      {seq_str(real_actions)}")
    print(f"  Simulated (1 run):  {seq_str(sim_actions_example)}")

    # Average simulated signatures
    def avg_field(sigs, key):
        vals = [s[key] for s in sigs if key in s]
        return mean(vals) if vals else 0.0

    def avg_freq(sigs, action):
        return mean(s["freqs"].get(action, 0) for s in sigs)

    print(f"\n  {'Metric':<25s} {'Real':>8s}  {'Simulated':>10s}  {'Match':>7s}")
    print(f"  {'-' * 55}")

    metrics = [
        ("Safe deck rate", real_sig["safe_rate"], avg_field(sim_sigs, "safe_rate")),
        ("Switch rate", real_sig["switch_rate"], avg_field(sim_sigs, "switch_rate")),
        ("Win-stay rate", real_sig["win_stay"], avg_field(sim_sigs, "win_stay")),
        ("Lose-shift rate", real_sig["lose_shift"], avg_field(sim_sigs, "lose_shift")),
        ("Avg streak length", real_sig["avg_streak"], avg_field(sim_sigs, "avg_streak")),
        ("Entropy (bits)", real_sig["entropy"], avg_field(sim_sigs, "entropy")),
        ("Learning delta", real_sig["learning_delta"], avg_field(sim_sigs, "learning_delta")),
    ]
    for name, real_v, sim_v in metrics:
        diff = abs(real_v - sim_v)
        match = "good" if diff < 0.15 else ("ok" if diff < 0.25 else "poor")
        print(f"  {name:<25s} {real_v:>8.2f}  {sim_v:>10.2f}  {match:>7s}")

    # Deck preferences
    print(f"\n  Deck preferences:")
    print(f"  {'Deck':<8s} {'Real':>8s}  {'Simulated':>10s}")
    for a in ACTIONS:
        real_f = real_sig["freqs"].get(a, 0)
        sim_f = avg_freq(sim_sigs, a)
        bar_r = "#" * int(real_f * 30)
        bar_s = "#" * int(sim_f * 30)
        print(f"    {a:<6s} {real_f:>7.0%}  {bar_r:<15s}  {sim_f:>7.0%}  {bar_s}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 74)
    print("  BEHAVIOR SIMULATION — Does the brain act like a human?")
    print("  Train on real participant, then generate simulated behavior")
    print("=" * 74)

    participants = base_val.load_steingroever_data(max_participants=9999, seed=0)
    payoff_model = build_payoff_model(participants)

    print(f"\n  Loaded {len(participants)} participants")
    print(f"  Payoff model: {sum(len(v) for v in payoff_model.values())} outcome samples")
    for a in ACTIONS:
        outs = payoff_model[a]
        print(f"    Deck {a}: mean={mean(outs):+.0f}, n={len(outs)}")

    # Pick representative participants
    learners = [p for p in participants if p.safe_fraction > 0.75]
    mixed = [p for p in participants if 0.40 <= p.safe_fraction <= 0.60]
    risk_seekers = [p for p in participants if p.safe_fraction < 0.25]

    # Sort for reproducibility
    targets = []
    if learners:
        targets.append(("LEARNER", sorted(learners, key=lambda p: -p.safe_fraction)[0]))
    if mixed:
        targets.append(("MIXED", sorted(mixed, key=lambda p: abs(p.safe_fraction - 0.50))[0]))
    if risk_seekers:
        targets.append(("RISK-SEEKER", sorted(risk_seekers, key=lambda p: p.safe_fraction)[0]))

    N_SIMS = 20  # Generate 20 simulated runs per participant

    for profile, participant in targets:
        uid = participant.participant_id
        real_actions = [DECK_MAP[c] for c in participant.choices]
        real_outcomes = [participant.wins[i] + participant.losses[i]
                         for i in range(participant.n_trials)]
        real_sig = behavioral_signature(real_actions, real_outcomes)

        sim_sigs = []
        sim_example_actions = None

        for sim_i in range(N_SIMS):
            # Fresh predictor for each simulation (train from scratch)
            predictor = UnifiedCognitivePredictor(
                actions=ACTIONS, D=6, lr=0.025, mix=0.3, adapt_pt=True,
            )

            # Train on ALL real trials (brain learns this person's patterns)
            train_uid = f"{uid}_train"
            for t in range(participant.n_trials):
                actual = DECK_MAP[participant.choices[t]]
                outcome = participant.wins[t] + participant.losses[t]
                predictor.predict(train_uid)
                predictor.update(train_uid, actual_action=actual, outcome=outcome)

            # KEEP the same user state — the brain has learned THIS person
            # Simulate by continuing to generate from their learned state
            rng = random.Random(sim_i * 1000 + hash(uid) % 10000)
            sim_actions, sim_outcomes = simulate_participant(
                predictor, train_uid,
                n_trials=participant.n_trials,
                payoff_model=payoff_model,
                rng=rng,
            )
            sig = behavioral_signature(sim_actions, sim_outcomes)
            sim_sigs.append(sig)
            if sim_i == 0:
                sim_example_actions = sim_actions

        print_signature_comparison(
            f"{profile}: {uid} (safe={participant.safe_fraction:.0%}, {participant.n_trials} trials)",
            real_sig, sim_sigs,
            real_actions, sim_example_actions,
        )

    # --- Aggregate across ALL participants ---
    print(f"\n\n  {'=' * 70}")
    print(f"  POPULATION-LEVEL: All {len(participants)} participants")
    print(f"  Train on each person's data, then simulate from their learned state")
    print(f"  {'=' * 70}")

    all_real_sigs = []
    all_sim_sigs = []
    rng = random.Random(42)

    for i, p in enumerate(participants):
        uid = p.participant_id
        real_actions = [DECK_MAP[c] for c in p.choices]
        real_outcomes = [p.wins[t] + p.losses[t] for t in range(p.n_trials)]
        all_real_sigs.append(behavioral_signature(real_actions, real_outcomes))

        # Train a fresh predictor on this participant
        pred = UnifiedCognitivePredictor(
            actions=ACTIONS, D=6, lr=0.025, mix=0.3, adapt_pt=True,
        )
        for t in range(p.n_trials):
            actual = DECK_MAP[p.choices[t]]
            outcome = p.wins[t] + p.losses[t]
            pred.predict(uid)
            pred.update(uid, actual_action=actual, outcome=outcome)

        # Simulate from the same learned state
        sim_actions, sim_outcomes = simulate_participant(
            pred, uid,
            n_trials=p.n_trials,
            payoff_model=payoff_model,
            rng=rng,
        )
        all_sim_sigs.append(behavioral_signature(sim_actions, sim_outcomes))

        if (i + 1) % 100 == 0:
            print(f"    [{i+1}/{len(participants)}] simulated", flush=True)

    # Compare population distributions
    def pop_compare(field_name, label):
        real_vals = [s[field_name] for s in all_real_sigs]
        sim_vals = [s[field_name] for s in all_sim_sigs]
        r_mean, s_mean = mean(real_vals), mean(sim_vals)
        diff = abs(r_mean - s_mean)
        match = "good" if diff < 0.05 else ("ok" if diff < 0.10 else "poor")
        print(f"  {label:<25s}  Real: {r_mean:.3f}   Sim: {s_mean:.3f}   diff={diff:.3f}  [{match}]")

    print(f"\n  {'Metric':<25s}  {'Real':>12s}   {'Simulated':>12s}   {'Diff':>10s}")
    print(f"  {'-' * 70}")
    pop_compare("safe_rate", "Safe deck rate")
    pop_compare("switch_rate", "Switch rate")
    pop_compare("win_stay", "Win-stay rate")
    pop_compare("lose_shift", "Lose-shift rate")
    pop_compare("avg_streak", "Avg streak length")
    pop_compare("entropy", "Choice entropy")
    pop_compare("learning_delta", "Learning (2nd-1st half)")

    # Deck preference distributions
    print(f"\n  Deck preferences (population mean):")
    for a in ACTIONS:
        r = mean(s["freqs"].get(a, 0) for s in all_real_sigs)
        s = mean(s["freqs"].get(a, 0) for s in all_sim_sigs)
        bar_r = "#" * int(r * 40)
        bar_s = "#" * int(s * 40)
        print(f"    {a}:  Real {r:.1%} {bar_r:<20s}   Sim {s:.1%} {bar_s}")

    print()
    print("=" * 74)


if __name__ == "__main__":
    main()
