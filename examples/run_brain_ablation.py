#!/usr/bin/env python3
"""Brain Ablation Study: Full Brain vs History-Only vs Pure Prior vs Random.

Measures the contribution of the brain's cognitive layer (state transitions,
branch competition, conflict resolution, control signals, regime detection)
beyond what simple behavioral history features can achieve.

Conditions:
  1. FULL BRAIN   — complete brain-first pipeline (HumanState, router, forest,
                     controller, memory, brain predictor, RL adapter).
  2. HISTORY-ONLY — same BrainPredictor and transition_model for EV/outcome
                     tracking, but bypasses the entire cognitive layer by
                     injecting a neutral/null BrainOutput (all signals at 0.5,
                     no conflicts, neutral regime).  Still uses prior,
                     outcome bias, transitions, context, recency, WSLS,
                     transfer learning.
  3. PURE PRIOR   — predicts the majority class from training base rates.
                     No brain, no learned weights.
  4. RANDOM        — uniform random baseline.

All conditions use seed=0, noise=0, same 50 participants, same 70/30 split.

Run:
    python examples/run_brain_ablation.py
"""

from __future__ import annotations

import os
import sys
import random
from dataclasses import dataclass
from statistics import mean, stdev
from typing import Any

# Path setup
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.prompt_forest.brain.brain_predictor import BrainPredictor
from src.prompt_forest.brain.controller import BrainController
from src.prompt_forest.brain.output import (
    BrainActionTendencies,
    BrainControlSignals,
    BrainOutput,
)
from src.prompt_forest.brain.rl_adapter import BrainRLAdapter
from src.prompt_forest.brain.transition_model import LearnedTransitionModel
from src.prompt_forest.modes.human_mode.branches import create_human_mode_forest
from src.prompt_forest.modes.human_mode.memory import HumanModeMemory
from src.prompt_forest.modes.human_mode.router import HumanModeRouter
from src.prompt_forest.state.human_state import HumanState
from src.prompt_forest.types import TaskInput

# Reuse data loader and helpers from the validation scripts
import run_adaptive_learning_validation as base_val
from run_brain_igt_validation import (
    MAX_PARTICIPANTS,
    SAFE_DECKS,
    WINDOW_SIZE,
    BrainParticipantResult,
    _warm_start_from_similar,
    run_brain_participant,
)

# ---------------------------------------------------------------------------
# Null BrainOutput factory
# ---------------------------------------------------------------------------

def _make_null_brain_output() -> BrainOutput:
    """Create a neutral/null BrainOutput with all signals at 0.5.

    This effectively zeros out the brain's contribution because:
      - All control signals are 0.5 (neutral)
      - All action tendencies are 0.5 (neutral)
      - Regime is "neutral" (no learned regime bias)
      - No conflicts (conflict_load = 0)
      - Empty branch activations and memory biases
      - State variables all at 0.5 (neutral)
    """
    return BrainOutput(
        regime="neutral",
        state={var: 0.5 for var in [
            "confidence", "stress", "frustration", "fear", "motivation",
            "impulse", "caution", "ambition", "curiosity", "fatigue",
            "trust", "self_protection", "reflection", "goal_commitment",
            "honesty", "self_justification",
        ]},
        dominant_drives=[],
        branch_activations={},
        active_branches=[],
        conflicts=[],
        control_signals=BrainControlSignals(
            approach_drive=0.5,
            avoidance_drive=0.5,
            exploration_drive=0.5,
            switch_pressure=0.5,
            persistence_drive=0.5,
            self_protection=0.5,
            social_openness=0.5,
            cognitive_effort=0.5,
        ),
        action_tendencies=BrainActionTendencies(
            act=0.5,
            inhibit=0.5,
            explore=0.5,
            exploit=0.5,
            reflect=0.5,
            react=0.5,
        ),
        memory_biases={},
        state_summary={},
        notes=["ablation: null brain output"],
    )


# ---------------------------------------------------------------------------
# Result container for ablation
# ---------------------------------------------------------------------------

@dataclass
class AblationResult:
    participant_id: str
    study: str
    safe_fraction: float
    full_brain_accuracy: float
    history_only_accuracy: float
    pure_prior_accuracy: float
    random_accuracy: float


# ---------------------------------------------------------------------------
# History-only participant runner
# ---------------------------------------------------------------------------

def run_history_only_participant(
    participant: base_val.RealIGTParticipant,
    transition_model: LearnedTransitionModel,
    brain_predictor: BrainPredictor,
    rl_adapter: BrainRLAdapter,
) -> float:
    """Run one participant through the history-only pipeline.

    Same as run_brain_participant but replaces the entire cognitive layer
    (HumanState, router, forest, controller, memory) with a constant
    neutral BrainOutput.  The BrainPredictor still uses prior, outcome bias,
    transitions, context features, recency, WSLS — but brain-derived signals
    (control weights, tendency weights, state weights, regime bias, conflict
    bias) will receive only neutral 0.5 values and thus learn nothing useful.
    """
    n_train = int(participant.n_trials * 0.7)
    n_holdout = participant.n_trials - n_train
    train_choices = participant.choices[:n_train]
    holdout_choices = participant.choices[n_train:]
    uid = participant.participant_id

    # No brain components — just the null output
    null_output = _make_null_brain_output()

    prev_outcome: float | None = None
    prev_action: str | None = None
    cumulative_score = 0.0
    deck_outcomes: dict[str, list[float]] = {"safe": [], "risky": []}
    consecutive_same = 0
    last_deck_action: str | None = None

    # ------- Training phase -------
    for idx, deck_choice in enumerate(train_choices):
        actual = "safe" if deck_choice in SAFE_DECKS else "risky"
        net_outcome = participant.wins[idx] + participant.losses[idx]
        cumulative_score += net_outcome

        # Still feed outcomes into transition_model for EV/outcome tracking
        if prev_outcome is not None and prev_action is not None:
            transition_model.compute_deltas(
                user_id=uid,
                state_vars={},  # no real state
                outcome=prev_outcome,
                action=prev_action,
            )

        # Track consecutive same action
        if last_deck_action == actual:
            consecutive_same += 1
        else:
            consecutive_same = 0
        last_deck_action = actual

        # Build context features (same as full brain)
        progress = idx / max(1, n_train)
        safe_so_far = sum(
            1 for c in train_choices[:idx + 1] if c in SAFE_DECKS
        )
        context = {
            "trial_progress": progress,
            "safe_rate_so_far": safe_so_far / max(1, idx + 1),
            "cumulative_score_norm": cumulative_score
            / max(1.0, abs(cumulative_score) + 100.0),
            "consecutive_same": min(1.0, consecutive_same / 5.0),
        }
        for act in ["safe", "risky"]:
            recent_do = deck_outcomes[act][-5:]
            if recent_do:
                m = sum(recent_do) / len(recent_do)
                context[f"deck_outcome_mean::{act}"] = m / (abs(m) + 100.0)
                context[f"deck_outcome_neg::{act}"] = sum(
                    1 for o in recent_do if o < 0
                ) / len(recent_do)
        context.update(transition_model.get_ev_features(uid))
        context.update(transition_model.get_outcome_features(uid))
        context.update(transition_model.get_per_action_features(uid))
        context.update(brain_predictor.compute_sequence_features(uid))

        # Predict with null brain output
        prediction = brain_predictor.predict(uid, null_output, context)

        # RL adapt with null brain output — this updates prior, outcome bias,
        # transitions, context weights, recency, WSLS (the behavioral signals)
        # but brain-signal weights (control, tendency, state, regime, conflict)
        # learn nothing useful since values are constant neutral.
        rl_adapter.adapt(
            user_id=uid,
            brain_output=null_output,
            predicted_action=prediction.predicted_action,
            actual_action=actual,
            outcome=net_outcome,
            context=context,
        )

        deck_outcomes[actual].append(net_outcome)

        # Transfer learning (same as full brain)
        if idx == 14 and brain_predictor.user_count > 3:
            _warm_start_from_similar(
                uid, brain_predictor, min_donor_obs=40, top_k=5
            )

        prev_outcome = net_outcome
        prev_action = actual

    # ------- Holdout phase -------
    holdout_correct = 0
    for idx, deck_choice in enumerate(holdout_choices):
        actual = "safe" if deck_choice in SAFE_DECKS else "risky"
        net_outcome = participant.wins[n_train + idx] + participant.losses[n_train + idx]

        # Still track EV
        if prev_outcome is not None and prev_action is not None:
            transition_model.compute_deltas(
                user_id=uid,
                state_vars={},
                outcome=prev_outcome,
                action=prev_action,
            )

        # Context features
        holdout_safe_count = sum(
            1 for c in holdout_choices[:idx] if c in SAFE_DECKS
        )
        total_safe = sum(1 for c in train_choices if c in SAFE_DECKS) + holdout_safe_count
        total_trials = n_train + idx
        context = {
            "trial_progress": 1.0,
            "safe_rate_so_far": total_safe / max(1, total_trials),
            "cumulative_score_norm": cumulative_score
            / max(1.0, abs(cumulative_score) + 100.0),
            "consecutive_same": min(1.0, consecutive_same / 5.0),
        }
        for act in ["safe", "risky"]:
            recent_do = deck_outcomes[act][-5:]
            if recent_do:
                m = sum(recent_do) / len(recent_do)
                context[f"deck_outcome_mean::{act}"] = m / (abs(m) + 100.0)
                context[f"deck_outcome_neg::{act}"] = sum(
                    1 for o in recent_do if o < 0
                ) / len(recent_do)
        context.update(transition_model.get_ev_features(uid))
        context.update(transition_model.get_outcome_features(uid))
        context.update(transition_model.get_per_action_features(uid))
        context.update(brain_predictor.compute_sequence_features(uid))

        # Predict with null brain output
        prediction = brain_predictor.predict(uid, null_output, context)
        if prediction.predicted_action == actual:
            holdout_correct += 1

        # Observe ground truth (same lightweight updates as full brain holdout)
        brain_predictor.observe(
            uid, actual, net_outcome,
            update_prior=True,
            update_outcome_bias=True,
        )
        transition_model.compute_deltas(
            user_id=uid,
            state_vars={},
            outcome=net_outcome,
            action=actual,
        )

        cumulative_score += net_outcome
        deck_outcomes[actual].append(net_outcome)
        if last_deck_action == actual:
            consecutive_same += 1
        else:
            consecutive_same = 0
        last_deck_action = actual

        prev_outcome = net_outcome
        prev_action = actual

    return holdout_correct / max(1, n_holdout)


# ---------------------------------------------------------------------------
# Pure prior baseline
# ---------------------------------------------------------------------------

def run_pure_prior_participant(
    participant: base_val.RealIGTParticipant,
) -> float:
    """Predict using only the majority class from training data."""
    n_train = int(participant.n_trials * 0.7)
    n_holdout = participant.n_trials - n_train
    train_choices = participant.choices[:n_train]
    holdout_choices = participant.choices[n_train:]

    # Compute training base rates
    safe_count = sum(1 for c in train_choices if c in SAFE_DECKS)
    safe_rate = safe_count / max(1, n_train)

    # Predict majority class for every holdout trial
    majority_action = "safe" if safe_rate >= 0.5 else "risky"

    correct = 0
    for deck_choice in holdout_choices:
        actual = "safe" if deck_choice in SAFE_DECKS else "risky"
        if majority_action == actual:
            correct += 1

    return correct / max(1, n_holdout)


# ---------------------------------------------------------------------------
# Random baseline
# ---------------------------------------------------------------------------

def run_random_participant(
    participant: base_val.RealIGTParticipant,
) -> float:
    """Uniform random baseline."""
    n_train = int(participant.n_trials * 0.7)
    holdout_choices = participant.choices[n_train:]
    uid = participant.participant_id

    rng = random.Random(hash(uid) % 100000 + 7777)
    correct = sum(
        1
        for c in holdout_choices
        if rng.choice(["safe", "risky"])
        == ("safe" if c in SAFE_DECKS else "risky")
    )
    return correct / max(1, len(holdout_choices))


# ---------------------------------------------------------------------------
# Run full ablation
# ---------------------------------------------------------------------------

def run_ablation(seed: int = 0) -> list[AblationResult]:
    """Run all four conditions on the same participants."""
    participants = base_val.load_steingroever_data(
        max_participants=MAX_PARTICIPANTS,
        seed=seed,
    )

    safe_fracs = [p.safe_fraction for p in participants]
    n_learn = sum(1 for f in safe_fracs if f > 0.65)
    n_mixed = sum(1 for f in safe_fracs if 0.35 <= f <= 0.65)
    n_risk = sum(1 for f in safe_fracs if f < 0.35)
    print(
        f"  Loaded {len(participants)} participants "
        f"(learners={n_learn}, mixed={n_mixed}, risk-seekers={n_risk})"
    )
    print()

    # ---- Condition 1: FULL BRAIN ----
    print("  === Condition 1: FULL BRAIN ===")
    fb_transition = LearnedTransitionModel(lr=0.02)
    fb_predictor = BrainPredictor(
        actions=["safe", "risky"],
        learning_rate=0.14,
        prior_lr=0.08,
        context_lr=0.06,
        min_lr=0.025,
        anneal_rate=0.005,
        outcome_lr=0.16,
        transition_lr=0.14,
    )
    fb_adapter = BrainRLAdapter(
        transition_model=fb_transition,
        brain_predictor=fb_predictor,
    )

    full_brain_results: list[BrainParticipantResult] = []
    for i, participant in enumerate(participants):
        print(
            f"    [{i+1:>2d}/{len(participants)}] {participant.participant_id:>8s} "
            f"({participant.study:>15s}, safe={participant.safe_fraction:.0%})",
            flush=True,
        )
        result = run_brain_participant(
            participant=participant,
            transition_model=fb_transition,
            brain_predictor=fb_predictor,
            rl_adapter=fb_adapter,
        )
        full_brain_results.append(result)
    print()

    # ---- Condition 2: HISTORY-ONLY ----
    print("  === Condition 2: HISTORY-ONLY ===")
    ho_transition = LearnedTransitionModel(lr=0.02)
    ho_predictor = BrainPredictor(
        actions=["safe", "risky"],
        learning_rate=0.14,
        prior_lr=0.08,
        context_lr=0.06,
        min_lr=0.025,
        anneal_rate=0.005,
        outcome_lr=0.16,
        transition_lr=0.14,
    )
    ho_adapter = BrainRLAdapter(
        transition_model=ho_transition,
        brain_predictor=ho_predictor,
    )

    history_only_accs: dict[str, float] = {}
    for i, participant in enumerate(participants):
        print(
            f"    [{i+1:>2d}/{len(participants)}] {participant.participant_id:>8s} "
            f"({participant.study:>15s}, safe={participant.safe_fraction:.0%})",
            flush=True,
        )
        acc = run_history_only_participant(
            participant=participant,
            transition_model=ho_transition,
            brain_predictor=ho_predictor,
            rl_adapter=ho_adapter,
        )
        history_only_accs[participant.participant_id] = acc
    print()

    # ---- Condition 3: PURE PRIOR ----
    print("  === Condition 3: PURE PRIOR ===")
    pure_prior_accs: dict[str, float] = {}
    for participant in participants:
        acc = run_pure_prior_participant(participant)
        pure_prior_accs[participant.participant_id] = acc
    print("    Done.")
    print()

    # ---- Condition 4: RANDOM ----
    print("  === Condition 4: RANDOM ===")
    random_accs: dict[str, float] = {}
    for participant in participants:
        acc = run_random_participant(participant)
        random_accs[participant.participant_id] = acc
    print("    Done.")
    print()

    # ---- Combine results ----
    results: list[AblationResult] = []
    for fb_result, participant in zip(full_brain_results, participants):
        uid = participant.participant_id
        results.append(AblationResult(
            participant_id=uid,
            study=participant.study,
            safe_fraction=participant.safe_fraction,
            full_brain_accuracy=fb_result.holdout_accuracy,
            history_only_accuracy=history_only_accs[uid],
            pure_prior_accuracy=pure_prior_accs[uid],
            random_accuracy=random_accs[uid],
        ))

    return results


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _group_stats(
    results: list[AblationResult],
    attr: str,
) -> tuple[float, float]:
    """Return (mean, stdev) for a given accuracy attribute."""
    vals = [getattr(r, attr) for r in results]
    m = mean(vals) if vals else 0.0
    s = stdev(vals) if len(vals) > 1 else 0.0
    return m, s


def _print_ablation_report(results: list[AblationResult]) -> None:
    n = len(results)

    # Group participants
    learners = [r for r in results if r.safe_fraction > 0.65]
    mixed = [r for r in results if 0.35 <= r.safe_fraction <= 0.65]
    risk_seekers = [r for r in results if r.safe_fraction < 0.35]

    conditions = [
        ("Full Brain", "full_brain_accuracy"),
        ("History-Only", "history_only_accuracy"),
        ("Pure Prior", "pure_prior_accuracy"),
        ("Random", "random_accuracy"),
    ]

    groups = [
        ("ALL", results),
        ("Learners (>65% safe)", learners),
        ("Mixed (35-65% safe)", mixed),
        ("Risk-seekers (<35%)", risk_seekers),
    ]

    print()
    print("=" * 90)
    print("  BRAIN ABLATION STUDY")
    print("  Iowa Gambling Task — Steingroever et al. (2015)")
    print("=" * 90)
    print()
    print("  Ablation design:")
    print("    Full Brain    = HumanState + Router + Forest + Controller + Memory + Predictor")
    print("    History-Only  = Null BrainOutput + same Predictor (prior, outcome bias,")
    print("                    transitions, context, recency, WSLS, transfer learning)")
    print("    Pure Prior    = majority-class from training base rate only")
    print("    Random        = uniform random (50% expected)")
    print()
    print(f"  Participants: {n}  |  Seed: 0  |  Train/Holdout: 70/30 trials")
    print(f"  Groups: learners={len(learners)}, mixed={len(mixed)}, "
          f"risk-seekers={len(risk_seekers)}")
    print()

    # ---- Main comparison table ----
    print("  " + "-" * 86)
    header = f"  {'Group':>25s}"
    for label, _ in conditions:
        header += f"  {label:>14s}"
    header += f"  {'Brain Contrib':>14s}"
    print(header)
    print("  " + "-" * 86)

    for group_label, group in groups:
        if not group:
            continue
        row = f"  {group_label:>25s}"
        accs = {}
        for cond_label, attr in conditions:
            m, s = _group_stats(group, attr)
            accs[cond_label] = m
            row += f"  {m:>11.1%} +/-{s:.1%}" if len(group) > 1 else f"  {m:>14.1%}"
        # Brain contribution = full brain - history only
        brain_contrib = accs["Full Brain"] - accs["History-Only"]
        row += f"  {brain_contrib:>+13.1%}"
        print(row)

    print("  " + "-" * 86)
    print()

    # ---- Lift table (simpler) ----
    print("  --- Lift over baselines ---")
    print()
    print(f"  {'Group':>25s}  {'vs History':>12s}  {'vs Prior':>12s}  {'vs Random':>12s}")
    print("  " + "-" * 66)
    for group_label, group in groups:
        if not group:
            continue
        fb = mean(r.full_brain_accuracy for r in group)
        ho = mean(r.history_only_accuracy for r in group)
        pp = mean(r.pure_prior_accuracy for r in group)
        rd = mean(r.random_accuracy for r in group)
        print(
            f"  {group_label:>25s}"
            f"  {fb - ho:>+11.1%}"
            f"  {fb - pp:>+11.1%}"
            f"  {fb - rd:>+11.1%}"
        )
    print("  " + "-" * 66)
    print()

    # ---- Per-participant detail (top 10 where brain helps most) ----
    results_sorted = sorted(
        results,
        key=lambda r: r.full_brain_accuracy - r.history_only_accuracy,
        reverse=True,
    )
    print("  --- Participants where brain helps most ---")
    print(f"  {'ID':>8s}  {'safe%':>5s}  {'Brain':>6s}  {'Hist':>6s}  {'Prior':>6s}  "
          f"{'Rand':>6s}  {'Contrib':>8s}")
    print("  " + "-" * 58)
    for r in results_sorted[:10]:
        contrib = r.full_brain_accuracy - r.history_only_accuracy
        print(
            f"  {r.participant_id:>8s}"
            f"  {r.safe_fraction:>5.0%}"
            f"  {r.full_brain_accuracy:>5.0%}"
            f"  {r.history_only_accuracy:>5.0%}"
            f"  {r.pure_prior_accuracy:>5.0%}"
            f"  {r.random_accuracy:>5.0%}"
            f"  {contrib:>+7.0%}"
        )
    print()

    # ---- Participants where brain hurts ----
    hurts = [r for r in results if r.full_brain_accuracy < r.history_only_accuracy]
    if hurts:
        print(f"  --- Participants where brain hurts ({len(hurts)}/{n}) ---")
        print(f"  {'ID':>8s}  {'safe%':>5s}  {'Brain':>6s}  {'Hist':>6s}  {'Contrib':>8s}")
        print("  " + "-" * 42)
        for r in sorted(hurts, key=lambda r: r.full_brain_accuracy - r.history_only_accuracy):
            contrib = r.full_brain_accuracy - r.history_only_accuracy
            print(
                f"  {r.participant_id:>8s}"
                f"  {r.safe_fraction:>5.0%}"
                f"  {r.full_brain_accuracy:>5.0%}"
                f"  {r.history_only_accuracy:>5.0%}"
                f"  {contrib:>+7.0%}"
            )
        print()

    # ---- Verdicts ----
    fb_mean = mean(r.full_brain_accuracy for r in results)
    ho_mean = mean(r.history_only_accuracy for r in results)
    pp_mean = mean(r.pure_prior_accuracy for r in results)
    rd_mean = mean(r.random_accuracy for r in results)
    brain_contrib = fb_mean - ho_mean

    print("  --- Verdicts ---")
    verdicts = [
        ("Full brain > history-only", fb_mean > ho_mean),
        ("Brain contribution > 1%", brain_contrib > 0.01),
        ("Brain contribution > 3%", brain_contrib > 0.03),
        ("Full brain > pure prior", fb_mean > pp_mean),
        ("Full brain > random", fb_mean > rd_mean),
        ("History-only > pure prior", ho_mean > pp_mean),
        ("History-only > random", ho_mean > rd_mean),
    ]
    for label, passed in verdicts:
        marker = "  [+]" if passed else "  [-]"
        print(f"  {marker} {label}: {'PASS' if passed else 'FAIL'}")
    print()

    # ---- Summary ----
    print("  --- Summary ---")
    print(f"  Full Brain mean accuracy:    {fb_mean:.1%}")
    print(f"  History-Only mean accuracy:  {ho_mean:.1%}")
    print(f"  Pure Prior mean accuracy:    {pp_mean:.1%}")
    print(f"  Random mean accuracy:        {rd_mean:.1%}")
    print()
    print(f"  Brain cognitive layer adds:  {brain_contrib:+.1%} over history-only")
    print(f"  Behavioral learning adds:    {ho_mean - pp_mean:+.1%} over pure prior")
    print(f"  Total system lift:           {fb_mean - rd_mean:+.1%} over random")
    print()

    # Breakdown of contributions
    total_lift = fb_mean - rd_mean
    if total_lift > 0:
        prior_share = (pp_mean - rd_mean) / total_lift * 100
        behavioral_share = (ho_mean - pp_mean) / total_lift * 100
        brain_share = (fb_mean - ho_mean) / total_lift * 100
        print("  Attribution of total lift over random:")
        print(f"    Prior (base rate):           {prior_share:>5.1f}%")
        print(f"    Behavioral learning:         {behavioral_share:>5.1f}%")
        print(f"    Brain cognitive layer:       {brain_share:>5.1f}%")
    print()
    print("=" * 90)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 90)
    print("  BRAIN ABLATION STUDY")
    print("  Loading Steingroever et al. (2015) IGT dataset...")
    print("=" * 90)
    print()

    results = run_ablation(seed=0)
    _print_ablation_report(results)


if __name__ == "__main__":
    main()
