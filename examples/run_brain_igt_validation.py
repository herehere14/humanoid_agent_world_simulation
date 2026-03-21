#!/usr/bin/env python3
"""Brain-First IGT Validation.

Validates the Prompt Forest Brain architecture against the Steingroever
et al. (2015) Iowa Gambling Task dataset.

Architecture (the brain IS the mechanism):
  outcome → learned state transition → brain state update
  → branch competition (router) → conflict resolution → control signals
  → brain predictor → behavioral prediction → RL adaptation

No simulated LLM backend.  No engine routing noise.  The brain's latent
state, control signals, and conflict dynamics are the PRIMARY prediction
mechanism.

Dataset: 504 real humans × 100 trials, sample 50 participants,
         train on first 70 trials, predict last 30.
"""

from __future__ import annotations

import os
import sys
import random
import tempfile
import shutil
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any

# Path setup
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.prompt_forest.brain.brain_predictor import BrainPredictor
from src.prompt_forest.brain.controller import BrainController
from src.prompt_forest.brain.rl_adapter import BrainRLAdapter
from src.prompt_forest.brain.transition_model import LearnedTransitionModel
from src.prompt_forest.modes.human_mode.branches import create_human_mode_forest
from src.prompt_forest.modes.human_mode.memory import HumanModeMemory
from src.prompt_forest.modes.human_mode.router import HumanModeRouter
from src.prompt_forest.state.human_state import HumanState
from src.prompt_forest.types import TaskInput

# Reuse data loader from the base validation
import run_adaptive_learning_validation as base_val

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_PARTICIPANTS = 50
SAFE_DECKS = {3, 4}
WINDOW_SIZE = 10


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class BrainParticipantResult:
    participant_id: str
    study: str
    safe_fraction: float
    holdout_accuracy: float
    random_accuracy: float
    window_accuracies: list[float]
    final_transition_params: dict[str, float]
    final_priors: dict[str, float]


# ---------------------------------------------------------------------------
# Brain-first experiment runner
# ---------------------------------------------------------------------------

def _warm_start_from_similar(
    target_uid: str,
    predictor: BrainPredictor,
    min_donor_obs: int = 40,
    top_k: int = 5,
    transfer_weight: float = 0.35,
    min_similarity: float = 0.5,
) -> None:
    """Transfer outcome biases and transitions from similar completed users."""
    import math

    target = predictor._users.get(target_uid)
    if target is None or target.n_observations < 5:
        return

    # Compute target signature from priors + outcome biases
    def _sig(model: Any) -> dict[str, float]:
        s: dict[str, float] = {}
        for a, v in model.prior.items():
            s[f"p_{a}"] = v
        for a in predictor.actions:
            for vbin in ["positive", "negative"]:
                s[f"ob_{a}_{vbin}"] = model.outcome_bias.get(a, {}).get(vbin, 0.0)
        for src in predictor.actions:
            for vbin in ["positive", "negative"]:
                for dst in predictor.actions:
                    s[f"tr_{src}_{vbin}_{dst}"] = (
                        model.transitions.get(src, {}).get(vbin, {}).get(dst, 0.0)
                    )
        return s

    def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
        keys = set(a) | set(b)
        dot = sum(a.get(k, 0) * b.get(k, 0) for k in keys)
        ma = math.sqrt(sum(a.get(k, 0) ** 2 for k in keys))
        mb = math.sqrt(sum(b.get(k, 0) ** 2 for k in keys))
        if ma < 1e-8 or mb < 1e-8:
            return 0.0
        return dot / (ma * mb)

    target_sig = _sig(target)
    donors: list[tuple[str, float]] = []
    for uid, model in predictor._users.items():
        if uid == target_uid or model.n_observations < min_donor_obs:
            continue
        sim = _cosine(target_sig, _sig(model))
        if sim >= min_similarity:
            donors.append((uid, sim))

    donors.sort(key=lambda x: x[1], reverse=True)
    donors = donors[:top_k]
    if not donors:
        return

    total_sim = sum(s for _, s in donors)
    tw = transfer_weight

    # Transfer outcome biases
    for action in predictor.actions:
        for vbin in ["positive", "negative", "neutral"]:
            donor_avg = 0.0
            for uid, sim in donors:
                d = predictor._users[uid]
                donor_avg += (sim / total_sim) * d.outcome_bias.get(action, {}).get(vbin, 0.0)
            cur = target.outcome_bias[action].get(vbin, 0.0)
            target.outcome_bias[action][vbin] = (1 - tw) * cur + tw * donor_avg

    # Transfer transition weights
    for src in predictor.actions:
        for vbin in ["positive", "negative", "neutral"]:
            for dst in predictor.actions:
                donor_avg = 0.0
                for uid, sim in donors:
                    d = predictor._users[uid]
                    donor_avg += (sim / total_sim) * d.transitions.get(src, {}).get(vbin, {}).get(dst, 0.0)
                cur = target.transitions[src][vbin].get(dst, 0.0)
                target.transitions[src][vbin][dst] = (1 - tw) * cur + tw * donor_avg

    # Transfer context weights
    for action in predictor.actions:
        donor_ctx: dict[str, float] = {}
        for uid, sim in donors:
            d = predictor._users[uid]
            for feat, w in d.context_weights.get(action, {}).items():
                donor_ctx[feat] = donor_ctx.get(feat, 0.0) + (sim / total_sim) * w
        for feat, avg_w in donor_ctx.items():
            cur = target.context_weights[action].get(feat, 0.0)
            target.context_weights[action][feat] = (1 - tw) * cur + tw * avg_w


def run_brain_participant(
    participant: base_val.RealIGTParticipant,
    transition_model: LearnedTransitionModel,
    brain_predictor: BrainPredictor,
    rl_adapter: BrainRLAdapter,
) -> BrainParticipantResult:
    """Run one participant through the brain-first pipeline.

    Causal chain per trial:
      1. Update brain state from previous outcome (learned transitions).
      2. Route through cognitive branch forest (state-conditioned).
      3. Resolve drive conflicts.
      4. Build brain output (control signals, tendencies, regime).
      5. Predict from brain output.
      6. RL adapt (transition sensitivities + predictor weights).
    """
    n_train = int(participant.n_trials * 0.7)
    n_holdout = participant.n_trials - n_train
    train_choices = participant.choices[:n_train]
    holdout_choices = participant.choices[n_train:]
    uid = participant.participant_id

    # Create per-participant brain components
    state = HumanState(
        decay_rate=0.05,
        momentum=0.45,
        noise_level=0.0,
        adaptive_baselines=True,
        baseline_lr=0.012,
        baseline_warmup=40,
    )
    forest = create_human_mode_forest()
    router = HumanModeRouter(top_k=4, noise_level=0.0)
    memory = HumanModeMemory(
        emotional_decay=0.92,
        trauma_amplification=1.5,
        experience_bias_strength=0.4,
    )
    controller = BrainController()

    correct_per_window: dict[int, list[bool]] = {}
    prev_outcome: float | None = None
    prev_action: str | None = None
    cumulative_score = 0.0
    # Per-deck outcome tracking for richer features
    deck_outcomes: dict[str, list[float]] = {"safe": [], "risky": []}
    consecutive_same = 0
    last_deck_action: str | None = None

    # ------- Training phase -------
    for idx, deck_choice in enumerate(train_choices):
        actual = "safe" if deck_choice in SAFE_DECKS else "risky"
        net_outcome = participant.wins[idx] + participant.losses[idx]
        cumulative_score += net_outcome

        # 1. Update brain state from previous outcome
        if prev_outcome is not None and prev_action is not None:
            deltas = transition_model.compute_deltas(
                user_id=uid,
                state_vars=state.variables,
                outcome=prev_outcome,
                action=prev_action,
            )
            state.update(deltas)

        # 2. Route through cognitive forest
        progress = idx / max(1, n_train)
        task = TaskInput(
            task_id=f"{uid}_t{idx}",
            text=f"IGT trial {idx+1}",
            task_type="decision",
            metadata={},
        )
        route, conflicts = router.route(task, forest, state)

        # 3. Resolve conflicts
        for conflict in conflicts:
            state.resolve_conflict(conflict, "weighted_compromise")

        # 4. Build brain output
        brain_output = controller.build_output(
            state=state,
            route=route,
            conflicts=state.active_conflicts,
            human_memory=memory,
            branch_weights={
                name: b.state.weight
                for name, b in forest.branches.items()
                if not b.state.metadata.get("category_node")
            },
        )

        # Track consecutive same action
        if last_deck_action == actual:
            consecutive_same += 1
        else:
            consecutive_same = 0
        last_deck_action = actual

        # 5. Build context features
        safe_so_far = sum(
            1 for c in train_choices[: idx + 1] if c in SAFE_DECKS
        )
        context = {
            "trial_progress": progress,
            "safe_rate_so_far": safe_so_far / max(1, idx + 1),
            "cumulative_score_norm": cumulative_score
            / max(1.0, abs(cumulative_score) + 100.0),
            "consecutive_same": min(1.0, consecutive_same / 5.0),
        }
        # Per-deck outcome means
        for act in ["safe", "risky"]:
            recent_do = deck_outcomes[act][-5:]
            if recent_do:
                m = sum(recent_do) / len(recent_do)
                context[f"deck_outcome_mean::{act}"] = m / (abs(m) + 100.0)
                context[f"deck_outcome_neg::{act}"] = sum(
                    1 for o in recent_do if o < 0
                ) / len(recent_do)
        # Add brain-derived features
        context.update(transition_model.get_ev_features(uid))
        context.update(transition_model.get_outcome_features(uid))
        context.update(transition_model.get_per_action_features(uid))
        context.update(brain_predictor.compute_sequence_features(uid))

        # 6. Predict from brain
        prediction = brain_predictor.predict(uid, brain_output, context)
        is_correct = prediction.predicted_action == actual
        correct_per_window.setdefault(idx // WINDOW_SIZE, []).append(
            is_correct
        )

        # 7. RL adapt
        rl_adapter.adapt(
            user_id=uid,
            brain_output=brain_output,
            predicted_action=prediction.predicted_action,
            actual_action=actual,
            outcome=net_outcome,
            context=context,
        )

        # Track per-deck outcomes
        deck_outcomes[actual].append(net_outcome)

        # Transfer learning: warm-start from similar users after 15 trials
        if idx == 14 and brain_predictor.user_count > 3:
            _warm_start_from_similar(
                uid, brain_predictor, min_donor_obs=40, top_k=5
            )

        # 8. Record experiential memory
        memory.record(
            event_id=task.task_id,
            task=task,
            state=state,
            reward=brain_predictor.prediction_accuracy_reward(
                prediction.predicted_action, actual, prediction.confidence
            ),
            selected_branch=(
                route.activated_branches[0]
                if route.activated_branches
                else ""
            ),
            active_branches=list(route.activated_branches),
        )

        prev_outcome = net_outcome
        prev_action = actual

    # ------- Holdout phase -------
    holdout_correct = 0
    for idx, deck_choice in enumerate(holdout_choices):
        actual = "safe" if deck_choice in SAFE_DECKS else "risky"
        net_outcome = participant.wins[n_train + idx] + participant.losses[n_train + idx]

        # Update state from previous outcome
        if prev_outcome is not None and prev_action is not None:
            deltas = transition_model.compute_deltas(
                user_id=uid,
                state_vars=state.variables,
                outcome=prev_outcome,
                action=prev_action,
            )
            state.update(deltas)

        # Route
        task = TaskInput(
            task_id=f"{uid}_h{idx}",
            text=f"IGT holdout trial {n_train+idx+1}",
            task_type="decision",
            metadata={},
        )
        route, conflicts = router.route(task, forest, state)
        for conflict in conflicts:
            state.resolve_conflict(conflict, "weighted_compromise")

        # Brain output
        brain_output = controller.build_output(
            state=state,
            route=route,
            conflicts=state.active_conflicts,
            human_memory=memory,
            branch_weights={
                name: b.state.weight
                for name, b in forest.branches.items()
                if not b.state.metadata.get("category_node")
            },
        )

        # Track consecutive same action (use prev_action from last observed)
        if last_deck_action == prev_action:
            # Don't update — we predict before seeing ground truth
            pass

        # Context — include ALL features that training uses
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
        # Per-deck outcome means (same as training)
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

        # Predict BEFORE observing ground truth
        prediction = brain_predictor.predict(uid, brain_output, context)
        if prediction.predicted_action == actual:
            holdout_correct += 1

        # Observe ground truth — lightweight updates only
        # (prior + outcome bias with dampened LR, no control/state/regime weights)
        brain_predictor.observe(
            uid, actual, net_outcome,
            update_prior=True,
            update_outcome_bias=True,
            regime=brain_output.regime,
            brain_state_summary=brain_output.state_summary,
        )
        transition_model.compute_deltas(
            user_id=uid,
            state_vars=state.variables,
            outcome=net_outcome,
            action=actual,
        )

        # Update holdout tracking state
        cumulative_score += net_outcome
        deck_outcomes[actual].append(net_outcome)
        if last_deck_action == actual:
            consecutive_same += 1
        else:
            consecutive_same = 0
        last_deck_action = actual

        prev_outcome = net_outcome
        prev_action = actual

    # Random baseline
    rng = random.Random(hash(uid) % 100000 + 7777)
    random_correct = sum(
        1
        for c in holdout_choices
        if rng.choice(["safe", "risky"])
        == ("safe" if c in SAFE_DECKS else "risky")
    )

    window_accs = [
        sum(items) / len(items) if items else 0.0
        for _, items in sorted(correct_per_window.items())
    ]

    return BrainParticipantResult(
        participant_id=uid,
        study=participant.study,
        safe_fraction=participant.safe_fraction,
        holdout_accuracy=holdout_correct / max(1, n_holdout),
        random_accuracy=random_correct / max(1, n_holdout),
        window_accuracies=window_accs,
        final_transition_params=transition_model.get_params_dict(uid),
        final_priors=brain_predictor.get_user_summary(uid).get("priors", {}),
    )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _print_report(results: list[BrainParticipantResult]) -> None:
    n = len(results)
    mean_acc = mean(r.holdout_accuracy for r in results)
    mean_rand = mean(r.random_accuracy for r in results)
    std_acc = stdev(r.holdout_accuracy for r in results) if n > 1 else 0.0

    learners = [r for r in results if r.safe_fraction > 0.65]
    mixed = [r for r in results if 0.35 <= r.safe_fraction <= 0.65]
    risk_seekers = [r for r in results if r.safe_fraction < 0.35]

    # Majority-class baseline
    majority_accs = [max(r.safe_fraction, 1.0 - r.safe_fraction) for r in results]
    mean_majority = mean(majority_accs)
    above_majority = sum(
        1 for r, m in zip(results, majority_accs) if r.holdout_accuracy >= m
    )

    print()
    print("=" * 78)
    print("  BRAIN-FIRST IGT VALIDATION")
    print("  Steingroever et al. (2015) — Prompt Forest Brain Architecture")
    print("=" * 78)
    print()
    print(f"  Participants:     {n}")
    print(f"  Train/Holdout:    70/30 trials per participant")
    print()
    print("  Architecture:")
    print("    outcome → learned state transition → brain state")
    print("    → branch competition → conflict resolution → control signals")
    print("    → brain predictor → prediction → RL adaptation")
    print("    (No LLM backend — the brain IS the mechanism)")
    print()

    print("  --- Results ---")
    print()
    print(f"  Holdout accuracy:     {mean_acc:.1%} (+/- {std_acc:.1%})")
    print(f"  Random baseline:      {mean_rand:.1%}")
    print(f"  Majority-class:       {mean_majority:.1%}")
    print(f"  Lift over random:     {mean_acc - mean_rand:+.1%}")
    print(f"  Lift over majority:   {mean_acc - mean_majority:+.1%}")
    print(f"  Above majority:       {above_majority}/{n}")
    print()

    for label, group in [
        ("Learners (>65% safe)", learners),
        ("Mixed (35-65% safe)", mixed),
        ("Risk-seekers (<35%)", risk_seekers),
    ]:
        if not group:
            continue
        g_acc = mean(r.holdout_accuracy for r in group)
        g_rand = mean(r.random_accuracy for r in group)
        print(
            f"    {label:>25s} (n={len(group):>2d}): "
            f"brain={g_acc:.1%}  random={g_rand:.1%}  "
            f"lift={g_acc - g_rand:+.1%}"
        )
    print()

    # Learning curve
    max_windows = max(len(r.window_accuracies) for r in results)
    print("  Learning curve (per-window training accuracy):")
    for w in range(min(max_windows, 7)):
        accs_at_w = [
            r.window_accuracies[w]
            for r in results
            if w < len(r.window_accuracies)
        ]
        if accs_at_w:
            print(f"    Window {w+1}: {mean(accs_at_w):.1%}")
    print()

    # Sample participant details
    print("  --- Sample Participants ---")
    for r in results[:8]:
        bars = "".join(
            "#" if a >= 0.6 else "." for a in r.window_accuracies
        )
        lift = r.holdout_accuracy - r.random_accuracy
        print(
            f"    {r.participant_id:>8s} "
            f"(safe={r.safe_fraction:.0%}): "
            f"brain={r.holdout_accuracy:.0%}  "
            f"lift={lift:+.0%}  "
            f"curve=[{bars}]  "
            f"priors={r.final_priors}"
        )
    print()

    # Verdicts
    print("  --- Verdicts ---")
    verdicts = [
        ("Brain outperforms random (50%)", mean_acc > 0.50),
        ("Brain outperforms random baseline", mean_acc > mean_rand + 0.01),
        ("Brain accuracy >= 70%", mean_acc >= 0.70),
        ("Learning curve improves", _learning_improves(results)),
        ("Approaches majority-class", mean_acc >= mean_majority * 0.90),
    ]
    all_pass = True
    for label, passed in verdicts:
        marker = "  [+]" if passed else "  [-]"
        print(f"  {marker} {label}: {'PASS' if passed else 'FAIL'}")
        if not passed:
            all_pass = False
    print()
    if all_pass:
        print("  OVERALL: ALL CHECKS PASSED")
    else:
        print("  OVERALL: SOME CHECKS NEED IMPROVEMENT")
    print("=" * 78)


def _learning_improves(results: list[BrainParticipantResult]) -> bool:
    slopes = []
    for r in results:
        accs = r.window_accuracies
        if len(accs) >= 2:
            slopes.append(accs[-1] - accs[0])
    return mean(slopes) > 0 if slopes else False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _run_single_seed(seed: int = 42) -> list[BrainParticipantResult]:
    """Run the full brain-first validation for a single sample seed."""
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

    # Shared brain components (accumulate across participants)
    transition_model = LearnedTransitionModel(lr=0.02)
    brain_predictor = BrainPredictor(
        actions=["safe", "risky"],
        learning_rate=0.14,
        prior_lr=0.08,
        context_lr=0.06,
        min_lr=0.025,
        anneal_rate=0.005,
        outcome_lr=0.16,
        transition_lr=0.14,
    )
    rl_adapter = BrainRLAdapter(
        transition_model=transition_model,
        brain_predictor=brain_predictor,
    )

    results: list[BrainParticipantResult] = []
    for i, participant in enumerate(participants):
        print(
            f"  [{i+1:>2d}/{len(participants)}] {participant.participant_id:>8s} "
            f"({participant.study:>15s}, safe={participant.safe_fraction:.0%})",
            flush=True,
        )
        result = run_brain_participant(
            participant=participant,
            transition_model=transition_model,
            brain_predictor=brain_predictor,
            rl_adapter=rl_adapter,
        )
        results.append(result)

    return results


def main() -> None:
    import sys

    if "--multi" in sys.argv:
        # Multi-seed robustness analysis
        n_seeds = 10
        print("=" * 78)
        print("  MULTI-SEED BRAIN-FIRST IGT VALIDATION")
        print(f"  Running {n_seeds} independent samples of {MAX_PARTICIPANTS} participants")
        print("=" * 78)
        print()

        all_accs: list[float] = []
        per_seed_accs: list[float] = []
        for seed in range(n_seeds):
            print(f"  --- Seed {seed} ---")
            print(f"  Loading Steingroever et al. (2015) IGT dataset...")
            results = _run_single_seed(seed)
            seed_acc = mean(r.holdout_accuracy for r in results)
            per_seed_accs.append(seed_acc)
            all_accs.extend(r.holdout_accuracy for r in results)
            print(f"  Seed {seed} accuracy: {seed_acc:.1%}")
            print()

        overall = mean(all_accs)
        overall_std = stdev(all_accs) if len(all_accs) > 1 else 0.0
        print("=" * 78)
        print(f"  OVERALL MEAN ACCURACY ({n_seeds} seeds × {MAX_PARTICIPANTS} participants): "
              f"{overall:.1%} (+/- {overall_std:.1%})")
        print(f"  Per-seed: {', '.join(f'{a:.1%}' for a in per_seed_accs)}")
        above_70 = sum(1 for a in per_seed_accs if a >= 0.70)
        print(f"  Seeds >= 70%: {above_70}/{n_seeds}")
        print("=" * 78)
        return

    # Default: single-seed detailed report
    print("Loading Steingroever et al. (2015) IGT dataset...")
    results = _run_single_seed(seed=0)
    _print_report(results)


if __name__ == "__main__":
    main()
