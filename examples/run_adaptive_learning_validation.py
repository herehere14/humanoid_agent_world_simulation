#!/usr/bin/env python3
"""Adaptive Learning Validation — Real Steingroever IGT Dataset (617 participants).

Validates the Prompt Forest RL system against REAL human behavioral data
from the Iowa Gambling Task (Steingroever et al., 2015).

Dataset: 504 healthy participants × 100 trials from 7 published studies.
  - Deck choices: 1=A(risky), 2=B(risky), 3=C(safe), 4=D(safe)
  - Real human decision patterns (not synthetic)
  - CC BY-SA 4.0 license

Architecture:
  - BehavioralPredictor: learns per-user branch→action mappings from observed
    behavior (no hardcoded mapping).
  - PromptForestEngine: branch weight adaptation + memory bias + exploration
    decay + user-specific routing.  Reward signal = prediction accuracy.
  - Closed loop: prediction accuracy → reward → weight update → routing.

Reference:
  Steingroever, H., Wetzels, R., Horstmann, A., Neumann, J., & Wagenmakers, E.-J.
  (2013). Performance of healthy participants on the Iowa Gambling Task.
  Psychological Assessment, 25(1), 180-193.

Run:
    python examples/run_adaptive_learning_validation.py
"""

from __future__ import annotations

import csv
import math
import os
import random
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, stdev
from typing import Any

# ---------------------------------------------------------------------------
# Path bootstrapping
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.prompt_forest.behavioral.predictor import BehavioralPredictor
from src.prompt_forest.config import load_config, EngineConfig
from src.prompt_forest.core.engine import PromptForestEngine
from src.prompt_forest.backend.simulated import DomainShiftBackend, shifted_quality_matrix

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "default.json"
DATA_DIR = ROOT / "data" / "igt_steingroever"

WINDOW_SIZE = 10  # trials per accuracy window
MAX_PARTICIPANTS = 50  # use a subset for reasonable runtime

# Deck mapping: 1=A(risky), 2=B(risky), 3=C(safe), 4=D(safe)
DECK_NAMES = {1: "A", 2: "B", 3: "C", 4: "D"}
SAFE_DECKS = {3, 4}
RISKY_DECKS = {1, 2}
DECK_TO_GROUP = {"A": "risky", "B": "risky", "C": "safe", "D": "safe"}


# ---------------------------------------------------------------------------
# Real IGT data loader
# ---------------------------------------------------------------------------

@dataclass
class RealIGTParticipant:
    participant_id: str
    study: str
    choices: list[int]        # 1-4 per trial
    wins: list[float]         # win amount per trial
    losses: list[float]       # loss amount per trial (negative)
    safe_fraction: float      # overall % safe choices
    n_trials: int

    def choice_label(self, trial_idx: int) -> str:
        """Return 'safe' or 'risky' for a given trial."""
        return "safe" if self.choices[trial_idx] in SAFE_DECKS else "risky"

    def net_outcome(self, trial_idx: int) -> float:
        """Return net payoff (win + loss) for a trial. Loss is negative."""
        return self.wins[trial_idx] + self.losses[trial_idx]

    def learning_curve(self, window: int = 10) -> list[float]:
        """Per-window safe-deck selection rate."""
        rates = []
        for i in range(0, len(self.choices), window):
            chunk = self.choices[i:i + window]
            if chunk:
                rates.append(sum(1 for c in chunk if c in SAFE_DECKS) / len(chunk))
        return rates


def load_steingroever_data(
    max_participants: int = MAX_PARTICIPANTS,
    min_trials: int = 95,
    seed: int = 42,
) -> list[RealIGTParticipant]:
    """Load real IGT data from Steingroever et al. (2015).

    Returns a diverse sample of participants spanning learners, non-learners,
    and risk-seekers to test whether the system adapts to each pattern.
    """
    choice_path = DATA_DIR / "choice_100.csv"
    win_path = DATA_DIR / "wi_100.csv"
    loss_path = DATA_DIR / "lo_100.csv"
    index_path = DATA_DIR / "index_100.csv"

    if not choice_path.exists():
        raise FileNotFoundError(
            f"IGT data not found at {choice_path}. "
            f"Download from https://osf.io/8t7rm/"
        )

    # Load study labels
    study_map: dict[int, str] = {}
    if index_path.exists():
        with open(index_path, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                study_map[int(row[0])] = row[1]

    # Load win/loss data
    win_data: dict[str, list[float]] = {}
    loss_data: dict[str, list[float]] = {}

    if win_path.exists():
        with open(win_path, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                label = row[0]
                win_data[label] = [float(x) for x in row[1:]]

    if loss_path.exists():
        with open(loss_path, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                label = row[0]
                loss_data[label] = [float(x) for x in row[1:]]

    # Load choices
    all_participants: list[RealIGTParticipant] = []
    with open(choice_path, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            label = row[0]
            subj_num = int(label.split("_")[1])
            choices = [int(x) for x in row[1:]]
            if len(choices) < min_trials:
                continue

            n = len(choices)
            wins = win_data.get(label, [0.0] * n)[:n]
            losses = loss_data.get(label, [0.0] * n)[:n]
            # Pad if needed
            wins.extend([0.0] * (n - len(wins)))
            losses.extend([0.0] * (n - len(losses)))

            safe_frac = sum(1 for c in choices if c in SAFE_DECKS) / len(choices)
            study = study_map.get(subj_num, "unknown")

            all_participants.append(RealIGTParticipant(
                participant_id=label,
                study=study,
                choices=choices,
                wins=wins,
                losses=losses,
                safe_fraction=safe_frac,
                n_trials=len(choices),
            ))

    # Stratified sample: ensure diversity of behavioral patterns
    rng = random.Random(seed)

    # Sort into behavioral categories
    learners = [p for p in all_participants if p.safe_fraction > 0.65]
    mixed = [p for p in all_participants if 0.35 <= p.safe_fraction <= 0.65]
    risk_seekers = [p for p in all_participants if p.safe_fraction < 0.35]

    # Sample proportionally but ensure representation
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
# Engine configuration
# ---------------------------------------------------------------------------

def _configure_engine(cfg: EngineConfig, policy: str) -> None:
    """Configure engine for behavioral prediction validation."""
    cfg.agent_runtimes.evaluator.enabled = False
    cfg.agent_runtimes.optimizer.enabled = False
    cfg.router.top_k = 3
    cfg.router.min_candidates = 2
    cfg.router.learned_weight_min_support = 0
    cfg.router.learned_weight_support_k = 0.0
    cfg.composer.enabled = False

    cfg.optimizer.candidate_failure_trigger = 999
    cfg.optimizer.max_active_candidates = 0
    cfg.optimizer.max_active_branches = 24
    cfg.optimizer.update_acceptance_min_gain = -1.0
    cfg.optimizer.rewrite_cooldown_episodes = 999

    if policy == "full":
        cfg.router.weight_coef = 1.0
        cfg.router.affinity_coef = 0.4
        cfg.router.memory_coef = 0.6
        cfg.router.memory_term_cap = 0.25
        cfg.router.bandit_value_coef = 0.3
        cfg.router.bandit_bonus_coef = 0.15
        cfg.router.bandit_bonus_cap = 0.12
        cfg.router.bandit_shrinkage_k = 8.0
        cfg.router.exploration = 0.20
        cfg.router.exploration_min = 0.04
        cfg.router.exploration_decay = 0.995

        cfg.memory.bias_scale = 1.2
        cfg.memory.bias_cap = 0.25
        cfg.memory.shrinkage_k = 8.0
        cfg.memory.recency_decay = 0.95
        cfg.memory.user_bias_mix = 0.8

        cfg.optimizer.learning_rate = 0.25
        cfg.optimizer.weight_decay = 0.002
        cfg.optimizer.advantage_baseline_beta = 0.15
        cfg.optimizer.branch_advantage_mix = 0.1
        cfg.optimizer.branch_baseline_beta = 0.08

    elif policy == "frozen":
        cfg.router.weight_coef = 1.0
        cfg.router.affinity_coef = 0.6
        cfg.router.memory_coef = 0.0
        cfg.router.memory_term_cap = 0.0
        cfg.router.bandit_value_coef = 0.0
        cfg.router.bandit_bonus_coef = 0.0
        cfg.router.exploration = 0.02
        cfg.router.exploration_min = 0.02
        cfg.router.exploration_decay = 1.0

        cfg.memory.bias_scale = 0.0
        cfg.memory.bias_cap = 0.0
        cfg.memory.user_bias_mix = 0.0
        cfg.memory.recency_decay = 0.98
        cfg.memory.shrinkage_k = 20.0

        cfg.optimizer.learning_rate = 0.0
        cfg.optimizer.weight_decay = 0.0

    else:
        raise ValueError(f"Unknown policy: {policy}")


def _create_engine(policy: str, seed: int, tmpdir: str) -> PromptForestEngine:
    cfg = load_config(CONFIG_PATH)
    _configure_engine(cfg, policy)
    run_dir = os.path.join(tmpdir, f"{policy}_{seed}")
    os.makedirs(run_dir, exist_ok=True)
    cfg.artifacts_dir = run_dir
    backend = DomainShiftBackend(shifted_quality_matrix(), noise=0.03, seed=seed)
    return PromptForestEngine(config=cfg, backend=backend)


# ---------------------------------------------------------------------------
# Predictor input extraction
# ---------------------------------------------------------------------------

def _extract_predictor_inputs(result: dict[str, Any]) -> tuple[dict[str, float], dict[str, float]]:
    """Extract predictor features from raw routing outputs.

    The behavioral predictor should learn from the router's activation signal,
    not from post-hoc branch quality judgments. We therefore use the raw
    routing scores as the primary branch features and keep route metadata and
    branch weights as separate contextual signals.
    """
    routing = result.get("routing", {})

    branch_scores: dict[str, float] = {}
    for name, score in routing.get("branch_scores", {}).items():
        if isinstance(score, (int, float)):
            branch_scores[name] = float(score)

    context_features: dict[str, float] = {}
    for name, data in result.get("branch_scores", {}).items():
        if not isinstance(data, dict):
            continue
        reward = data.get("reward")
        confidence = data.get("confidence")
        if isinstance(reward, (int, float)):
            context_features[f"judge_reward::{name}"] = float(reward)
        if isinstance(confidence, (int, float)):
            context_features[f"judge_conf::{name}"] = float(confidence)

    branch_weights = result.get("branch_weights", {})
    for name, weight in branch_weights.items():
        if isinstance(weight, (int, float)):
            context_features[f"branch_weight::{name}"] = float(weight)

    activated_branches = routing.get("activated_branches", []) or []
    activated_paths = routing.get("activated_paths", []) or []
    context_features["route_num_activated_branches"] = float(len(activated_branches))
    context_features["route_num_paths"] = float(len(activated_paths))

    sorted_scores = sorted(branch_scores.values(), reverse=True)
    if sorted_scores:
        context_features["route_top_score"] = float(sorted_scores[0])
    if len(sorted_scores) >= 2:
        context_features["route_top_gap"] = float(sorted_scores[0] - sorted_scores[1])

    selected = result.get("evaluation_signal", {}).get("selected_branch", "")
    if selected:
        context_features[f"selected_branch::{selected}"] = 1.0

    for branch_name in activated_branches:
        context_features[f"activated_branch::{branch_name}"] = 1.0

    sibling_decisions = routing.get("sibling_decisions", {}) or {}
    for parent_id, meta in sibling_decisions.items():
        if not isinstance(meta, dict):
            continue
        support = meta.get("support")
        win_rate = meta.get("win_rate")
        expected_margin = meta.get("expected_margin")
        if isinstance(support, (int, float)):
            context_features[f"sibling_support::{parent_id}"] = float(support)
        if isinstance(win_rate, (int, float)):
            context_features[f"sibling_win_rate::{parent_id}"] = float(win_rate)
        if isinstance(expected_margin, (int, float)):
            context_features[f"sibling_margin::{parent_id}"] = float(expected_margin)

    return branch_scores, context_features


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

@dataclass
class ParticipantResult:
    participant_id: str
    study: str
    safe_fraction: float
    n_trials: int
    n_train: int
    n_holdout: int
    # Per-window training accuracy (full policy only)
    window_accuracies: list[float]
    # Holdout metrics
    holdout_accuracy_full: float
    holdout_accuracy_frozen: float
    # Learning dynamics
    first_half_acc: float
    second_half_acc: float
    learning_slope: float
    convergence_window: int   # first window >= 60%, or -1
    predictor_priors: dict[str, float]


def run_participant_experiment(
    participant: RealIGTParticipant,
    tmpdir: str,
    shared_predictor: BehavioralPredictor | None = None,
    holdout_fraction: float = 0.3,
) -> ParticipantResult:
    """Run the full experiment for one real participant.

    Parameters
    ----------
    shared_predictor:
        If provided, uses this predictor (which contains learned models from
        previous participants) to enable transfer learning. The new user's
        model will be warm-started from similar previously-seen users after
        an initial observation period.
    """
    n_train = int(participant.n_trials * (1.0 - holdout_fraction))
    n_holdout = participant.n_trials - n_train
    train_choices = participant.choices[:n_train]
    holdout_choices = participant.choices[n_train:]

    seed_base = hash(participant.participant_id) % 100000
    holdout_accuracies: dict[str, float] = {}
    window_accs: list[float] = []
    predictor_priors: dict[str, float] = {}

    for policy in ["full", "frozen"]:
        engine = _create_engine(policy, seed_base, tmpdir)
        adapt = policy == "full"
        update_memory = policy == "full"

        # Use shared predictor for transfer learning, or create fresh one
        if shared_predictor is not None and policy == "full":
            predictor = shared_predictor
        else:
            predictor = BehavioralPredictor(
                actions=["safe", "risky"],
                learning_rate=0.15,
                prior_learning_rate=0.08,
                prior_smoothing=0.01,
                association_decay=0.005,
                min_learning_rate=0.03,
                anneal_rate=0.008,
                outcome_learning_rate=0.12,
                sequence_window=12,
                transition_learning_rate=0.10,
            )

        transfer_done = False

        # --- Training ---
        correct_per_window: dict[int, list[bool]] = {}
        cumulative_score = 0.0

        for idx, deck_choice in enumerate(train_choices):
            actual_group = "safe" if deck_choice in SAFE_DECKS else "risky"
            window_idx = idx // WINDOW_SIZE

            # Net outcome for this trial
            net_outcome = participant.wins[idx] + participant.losses[idx]
            cumulative_score += net_outcome

            # Cumulative stats for context
            safe_so_far = sum(1 for c in train_choices[:idx + 1] if c in SAFE_DECKS)
            progress = idx / max(1, n_train)

            context = {
                "trial_progress": progress,
                "safe_rate_so_far": safe_so_far / max(1, idx + 1),
                "cumulative_score_norm": cumulative_score / max(1.0, abs(cumulative_score) + 100.0),
            }

            text = (
                f"Iowa Gambling Task trial {idx + 1}/{participant.n_trials}: "
                f"Participant choosing between 4 decks. "
                f"Decks A,B give high rewards but high losses. "
                f"Decks C,D give low rewards but low losses. "
                f"Trial progress: {progress:.0%}."
            )

            result = engine.run_task_controlled(
                text=text,
                task_type="general",
                metadata={
                    "user_id": participant.participant_id,
                    "trial_num": idx + 1,
                    "expected_keywords": [f"igt_real_{idx % 14}"],
                    "required_substrings": ["confidence", "key-points"],
                },
                adapt=adapt,
                update_memory=update_memory,
            )

            branch_scores, route_context = _extract_predictor_inputs(result)
            predictor_context = {**context, **route_context}

            # Improvement #5: boost branches that are more predictive
            # of this user's behavior (predictor→optimizer connection)
            if adapt and idx >= 15:
                branch_scores = predictor.predictiveness_weight_bonus(
                    participant.participant_id, branch_scores
                )

            pred_result = predictor.predict(
                user_id=participant.participant_id,
                branch_scores=branch_scores,
                context=predictor_context,
            )
            prediction = pred_result.predicted_action
            is_correct = prediction == actual_group
            correct_per_window.setdefault(window_idx, []).append(is_correct)

            reward_score = predictor.prediction_accuracy_reward(
                predicted=prediction,
                actual=actual_group,
                confidence=pred_result.confidence,
            )

            if adapt or update_memory:
                predictor.update(
                    user_id=participant.participant_id,
                    branch_scores=branch_scores,
                    actual_action=actual_group,
                    context=predictor_context,
                    outcome=net_outcome,
                )

                # Transfer learning: after 15 observations, warm-start
                # from similar previously-completed users.
                # At 15 trials we have enough signal to compute a meaningful
                # similarity but still early enough to benefit from transfer.
                if (
                    not transfer_done
                    and shared_predictor is not None
                    and policy == "full"
                    and idx == 14  # after 15th trial
                    and shared_predictor.user_count > 3
                ):
                    transfer_stats = predictor.warm_start_from_similar(
                        user_id=participant.participant_id,
                        top_k=5,
                        min_donor_observations=40,
                        transfer_weight=0.4,
                        min_similarity=0.5,
                    )
                    transfer_done = True

                engine.apply_feedback(
                    task_id=result["task"]["task_id"],
                    score=reward_score,
                    accepted=is_correct,
                    corrected_answer="" if is_correct else f"actual={actual_group}",
                    feedback_text=f"trial {idx + 1}: pred={prediction}, actual={actual_group}",
                    user_id=participant.participant_id,
                )

        if policy == "full":
            for w_idx in sorted(correct_per_window.keys()):
                cl = correct_per_window[w_idx]
                window_accs.append(sum(cl) / len(cl) if cl else 0.0)
            predictor_priors = predictor.get_user_model_summary(
                participant.participant_id
            ).get("priors", {})

        # --- Holdout ---
        holdout_correct = 0
        holdout_rng = random.Random(seed_base + 7777)

        for idx, deck_choice in enumerate(holdout_choices):
            actual_group = "safe" if deck_choice in SAFE_DECKS else "risky"

            if policy == "frozen":
                # Random baseline
                prediction = holdout_rng.choice(["safe", "risky"])
            else:
                context_h = {
                    "trial_progress": 1.0,
                    "safe_rate_so_far": participant.safe_fraction,
                }
                text = (
                    f"Iowa Gambling Task trial {n_train + idx + 1}/{participant.n_trials}: "
                    f"Participant choosing between 4 decks."
                )
                result = engine.run_task_controlled(
                    text=text,
                    task_type="general",
                    metadata={
                        "user_id": participant.participant_id,
                        "trial_num": n_train + idx + 1,
                        "expected_keywords": [f"igt_holdout_{idx % 14}"],
                        "required_substrings": ["confidence", "key-points"],
                    },
                    adapt=False,
                    update_memory=False,
                )
                branch_scores, route_context = _extract_predictor_inputs(result)
                predictor_context = {**context_h, **route_context}
                pred_result = predictor.predict(
                    user_id=participant.participant_id,
                    branch_scores=branch_scores,
                    context=predictor_context,
                )
                prediction = pred_result.predicted_action

            if prediction == actual_group:
                holdout_correct += 1

        holdout_accuracies[policy] = holdout_correct / max(1, n_holdout)

    # Compute learning metrics
    n_windows = len(window_accs)
    first_half = window_accs[:n_windows // 2] if n_windows >= 2 else window_accs
    second_half = window_accs[n_windows // 2:] if n_windows >= 2 else window_accs
    first_half_acc = mean(first_half) if first_half else 0.0
    second_half_acc = mean(second_half) if second_half else 0.0
    learning_slope = 0.0
    if n_windows >= 2:
        learning_slope = (window_accs[-1] - window_accs[0]) / max(1, n_windows - 1)

    convergence_window = -1
    for i, acc in enumerate(window_accs):
        if acc >= 0.60:
            convergence_window = i
            break

    return ParticipantResult(
        participant_id=participant.participant_id,
        study=participant.study,
        safe_fraction=participant.safe_fraction,
        n_trials=participant.n_trials,
        n_train=n_train,
        n_holdout=n_holdout,
        window_accuracies=window_accs,
        holdout_accuracy_full=holdout_accuracies.get("full", 0.0),
        holdout_accuracy_frozen=holdout_accuracies.get("frozen", 0.0),
        first_half_acc=first_half_acc,
        second_half_acc=second_half_acc,
        learning_slope=learning_slope,
        convergence_window=convergence_window,
        predictor_priors=predictor_priors,
    )


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def _print_header(participants: list[RealIGTParticipant]) -> None:
    print("=" * 76)
    print("  ADAPTIVE LEARNING VALIDATION — Real Human Data")
    print("  Steingroever et al. (2015) Iowa Gambling Task")
    print("=" * 76)
    print()
    print(f"  Dataset: {len(participants)} real participants (from 504 total)")
    print(f"  Trials per participant: 100")
    print(f"  Studies: {sorted(set(p.study for p in participants))}")
    print()

    # Dataset stats
    safe_fracs = [p.safe_fraction for p in participants]
    print(f"  Safe-deck selection rates:")
    print(f"    Mean:   {mean(safe_fracs):.1%}")
    print(f"    Std:    {stdev(safe_fracs):.1%}" if len(safe_fracs) > 1 else "")
    print(f"    Range:  {min(safe_fracs):.0%} - {max(safe_fracs):.0%}")
    n_learn = sum(1 for f in safe_fracs if f > 0.65)
    n_mixed = sum(1 for f in safe_fracs if 0.35 <= f <= 0.65)
    n_risk = sum(1 for f in safe_fracs if f < 0.35)
    print(f"    >65% safe (learners):     {n_learn}")
    print(f"    35-65% safe (mixed):      {n_mixed}")
    print(f"    <35% safe (risk-seekers): {n_risk}")
    print()
    print("  Architecture:")
    print("    BehavioralPredictor: learns per-user branch->action mappings")
    print("    PromptForestEngine:  branch weight + memory adaptation")
    print("    No hardcoded branch-to-action mapping — fully learned")
    print()


def _print_results(results: list[ParticipantResult]) -> None:
    print("-" * 76)
    print("  Per-Participant Results")
    print("-" * 76)
    print()

    # Group by behavioral type
    learners = [r for r in results if r.safe_fraction > 0.65]
    mixed = [r for r in results if 0.35 <= r.safe_fraction <= 0.65]
    risk_seekers = [r for r in results if r.safe_fraction < 0.35]

    for group_name, group in [("LEARNERS (>65% safe)", learners),
                               ("MIXED (35-65% safe)", mixed),
                               ("RISK-SEEKERS (<35% safe)", risk_seekers)]:
        if not group:
            continue
        print(f"  --- {group_name} ---")
        for r in group[:5]:  # show first 5 per group
            lift = r.holdout_accuracy_full - r.holdout_accuracy_frozen
            bars = "".join(
                "#" if a >= 0.6 else "." for a in r.window_accuracies
            )
            print(
                f"    {r.participant_id:>8s} ({r.study:>15s}, safe={r.safe_fraction:.0%}): "
                f"full={r.holdout_accuracy_full:.0%} frozen={r.holdout_accuracy_frozen:.0%} "
                f"lift={lift:+.0%}  learning=[{bars}]  "
                f"priors={r.predictor_priors}"
            )
        if len(group) > 5:
            remaining_lifts = [r.holdout_accuracy_full - r.holdout_accuracy_frozen for r in group[5:]]
            print(f"    ... and {len(group) - 5} more (mean lift: {mean(remaining_lifts):+.1%})")
        print()


def _print_summary(results: list[ParticipantResult]) -> None:
    print("-" * 76)
    print("  Overall Summary")
    print("-" * 76)
    print()

    n = len(results)
    mean_full = mean(r.holdout_accuracy_full for r in results)
    mean_frozen = mean(r.holdout_accuracy_frozen for r in results)
    mean_lift = mean_full - mean_frozen

    # Learning dynamics
    slopes = [r.learning_slope for r in results]
    mean_slope = mean(slopes)
    first_half_accs = [r.first_half_acc for r in results]
    second_half_accs = [r.second_half_acc for r in results]
    mean_improvement = mean(second_half_accs) - mean(first_half_accs)
    converged = sum(1 for r in results if r.convergence_window >= 0)

    # Per-group breakdown
    learners = [r for r in results if r.safe_fraction > 0.65]
    mixed = [r for r in results if 0.35 <= r.safe_fraction <= 0.65]
    risk_seekers = [r for r in results if r.safe_fraction < 0.35]

    print(f"  Participants: {n}")
    print()
    print(f"  Holdout accuracy:")
    print(f"    Full (adaptive):  {mean_full:.1%}")
    print(f"    Frozen (random):  {mean_frozen:.1%}")
    print(f"    Lift:             {mean_lift:+.1%}")
    print()

    for group_name, group in [("Learners", learners), ("Mixed", mixed), ("Risk-seekers", risk_seekers)]:
        if not group:
            continue
        g_full = mean(r.holdout_accuracy_full for r in group)
        g_frozen = mean(r.holdout_accuracy_frozen for r in group)
        g_lift = g_full - g_frozen
        print(f"    {group_name:>14s} (n={len(group):>2d}): "
              f"full={g_full:.1%}  frozen={g_frozen:.1%}  lift={g_lift:+.1%}")
    print()

    print(f"  Learning dynamics:")
    print(f"    Mean learning slope:        {mean_slope:+.2%} per window")
    print(f"    Mean 1st→2nd half improve:  {mean_improvement:+.1%}")
    print(f"    Convergence (>=60%):        {converged}/{n} participants")
    print()

    # Majority-class baseline comparison
    # If the predictor just predicted the majority class for each participant,
    # what would accuracy be?
    majority_class_accs = []
    for r in results:
        # Majority class accuracy = max(safe_frac, 1-safe_frac)
        majority_class_accs.append(max(r.safe_fraction, 1.0 - r.safe_fraction))
    mean_majority = mean(majority_class_accs)
    above_majority = sum(1 for r, m in zip(results, majority_class_accs)
                         if r.holdout_accuracy_full >= m)

    print(f"  Baseline comparisons:")
    print(f"    Random chance:              50.0%")
    print(f"    Majority-class baseline:    {mean_majority:.1%}")
    print(f"    Our system (full):          {mean_full:.1%}")
    print(f"    Participants above majority: {above_majority}/{n}")
    print()

    # Verdicts
    print("-" * 76)
    print("  Verdicts")
    print("-" * 76)
    verdicts = [
        ("Full outperforms random (50%)", mean_full > 0.50),
        ("Full outperforms frozen baseline", mean_lift > 0),
        ("Learning curve improves (slope > 0)", mean_slope > 0),
        ("2nd half > 1st half accuracy", mean_improvement > 0),
        ("At least 50% converge to 60%+", converged >= n * 0.5),
        ("System approaches majority-class", mean_full >= mean_majority * 0.90),
    ]
    all_pass = True
    for label, passed in verdicts:
        status = "PASS" if passed else "FAIL"
        marker = "  [+]" if passed else "  [-]"
        print(f"  {marker} {label}: {status}")
        if not passed:
            all_pass = False
    print()
    if all_pass:
        print("  OVERALL: ALL CHECKS PASSED")
    else:
        print("  OVERALL: SOME CHECKS FAILED -- see details above")
    print("=" * 76)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Load real data
    print("  Loading Steingroever et al. (2015) IGT dataset...")
    participants = load_steingroever_data(max_participants=MAX_PARTICIPANTS)
    _print_header(participants)

    # Run experiments
    tmpdir = tempfile.mkdtemp(prefix="igt_real_val_")
    print(f"  Artifacts directory: {tmpdir}")
    print(f"  Running {len(participants)} participants × 2 policies × 100 trials...")
    print(f"  Transfer learning: enabled (warm-start from similar users after 10 trials)")
    print()

    # Shared predictor: accumulates learned models across participants
    # so later participants can warm-start from similar earlier ones
    shared_predictor = BehavioralPredictor(
        actions=["safe", "risky"],
        learning_rate=0.15,
        prior_learning_rate=0.08,
        prior_smoothing=0.01,
        association_decay=0.005,
        min_learning_rate=0.03,
        anneal_rate=0.008,
        outcome_learning_rate=0.12,
        sequence_window=12,
        transition_learning_rate=0.10,
    )

    results: list[ParticipantResult] = []
    for i, participant in enumerate(participants):
        safe_pct = f"{participant.safe_fraction:.0%}"
        n_donors = shared_predictor.user_count
        donor_str = f" donors={n_donors}" if n_donors > 0 else ""
        print(
            f"    [{i + 1:>2d}/{len(participants)}] {participant.participant_id:>8s} "
            f"({participant.study:>15s}, safe={safe_pct:>3s}{donor_str})...",
            flush=True,
        )
        result = run_participant_experiment(
            participant, tmpdir, shared_predictor=shared_predictor
        )
        results.append(result)
    print()

    # Print results
    _print_results(results)
    _print_summary(results)

    # Cleanup
    try:
        shutil.rmtree(tmpdir)
    except OSError:
        pass


if __name__ == "__main__":
    main()
