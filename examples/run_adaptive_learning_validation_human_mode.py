#!/usr/bin/env python3
"""Adaptive Learning Validation using Human Mode.

Runs the Steingroever IGT benchmark through the human-mode stack:
  - human-mode branch forest
  - state-conditioned human router
  - experiential/emotional memory
  - behavioral predictor over routing + human-state features
"""

from __future__ import annotations

import os
import random
import shutil
import sys
import tempfile
from hashlib import blake2b
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

EXAMPLES_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.join(EXAMPLES_DIR, "..")
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, EXAMPLES_DIR)

from src.prompt_forest.backend.simulated import DomainShiftBackend, shifted_quality_matrix
from src.prompt_forest.behavioral.predictor import BehavioralPredictor
from src.prompt_forest.config import load_config
from src.prompt_forest.modes.orchestrator import ModeOrchestrator

import run_adaptive_learning_validation as base_validation

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "default.json"
MAX_PARTICIPANTS = base_validation.MAX_PARTICIPANTS
WINDOW_SIZE = base_validation.WINDOW_SIZE
SAFE_DECKS = base_validation.SAFE_DECKS


@dataclass
class HumanModeParticipantResult:
    participant_id: str
    study: str
    safe_fraction: float
    holdout_accuracy_full: float
    holdout_accuracy_frozen: float
    window_accuracies: list[float]


@dataclass(frozen=True)
class HumanModeValidationConfig:
    name: str = "full_human_mode"
    use_branch_scores: bool = True
    use_route_context: bool = True
    use_state_features: bool = True
    adapt: bool = True
    update_memory: bool = True
    apply_behavior_feedback: bool = True


def _configure_human_engine() -> object:
    cfg = load_config(CONFIG_PATH)
    base_validation._configure_engine(cfg, "full")
    cfg.router.top_k = 4
    cfg.router.min_candidates = 3
    cfg.composer.enabled = False
    return cfg


def _create_human_orchestrator(seed: int, tmpdir: str) -> ModeOrchestrator:
    cfg = _configure_human_engine()
    run_dir = os.path.join(tmpdir, f"human_mode_{seed}")
    os.makedirs(run_dir, exist_ok=True)
    cfg.artifacts_dir = run_dir
    backend = DomainShiftBackend(shifted_quality_matrix(), noise=0.03, seed=seed)
    return ModeOrchestrator(mode="human_mode", engine_config=cfg, backend=backend)


def _stable_seed(value: str) -> int:
    digest = blake2b(value.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % 100000


def _outcome_event(net_outcome: float) -> tuple[str, float]:
    mag = min(1.0, abs(net_outcome) / (abs(net_outcome) + 100.0))
    if net_outcome > 0:
        return "reward", max(0.15, mag)
    if net_outcome < 0:
        return "threat", max(0.15, mag)
    return "novelty", 0.2


def _human_state_features(result: dict) -> dict[str, float]:
    features: dict[str, float] = {}
    human_state = result.get("human_state", {}) or {}
    before = human_state.get("before", {}) or {}
    variables = before.get("variables", {}) or {}
    for key, value in variables.items():
        if isinstance(value, (int, float)):
            features[f"state::{key}"] = float(value)

    mood = before.get("mood_valence")
    if isinstance(mood, (int, float)):
        features["state::mood_valence"] = float(mood)

    dominant = before.get("dominant_drives", []) or []
    for drive in dominant:
        features[f"drive::{drive}"] = 1.0

    conflicts = result.get("conflicts", []) or []
    features["conflict::count"] = float(len(conflicts))
    max_intensity = 0.0
    for conflict in conflicts:
        if not isinstance(conflict, dict):
            continue
        drive_a = str(conflict.get("drive_a", "")).strip()
        drive_b = str(conflict.get("drive_b", "")).strip()
        intensity = conflict.get("intensity")
        resolution_weight = conflict.get("resolution_weight")
        if drive_a and drive_b:
            features[f"conflict_pair::{drive_a}_vs_{drive_b}"] = 1.0
        if isinstance(intensity, (int, float)):
            max_intensity = max(max_intensity, float(intensity))
        if isinstance(resolution_weight, (int, float)) and drive_a and drive_b:
            features[f"conflict_resolution::{drive_a}_vs_{drive_b}"] = float(resolution_weight)
    features["conflict::max_intensity"] = max_intensity

    coherence = (result.get("evaluation_signal", {}) or {}).get("coherence_details", {}) or {}
    for key in ["coherence", "consistency", "believability", "conflict_handling", "mood_valence", "arousal"]:
        value = coherence.get(key)
        if isinstance(value, (int, float)):
            features[f"human_eval::{key}"] = float(value)

    memory = result.get("experiential_memory", {}) or {}
    count = memory.get("count")
    if isinstance(count, (int, float)):
        features["experience::memory_count"] = float(count)
    for tag in memory.get("latest_tags", []) or []:
        features[f"experience_tag::{tag}"] = 1.0

    return features


def _make_behavior_acceptance_runner(
    orch: ModeOrchestrator,
    predictor: BehavioralPredictor,
):
    def runner(
        *,
        branch_name: str,
        task_type: str,
        user_id: str,
        old_weight: float,
        new_weight: float,
        prompt_rewritten: bool,
        **_: object,
    ) -> bool:
        records = orch.engine.memory.retrieve_similar(task_type, limit=8, user_id=user_id)
        old_hits: list[float] = []
        new_hits: list[float] = []
        weight_ratio = (new_weight / old_weight) if abs(old_weight) > 1e-6 else 1.0

        for record in reversed(records):
            meta = record.task_metadata or {}
            actual = meta.get("behavior_actual_action")
            branch_scores = meta.get("behavior_branch_scores")
            predictor_context = meta.get("behavior_context")
            if not isinstance(actual, str):
                continue
            if not isinstance(branch_scores, dict) or not isinstance(predictor_context, dict):
                continue

            base_scores = {
                key: float(value)
                for key, value in branch_scores.items()
                if isinstance(value, (int, float))
            }
            base_context = {
                key: float(value)
                for key, value in predictor_context.items()
                if isinstance(value, (int, float))
            }
            if not base_scores and not base_context:
                continue

            old_scores = dict(base_scores)
            new_scores = dict(base_scores)
            old_context = dict(base_context)
            new_context = dict(base_context)

            if branch_name in new_scores:
                new_scores[branch_name] = new_scores[branch_name] * weight_ratio
            old_context[f"branch_weight::{branch_name}"] = float(old_weight)
            new_context[f"branch_weight::{branch_name}"] = float(new_weight)

            old_pred = predictor.predict(user_id, old_scores, old_context).predicted_action
            new_pred = predictor.predict(user_id, new_scores, new_context).predicted_action
            old_hits.append(float(old_pred == actual))
            new_hits.append(float(new_pred == actual))

        if len(old_hits) < 2 or len(new_hits) < 2:
            return True

        min_gain = 0.001 if prompt_rewritten else 0.0
        return mean(new_hits) >= (mean(old_hits) + min_gain)

    return runner


def run_human_mode_participant_experiment(
    participant: base_validation.RealIGTParticipant,
    tmpdir: str,
    shared_predictor: BehavioralPredictor,
    config: HumanModeValidationConfig | None = None,
) -> HumanModeParticipantResult:
    run_config = config or HumanModeValidationConfig()
    n_train = int(participant.n_trials * 0.7)
    n_holdout = participant.n_trials - n_train
    train_choices = participant.choices[:n_train]
    holdout_choices = participant.choices[n_train:]

    seed_base = _stable_seed(participant.participant_id)
    orch = _create_human_orchestrator(seed_base, tmpdir)
    predictor = shared_predictor
    acceptance_runner = _make_behavior_acceptance_runner(orch, predictor)
    transfer_done = False
    correct_per_window: dict[int, list[bool]] = {}
    cumulative_score = 0.0
    prev_outcome: float | None = None

    for idx, deck_choice in enumerate(train_choices):
        if prev_outcome is not None:
            evt_type, intensity = _outcome_event(prev_outcome)
            orch.inject_event(evt_type, intensity)

        actual = "safe" if deck_choice in SAFE_DECKS else "risky"
        net_outcome = participant.wins[idx] + participant.losses[idx]
        cumulative_score += net_outcome
        safe_so_far = sum(1 for c in train_choices[: idx + 1] if c in SAFE_DECKS)
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
        result = orch.run_task(
            text=text,
            task_type="general",
            metadata={"user_id": participant.participant_id, "trial_num": idx + 1},
            adapt=run_config.adapt,
            update_memory=run_config.update_memory,
            acceptance_runner=acceptance_runner if run_config.adapt else None,
        )
        raw_branch_scores, route_context = base_validation._extract_predictor_inputs(result)
        branch_scores = dict(raw_branch_scores) if run_config.use_branch_scores else {}
        if run_config.use_branch_scores and idx >= 15:
            branch_scores = predictor.predictiveness_weight_bonus(participant.participant_id, branch_scores)

        predictor_context = dict(context)
        if run_config.use_route_context:
            predictor_context.update(route_context)
        if run_config.use_state_features:
            predictor_context.update(_human_state_features(result))
        pred_result = predictor.predict(participant.participant_id, branch_scores, predictor_context)
        is_correct = pred_result.predicted_action == actual
        correct_per_window.setdefault(idx // WINDOW_SIZE, []).append(is_correct)
        reward_score = predictor.prediction_accuracy_reward(
            predicted=pred_result.predicted_action,
            actual=actual,
            confidence=pred_result.confidence,
        )

        predictor.update(
            participant.participant_id,
            branch_scores,
            actual,
            predictor_context,
            net_outcome,
        )
        if run_config.update_memory:
            orch.engine.memory.annotate_record_metadata(
                result["task"]["task_id"],
                {
                    "behavior_actual_action": actual,
                    "behavior_branch_scores": dict(branch_scores),
                    "behavior_context": dict(predictor_context),
                    "behavior_outcome": float(net_outcome),
                },
            )
        if not transfer_done and idx == 14 and shared_predictor.user_count > 3:
            predictor.warm_start_from_similar(
                user_id=participant.participant_id,
                top_k=5,
                min_donor_observations=40,
                transfer_weight=0.4,
                min_similarity=0.5,
            )
            transfer_done = True
        if run_config.apply_behavior_feedback and run_config.update_memory:
            orch.engine.apply_feedback(
                task_id=result["task"]["task_id"],
                score=reward_score,
                accepted=is_correct,
                corrected_answer="" if is_correct else f"actual={actual}",
                feedback_text=(
                    f"trial {idx + 1}: pred={pred_result.predicted_action}, actual={actual}"
                ),
                user_id=participant.participant_id,
            )
        prev_outcome = net_outcome

    holdout_correct = 0
    prev_holdout_outcome = prev_outcome
    for idx, deck_choice in enumerate(holdout_choices):
        if prev_holdout_outcome is not None:
            evt_type, intensity = _outcome_event(prev_holdout_outcome)
            orch.inject_event(evt_type, intensity)

        actual = "safe" if deck_choice in SAFE_DECKS else "risky"
        context_h = {
            "trial_progress": 1.0,
            "safe_rate_so_far": participant.safe_fraction,
        }
        text = f"Iowa Gambling Task trial {n_train + idx + 1}/{participant.n_trials}: Participant choosing between 4 decks."
        result = orch.run_task(
            text=text,
            task_type="general",
            metadata={"user_id": participant.participant_id, "trial_num": n_train + idx + 1},
            adapt=False,
            update_memory=False,
        )
        raw_branch_scores, route_context = base_validation._extract_predictor_inputs(result)
        branch_scores = dict(raw_branch_scores) if run_config.use_branch_scores else {}
        predictor_context = dict(context_h)
        if run_config.use_route_context:
            predictor_context.update(route_context)
        if run_config.use_state_features:
            predictor_context.update(_human_state_features(result))
        prediction = predictor.predict(participant.participant_id, branch_scores, predictor_context).predicted_action
        holdout_correct += int(prediction == actual)
        prev_holdout_outcome = participant.wins[n_train + idx] + participant.losses[n_train + idx]

    rng = random.Random(seed_base + 7777)
    frozen_correct = sum(
        1
        for c in holdout_choices
        if rng.choice(["safe", "risky"]) == ("safe" if c in SAFE_DECKS else "risky")
    )

    return HumanModeParticipantResult(
        participant_id=participant.participant_id,
        study=participant.study,
        safe_fraction=participant.safe_fraction,
        holdout_accuracy_full=holdout_correct / max(1, n_holdout),
        holdout_accuracy_frozen=frozen_correct / max(1, n_holdout),
        window_accuracies=[
            sum(items) / len(items) if items else 0.0
            for _, items in sorted(correct_per_window.items())
        ],
    )


def _print_summary(results: list[HumanModeParticipantResult]) -> None:
    mean_full = mean(r.holdout_accuracy_full for r in results)
    mean_frozen = mean(r.holdout_accuracy_frozen for r in results)
    learners = [r for r in results if r.safe_fraction > 0.65]
    mixed = [r for r in results if 0.35 <= r.safe_fraction <= 0.65]
    risk = [r for r in results if r.safe_fraction < 0.35]

    print("=" * 76)
    print("  ADAPTIVE LEARNING VALIDATION — Human Mode")
    print("=" * 76)
    print(f"  Participants: {len(results)}")
    print()
    print(f"  Holdout accuracy:")
    print(f"    Full (human_mode): {mean_full:.1%}")
    print(f"    Frozen (random):   {mean_frozen:.1%}")
    print(f"    Lift:              {mean_full - mean_frozen:+.1%}")
    print()
    for label, group in [("Learners", learners), ("Mixed", mixed), ("Risk-seekers", risk)]:
        if not group:
            continue
        print(
            f"    {label:>12s} (n={len(group):>2d}): "
            f"{mean(r.holdout_accuracy_full for r in group):.1%}"
        )
    print("=" * 76)


def main() -> None:
    participants = base_validation.load_steingroever_data(max_participants=MAX_PARTICIPANTS)
    tmpdir = tempfile.mkdtemp(prefix="igt_human_mode_val_")
    try:
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
        results: list[HumanModeParticipantResult] = []
        for idx, participant in enumerate(participants):
            print(
                f"[{idx + 1:>2d}/{len(participants)}] {participant.participant_id} "
                f"({participant.study}, safe={participant.safe_fraction:.0%})",
                flush=True,
            )
            results.append(run_human_mode_participant_experiment(participant, tmpdir, shared_predictor))
        _print_summary(results)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
