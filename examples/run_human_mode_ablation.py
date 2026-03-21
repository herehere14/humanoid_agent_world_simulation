#!/usr/bin/env python3
"""Human-mode ablations for behavioral prediction."""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path
from statistics import mean

EXAMPLES_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.join(EXAMPLES_DIR, "..")
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, EXAMPLES_DIR)

from src.prompt_forest.behavioral.predictor import BehavioralPredictor

import run_adaptive_learning_validation as base_validation
from run_adaptive_learning_validation_human_mode import (
    HumanModeParticipantResult,
    HumanModeValidationConfig,
    MAX_PARTICIPANTS,
    run_human_mode_participant_experiment,
)

ROOT = Path(__file__).resolve().parents[1]


def _new_predictor() -> BehavioralPredictor:
    return BehavioralPredictor(
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


def _summarize(results: list[HumanModeParticipantResult]) -> dict[str, float]:
    learners = [r for r in results if r.safe_fraction > 0.65]
    mixed = [r for r in results if 0.35 <= r.safe_fraction <= 0.65]
    risk = [r for r in results if r.safe_fraction < 0.35]
    return {
        "overall": mean(r.holdout_accuracy_full for r in results),
        "learners": mean(r.holdout_accuracy_full for r in learners) if learners else 0.0,
        "mixed": mean(r.holdout_accuracy_full for r in mixed) if mixed else 0.0,
        "risk": mean(r.holdout_accuracy_full for r in risk) if risk else 0.0,
    }


def main() -> None:
    participants = base_validation.load_steingroever_data(max_participants=MAX_PARTICIPANTS)
    variants = [
        HumanModeValidationConfig(
            name="history_only",
            use_branch_scores=False,
            use_route_context=False,
            use_state_features=False,
            adapt=False,
            update_memory=False,
            apply_behavior_feedback=False,
        ),
        HumanModeValidationConfig(
            name="history_plus_state",
            use_branch_scores=False,
            use_route_context=False,
            use_state_features=True,
            adapt=False,
            update_memory=False,
            apply_behavior_feedback=False,
        ),
        HumanModeValidationConfig(
            name="history_plus_branches",
            use_branch_scores=True,
            use_route_context=True,
            use_state_features=False,
            adapt=True,
            update_memory=True,
            apply_behavior_feedback=True,
        ),
        HumanModeValidationConfig(
            name="full_human_mode",
            use_branch_scores=True,
            use_route_context=True,
            use_state_features=True,
            adapt=True,
            update_memory=True,
            apply_behavior_feedback=True,
        ),
    ]

    tmpdir = tempfile.mkdtemp(prefix="igt_human_mode_ablation_")
    try:
        summaries: list[tuple[str, dict[str, float]]] = []
        for variant in variants:
            print(f"\n=== {variant.name} ===", flush=True)
            predictor = _new_predictor()
            results: list[HumanModeParticipantResult] = []
            for idx, participant in enumerate(participants):
                print(
                    f"[{idx + 1:>2d}/{len(participants)}] {participant.participant_id} "
                    f"({participant.study}, safe={participant.safe_fraction:.0%})",
                    flush=True,
                )
                results.append(
                    run_human_mode_participant_experiment(
                        participant=participant,
                        tmpdir=tmpdir,
                        shared_predictor=predictor,
                        config=variant,
                    )
                )
            summaries.append((variant.name, _summarize(results)))

        print("\n" + "=" * 76)
        print("  HUMAN MODE ABLATIONS")
        print("=" * 76)
        print(f"{'Variant':<24} {'Overall':>8} {'Learners':>10} {'Mixed':>8} {'Risk':>8}")
        for name, summary in summaries:
            print(
                f"{name:<24} "
                f"{summary['overall']:>8.1%} "
                f"{summary['learners']:>10.1%} "
                f"{summary['mixed']:>8.1%} "
                f"{summary['risk']:>8.1%}"
            )
        print("=" * 76)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
