from __future__ import annotations

from pathlib import Path
from statistics import pstdev

from prompt_forest.experiments.rl_validation import RLLearningValidator


def test_full_policy_beats_frozen_and_memory_only_on_holdout():
    root = Path(__file__).resolve().parents[1]
    validator = RLLearningValidator(root)
    report = validator.run(seeds=[3, 5, 7], episodes_per_seed=120)

    policy_metrics = report["aggregate"]["policy_metrics"]
    assert policy_metrics["full"]["mean_holdout_reward"] > policy_metrics["frozen"]["mean_holdout_reward"]
    assert policy_metrics["full"]["mean_holdout_reward"] > policy_metrics["memory_only"]["mean_holdout_reward"]


def test_full_policy_gain_is_stable_across_seeds():
    root = Path(__file__).resolve().parents[1]
    validator = RLLearningValidator(root)
    report = validator.run(seeds=[13, 17, 19, 23], episodes_per_seed=120)

    full_gains = [trial["gains_vs_frozen"]["full"]["holdout_reward_gain"] for trial in report["per_trial"]]
    win_rate = sum(1 for gain in full_gains if gain > 0.0) / len(full_gains)

    assert win_rate >= 0.75
    assert pstdev(full_gains) < 0.08


def test_validation_report_exposes_relative_gain_target_gate():
    root = Path(__file__).resolve().parents[1]
    validator = RLLearningValidator(root)
    report = validator.run(seeds=[3, 5], episodes_per_seed=80)

    agg = report["aggregate"]
    assert "full_over_frozen_relative_gain" in agg
    assert "target_relative_gain" in agg
    assert "passes_target_relative_gain" in agg
    assert agg["target_relative_gain"] == 0.2
