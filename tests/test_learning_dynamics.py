from __future__ import annotations

from pathlib import Path

from prompt_forest.experiments.rl_validation import RLLearningValidator


def test_adaptive_policy_beats_frozen_baseline():
    root = Path(__file__).resolve().parents[1]
    validator = RLLearningValidator(root)
    report = validator.run(seeds=[3, 5, 7], episodes_per_seed=120)

    agg = report["aggregate"]
    assert agg["mean_reward_gain"] > 0.1
    assert agg["reward_gain_win_rate"] >= 0.9


def test_adaptive_policy_improves_over_time_and_alignment():
    root = Path(__file__).resolve().parents[1]
    validator = RLLearningValidator(root)
    report = validator.run(seeds=[13, 17, 19], episodes_per_seed=120)

    agg = report["aggregate"]
    assert agg["mean_adaptive_reward_trend"] > 0.03
    assert agg["mean_adaptive_optimal_hit_trend"] > 0.15
