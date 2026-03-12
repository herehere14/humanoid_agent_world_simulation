from __future__ import annotations

from pathlib import Path

from prompt_forest.experiments.detailed_validation import DetailedHierarchicalValidator


def test_branch_inventory_counts_are_correct():
    root = Path(__file__).resolve().parents[1]
    validator = DetailedHierarchicalValidator(root)
    report = validator.run(seeds=[3], episodes_per_seed=60)

    inventory = report["branch_inventory"]
    assert inventory["total_trainable_branches"] == 18
    assert inventory["macro_branch_count"] == 6
    assert inventory["niche_sub_branch_count"] == 12


def test_hierarchical_adaptation_improves_reward_and_hit_rate():
    root = Path(__file__).resolve().parents[1]
    validator = DetailedHierarchicalValidator(root)
    report = validator.run(seeds=[3, 5], episodes_per_seed=90)

    learning = report["learning"]
    assert learning["mean_reward_gain"] > 0.12
    assert learning["reward_gain_win_rate"] >= 1.0
    assert learning["mean_optimal_hit_gain"] > 0.3
    assert learning["mean_late_minus_early"] > 0.05


def test_sub_branches_and_key_subtrees_have_positive_effect():
    root = Path(__file__).resolve().parents[1]
    validator = DetailedHierarchicalValidator(root)
    report = validator.run(seeds=[3, 5], episodes_per_seed=90)

    effect = report["branch_effect"]
    assert effect["full_minus_no_verification"] > 0.02
    assert effect["full_minus_no_retrieval"] > 0.01


def test_growth_probe_adds_branches_and_improves_reward_trend():
    root = Path(__file__).resolve().parents[1]
    validator = DetailedHierarchicalValidator(root)
    report = validator.run(seeds=[3, 5], episodes_per_seed=90)

    growth = report["growth"]
    assert growth["mean_new_branches_created"] >= 5
    assert growth["mean_depth_gain"] >= 1
    assert growth["mean_growth_reward_trend"] > 0.02
