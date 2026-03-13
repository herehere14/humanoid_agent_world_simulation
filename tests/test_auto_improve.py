from __future__ import annotations

from pathlib import Path

from prompt_forest.experiments.auto_improve import AutoImprover
from prompt_forest.experiments.rl_validation import RLLearningValidator


def test_rl_validation_emits_bias_audit_fields():
    root = Path(__file__).resolve().parents[1]
    validator = RLLearningValidator(root)
    report = validator.run(seeds=[3], episodes_per_seed=60, config_patch={"router.memory_coef": 0.16})

    agg = report["aggregate"]
    assert "policy_holdout_by_task" in agg
    assert "policy_branch_concentration_hhi" in agg
    assert "full" in agg["policy_holdout_by_task"]
    assert 0.0 <= agg["policy_branch_concentration_hhi"]["full"] <= 1.0


def test_auto_improver_smoke_run(tmp_path):
    root = Path(__file__).resolve().parents[1]
    improver = AutoImprover(root)
    report = improver.run(
        rounds=1,
        candidates_per_round=3,
        seeds=[3],
        episodes_per_seed=50,
        final_eval_seeds=[3],
        final_eval_episodes=60,
        apply_best=False,
    )

    assert isinstance(report["best_candidate"]["full_minus_frozen"], float)
    assert isinstance(report["best_candidate"]["objective"], float)
    assert Path(report["final_validation_path"]).exists()
