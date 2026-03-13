from __future__ import annotations

from prompt_forest.core.engine import PromptForestEngine


def test_engine_runs_single_task(tmp_path):
    artifacts = tmp_path / "artifacts"
    engine = PromptForestEngine()
    engine.artifacts_dir = artifacts

    result = engine.run_task(
        text="Calculate derivative of x^2",
        task_type="math",
        metadata={"expected_keywords": ["derivative", "2x"], "required_substrings": ["confidence"]},
    )

    assert result["routing"]["activated_branches"]
    assert 0.0 <= result["evaluation_signal"]["reward_score"] <= 1.0
    assert result["evaluation_signal"]["selected_branch"] in result["routing"]["activated_branches"]
    assert "composer-fusion" in result["evaluation_signal"]["selected_output"]
    assert result["composer"].get("composer_enabled") is True
