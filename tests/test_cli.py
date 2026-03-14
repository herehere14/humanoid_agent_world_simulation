from __future__ import annotations

from pathlib import Path

from prompt_forest.backend.mock import MockLLMBackend
from prompt_forest.cli import _base_model_comparison, _clone_backend_for_compare, _render_direct_prompt
from prompt_forest.config import load_config
from prompt_forest.core.engine import PromptForestEngine
from prompt_forest.types import TaskInput


def test_clone_backend_for_compare_creates_fresh_mock_backend():
    backend = MockLLMBackend(seed=17)

    cloned = _clone_backend_for_compare(backend)

    assert isinstance(cloned, MockLLMBackend)
    assert cloned is not backend
    assert cloned.seed == backend.seed


def test_render_direct_prompt_mentions_direct_answering():
    task = TaskInput(task_id="t1", text="Plan a migration with confidence.", task_type="planning")

    prompt = _render_direct_prompt(task)

    assert "without any routing or helper branches" in prompt
    assert "Task type: planning" in prompt
    assert "Plan a migration with confidence." in prompt


def test_base_model_comparison_returns_adaptive_and_base_metrics(tmp_path):
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "default.json")
    cfg.artifacts_dir = str(tmp_path / "artifacts")
    engine = PromptForestEngine(config=cfg, backend=MockLLMBackend(seed=11))

    result = engine.run_task(
        text="Plan a migration timeline with confidence.",
        task_type="planning",
        metadata={"expected_keywords": ["plan", "timeline", "confidence"], "required_substrings": ["confidence"]},
    )
    comparison = _base_model_comparison(engine, result)

    assert comparison["adaptive_system"]["selected_branch"] == result["evaluation_signal"]["selected_branch"]
    assert comparison["base_model"]["selected_branch"] == "base_model_direct"
    assert "hybrid_verifier_reward" in comparison["adaptive_system"]["objective_metrics"]
    assert "hybrid_verifier_reward" in comparison["base_model"]["objective_metrics"]
    assert "winner" in comparison
