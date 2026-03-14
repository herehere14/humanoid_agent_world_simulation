from __future__ import annotations

from pathlib import Path

from prompt_forest.backend.mock import MockLLMBackend
from prompt_forest.backend.openai_chat import OpenAIChatBackend
from prompt_forest.cli import (
    _base_model_comparison,
    _build_primary_backend_from_args,
    _build_split_debug_panel,
    _clone_backend_for_compare,
    _render_direct_prompt,
    build_parser,
)
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


def test_build_split_debug_panel_includes_comparison_and_trace(tmp_path):
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "default.json")
    cfg.artifacts_dir = str(tmp_path / "artifacts")
    engine = PromptForestEngine(config=cfg, backend=MockLLMBackend(seed=13))

    result = engine.run_task(
        text="Audit a rollout plan for contradictions and confidence.",
        task_type="code",
        metadata={"expected_keywords": ["contradiction", "confidence"], "required_substrings": ["confidence"]},
    )
    comparison = _base_model_comparison(engine, result)

    panel = _build_split_debug_panel(
        result,
        comparison,
        visibility="full",
        top_branches=4,
        task_type="code",
        compare_enabled=True,
        status="Running latest turn.",
    )

    assert "Session" in panel
    assert "Base Model" in panel
    assert "Base vs Adaptive" in panel
    assert "[compare] adaptive_reward=" in panel
    assert "[routing] path=" in panel
    assert "[optimizer] task_baseline=" in panel


def test_build_parser_accepts_split_view_flag():
    parser = build_parser()

    args = parser.parse_args(["chat", "--split-view", "--compare-base"])

    assert args.command == "chat"
    assert args.split_view is True
    assert args.compare_base is True


def test_build_primary_backend_from_args_uses_openai_backend():
    parser = build_parser()
    args = parser.parse_args(
        [
            "chat",
            "--model",
            "gpt-4.1-mini",
            "--api-key-env",
            "OPENAI_API_KEY",
            "--api-mode",
            "responses",
            "--reasoning-effort",
            "medium",
        ]
    )

    backend = _build_primary_backend_from_args(args)

    assert isinstance(backend, OpenAIChatBackend)
    assert backend.model == "gpt-4.1-mini"
    assert backend.api_key_env == "OPENAI_API_KEY"
    assert backend.api_mode == "responses"
    assert backend.reasoning_effort == "medium"
