from __future__ import annotations

from pathlib import Path

from prompt_forest.config import load_config
from prompt_forest.core.engine import PromptForestEngine
from prompt_forest.observability.trace import format_comparison_trace, format_turn_trace


def test_turn_trace_includes_evaluator_and_optimizer_sections(tmp_path):
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "default.json")
    cfg.artifacts_dir = str(tmp_path / "artifacts")
    engine = PromptForestEngine(config=cfg)

    payload = engine.run_task(
        text="Plan a migration timeline with confidence statement.",
        task_type="planning",
        metadata={
            "expected_keywords": ["plan", "timeline", "confidence"],
            "required_substrings": ["confidence"],
        },
    )
    trace = format_turn_trace(payload, visibility="full", top_branches=4)

    assert "[routing] path=" in trace
    assert "[routing] sibling_decisions:" in trace
    assert "[evaluator] selected=" in trace
    assert "[evaluator] reward_components=" in trace
    assert "[optimizer] task_baseline=" in trace
    assert "[optimizer] branch_updates:" in trace


def test_turn_trace_eval_mode_omits_optimizer_section():
    payload = {
        "task": {"task_id": "abc123", "task_type": "math"},
        "routing": {
            "task_type": "math",
            "activated_branches": ["analytical", "analytical_symbolic_solver"],
            "branch_scores": {"analytical": 1.2, "planner": 0.8},
        },
        "branch_scores": {"analytical": {"reward": 0.7, "reason": "medium_quality"}},
        "evaluation_signal": {
            "selected_branch": "analytical",
            "reward_score": 0.7,
            "confidence": 0.64,
            "failure_reason": "",
            "branch_feedback": {
                "analytical": {
                    "reward": 0.7,
                    "confidence": 0.62,
                    "failure_reason": "",
                    "suggested_improvement_direction": "preserve_success_pattern",
                }
            },
        },
        "optimization": {"task_baseline_before": 0.5, "task_baseline_after": 0.53},
    }

    trace = format_turn_trace(payload, visibility="eval", top_branches=2)

    assert "[evaluator] selected=analytical" in trace
    assert "[optimizer]" not in trace


def test_format_comparison_trace_shows_delta_and_breakdown():
    trace = format_comparison_trace(
        {
            "adaptive_system": {
                "selected_branch": "planner_risk_allocator",
                "objective_metrics": {
                    "hybrid_verifier_reward": 0.81,
                    "keyword_score": 1.0,
                    "rule_score": 1.0,
                    "external_verifier_score": 0.67,
                    "hybrid_verifier_reason": "medium_quality|rules_passed",
                },
            },
            "base_model": {
                "selected_branch": "base_model_direct",
                "objective_metrics": {
                    "hybrid_verifier_reward": 0.65,
                    "keyword_score": 0.5,
                    "rule_score": 0.5,
                    "external_verifier_score": 0.33,
                    "hybrid_verifier_reason": "low_quality|rule_miss:confidence",
                },
            },
            "delta": {"hybrid_verifier_reward": 0.16},
            "winner": "adaptive_system",
        }
    )

    assert "[compare] adaptive_reward=0.810" in trace
    assert "winner=adaptive_system" in trace
    assert "adaptive_breakdown=keyword:1.000" in trace
    assert "base_breakdown=keyword:0.500" in trace
