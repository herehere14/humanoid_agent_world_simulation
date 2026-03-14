from __future__ import annotations

from pathlib import Path

from prompt_forest.config import apply_latency_profile, load_config
from prompt_forest.experiments.latency_validation import LatencyValidator


def test_apply_latency_profile_fast_disables_expensive_features():
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "default.json")

    fast = apply_latency_profile(cfg, "fast")

    assert fast.router.top_k == 1
    assert fast.router.min_candidates == 1
    assert fast.router.sibling_probe_top_n == 0
    assert fast.composer.enabled is False
    assert fast.execution_adaptation.enabled is False
    assert fast.agent_runtimes.evaluator.enabled is False
    assert fast.agent_runtimes.optimizer.enabled is False


def test_latency_validator_smoke_generates_report(tmp_path):
    validator = LatencyValidator(Path(__file__).resolve().parents[1])

    report = validator.run(
        dataset_path=Path(__file__).resolve().parents[1] / "examples" / "demo_tasks.json",
        rounds=1,
        output_subdir=str(tmp_path / "latency_validation"),
        report_prefix="smoke_latency_report",
    )

    assert report["mode"] == "mock_backend"
    assert "cold" in report["aggregate"]
    assert "session" in report["aggregate"]
    assert report["aggregate"]["cold"]["fast"]["mean_calls"]["primary_backend_calls"] <= report["aggregate"]["cold"]["full"]["mean_calls"]["primary_backend_calls"]
    assert report["report_paths"]["json"].endswith("smoke_latency_report.json")
