from __future__ import annotations

from pathlib import Path

from prompt_forest.experiments.hard_slice_validation import HardSliceValidator


def test_hard_slice_validation_reports_target_gate_and_positive_gain():
    root = Path(__file__).resolve().parents[1]
    validator = HardSliceValidator(root)
    report = validator.run(seeds=[3, 5], episodes_per_seed=100)

    agg = report["aggregate"]
    assert "target_relative_gain" in agg
    assert agg["target_relative_gain"] == 0.2
    assert isinstance(agg["full_minus_frozen_holdout"], float)
    assert "passes_target_relative_gain" in agg
