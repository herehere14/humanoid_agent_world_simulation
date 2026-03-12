from __future__ import annotations

from prompt_forest.core.engine import PromptForestEngine


def test_hierarchical_route_is_sequential_path():
    engine = PromptForestEngine()
    result = engine.run_task(
        text="Design a robust plan and verify constraints",
        task_type="planning",
        metadata={"expected_keywords": ["plan", "constraints", "confidence"], "required_substrings": ["confidence"]},
    )

    path = result["routing"]["activated_branches"]
    assert len(path) >= 2
    assert path[0] in {"analytical", "planner", "retrieval", "critique", "verification", "creative"}


def test_path_backprop_updates_multiple_layers():
    engine = PromptForestEngine()
    result = engine.run_task(
        text="Check factual consistency with evidence",
        task_type="factual",
        metadata={"expected_keywords": ["evidence", "factual", "confidence"], "required_substrings": ["confidence"]},
    )

    path = result["routing"]["activated_branches"]
    updates = result["optimization"]["updated_weights"]

    # At least first two layers should be updated in sequential path mode.
    assert len(path) >= 2
    assert path[0] in updates
    assert path[1] in updates
