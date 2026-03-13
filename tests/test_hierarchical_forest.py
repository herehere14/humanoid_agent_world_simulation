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
    paths = result["routing"].get("activated_paths", [])
    assert len(path) >= 2
    assert path[0] in {"analytical", "planner", "retrieval", "critique", "verification", "creative"}
    assert paths
    assert len(paths) >= 1
    assert all(len(p) >= 2 for p in paths)


def test_path_backprop_updates_multiple_layers():
    engine = PromptForestEngine()
    result = engine.run_task(
        text="Check factual consistency with evidence",
        task_type="factual",
        metadata={"expected_keywords": ["evidence", "factual", "confidence"], "required_substrings": ["confidence"]},
    )

    path = result["routing"]["activated_branches"]
    updates = result["optimization"]["updated_weights"]

    assert len(path) >= 2
    assert any(branch in updates for branch in path[:2])
