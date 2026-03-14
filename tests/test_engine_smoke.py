from __future__ import annotations

from prompt_forest.core.engine import PromptForestEngine
from prompt_forest.types import MemoryRecord, TaskInput


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


def test_engine_augments_branch_context_with_memory_and_adaptive_hints():
    engine = PromptForestEngine()
    task = TaskInput(
        task_id="t1",
        text="Create a launch risk register with mitigation, owner, rollback, and confidence.",
        task_type="planning",
        metadata={"aspect": "planning", "user_id": "test_user", "required_substrings": ["owner", "rollback", "confidence"]},
    )
    branch = engine.branches["planner_risk_allocator"]
    branch.state.metadata["adaptive_execution_hint"] = "Assign explicit owners and end with calibrated confidence."

    engine.memory.add(
        MemoryRecord(
            task_id="m1",
            task_type="planning",
            input_text="Create a launch risk register with mitigation, owner, rollback, and confidence.",
            activated_branches=["planner", "planner_risk_allocator"],
            branch_outputs={"planner_risk_allocator": "## Risk Register\nOwner: PM\nRollback: revert\nConfidence: 0.82"},
            selected_branch="planner_risk_allocator",
            selected_output="## Risk Register\nOwner: PM\nRollback: revert\nConfidence: 0.82",
            reward_score=0.84,
            failure_reason="",
            confidence=0.8,
            useful_patterns=[],
            branch_rewards={"planner_risk_allocator": 0.84},
            selected_path=["planner", "planner_risk_allocator"],
            routing_context_key=engine.memory.routing_context_key_for_task(task, "planner"),
            user_id="test_user",
            task_metadata=dict(task.metadata),
            reward_components={},
        )
    )

    context = engine._augment_context_for_branch(task, branch, "upstream summary")

    assert "upstream summary" in context
    assert "Adaptive branch hint" in context
    assert "Similar success" in context
