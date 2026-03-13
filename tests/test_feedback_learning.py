from __future__ import annotations

from pathlib import Path

from prompt_forest.config import load_config
from prompt_forest.core.engine import PromptForestEngine


def test_feedback_updates_memory_record_and_reward(tmp_path):
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "default.json")
    cfg.artifacts_dir = str(tmp_path / "artifacts")
    engine = PromptForestEngine(config=cfg)

    result = engine.run_task(
        text="Create a migration checklist with confidence.",
        task_type="planning",
        metadata={"user_id": "alice", "expected_keywords": ["checklist", "confidence"], "required_substrings": ["confidence"]},
    )
    task_id = result["task"]["task_id"]
    old_reward = result["evaluation_signal"]["reward_score"]

    fb = engine.apply_feedback(
        task_id=task_id,
        score=0.1,
        accepted=False,
        corrected_answer="Use phased rollout, rollback checks, and confidence calibration.",
        feedback_text="Missing practical constraints.",
        user_id="alice",
    )
    assert fb["ok"] is True
    assert fb["new_reward"] < old_reward

    record = engine.memory.retrieve_similar("planning", user_id="alice")[-1]
    assert record.feedback_score is not None
    assert record.accepted is False
    assert "rollback" in record.corrected_answer.lower()
    assert record.user_id == "alice"


def test_user_specific_bias_is_distinct_from_global(tmp_path):
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "default.json")
    cfg.artifacts_dir = str(tmp_path / "artifacts")
    cfg.memory.user_bias_mix = 0.8
    engine = PromptForestEngine(config=cfg)

    # user alice strongly rewards planner path
    for _ in range(6):
        result = engine.run_task(
            text="Plan a release timeline with confidence.",
            task_type="planning",
            metadata={"user_id": "alice", "expected_keywords": ["plan", "timeline", "confidence"], "required_substrings": ["confidence"]},
        )
        engine.apply_feedback(task_id=result["task"]["task_id"], score=1.0, accepted=True, user_id="alice")

    # user bob strongly dislikes planner path
    for _ in range(6):
        result = engine.run_task(
            text="Plan a release timeline with confidence.",
            task_type="planning",
            metadata={"user_id": "bob", "expected_keywords": ["plan", "timeline", "confidence"], "required_substrings": ["confidence"]},
        )
        engine.apply_feedback(task_id=result["task"]["task_id"], score=0.0, accepted=False, user_id="bob")

    bias_alice = engine.memory.branch_success_bias("planning", user_id="alice")
    bias_bob = engine.memory.branch_success_bias("planning", user_id="bob")

    # Planner subtree should be meaningfully separated by user-specific feedback.
    planner_like_a = bias_alice.get("planner", 0.0) + bias_alice.get("planner_timeline_optimizer", 0.0)
    planner_like_b = bias_bob.get("planner", 0.0) + bias_bob.get("planner_timeline_optimizer", 0.0)
    assert planner_like_a > planner_like_b
