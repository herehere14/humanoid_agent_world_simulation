from __future__ import annotations

from pathlib import Path

from prompt_forest.experiments.codex_routing_divergence_validation import CodexRoutingDivergenceValidator
from prompt_forest.types import TaskInput


def _run(task_id: str, *, aspect: str, branch: str, reward: float) -> dict:
    return {
        "task_id": task_id,
        "task_type": "planning" if aspect.startswith("planning") else "general",
        "aspect": aspect,
        "selected_branch": branch,
        "activated_paths": [["planner", branch]],
        "objective_metrics": {"hybrid_verifier_reward": reward},
    }


def test_holdout_divergence_summary_counts_branch_changes():
    summary = CodexRoutingDivergenceValidator._holdout_divergence_summary(  # noqa: SLF001
        [
            _run("t1", aspect="planning_risk", branch="planner_risk_allocator", reward=0.8),
            _run("t2", aspect="planning_timeline", branch="planner_timeline_optimizer", reward=0.75),
        ],
        [
            _run("t1", aspect="planning_risk", branch="planner_timeline_optimizer", reward=0.7),
            _run("t2", aspect="planning_timeline", branch="planner_timeline_optimizer", reward=0.75),
        ],
    )

    assert summary["holdout_tasks"] == 2
    assert summary["selected_branch_diff_count"] == 1
    assert summary["selected_branch_diff_rate"] == 0.5
    assert summary["mean_reward_delta_on_divergent"] == 0.1
    assert summary["divergent_tasks"][0]["task_id"] == "t1"


def test_aspect_summary_groups_rewards_and_branch_counts():
    summary = CodexRoutingDivergenceValidator._aspect_summary(  # noqa: SLF001
        [
            _run("t1", aspect="planning_risk", branch="planner_risk_allocator", reward=0.8),
            _run("t2", aspect="planning_risk", branch="planner_risk_allocator", reward=0.7),
            _run("t3", aspect="planning_timeline", branch="planner_timeline_optimizer", reward=0.78),
        ],
        [
            _run("t1", aspect="planning_risk", branch="planner_timeline_optimizer", reward=0.72),
            _run("t2", aspect="planning_risk", branch="planner_timeline_optimizer", reward=0.68),
            _run("t3", aspect="planning_timeline", branch="planner_timeline_optimizer", reward=0.78),
        ],
    )

    assert summary["planning_risk"]["n"] == 2
    assert summary["planning_risk"]["branch_diff_count"] == 2
    assert summary["planning_risk"]["adaptive_branch_counts"]["planner_risk_allocator"] == 2
    assert summary["planning_timeline"]["adaptive_minus_frozen"] == 0.0


def test_validator_can_be_constructed():
    root = Path(__file__).resolve().parents[1]
    validator = CodexRoutingDivergenceValidator(root)
    assert validator.root_dir == root


def test_forced_parent_and_children_can_be_inferred_from_aspect():
    parent, children = CodexRoutingDivergenceValidator._forced_parent_and_children(  # noqa: SLF001
        TaskInput(
            task_id="t4",
            text="Audit consistency",
            task_type="general",
            metadata={"aspect": "general_consistency_audit"},
        )
    )

    assert parent == "verification"
    assert children == ["verification_constraint_checker", "verification_consistency_auditor"]
