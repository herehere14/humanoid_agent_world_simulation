from __future__ import annotations

from prompt_forest.agents.optimizer_agent import OptimizerAgent
from prompt_forest.branches.library import create_default_branches, make_candidate_branch
from prompt_forest.config import MemoryConfig, OptimizerConfig
from prompt_forest.memory.store import MemoryStore
from prompt_forest.types import (
    BranchFeedback,
    EvaluationSignal,
    MemoryRecord,
    RoutingDecision,
    TaskInput,
)


def _signal(branches: list[str], reward: float, reason: str = "low_quality") -> EvaluationSignal:
    feedback = {
        b: BranchFeedback(
            branch_name=b,
            reward=reward,
            confidence=0.7,
            failure_reason=reason,
            suggested_improvement_direction="improve_keyword_coverage",
        )
        for b in branches
    }
    return EvaluationSignal(
        reward_score=reward,
        confidence=0.7,
        selected_branch=branches[0],
        selected_output="x",
        failure_reason=reason,
        suggested_improvement_direction="improve_keyword_coverage",
        branch_feedback=feedback,
    )


def test_optimizer_updates_only_active_branches(tmp_path):
    branches = create_default_branches()
    memory = MemoryStore(MemoryConfig(), memory_path=tmp_path / "m.jsonl")
    optimizer = OptimizerAgent(OptimizerConfig(learning_rate=0.2))

    route = RoutingDecision(task_type="math", activated_branches=["analytical", "verification"], branch_scores={})
    signal = _signal(["analytical", "verification"], reward=0.9, reason="high_quality")

    before = {k: v.state.weight for k, v in branches.items()}
    optimizer.optimize(TaskInput("1", "task", "math"), route, signal, branches, memory)

    assert branches["analytical"].state.weight > before["analytical"]
    assert branches["verification"].state.weight > before["verification"]
    assert branches["planner"].state.weight == before["planner"]


def test_candidate_creation_after_repeated_failures(tmp_path):
    branches = create_default_branches()
    memory = MemoryStore(MemoryConfig(), memory_path=tmp_path / "m.jsonl")
    optimizer = OptimizerAgent(
        OptimizerConfig(candidate_failure_trigger=3, candidate_trial_episodes=2, learning_rate=0.1)
    )

    for i in range(4):
        memory.add(
            MemoryRecord(
                task_id=f"f{i}",
                task_type="general",
                input_text="x",
                activated_branches=["analytical", "retrieval"],
                branch_outputs={},
                selected_branch="analytical",
                selected_output="",
                reward_score=0.2,
                failure_reason="low_quality|keyword_coverage:0/3",
                confidence=0.4,
                useful_patterns=[],
            )
        )

    route = RoutingDecision(task_type="general", activated_branches=["analytical", "retrieval"], branch_scores={})
    signal = _signal(["analytical", "retrieval"], reward=0.2, reason="low_quality|keyword_coverage:0/3")

    event = optimizer.optimize(TaskInput("x", "task", "general"), route, signal, branches, memory)

    assert event.created_candidates
    created_name = event.created_candidates[0]
    assert branches[created_name].state.trial_remaining == 2


def test_candidate_promotes_after_successful_trial(tmp_path):
    branches = create_default_branches()
    branches["candidate_x"] = make_candidate_branch(
        name="candidate_x",
        purpose="extra verifier",
        prompt_template="Task {task}",
        trial_episodes=2,
    )
    memory = MemoryStore(MemoryConfig(), memory_path=tmp_path / "m.jsonl")
    optimizer = OptimizerAgent(
        OptimizerConfig(
            candidate_failure_trigger=99,
            candidate_trial_episodes=2,
            candidate_promote_threshold=0.6,
            learning_rate=0.1,
        )
    )

    route = RoutingDecision(task_type="general", activated_branches=["candidate_x"], branch_scores={})
    signal = _signal(["candidate_x"], reward=0.9, reason="high_quality")
    optimizer.optimize(TaskInput("1", "x", "general"), route, signal, branches, memory)
    optimizer.optimize(TaskInput("2", "x", "general"), route, signal, branches, memory)

    assert branches["candidate_x"].state.status.value == "active"
