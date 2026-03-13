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


class _FakeAdvisor:
    def is_enabled(self) -> bool:
        return True

    def advise(self, task, route, signal, branches):
        return {
            "branch_directives": [
                {"branch_name": "planner", "extra_weight_delta": 0.1, "rewrite_hint": "bad", "confidence": 0.95}
            ],
            "candidate_proposals": [
                {
                    "base_name": "advisor_local_candidate",
                    "capability_tag": "advisor_local_cap",
                    "purpose": "Advisor proposed local branch",
                    "prompt_template": "Task={task}; Type={task_type}; Context={context}",
                    "parent_hint": "planner",
                }
            ],
        }


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
    event = optimizer.optimize(TaskInput("1", "task", "math"), route, signal, branches, memory)

    assert branches["analytical"].state.weight > before["analytical"]
    assert branches["verification"].state.weight > before["verification"]
    assert branches["planner"].state.weight == before["planner"]
    assert "analytical" in event.update_details
    assert event.update_details["analytical"]["status_before"] == "active"
    assert event.update_details["analytical"]["status_after"] == "active"
    assert event.update_details["analytical"]["new_weight"] > event.update_details["analytical"]["old_weight"]


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


def test_optimizer_can_spawn_multiple_candidates_in_one_event(tmp_path):
    branches = create_default_branches()
    memory = MemoryStore(MemoryConfig(), memory_path=tmp_path / "m.jsonl")
    optimizer = OptimizerAgent(
        OptimizerConfig(
            candidate_failure_trigger=3,
            candidate_trial_episodes=2,
            learning_rate=0.1,
            candidate_spawn_per_event=2,
            max_active_candidates=6,
            max_active_branches=40,
        )
    )

    for i in range(4):
        memory.add(
            MemoryRecord(
                task_id=f"mx{i}",
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
    event = optimizer.optimize(TaskInput("m", "task", "general"), route, signal, branches, memory)

    assert len(event.created_candidates) >= 2
    for name in event.created_candidates:
        assert branches[name].state.status.value == "candidate"
        assert branches[name].state.trial_remaining == 2


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


def test_advantage_updates_shrink_as_task_baseline_rises(tmp_path):
    branches = create_default_branches()
    memory = MemoryStore(MemoryConfig(), memory_path=tmp_path / "m.jsonl")
    optimizer = OptimizerAgent(OptimizerConfig(learning_rate=0.2, advantage_baseline_beta=0.3))

    route = RoutingDecision(task_type="math", activated_branches=["analytical"], branch_scores={})
    signal = _signal(["analytical"], reward=0.8, reason="medium_quality")

    w0 = branches["analytical"].state.weight
    optimizer.optimize(TaskInput("1", "task", "math"), route, signal, branches, memory)
    w1 = branches["analytical"].state.weight
    optimizer.optimize(TaskInput("2", "task", "math"), route, signal, branches, memory)
    w2 = branches["analytical"].state.weight

    first_gain = w1 - w0
    second_gain = w2 - w1
    assert first_gain > second_gain


def test_candidate_trial_extends_near_promotion_threshold(tmp_path):
    branches = create_default_branches()
    branches["candidate_x"] = make_candidate_branch(
        name="candidate_x",
        purpose="extra verifier",
        prompt_template="Task {task}",
        trial_episodes=1,
    )
    memory = MemoryStore(MemoryConfig(), memory_path=tmp_path / "m.jsonl")
    optimizer = OptimizerAgent(
        OptimizerConfig(
            candidate_failure_trigger=99,
            candidate_trial_episodes=1,
            candidate_promote_threshold=0.62,
            candidate_neutral_band=0.05,
            candidate_extension_episodes=2,
            candidate_max_extensions=1,
            learning_rate=0.1,
        )
    )

    route = RoutingDecision(task_type="general", activated_branches=["candidate_x"], branch_scores={})
    signal = _signal(["candidate_x"], reward=0.6, reason="medium_quality")
    optimizer.optimize(TaskInput("1", "x", "general"), route, signal, branches, memory)

    assert branches["candidate_x"].state.status.value == "candidate"
    assert branches["candidate_x"].state.trial_remaining == 2


def test_advisor_directives_are_scoped_to_active_path_and_parent_is_local(tmp_path):
    branches = create_default_branches()
    memory = MemoryStore(MemoryConfig(), memory_path=tmp_path / "m.jsonl")
    optimizer = OptimizerAgent(
        OptimizerConfig(
            learning_rate=0.1,
            candidate_failure_trigger=3,
            candidate_spawn_per_event=1,
            max_active_candidates=6,
            max_active_branches=40,
        ),
        advisor=_FakeAdvisor(),
    )

    for i in range(4):
        memory.add(
            MemoryRecord(
                task_id=f"adv{i}",
                task_type="math",
                input_text="x",
                activated_branches=["analytical", "verification"],
                branch_outputs={},
                selected_branch="analytical",
                selected_output="",
                reward_score=0.2,
                failure_reason="rule_miss|required_substring:0/1",
                confidence=0.4,
                useful_patterns=[],
            )
        )

    route = RoutingDecision(task_type="math", activated_branches=["analytical", "verification"], branch_scores={})
    signal = _signal(["analytical", "verification"], reward=0.2, reason="rule_miss|required_substring:0/1")
    planner_before = branches["planner"].state.weight

    event = optimizer.optimize(TaskInput("1", "task", "math"), route, signal, branches, memory)

    # Non-active branch cannot be modified by advisor directives.
    assert branches["planner"].state.weight == planner_before
    # Candidate proposal with non-local parent gets forced to route leaf parent.
    assert event.created_candidates
    created = event.created_candidates[0]
    assert branches[created].state.metadata.get("parent_hint") == "verification"
