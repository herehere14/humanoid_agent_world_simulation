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

    def is_proposal_only(self) -> bool:
        return False

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


def test_optimizer_updates_only_selected_leaf_branch(tmp_path):
    branches = create_default_branches()
    memory = MemoryStore(MemoryConfig(), memory_path=tmp_path / "m.jsonl")
    optimizer = OptimizerAgent(OptimizerConfig(learning_rate=0.2))

    route = RoutingDecision(task_type="math", activated_branches=["analytical", "verification"], branch_scores={})
    signal = _signal(["analytical", "verification"], reward=0.9, reason="high_quality")

    before = {k: v.state.weight for k, v in branches.items()}
    event = optimizer.optimize(TaskInput("1", "task", "math"), route, signal, branches, memory)

    assert branches["analytical"].state.weight > before["analytical"]
    assert branches["verification"].state.weight == before["verification"]
    assert branches["planner"].state.weight == before["planner"]
    assert "analytical" in event.update_details
    assert "verification" not in event.update_details
    assert event.update_details["analytical"]["status_before"] == "active"
    assert event.update_details["analytical"]["status_after"] == "active"
    assert event.update_details["analytical"]["new_weight"] > event.update_details["analytical"]["old_weight"]


def test_optimizer_blocks_middling_positive_weight_update(tmp_path):
    branches = create_default_branches()
    memory = MemoryStore(MemoryConfig(), memory_path=tmp_path / "m.jsonl")
    optimizer = OptimizerAgent(OptimizerConfig(learning_rate=0.2))

    route = RoutingDecision(task_type="math", activated_branches=["analytical"], branch_scores={})
    signal = _signal(["analytical"], reward=0.7, reason="medium_quality")
    old_weight = branches["analytical"].state.weight

    event = optimizer.optimize(TaskInput("1", "task", "math"), route, signal, branches, memory)

    assert branches["analytical"].state.weight == old_weight
    assert event.update_details["analytical"]["weight_update_allowed"] is False
    assert event.update_details["analytical"]["weight_update_block_reason"] == "reward_below_floor"


def test_optimizer_records_selected_branch_execution_hint(tmp_path):
    branches = create_default_branches()
    memory = MemoryStore(MemoryConfig(), memory_path=tmp_path / "m.jsonl")
    optimizer = OptimizerAgent(OptimizerConfig(learning_rate=0.2))

    route = RoutingDecision(
        task_type="general",
        activated_branches=["analytical"],
        branch_scores={},
    )
    signal = EvaluationSignal(
        reward_score=0.86,
        confidence=0.8,
        selected_branch="analytical",
        selected_output=(
            "## Risk Register\n"
            "| Risk | Owner | Mitigation | Rollback |\n"
            "| --- | --- | --- | --- |\n"
            "Confidence: 0.82"
        ),
        failure_reason="",
        suggested_improvement_direction="preserve_success_pattern",
        branch_feedback={
            "analytical": BranchFeedback(
                branch_name="analytical",
                reward=0.86,
                confidence=0.8,
                failure_reason="",
                suggested_improvement_direction="preserve_success_pattern",
            ),
        },
    )
    task = TaskInput(
            "1",
            "Create a launch risk register with mitigation, owner, rollback, and confidence.",
            "general",
            metadata={"required_substrings": ["owner", "rollback", "confidence"]},
        )

    optimizer.optimize(task, route, signal, branches, memory)

    hint = str(branches["analytical"].state.metadata.get("adaptive_execution_hint", ""))
    assert "owner" in hint.lower()
    assert "confidence" in hint.lower()


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


def test_branch_specific_baseline_reduces_repeated_delta_even_with_fixed_task_baseline(tmp_path):
    route = RoutingDecision(task_type="math", activated_branches=["analytical"], branch_scores={})
    signal = _signal(["analytical"], reward=0.8, reason="medium_quality")

    # No branch-specific baseline mixing: repeated gains stay roughly constant.
    branches_plain = create_default_branches()
    memory_plain = MemoryStore(MemoryConfig(), memory_path=tmp_path / "m_plain.jsonl")
    optimizer_plain = OptimizerAgent(
        OptimizerConfig(
            learning_rate=0.2,
            weight_decay=0.0,
            advantage_baseline_beta=0.0,
            branch_advantage_mix=0.0,
            update_acceptance_min_gain=-1.0,
        )
    )
    p0 = branches_plain["analytical"].state.weight
    optimizer_plain.optimize(TaskInput("1", "task", "math"), route, signal, branches_plain, memory_plain)
    p1 = branches_plain["analytical"].state.weight
    optimizer_plain.optimize(TaskInput("2", "task", "math"), route, signal, branches_plain, memory_plain)
    p2 = branches_plain["analytical"].state.weight

    # Full branch-specific baseline mixing: second gain should shrink due to raised branch baseline.
    branches_mixed = create_default_branches()
    memory_mixed = MemoryStore(MemoryConfig(), memory_path=tmp_path / "m_mixed.jsonl")
    optimizer_mixed = OptimizerAgent(
        OptimizerConfig(
            learning_rate=0.2,
            weight_decay=0.0,
            advantage_baseline_beta=0.0,
            branch_advantage_mix=1.0,
            branch_baseline_beta=0.5,
            update_acceptance_min_gain=-1.0,
        )
    )
    m0 = branches_mixed["analytical"].state.weight
    optimizer_mixed.optimize(TaskInput("1", "task", "math"), route, signal, branches_mixed, memory_mixed)
    m1 = branches_mixed["analytical"].state.weight
    optimizer_mixed.optimize(TaskInput("2", "task", "math"), route, signal, branches_mixed, memory_mixed)
    m2 = branches_mixed["analytical"].state.weight

    plain_gain1 = p1 - p0
    plain_gain2 = p2 - p1
    mixed_gain1 = m1 - m0
    mixed_gain2 = m2 - m1

    assert abs(plain_gain1 - plain_gain2) < 1e-6
    assert mixed_gain1 > mixed_gain2


def test_llm_variance_scaling_dampens_updates_under_noisy_history(tmp_path):
    route = RoutingDecision(task_type="math", activated_branches=["analytical"], branch_scores={})
    signal = _signal(["analytical"], reward=0.8, reason="medium_quality")
    task = TaskInput("live-1", "task", "math", metadata={"llm_runtime_active": True, "user_id": "global"})

    cfg = OptimizerConfig(
        learning_rate=0.2,
        weight_decay=0.0,
        advantage_baseline_beta=0.0,
        branch_advantage_mix=0.0,
        llm_variance_sensitivity=8.0,
        llm_min_variance_scale=0.4,
        update_acceptance_min_gain=-1.0,
    )

    stable_branches = create_default_branches()
    stable_memory = MemoryStore(MemoryConfig(recency_decay=1.0), memory_path=tmp_path / "m_stable.jsonl")
    for i in range(8):
        stable_memory.add(
            MemoryRecord(
                task_id=f"s-{i}",
                task_type="math",
                input_text="x",
                activated_branches=["analytical"],
                branch_outputs={},
                selected_branch="analytical",
                selected_output="",
                reward_score=0.75,
                failure_reason="",
                confidence=0.7,
                useful_patterns=[],
            )
        )

    noisy_branches = create_default_branches()
    noisy_memory = MemoryStore(MemoryConfig(recency_decay=1.0), memory_path=tmp_path / "m_noisy.jsonl")
    for i, reward in enumerate([0.1, 0.9, 0.2, 0.8, 0.15, 0.85, 0.25, 0.75]):
        noisy_memory.add(
            MemoryRecord(
                task_id=f"n-{i}",
                task_type="math",
                input_text="x",
                activated_branches=["analytical"],
                branch_outputs={},
                selected_branch="analytical",
                selected_output="",
                reward_score=reward,
                failure_reason="",
                confidence=0.7,
                useful_patterns=[],
            )
        )

    stable_opt = OptimizerAgent(cfg)
    noisy_opt = OptimizerAgent(cfg)

    s0 = stable_branches["analytical"].state.weight
    stable_event = stable_opt.optimize(task, route, signal, stable_branches, stable_memory)
    s1 = stable_branches["analytical"].state.weight

    n0 = noisy_branches["analytical"].state.weight
    noisy_event = noisy_opt.optimize(task, route, signal, noisy_branches, noisy_memory)
    n1 = noisy_branches["analytical"].state.weight

    stable_gain = s1 - s0
    noisy_gain = n1 - n0

    assert noisy_gain < stable_gain
    assert noisy_event.update_details["analytical"]["variance_scale"] < stable_event.update_details["analytical"]["variance_scale"]


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


def test_rewrite_requires_streak_and_respects_cooldown(tmp_path):
    branches = create_default_branches()
    memory = MemoryStore(MemoryConfig(), memory_path=tmp_path / "m.jsonl")
    optimizer = OptimizerAgent(
        OptimizerConfig(
            learning_rate=0.1,
            prompt_rewrite_threshold=0.7,
            rewrite_failure_streak_trigger=3,
            rewrite_cooldown_episodes=4,
        )
    )
    route = RoutingDecision(task_type="math", activated_branches=["analytical"], branch_scores={})
    low = _signal(["analytical"], reward=0.2, reason="low_quality")

    optimizer.optimize(TaskInput("1", "task", "math"), route, low, branches, memory)
    optimizer.optimize(TaskInput("2", "task", "math"), route, low, branches, memory)
    assert branches["analytical"].state.rewrite_history == []

    optimizer.optimize(TaskInput("3", "task", "math"), route, low, branches, memory)
    assert len(branches["analytical"].state.rewrite_history) == 1

    optimizer.optimize(TaskInput("4", "task", "math"), route, low, branches, memory)
    assert len(branches["analytical"].state.rewrite_history) == 1


def test_acceptance_gate_rolls_back_low_gain_updates(tmp_path):
    branches = create_default_branches()
    memory = MemoryStore(MemoryConfig(), memory_path=tmp_path / "m.jsonl")
    for i in range(6):
        memory.add(
            MemoryRecord(
                task_id=f"a{i}",
                task_type="math",
                input_text="x",
                activated_branches=["analytical"],
                branch_outputs={},
                selected_branch="analytical",
                selected_output="",
                reward_score=0.6,
                failure_reason="",
                confidence=0.5,
                useful_patterns=[],
                branch_rewards={"analytical": 0.6},
            )
        )

    optimizer = OptimizerAgent(
        OptimizerConfig(
            learning_rate=0.01,
            prompt_rewrite_threshold=0.1,
            update_acceptance_min_gain=0.05,
            update_acceptance_window=6,
        )
    )
    route = RoutingDecision(task_type="math", activated_branches=["analytical"], branch_scores={})
    signal = _signal(["analytical"], reward=0.8, reason="ok")
    old_weight = branches["analytical"].state.weight

    event = optimizer.optimize(
        TaskInput("now", "task", "math", metadata={"user_id": "global"}),
        route,
        signal,
        branches,
        memory,
    )

    assert branches["analytical"].state.weight == old_weight
    assert event.update_details["analytical"]["update_accepted"] is False


def test_candidate_creation_requires_clustered_failures(tmp_path):
    branches = create_default_branches()
    memory = MemoryStore(MemoryConfig(), memory_path=tmp_path / "m.jsonl")
    optimizer = OptimizerAgent(
        OptimizerConfig(candidate_failure_trigger=3, candidate_trial_episodes=2, learning_rate=0.1)
    )

    # Single failure pattern at exactly threshold should not pass clustered-failure gate.
    for i in range(3):
        memory.add(
            MemoryRecord(
                task_id=f"c{i}",
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
    assert event.created_candidates == []


def test_candidate_parent_gate_blocks_promotion_without_enough_comparisons(tmp_path):
    branches = create_default_branches()
    branches["candidate_x"] = make_candidate_branch(
        name="candidate_x",
        purpose="extra verifier",
        prompt_template="Task {task}",
        trial_episodes=2,
        metadata={"parent_hint": "verification"},
    )
    memory = MemoryStore(MemoryConfig(), memory_path=tmp_path / "m.jsonl")
    optimizer = OptimizerAgent(
        OptimizerConfig(
            candidate_failure_trigger=99,
            candidate_trial_episodes=2,
            candidate_promote_threshold=0.6,
            candidate_parent_min_comparisons=3,
            learning_rate=0.1,
        )
    )

    feedback = {
        "verification": BranchFeedback(
            branch_name="verification",
            reward=0.5,
            confidence=0.7,
            failure_reason="ok",
            suggested_improvement_direction="",
        ),
        "candidate_x": BranchFeedback(
            branch_name="candidate_x",
            reward=0.9,
            confidence=0.7,
            failure_reason="ok",
            suggested_improvement_direction="",
        ),
    }
    signal = EvaluationSignal(
        reward_score=0.9,
        confidence=0.7,
        selected_branch="candidate_x",
        selected_output="x",
        failure_reason="ok",
        suggested_improvement_direction="",
        branch_feedback=feedback,
    )
    route = RoutingDecision(task_type="general", activated_branches=["verification", "candidate_x"], branch_scores={})

    optimizer.optimize(TaskInput("1", "x", "general"), route, signal, branches, memory)
    optimizer.optimize(TaskInput("2", "x", "general"), route, signal, branches, memory)

    # Strong rewards but not enough parent comparisons, so promotion is blocked.
    assert branches["candidate_x"].state.status.value != "active"


def test_candidate_parent_gate_requires_positive_reward_gap(tmp_path):
    branches = create_default_branches()
    branches["candidate_x"] = make_candidate_branch(
        name="candidate_x",
        purpose="extra verifier",
        prompt_template="Task {task}",
        trial_episodes=3,
        metadata={"parent_hint": "verification"},
    )
    memory = MemoryStore(MemoryConfig(), memory_path=tmp_path / "m.jsonl")
    optimizer = OptimizerAgent(
        OptimizerConfig(
            candidate_failure_trigger=99,
            candidate_trial_episodes=3,
            candidate_promote_threshold=0.6,
            candidate_parent_min_comparisons=3,
            candidate_parent_win_rate_threshold=0.6,
            candidate_parent_min_reward_gap=0.02,
            learning_rate=0.1,
        )
    )

    route = RoutingDecision(task_type="general", activated_branches=["verification", "candidate_x"], branch_scores={})

    for i in range(3):
        feedback = {
            "verification": BranchFeedback(
                branch_name="verification",
                reward=0.74,
                confidence=0.7,
                failure_reason="ok",
                suggested_improvement_direction="",
            ),
            "candidate_x": BranchFeedback(
                branch_name="candidate_x",
                reward=0.75,
                confidence=0.7,
                failure_reason="ok",
                suggested_improvement_direction="",
            ),
        }
        signal = EvaluationSignal(
            reward_score=0.75,
            confidence=0.7,
            selected_branch="candidate_x",
            selected_output="x",
            failure_reason="ok",
            suggested_improvement_direction="",
            branch_feedback=feedback,
        )
        optimizer.optimize(TaskInput(str(i), "x", "general"), route, signal, branches, memory)

    # Candidate is slightly better but does not clear minimum reward-gap gate.
    assert branches["candidate_x"].state.status.value != "active"


def test_acceptance_runner_can_block_weight_update(tmp_path):
    branches = create_default_branches()
    memory = MemoryStore(MemoryConfig(), memory_path=tmp_path / "m.jsonl")
    optimizer = OptimizerAgent(OptimizerConfig(learning_rate=0.2, update_acceptance_min_gain=-1.0))

    route = RoutingDecision(task_type="math", activated_branches=["analytical"], branch_scores={})
    signal = _signal(["analytical"], reward=0.9, reason="high_quality")
    old_weight = branches["analytical"].state.weight

    event = optimizer.optimize(
        TaskInput("1", "task", "math", metadata={"user_id": "global"}),
        route,
        signal,
        branches,
        memory,
        acceptance_runner=lambda **_: False,
    )

    assert branches["analytical"].state.weight == old_weight
    assert event.update_details["analytical"]["update_accepted"] is False


def test_proposal_only_advisor_does_not_add_weight_delta(tmp_path):
    class _ProposalOnlyAdvisor:
        def is_enabled(self) -> bool:
            return True

        def is_proposal_only(self) -> bool:
            return True

        def advise(self, task, route, signal, branches):
            return {
                "branch_directives": [
                    {
                        "branch_name": "analytical",
                        "extra_weight_delta": 0.2,
                        "rewrite_hint": "tighten format",
                        "confidence": 0.99,
                    }
                ],
                "candidate_proposals": [],
            }

    branches = create_default_branches()
    memory = MemoryStore(MemoryConfig(), memory_path=tmp_path / "m.jsonl")
    optimizer = OptimizerAgent(
        OptimizerConfig(learning_rate=0.0, weight_decay=0.0, prompt_rewrite_threshold=0.0),
        advisor=_ProposalOnlyAdvisor(),
    )
    route = RoutingDecision(task_type="math", activated_branches=["analytical"], branch_scores={})
    signal = _signal(["analytical"], reward=0.5, reason="ok")
    old_weight = branches["analytical"].state.weight

    event = optimizer.optimize(
        TaskInput("1", "task", "math", metadata={"user_id": "global"}),
        route,
        signal,
        branches,
        memory,
    )

    assert branches["analytical"].state.weight == old_weight
    assert event.update_details["analytical"]["advisory_extra_delta"] == 0.0
