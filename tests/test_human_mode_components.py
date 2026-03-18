"""Tests for Human Mode components: router, evaluator, memory, branches."""

from __future__ import annotations

import pytest

from prompt_forest.backend.mock import MockLLMBackend
from prompt_forest.modes.human_mode.branches import (
    create_human_mode_branches,
    create_human_mode_forest,
)
from prompt_forest.modes.human_mode.evaluator import HumanModeEvaluator
from prompt_forest.modes.human_mode.memory import ExperientialMemory, HumanModeMemory
from prompt_forest.modes.human_mode.router import HumanModeRouter
from prompt_forest.state.human_state import DriveConflict, HumanState
from prompt_forest.types import TaskInput


# ── Branches ─────────────────────────────────────────────────────────────────

def test_create_branches_returns_all_expected():
    branches = create_human_mode_branches()
    expected = {
        "reflective_reasoning", "working_memory", "long_term_memory",
        "emotional_modulation", "fear_risk", "ambition_reward",
        "curiosity_exploration", "impulse_response", "empathy_social",
        "moral_evaluation", "self_protection", "self_justification",
        "long_term_goals", "conflict_resolver",
    }
    assert expected.issubset(set(branches.keys()))


def test_create_forest_hierarchy():
    forest = create_human_mode_forest()
    assert len(forest.branches) > 10
    # Category nodes exist
    for cat in ["cognition", "affect", "social", "self_narrative", "meta"]:
        assert cat in forest.branches, f"Missing category node: {cat}"
    # Leaf branches exist under categories
    path = forest.path_to_root("reflective_reasoning")
    assert "cognition" in path


def test_branch_prompt_has_placeholders():
    branches = create_human_mode_branches()
    for name, branch in branches.items():
        template = branch.state.prompt_template
        assert "{task}" in template, f"{name} missing {{task}} placeholder"
        assert "{context}" in template, f"{name} missing {{context}} placeholder"


# ── Router ───────────────────────────────────────────────────────────────────

def test_router_returns_decision_and_conflicts():
    forest = create_human_mode_forest()
    state = HumanState(initial_values={"curiosity": 0.8, "fear": 0.7})
    router = HumanModeRouter(top_k=4, noise_level=0.0)

    task = TaskInput(task_id="t1", text="Explore a new area", task_type="auto")
    decision, conflicts = router.route(task, forest, state)

    assert len(decision.activated_branches) >= 1
    assert decision.task_type != ""
    assert isinstance(decision.branch_scores, dict)


def test_router_high_arousal_favours_fast_branches():
    forest = create_human_mode_forest()
    # High arousal state
    state_high = HumanState(
        initial_values={"stress": 0.8, "curiosity": 0.8, "fear": 0.7, "impulse": 0.7},
        noise_level=0.0,
    )
    router = HumanModeRouter(top_k=6, noise_level=0.0)
    task = TaskInput(task_id="t1", text="Quick decision needed", task_type="auto")
    decision_high, _ = router.route(task, forest, state_high)

    # Low arousal state
    state_low = HumanState(
        initial_values={"stress": 0.1, "curiosity": 0.2, "fear": 0.1, "impulse": 0.1},
        noise_level=0.0,
    )
    decision_low, _ = router.route(task, forest, state_low)

    # impulse_response should score higher under high arousal
    score_impulse_high = decision_high.branch_scores.get("impulse_response", 0)
    score_impulse_low = decision_low.branch_scores.get("impulse_response", 0)
    assert score_impulse_high > score_impulse_low


def test_router_activates_conflict_resolver_on_conflict():
    forest = create_human_mode_forest()
    state = HumanState(
        initial_values={"curiosity": 0.75, "fear": 0.70},
        noise_level=0.0,
    )
    # Force conflict detection
    state.update({})

    router = HumanModeRouter(top_k=4, noise_level=0.0)
    task = TaskInput(task_id="t1", text="Should I explore?", task_type="auto")
    decision, conflicts = router.route(task, forest, state)

    if conflicts:
        # conflict_resolver should have boosted score
        assert "conflict_resolver" in decision.branch_scores


def test_router_cognitive_context_classification():
    forest = create_human_mode_forest()
    router = HumanModeRouter(top_k=4, noise_level=0.0)

    # Threat state
    state = HumanState(initial_values={"stress": 0.8, "fear": 0.8})
    task = TaskInput(task_id="t1", text="Something dangerous", task_type="auto")
    decision, _ = router.route(task, forest, state)
    assert decision.task_type == "threat_response"

    # Exploration state
    state2 = HumanState(initial_values={"curiosity": 0.9, "stress": 0.1, "fear": 0.1})
    decision2, _ = router.route(task, forest, state2)
    assert decision2.task_type == "exploration"


# ── Evaluator ────────────────────────────────────────────────────────────────

def test_evaluator_returns_signal():
    from prompt_forest.aggregator.strategies import AggregationResult
    from prompt_forest.evaluator.judge import BranchScore
    from prompt_forest.types import RoutingDecision

    evaluator = HumanModeEvaluator()
    state = HumanState()

    route = RoutingDecision(
        task_type="general",
        activated_branches=["reflective_reasoning"],
        branch_scores={"reflective_reasoning": 0.8},
    )
    branch_scores = {"reflective_reasoning": BranchScore(0.7, "ok")}
    aggregation = AggregationResult(
        selected_branch="reflective_reasoning",
        selected_output="This is a carefully considered response.",
        notes={},
    )
    task = TaskInput(task_id="t1", text="test", task_type="general")

    signal = evaluator.evaluate(
        task=task, route=route, branch_scores=branch_scores,
        aggregation=aggregation, state=state, conflicts=[],
    )
    assert 0.0 <= signal.reward_score <= 1.0
    assert signal.selected_branch == "reflective_reasoning"


def test_evaluator_conflict_handling_score():
    from prompt_forest.aggregator.strategies import AggregationResult
    from prompt_forest.evaluator.judge import BranchScore
    from prompt_forest.types import RoutingDecision

    evaluator = HumanModeEvaluator()
    state = HumanState()

    route = RoutingDecision(
        task_type="general",
        activated_branches=["reflective_reasoning"],
        branch_scores={"reflective_reasoning": 0.8},
    )
    branch_scores = {"reflective_reasoning": BranchScore(0.7, "ok")}

    # Output that acknowledges conflict
    aggregation_conflict = AggregationResult(
        selected_branch="reflective_reasoning",
        selected_output="On the other hand, there is tension between these approaches. However, we must find a compromise.",
        notes={},
    )
    task = TaskInput(task_id="t1", text="test", task_type="general")
    conflict = DriveConflict(drive_a="curiosity", drive_b="fear", intensity=0.5)

    signal_good = evaluator.evaluate(
        task=task, route=route, branch_scores=branch_scores,
        aggregation=aggregation_conflict, state=state, conflicts=[conflict],
    )

    # Output that ignores conflict
    aggregation_ignore = AggregationResult(
        selected_branch="reflective_reasoning",
        selected_output="The answer is simple and straightforward.",
        notes={},
    )
    signal_bad = evaluator.evaluate(
        task=task, route=route, branch_scores=branch_scores,
        aggregation=aggregation_ignore, state=state, conflicts=[conflict],
    )

    assert signal_good.reward_score >= signal_bad.reward_score


# ── Memory ───────────────────────────────────────────────────────────────────

def test_memory_record_and_count():
    mem = HumanModeMemory(max_memories=100)
    state = HumanState()
    task = TaskInput(task_id="t1", text="test task", task_type="general")

    entry = mem.record(
        event_id="e1", task=task, state=state,
        reward=0.8, selected_branch="reflective_reasoning",
        active_branches=["reflective_reasoning", "curiosity_exploration"],
    )
    assert mem.memory_count == 1
    assert isinstance(entry, ExperientialMemory)
    assert "success" in entry.tags


def test_memory_recall_similar():
    mem = HumanModeMemory(max_memories=100)
    state = HumanState()

    for i in range(5):
        task = TaskInput(task_id=f"t{i}", text=f"task {i}", task_type="general")
        mem.record(
            event_id=f"e{i}", task=task, state=state,
            reward=0.6, selected_branch="reflective_reasoning",
            active_branches=["reflective_reasoning"],
        )

    recalled = mem.recall_similar("general", state, limit=3)
    assert len(recalled) <= 3
    assert all(isinstance(r, ExperientialMemory) for r in recalled)


def test_memory_experiential_bias():
    mem = HumanModeMemory(max_memories=100)
    state = HumanState()

    # Record positive outcomes for branch A
    for i in range(3):
        task = TaskInput(task_id=f"t{i}", text=f"task", task_type="general")
        mem.record(
            event_id=f"e{i}", task=task, state=state,
            reward=0.9, selected_branch="curiosity_exploration",
            active_branches=["curiosity_exploration"],
        )

    # Record negative outcomes for branch B
    for i in range(3, 6):
        task = TaskInput(task_id=f"t{i}", text=f"task", task_type="general")
        mem.record(
            event_id=f"e{i}", task=task, state=state,
            reward=0.1, selected_branch="fear_risk",
            active_branches=["fear_risk"],
        )

    biases = mem.experiential_bias(state)
    # Positive-outcome branch should have positive bias
    assert biases.get("curiosity_exploration", 0) > 0
    # Negative-outcome branch should have negative bias
    assert biases.get("fear_risk", 0) < 0


def test_memory_eviction():
    mem = HumanModeMemory(max_memories=5)
    state = HumanState()

    for i in range(10):
        task = TaskInput(task_id=f"t{i}", text=f"task {i}", task_type="general")
        mem.record(
            event_id=f"e{i}", task=task, state=state,
            reward=0.5, selected_branch="reflective_reasoning",
            active_branches=["reflective_reasoning"],
        )

    assert mem.memory_count <= 5


def test_memory_auto_tags():
    mem = HumanModeMemory()
    # Set strongly negative state: high negative vars, low positive vars
    state = HumanState(initial_values={
        "stress": 0.9, "frustration": 0.9, "fear": 0.8,
        "fatigue": 0.8, "self_protection": 0.8,
        "confidence": 0.1, "motivation": 0.1, "curiosity": 0.1,
        "trust": 0.1, "ambition": 0.1,
    })
    task = TaskInput(task_id="t1", text="test", task_type="general")

    entry = mem.record(
        event_id="e1", task=task, state=state,
        reward=0.1, selected_branch="reflective_reasoning",
        active_branches=["reflective_reasoning"],
        failure_reason="constraint_violation",
    )
    assert "failure" in entry.tags
    assert "negative_mood" in entry.tags
    assert any(t.startswith("fail:") for t in entry.tags)


def test_memory_failure_patterns():
    mem = HumanModeMemory()
    state = HumanState()

    for reason in ["timeout", "timeout", "constraint", "timeout"]:
        task = TaskInput(task_id="t1", text="test", task_type="general")
        mem.record(
            event_id="e1", task=task, state=state,
            reward=0.2, selected_branch="x",
            active_branches=["x"],
            failure_reason=reason,
        )

    patterns = mem.failure_patterns()
    assert patterns["timeout"] == 3
    assert patterns["constraint"] == 1


def test_memory_emotional_trajectory():
    mem = HumanModeMemory()
    state = HumanState()

    for i in range(5):
        task = TaskInput(task_id=f"t{i}", text="test", task_type="general")
        mem.record(
            event_id=f"e{i}", task=task, state=state,
            reward=0.5, selected_branch="x",
            active_branches=["x"],
        )

    traj = mem.emotional_trajectory(window=3)
    assert len(traj) == 3
    assert all(isinstance(v, float) for v in traj)
