from __future__ import annotations

from prompt_forest.branches.hierarchical import create_default_hierarchical_forest
from prompt_forest.config import MemoryConfig, RouterConfig
from prompt_forest.memory.store import MemoryStore
from prompt_forest.router.hierarchical_router import HierarchicalRouter
from prompt_forest.types import MemoryRecord, TaskInput


def _memory_record(
    store: MemoryStore,
    *,
    task_id: str,
    text: str,
    task_type: str,
    selected_path: list[str],
    reward: float,
    user_id: str = "test_user",
    selected_output: str = "output",
) -> MemoryRecord:
    task = TaskInput(
        task_id=task_id,
        text=text,
        task_type=task_type,
        metadata={"aspect": task_type, "user_id": user_id},
    )
    parent_id = selected_path[-2] if len(selected_path) >= 2 else ""
    return MemoryRecord(
        task_id=task_id,
        task_type=task_type,
        input_text=text,
        activated_branches=list(selected_path),
        branch_outputs={selected_path[-1]: selected_output},
        selected_branch=selected_path[-1],
        selected_output=selected_output,
        reward_score=reward,
        failure_reason="",
        confidence=0.8,
        useful_patterns=[],
        branch_rewards={selected_path[-1]: reward},
        selected_path=list(selected_path),
        routing_context_key=store.routing_context_key_for_task(task, parent_id) if parent_id else "",
        user_id=user_id,
        task_metadata=dict(task.metadata),
        reward_components={},
    )


def test_contextual_route_memory_is_parent_scoped():
    store = MemoryStore(
        MemoryConfig(
            routing_min_support=2,
            routing_min_similarity=0.2,
            routing_shrinkage_k=1.0,
        )
    )
    store.add(
        _memory_record(
            store,
            task_id="r1",
            text="Create a 48-hour incident response plan with owners, rollback, risks, and confidence.",
            task_type="planning",
            selected_path=["planner", "planner_timeline_optimizer"],
            reward=0.82,
        )
    )
    store.add(
        _memory_record(
            store,
            task_id="r2",
            text="Plan a 30-day onboarding schedule with milestones, owners, risks, and confidence.",
            task_type="planning",
            selected_path=["planner", "planner_timeline_optimizer"],
            reward=0.8,
        )
    )
    store.add(
        _memory_record(
            store,
            task_id="r3",
            text="Respond only as minified JSON with answer and confidence.",
            task_type="planning",
            selected_path=["verification", "json_lock"],
            reward=0.95,
        )
    )

    task = TaskInput(
        task_id="t1",
        text="Create a concise risk register for a launch next week. Include mitigation, owner, and confidence.",
        task_type="planning",
        metadata={"aspect": "planning", "user_id": "test_user"},
    )

    planner_scores = store.sibling_routing_scores(
        task=task,
        task_type="planning",
        parent_id="planner",
        child_ids=["planner_timeline_optimizer", "planner_risk_allocator"],
        user_id="test_user",
    )
    verification_scores = store.sibling_routing_scores(
        task=task,
        task_type="planning",
        parent_id="verification",
        child_ids=["json_lock", "csv_lock"],
        user_id="test_user",
    )

    assert planner_scores["planner_timeline_optimizer"] > 0.0
    assert verification_scores == {}


def test_router_penalizes_lock_leaves_without_contract():
    router = HierarchicalRouter(
        RouterConfig(
            top_k=1,
            min_candidates=1,
            exploration=0.0,
            exploration_min=0.0,
        )
    )
    forest = create_default_hierarchical_forest()
    memory = MemoryStore(MemoryConfig())
    task = TaskInput(
        task_id="math_task",
        text="Differentiate x^2 + 3x. Show the result briefly and include a confidence line.",
        task_type="math",
        metadata={"user_id": "test_user"},
    )

    route = router.route(task, forest, memory)

    assert route.activated_paths[0][0] == "verification"
    assert route.activated_paths[0][-1] != "json_lock"


def test_router_uses_contextual_memory_to_rerank_leaf_siblings():
    config = RouterConfig(
        top_k=1,
        min_candidates=1,
        exploration=0.0,
        exploration_min=0.0,
        memory_coef=1.0,
    )
    forest = create_default_hierarchical_forest()
    empty_memory = MemoryStore(MemoryConfig())
    task = TaskInput(
        task_id="plan_task",
        text="Create a concise risk register for launching a new pricing page. Include mitigation, owner, and confidence.",
        task_type="planning",
        metadata={"aspect": "planning", "user_id": "test_user"},
    )

    baseline_route = HierarchicalRouter(config).route(task, forest, empty_memory)
    assert baseline_route.activated_paths[0][-1] == "planner_timeline_optimizer"

    memory = MemoryStore(
        MemoryConfig(
            routing_min_support=2,
            routing_min_similarity=0.2,
            routing_shrinkage_k=1.0,
        )
    )
    memory.add(
        _memory_record(
            memory,
            task_id="m1",
            text="Create a launch risk register with mitigation, owner, rollout risk, and confidence.",
            task_type="planning",
            selected_path=["planner", "planner_risk_allocator"],
            reward=0.84,
        )
    )
    memory.add(
        _memory_record(
            memory,
            task_id="m2",
            text="Write a rollout risk register with mitigations, owners, fallback plans, and confidence.",
            task_type="planning",
            selected_path=["planner", "planner_risk_allocator"],
            reward=0.83,
        )
    )

    route = HierarchicalRouter(config).route(task, forest, memory)

    assert route.activated_paths[0][0] == "planner"
    assert route.activated_paths[0][-1] == "planner_risk_allocator"


def test_router_ignores_unsupported_learned_weight_delta_until_support_exists():
    config = RouterConfig(
        top_k=1,
        min_candidates=1,
        exploration=0.0,
        exploration_min=0.0,
        learned_weight_min_support=2,
        learned_weight_support_k=0.0,
    )
    forest = create_default_hierarchical_forest()
    memory = MemoryStore(MemoryConfig())
    task = TaskInput(
        task_id="plan_weight_task",
        text="Create a concise risk register for launching a new pricing page. Include mitigation, owner, and confidence.",
        task_type="planning",
        metadata={"aspect": "planning", "user_id": "test_user"},
    )

    baseline_route = HierarchicalRouter(config).route(task, forest, memory)
    assert baseline_route.activated_paths[0][-1] == "planner_timeline_optimizer"

    forest.branches["planner_risk_allocator"].state.weight = 1.35
    unsupported_route = HierarchicalRouter(config).route(task, forest, memory)
    assert unsupported_route.activated_paths[0][-1] == "planner_timeline_optimizer"

    forest.branches["planner_risk_allocator"].state.historical_rewards.extend([0.82, 0.84])
    supported_route = HierarchicalRouter(config).route(task, forest, memory)
    assert supported_route.activated_paths[0][-1] == "planner_risk_allocator"


def test_memory_execution_guidance_uses_similar_successful_leaf_examples():
    store = MemoryStore(
        MemoryConfig(
            routing_min_similarity=0.2,
            execution_hint_min_reward=0.72,
        )
    )
    store.add(
        _memory_record(
            store,
            task_id="g1",
            text="Create a launch risk register with mitigation, owner, rollback, and confidence.",
            task_type="planning",
            selected_path=["planner", "planner_risk_allocator"],
            reward=0.84,
            selected_output=(
                "## Risk Register\n"
                "| Risk | Owner | Mitigation | Rollback |\n"
                "| --- | --- | --- | --- |\n"
                "Confidence: 0.82"
            ),
        )
    )
    store.add(
        _memory_record(
            store,
            task_id="g2",
            text="Plan an onboarding timeline with milestones, owner, and confidence.",
            task_type="planning",
            selected_path=["planner", "planner_timeline_optimizer"],
            reward=0.88,
            selected_output="## Timeline\nWeek 1\nOwner: Eng Manager\nConfidence: 0.8",
        )
    )
    store.add(
        _memory_record(
            store,
            task_id="g3",
            text="Create a launch risk register with mitigation, owner, rollback, and confidence.",
            task_type="planning",
            selected_path=["planner", "planner_risk_allocator"],
            reward=0.55,
            selected_output="weak answer",
        )
    )

    task = TaskInput(
        task_id="tg",
        text="Draft a concise risk register for next week's launch. Include mitigation, owner, rollback, and confidence.",
        task_type="planning",
        metadata={"aspect": "planning", "user_id": "test_user"},
    )

    guidance = store.execution_guidance(task, "planner_risk_allocator", user_id="test_user")

    assert guidance
    assert "owner" in guidance[0].lower()
    assert "rollback" in guidance[0].lower()
    assert "table" in guidance[0].lower()
