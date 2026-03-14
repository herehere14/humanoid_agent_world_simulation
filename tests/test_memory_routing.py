from __future__ import annotations

from prompt_forest.branches.hierarchical import create_default_hierarchical_forest
from prompt_forest.config import MemoryConfig, RouterConfig
from prompt_forest.memory.store import MemoryStore, _task_checklist_items
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


def _add_sibling_probe(
    store: MemoryStore,
    *,
    task_id: str,
    text: str,
    task_type: str,
    parent_id: str,
    reward_by_child: dict[str, float],
    user_id: str = "test_user",
):
    best_child = max(reward_by_child.items(), key=lambda item: item[1])[0]
    task = TaskInput(
        task_id=task_id,
        text=text,
        task_type=task_type,
        metadata={"aspect": task_type, "user_id": user_id},
    )
    record = MemoryRecord(
        task_id=task_id,
        task_type=task_type,
        input_text=text,
        activated_branches=[parent_id, *reward_by_child.keys()],
        branch_outputs={child_id: f"{child_id} output" for child_id in reward_by_child},
        selected_branch=best_child,
        selected_output=f"{best_child} output",
        reward_score=float(reward_by_child[best_child]),
        failure_reason="",
        confidence=0.8,
        useful_patterns=[],
        branch_rewards={child_id: float(score) for child_id, score in reward_by_child.items()},
        selected_path=[parent_id, best_child],
        routing_context_key=store.routing_context_key_for_task(task, parent_id),
        user_id=user_id,
        task_metadata=dict(task.metadata),
        reward_components={},
    )
    store.add(record)
    return store.record_sibling_probe(
        task=task,
        parent_id=parent_id,
        reward_by_child=reward_by_child,
        user_id=user_id,
        source="test_probe",
    )


def test_pairwise_route_memory_is_parent_scoped():
    store = MemoryStore(
        MemoryConfig(
            routing_min_support=2,
            routing_min_similarity=0.2,
            routing_shrinkage_k=1.0,
            routing_promotion_min_support=3,
            routing_pair_replay_min_samples=3,
        )
    )
    for idx, text in enumerate(
        [
            "Create a concise risk register for launch. Include mitigation, owner, fallback, and confidence.",
            "Write a rollout risk register with owners, mitigations, fallback plans, and confidence.",
            "Draft a recovery risk register with owners, rollback, dependencies, and confidence.",
        ],
        start=1,
    ):
        _add_sibling_probe(
            store,
            task_id=f"r{idx}",
            text=text,
            task_type="planning",
            parent_id="planner",
            reward_by_child={
                "planner_timeline_optimizer": 0.58 + (idx * 0.01),
                "planner_risk_allocator": 0.83 + (idx * 0.01),
            },
        )

    task = TaskInput(
        task_id="t1",
        text="Create a concise risk register for a launch next week. Include mitigation, owner, and confidence.",
        task_type="planning",
        metadata={"aspect": "planning", "user_id": "test_user"},
    )

    planner_signal = store.sibling_preference_signal(
        task=task,
        task_type="planning",
        parent_id="planner",
        child_ids=["planner_timeline_optimizer", "planner_risk_allocator"],
        user_id="test_user",
    )
    verification_signal = store.sibling_preference_signal(
        task=task,
        task_type="planning",
        parent_id="verification",
        child_ids=["json_lock", "csv_lock"],
        user_id="test_user",
    )

    assert planner_signal.preferred_child == "planner_risk_allocator"
    assert planner_signal.support >= 3
    assert verification_signal.scores == {}


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


def test_router_does_not_learn_routing_from_plain_execution_records():
    config = RouterConfig(
        top_k=1,
        min_candidates=1,
        exploration=0.0,
        exploration_min=0.0,
        memory_coef=1.0,
    )
    forest = create_default_hierarchical_forest()
    memory = MemoryStore(MemoryConfig())
    task = TaskInput(
        task_id="plan_task",
        text="Create a concise risk register for launching a new pricing page. Include mitigation, owner, and confidence.",
        task_type="planning",
        metadata={"aspect": "planning", "user_id": "test_user"},
    )

    baseline_route = HierarchicalRouter(config).route(task, forest, memory)
    assert baseline_route.activated_paths[0][-1] == "planner_timeline_optimizer"

    memory.add(
        _memory_record(
            memory,
            task_id="m1",
            text="Create a launch risk register with mitigation, owner, rollout risk, and confidence.",
            task_type="planning",
            selected_path=["planner", "planner_risk_allocator"],
            reward=0.9,
        )
    )
    memory.add(
        _memory_record(
            memory,
            task_id="m2",
            text="Write a rollout risk register with mitigations, owners, fallback plans, and confidence.",
            task_type="planning",
            selected_path=["planner", "planner_risk_allocator"],
            reward=0.88,
        )
    )

    route = HierarchicalRouter(config).route(task, forest, memory)

    assert route.activated_paths[0][-1] == "planner_timeline_optimizer"


def test_router_uses_promoted_pairwise_preference_to_rerank_leaf_siblings():
    config = RouterConfig(
        top_k=1,
        min_candidates=1,
        exploration=0.0,
        exploration_min=0.0,
        memory_coef=0.6,
    )
    forest = create_default_hierarchical_forest()
    memory = MemoryStore(
        MemoryConfig(
            routing_min_support=2,
            routing_min_similarity=0.2,
            routing_shrinkage_k=1.0,
            routing_promotion_min_support=4,
            routing_pair_replay_min_samples=3,
        )
    )
    task = TaskInput(
        task_id="plan_task",
        text="Create a concise risk register for launching a new pricing page. Include mitigation, owner, and confidence.",
        task_type="planning",
        metadata={"aspect": "planning", "user_id": "test_user"},
    )

    baseline_route = HierarchicalRouter(config).route(task, forest, memory)
    assert baseline_route.activated_paths[0][-1] == "planner_timeline_optimizer"

    probe_texts = [
        "Create a launch risk register with mitigation, owner, rollout risk, and confidence.",
        "Write a rollout risk register with mitigations, owners, fallback plans, and confidence.",
        "Draft a recovery risk register with owners, mitigations, rollback, and confidence.",
        "Create an incident risk register with owner, fallback, mitigations, and confidence.",
    ]
    for idx, text in enumerate(probe_texts, start=1):
        _add_sibling_probe(
            memory,
            task_id=f"probe_{idx}",
            text=text,
            task_type="planning",
            parent_id="planner",
            reward_by_child={
                "planner_timeline_optimizer": 0.57 + (idx * 0.01),
                "planner_risk_allocator": 0.82 + (idx * 0.01),
            },
        )

    route = HierarchicalRouter(config).route(task, forest, memory)

    assert route.activated_paths[0][0] == "planner"
    assert route.activated_paths[0][-1] == "planner_risk_allocator"


def test_router_hard_override_gate_can_flip_selection_even_without_memory_term():
    config = RouterConfig(
        top_k=1,
        min_candidates=1,
        exploration=0.0,
        exploration_min=0.0,
        memory_coef=0.0,
        route_override_min_support=4,
        route_override_min_win_rate=0.7,
        route_override_min_margin=0.03,
    )
    forest = create_default_hierarchical_forest()
    memory = MemoryStore(
        MemoryConfig(
            routing_min_support=2,
            routing_min_similarity=0.2,
            routing_shrinkage_k=1.0,
            routing_promotion_min_support=4,
            routing_pair_replay_min_samples=3,
        )
    )
    task = TaskInput(
        task_id="override_task",
        text="Create a concise risk register for launching a new pricing page. Include mitigation, owner, and confidence.",
        task_type="planning",
        metadata={"aspect": "planning", "user_id": "test_user"},
    )

    for idx in range(4):
        _add_sibling_probe(
            memory,
            task_id=f"override_{idx}",
            text="Create a risk register with mitigation, owner, fallback, and confidence.",
            task_type="planning",
            parent_id="planner",
            reward_by_child={
                "planner_timeline_optimizer": 0.58,
                "planner_risk_allocator": 0.86,
            },
        )

    route = HierarchicalRouter(config).route(task, forest, memory)

    assert route.activated_paths[0][-1] == "planner_risk_allocator"
    assert route.sibling_decisions["planner"]["override_child"] == "planner_risk_allocator"


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


def test_pairwise_preference_stays_shadow_until_replay_support_exists():
    store = MemoryStore(
        MemoryConfig(
            routing_min_support=2,
            routing_min_similarity=0.2,
            routing_promotion_min_support=4,
            routing_pair_replay_min_samples=3,
        )
    )
    task = TaskInput(
        task_id="shadow",
        text="Create a risk register with mitigation, owner, fallback, and confidence.",
        task_type="planning",
        metadata={"aspect": "planning", "user_id": "test_user"},
    )

    store.record_sibling_probe(
        task=task,
        parent_id="planner",
        reward_by_child={
            "planner_timeline_optimizer": 0.6,
            "planner_risk_allocator": 0.86,
        },
        user_id="test_user",
        source="test_probe",
    )

    signal = store.sibling_preference_signal(
        task=task,
        task_type="planning",
        parent_id="planner",
        child_ids=["planner_timeline_optimizer", "planner_risk_allocator"],
        user_id="test_user",
    )

    assert signal.scores == {}


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


def test_memory_execution_playbook_aggregates_coverage_structure_and_failures():
    store = MemoryStore(
        MemoryConfig(
            routing_min_similarity=0.2,
            execution_hint_min_reward=0.72,
        )
    )
    store.add(
        _memory_record(
            store,
            task_id="p1",
            text="Create a launch risk register with mitigation, owner, rollback, and confidence.",
            task_type="planning",
            selected_path=["planner", "planner_risk_allocator"],
            reward=0.86,
            selected_output=(
                "## Risk Register\n"
                "- Owner: PM\n"
                "- Rollback: revert release\n"
                "- Risk: delayed rollout\n"
                "Confidence: 0.82"
            ),
        )
    )
    store.add(
        _memory_record(
            store,
            task_id="p2",
            text="Create a launch risk register with mitigation, owner, rollback, and confidence.",
            task_type="planning",
            selected_path=["planner", "planner_risk_allocator"],
            reward=0.84,
            selected_output=(
                "## Risks\n"
                "| Risk | Owner | Mitigation | Rollback |\n"
                "| --- | --- | --- | --- |\n"
                "Confidence: 0.8"
            ),
        )
    )
    store.add(
        _memory_record(
            store,
            task_id="p3",
            text="Create a launch risk register with mitigation, owner, rollback, and confidence.",
            task_type="planning",
            selected_path=["planner", "planner_risk_allocator"],
            reward=0.52,
            selected_output="Owner: PM\nConfidence: 0.65",
        )
    )

    task = TaskInput(
        task_id="target",
        text="Draft a concise risk register for next week's launch. Include owner, rollback, communication, risks, and confidence.",
        task_type="planning",
        metadata={"aspect": "planning", "user_id": "test_user"},
    )

    playbook = store.execution_playbook(task, "planner_risk_allocator", user_id="test_user", min_similarity=0.2)

    assert playbook.support == 2
    assert any("owner" in item for item in playbook.coverage_items)
    assert any("rollback" in item for item in playbook.coverage_items)
    assert any("confidence" in item for item in playbook.coverage_items)
    assert playbook.structure_cues
    assert playbook.success_examples
    assert any("do not omit rollback" in item for item in playbook.anti_patterns)


def test_task_checklist_items_extracts_explain_clauses():
    items = _task_checklist_items(
        "Audit a migration update for consistency when the backfill is complete but one shard is still lagging. "
        "Explain contradictions, uncertainty calibration, and confidence."
    )

    assert "contradictions" in items
    assert "uncertainty calibration" in items
    assert "confidence" in items
