from __future__ import annotations

from prompt_forest.backend.mock import MockLLMBackend
from prompt_forest.core.engine import PromptForestEngine
from prompt_forest.types import BranchOutput, MemoryRecord, RoutingDecision, TaskInput


class RefinementBackend(MockLLMBackend):
    def generate(self, prompt, task, branch_name):
        if branch_name == "planner_risk_allocator__adaptive_refine":
            return (
                "## Recovery Plan\n"
                "- Owner: SRE lead\n"
                "- Rollback: revert the deployment immediately if error rate stays elevated\n"
                "- Communication: update status page and customer support\n"
                "- Risk: queue backlog during recovery\n"
                "Confidence: 0.82",
                {"branch": branch_name},
            )
        return super().generate(prompt, task, branch_name)


class LossyVerificationRefinementBackend(MockLLMBackend):
    def generate(self, prompt, task, branch_name):
        if branch_name == "verification_constraint_checker__adaptive_refine":
            return (
                "Constraint audit summary:\n"
                "- Contradiction found between claimed completion and lagging shard.\n"
                "- Manual repair remains unresolved.\n"
                "- Revise the status note to be more explicit.",
                {"branch": branch_name},
            )
        return super().generate(prompt, task, branch_name)


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


def test_engine_refines_leaf_output_with_learned_playbook_when_score_improves():
    engine = PromptForestEngine(backend=RefinementBackend())
    task = TaskInput(
        task_id="t_refine",
        text="Create a next-7-days recovery plan after a failed production deploy. Include owners, rollback, communication, risks, and confidence.",
        task_type="planning",
        metadata={"aspect": "planning", "user_id": "test_user"},
    )
    branch = engine.branches["planner_risk_allocator"]

    for idx in range(2):
        engine.memory.add(
            MemoryRecord(
                task_id=f"seed_{idx}",
                task_type="planning",
                input_text="Create a 48-hour incident response plan. Include owners, rollback, risks, and confidence.",
                activated_branches=["planner", "planner_risk_allocator"],
                branch_outputs={
                    "planner_risk_allocator": (
                        "## Plan\n"
                        "- Owner: Incident commander\n"
                        "- Rollback: revert release\n"
                        "- Risk: customer impact\n"
                        "Confidence: 0.8"
                    )
                },
                selected_branch="planner_risk_allocator",
                selected_output=(
                    "## Plan\n"
                    "- Owner: Incident commander\n"
                    "- Rollback: revert release\n"
                    "- Risk: customer impact\n"
                    "Confidence: 0.8"
                ),
                reward_score=0.84,
                failure_reason="",
                confidence=0.8,
                useful_patterns=[],
                branch_rewards={"planner_risk_allocator": 0.84},
                selected_path=["planner", "planner_risk_allocator"],
                routing_context_key=engine.memory.routing_context_key_for_task(task, "planner"),
                user_id="test_user",
                task_metadata={"aspect": "planning", "user_id": "test_user"},
                reward_components={},
            )
        )

    original = BranchOutput(
        branch_name="planner_risk_allocator",
        prompt="seed",
        output="- Owner: SRE lead\n- Risk: queue backlog\nConfidence: 0.62",
        task_type="planning",
        model_meta={},
    )

    refined = engine._maybe_refine_leaf_output(task, branch, original)

    assert refined.output != original.output
    assert refined.model_meta.get("adaptive_refined") is True
    assert "rollback" in refined.output.lower()
    assert "communication" in refined.output.lower()


def test_support_branch_selection_prefers_review_and_tradeoff_specialists():
    code_task = TaskInput(
        task_id="code_support",
        text="Give a pull-request review checklist for adding a background job queue. Include failure handling, tests, rollout risk, and confidence.",
        task_type="code",
        metadata={},
    )
    consistency_task = TaskInput(
        task_id="code_consistency_support",
        text="Audit a migration update for consistency when it says the backfill is complete, but a shard is still lagging. Explain contradictions, uncertainty calibration, and confidence.",
        task_type="code",
        metadata={},
    )
    general_task = TaskInput(
        task_id="general_support",
        text="Write a short decision note comparing ship now vs delay one week. Include tradeoffs, recommendation, and confidence.",
        task_type="general",
        metadata={},
    )

    assert PromptForestEngine._support_branch_for_task(code_task, "verification_constraint_checker", ["tests"]) == "critique_failure_hunter"
    assert PromptForestEngine._support_branch_for_task(consistency_task, "verification_constraint_checker", ["confidence"]) == "verification_consistency_auditor"
    assert PromptForestEngine._support_branch_for_task(general_task, "retrieval_evidence_tracer", ["recommendation"]) == "verification_consistency_auditor"


def test_refinement_rejects_candidate_that_drops_task_native_coverage():
    engine = PromptForestEngine(backend=LossyVerificationRefinementBackend())
    task = TaskInput(
        task_id="t_consistency_refine",
        text="Audit a migration update for consistency when it says the backfill is complete, but one shard is still lagging and manual repair steps are pending. Explain contradictions, uncertainty calibration, and confidence.",
        task_type="code",
        metadata={
            "aspect": "code_consistency_audit",
            "user_id": "test_user",
            "expected_keywords": ["contradictions", "uncertainty calibration", "confidence"],
            "required_substrings": ["confidence"],
        },
    )
    branch = engine.branches["verification_constraint_checker"]

    for idx in range(2):
        engine.memory.add(
            MemoryRecord(
                task_id=f"verification_seed_{idx}",
                task_type="code",
                input_text="Audit a rollout note for consistency. Explain contradictions, uncertainty calibration, and confidence.",
                activated_branches=["verification", "verification_constraint_checker"],
                branch_outputs={
                    "verification_constraint_checker": (
                        "Constraint audit:\n"
                        "- Contradictions: rollout claims success despite unresolved rollback risk.\n"
                        "- Uncertainty calibration: medium.\n"
                        "confidence=0.54"
                    )
                },
                selected_branch="verification_constraint_checker",
                selected_output=(
                    "Constraint audit:\n"
                    "- Contradictions: rollout claims success despite unresolved rollback risk.\n"
                    "- Uncertainty calibration: medium.\n"
                    "confidence=0.54"
                ),
                reward_score=0.82,
                failure_reason="",
                confidence=0.8,
                useful_patterns=[],
                branch_rewards={"verification_constraint_checker": 0.82},
                selected_path=["verification", "verification_constraint_checker"],
                routing_context_key=engine.memory.routing_context_key_for_task(task, "verification"),
                user_id="test_user",
                task_metadata={"aspect": "code_consistency_audit", "user_id": "test_user"},
                reward_components={},
            )
        )

    original = BranchOutput(
        branch_name="verification_constraint_checker",
        prompt="seed",
        output=(
            "Constraint audit:\n"
            "- Contradictions: the update claims completion while a shard still lags.\n"
            "- Uncertainty calibration: low because repair is pending.\n"
            "confidence=0.41"
        ),
        task_type="code",
        model_meta={},
    )

    refined = engine._maybe_refine_leaf_output(task, branch, original)

    assert refined.output == original.output
    assert refined.model_meta.get("adaptive_refined") is None


def test_inference_route_can_expand_with_support_path_when_memory_exists():
    engine = PromptForestEngine()
    engine.config.execution_adaptation.enable_support_pass = True
    task = TaskInput(
        task_id="route_support",
        text="Write a short decision note comparing ship now vs delay one week for testing. Include tradeoffs, recommendation, and confidence.",
        task_type="general",
        metadata={"user_id": "test_user"},
    )
    engine.memory.add(
        MemoryRecord(
            task_id="support_seed_1",
            task_type="general",
            input_text="Write a short decision note comparing ship now vs delay one week for testing. Include tradeoffs, recommendation, and confidence.",
            activated_branches=["verification", "verification_consistency_auditor"],
            branch_outputs={"verification_consistency_auditor": "Recommendation: delay.\nConfidence: 0.8"},
            selected_branch="verification_consistency_auditor",
            selected_output="Recommendation: delay.\nConfidence: 0.8",
            reward_score=0.82,
            failure_reason="",
            confidence=0.8,
            useful_patterns=[],
            branch_rewards={"verification_consistency_auditor": 0.82},
            selected_path=["verification", "verification_consistency_auditor"],
            routing_context_key="",
            user_id="test_user",
            task_metadata={"user_id": "test_user"},
            reward_components={},
        )
    )
    engine.memory.add(
        MemoryRecord(
            task_id="support_seed_2",
            task_type="general",
            input_text="Compare two launch options and give a recommendation with confidence.",
            activated_branches=["verification", "verification_consistency_auditor"],
            branch_outputs={"verification_consistency_auditor": "Recommendation: wait.\nConfidence: 0.79"},
            selected_branch="verification_consistency_auditor",
            selected_output="Recommendation: wait.\nConfidence: 0.79",
            reward_score=0.8,
            failure_reason="",
            confidence=0.8,
            useful_patterns=[],
            branch_rewards={"verification_consistency_auditor": 0.8},
            selected_path=["verification", "verification_consistency_auditor"],
            routing_context_key="",
            user_id="test_user",
            task_metadata={"user_id": "test_user"},
            reward_components={},
        )
    )

    route = RoutingDecision(
        task_type="general",
        activated_branches=["retrieval", "retrieval_evidence_tracer"],
        branch_scores={"retrieval": 1.0, "retrieval_evidence_tracer": 1.0},
        activated_paths=[["retrieval", "retrieval_evidence_tracer"]],
    )

    expanded = engine._augment_route_with_support_paths(task, route, adapt=False, update_memory=False)

    assert ["verification", "verification_consistency_auditor"] in expanded.activated_paths


def test_engine_probes_close_siblings_and_records_route_preferences():
    engine = PromptForestEngine(backend=MockLLMBackend(seed=13))
    task_text = "Create a concise risk register for launching a new pricing page. Include mitigation, owner, rollback, and confidence."
    route = RoutingDecision(
        task_type="planning",
        activated_branches=["planner", "planner_timeline_optimizer"],
        branch_scores={
            "planner": 1.0,
            "planner_timeline_optimizer": 1.04,
            "planner_risk_allocator": 1.01,
        },
        activated_paths=[["planner", "planner_timeline_optimizer"]],
        sibling_decisions={
            "planner": {
                "selected_by_score": "planner_timeline_optimizer",
                "selected_child": "planner_timeline_optimizer",
                "probe_candidates": ["planner_timeline_optimizer", "planner_risk_allocator"],
                "score_gap": 0.03,
            }
        },
    )
    original_route = engine.router.route
    engine.router.route = lambda *_args, **_kwargs: route  # type: ignore[assignment]
    try:
        result = engine.run_task_controlled(
            text=task_text,
            task_type="planning",
            metadata={"aspect": "planning", "user_id": "test_user"},
            adapt=True,
            update_memory=True,
        )
    finally:
        engine.router.route = original_route  # type: ignore[assignment]

    assert result["routing_probes"]
    assert result["routing_probes"][0]["added_children"] == ["planner_risk_allocator"]
    assert result["routing_preferences"]
    assert set(result["routing_preferences"][0]["reward_by_child"]) == {
        "planner_timeline_optimizer",
        "planner_risk_allocator",
    }
