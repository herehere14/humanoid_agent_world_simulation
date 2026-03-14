from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Any
from uuid import uuid4

from ..agents.evaluator_agent import EvaluatorAgent
from ..agents.llm_evaluator_agent import LLMEvaluatorAgent
from ..agents.llm_optimizer_advisor import LLMOptimizerAdvisor
from ..agents.optimizer_agent import OptimizationEvent, OptimizerAgent
from ..aggregator.strategies import AggregationResult, Aggregator
from ..backend.base import LLMBackend
from ..backend.mock import MockLLMBackend
from ..branches.base import PromptBranch
from ..branches.hierarchical import HierarchicalPromptForest, create_default_hierarchical_forest
from ..config import EngineConfig, load_config
from ..contracts import infer_output_contract
from ..evaluator.judge import OutputJudge
from ..memory.store import MemoryStore
from ..rewards.modes import KeywordReward, RuleBasedReward, TaskSpecificReward
from ..rewards.verifiers import ExternalVerifierReward
from ..router.hierarchical_router import HierarchicalRouter
from ..types import MemoryRecord, RoutingDecision, TaskInput
from ..utils.io import append_jsonl, ensure_parent
from .composer import FinalComposer
from .executor import PromptExecutor


class PromptForestEngine:
    def __init__(
        self,
        config: EngineConfig | None = None,
        config_path: str | Path | None = None,
        backend: LLMBackend | None = None,
        branches: dict[str, PromptBranch] | None = None,
    ) -> None:
        self.config = config or load_config(config_path)
        self.artifacts_dir = Path(self.config.artifacts_dir)
        ensure_parent(self.artifacts_dir / "dummy")

        # Backward compatibility: if branches are supplied, wrap them as a 1-layer forest.
        self.forest = HierarchicalPromptForest.from_flat(branches) if branches is not None else create_default_hierarchical_forest()
        self.branches = self.forest.branches

        self.backend = backend or MockLLMBackend()
        self.executor = PromptExecutor(self.backend)
        self.composer = FinalComposer(self.backend, self.config.composer)

        self.memory = MemoryStore(self.config.memory, memory_path=self.artifacts_dir / "memory_records.jsonl")
        self.router = HierarchicalRouter(self.config.router)
        self.judge = OutputJudge(self.config.evaluator.reward_mode)
        self.evaluator_agent = LLMEvaluatorAgent(
            runtime_config=self.config.agent_runtimes.evaluator,
            fallback=EvaluatorAgent(),
        )
        optimizer_advisor = LLMOptimizerAdvisor(self.config.agent_runtimes.optimizer)
        self.optimizer_agent = OptimizerAgent(self.config.optimizer, advisor=optimizer_advisor)
        self.aggregator = Aggregator(self.config.evaluator.aggregation_strategy)

        self._route_history: list[RoutingDecision] = []
        self._event_log = self.artifacts_dir / "events.jsonl"

    def run_task(self, text: str, task_type: str = "auto", metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.run_task_controlled(text=text, task_type=task_type, metadata=metadata, adapt=True, update_memory=True)

    def run_task_controlled(
        self,
        text: str,
        task_type: str = "auto",
        metadata: dict[str, Any] | None = None,
        adapt: bool = True,
        update_memory: bool = True,
    ) -> dict[str, Any]:
        task_metadata = dict(metadata or {})
        user_id = str(task_metadata.get("user_id", "global")).strip() or "global"
        task_metadata["user_id"] = user_id
        task_metadata["llm_runtime_active"] = bool(
            self.config.agent_runtimes.evaluator.enabled or self.config.agent_runtimes.optimizer.enabled
        )
        user_profile = self.memory.get_user_profile(user_id)
        if user_profile:
            task_metadata["user_preferences"] = user_profile
        task = TaskInput(task_id=str(uuid4()), text=text, task_type=task_type, metadata=task_metadata)

        route = self.router.route(task, self.forest, self.memory)
        route = self._augment_route_with_support_paths(task, route, adapt=adapt, update_memory=update_memory)
        task.task_type = route.task_type
        self._route_history.append(route)

        outputs = self._run_path(route, task)
        route, outputs, routing_probes = self._maybe_probe_sibling_branches(
            task,
            route,
            outputs,
            adapt=adapt,
            update_memory=update_memory,
        )
        composer_notes: dict[str, Any] = {}
        contract_hint = infer_output_contract(task.text, task.metadata)
        composed = None
        if contract_hint is None:
            composed = self.composer.compose(task=task, route=route, outputs=outputs)
        if composed is not None and route.activated_branches:
            if route.activated_paths and route.activated_paths[0]:
                leaf = route.activated_paths[0][-1]
            else:
                leaf = route.activated_branches[-1]
            outputs[leaf] = composed.output
            composer_notes = composed.notes

        branch_scores = self.judge.score_all(outputs, task)
        leaf_candidates = self._candidate_leaves(route, outputs)
        numeric_scores = {k: v.reward for k, v in branch_scores.items() if k in leaf_candidates}
        if composed is not None and route.activated_branches:
            if route.activated_paths and route.activated_paths[0]:
                composed_leaf = route.activated_paths[0][-1]
            else:
                composed_leaf = route.activated_branches[-1]
            aggregation = AggregationResult(
                selected_branch=composed_leaf,
                selected_output=outputs[composed_leaf].output,
                notes={"strategy": "composer_primary", "path": route.activated_branches},
            )
        else:
            aggregation = self._aggregate(route, outputs, numeric_scores, candidate_branches=leaf_candidates)
        signal = self.evaluator_agent.evaluate(
            task,
            route,
            branch_scores,
            aggregation,
            branch_outputs={k: v.output for k, v in outputs.items()},
        )
        if composer_notes:
            signal.aggregator_notes["composer"] = composer_notes
        self._propagate_rewards_along_path(route, signal, gamma=0.9, local_mix=0.55)
        reward_components = self._compute_reward_components(task, signal.selected_output, signal.reward_score)
        selected_path = self._selected_path(route, signal.selected_branch)
        routing_context_key = ""
        if len(selected_path) >= 2:
            routing_context_key = self.memory.routing_context_key_for_task(task, parent_id=selected_path[-2])

        if adapt:
            optimize_event: OptimizationEvent = self.optimizer_agent.optimize(
                task=task,
                route=route,
                signal=signal,
                branches=self.branches,
                memory=self.memory,
                acceptance_runner=self._mini_holdout_acceptance,
            )
            self._attach_created_candidates(route, optimize_event, selected_branch=signal.selected_branch)
        else:
            optimize_event = OptimizationEvent(
                updated_weights={},
                update_details={},
                rewritten_prompts=[],
                promoted_candidates=[],
                archived_candidates=[],
                created_candidates=[],
                advisor_used=False,
                advisor_error="",
            )

        record = MemoryRecord(
            task_id=task.task_id,
            task_type=route.task_type,
            input_text=task.text,
            activated_branches=route.activated_branches,
            branch_outputs={name: out.output for name, out in outputs.items()},
            selected_branch=signal.selected_branch,
            selected_output=signal.selected_output,
            reward_score=signal.reward_score,
            failure_reason=signal.failure_reason,
            confidence=signal.confidence,
            useful_patterns=self.memory.useful_patterns(route.task_type),
            branch_rewards={name: fb.reward for name, fb in signal.branch_feedback.items()},
            selected_path=selected_path,
            routing_context_key=routing_context_key,
            user_id=user_id,
            task_metadata=task.metadata,
            reward_components=reward_components,
        )
        if update_memory:
            self.memory.add(record)
            routing_preferences = self._record_route_preferences(task, route, signal)
        else:
            routing_preferences = []

        payload = {
            "task": asdict(task),
            "routing": asdict(route),
            "routing_probes": routing_probes,
            "routing_preferences": routing_preferences,
            "branch_scores": {k: asdict(v) for k, v in branch_scores.items()},
            "evaluation_signal": {
                "reward_score": signal.reward_score,
                "confidence": signal.confidence,
                "selected_branch": signal.selected_branch,
                "selected_output": signal.selected_output,
                "failure_reason": signal.failure_reason,
                "suggested_improvement_direction": signal.suggested_improvement_direction,
                "branch_feedback": {k: asdict(v) for k, v in signal.branch_feedback.items()},
                "aggregator_notes": signal.aggregator_notes,
            },
            "optimization": asdict(optimize_event),
            "reward_components": reward_components,
            "selected_path": selected_path,
            "routing_context_key": routing_context_key,
            "runtime": {
                "primary_backend": self.backend.__class__.__name__,
                "primary_model": getattr(self.backend, "model", "mock"),
                "evaluator_llm_enabled": self.config.agent_runtimes.evaluator.enabled,
                "optimizer_llm_enabled": self.config.agent_runtimes.optimizer.enabled,
                "evaluator_provider": self.config.agent_runtimes.evaluator.provider,
                "optimizer_provider": self.config.agent_runtimes.optimizer.provider,
            },
            "composer": composer_notes,
            "branch_weights": {name: round(branch.state.weight, 4) for name, branch in self.branches.items()},
        }
        append_jsonl(self._event_log, payload)

        return payload

    def _augment_route_with_support_paths(
        self,
        task: TaskInput,
        route: RoutingDecision,
        *,
        adapt: bool,
        update_memory: bool,
    ) -> RoutingDecision:
        if not self.config.execution_adaptation.enable_support_pass:
            return route
        if adapt or update_memory:
            return route
        if not route.activated_paths or not route.activated_paths[0]:
            return route

        user_id = str(task.metadata.get("user_id", "global")).strip() or "global"
        similar_records = self.memory.retrieve_similar(task.task_type, limit=6, user_id=user_id)
        if len(similar_records) < self.config.execution_adaptation.min_success_support:
            similar_records = self.memory.retrieve_similar(task.task_type, limit=6, user_id=None)
        if len(similar_records) < self.config.execution_adaptation.min_success_support:
            return route

        primary_path = list(route.activated_paths[0])
        primary_leaf = primary_path[-1]
        support_leaf = self._support_branch_for_task(task, primary_leaf, [])
        if not support_leaf or support_leaf == primary_leaf:
            return route
        if support_leaf not in self.forest.nodes:
            return route

        support_path = self.forest.path_to_root(support_leaf)
        if not support_path or support_path in route.activated_paths:
            return route

        activated_paths = [list(path) for path in route.activated_paths]
        activated_paths.append(support_path)
        activated_branches = list(route.activated_branches)
        for branch_name in support_path:
            if branch_name not in activated_branches:
                activated_branches.append(branch_name)

        return RoutingDecision(
            task_type=route.task_type,
            activated_branches=activated_branches,
            branch_scores=dict(route.branch_scores),
            activated_paths=activated_paths,
            sibling_decisions=dict(route.sibling_decisions),
        )

    def _maybe_probe_sibling_branches(
        self,
        task: TaskInput,
        route: RoutingDecision,
        outputs: dict[str, Any],
        *,
        adapt: bool,
        update_memory: bool,
    ) -> tuple[RoutingDecision, dict[str, Any], list[dict[str, Any]]]:
        if not adapt or not update_memory:
            return route, outputs, []
        if not route.activated_paths or not route.activated_paths[0]:
            return route, outputs, []

        primary_path = route.activated_paths[0]
        if len(primary_path) < 2:
            return route, outputs, []

        parent_id = primary_path[-2]
        decision = dict(route.sibling_decisions.get(parent_id, {}))
        probe_candidates = [
            str(child).strip()
            for child in decision.get("probe_candidates", [])
            if str(child).strip()
        ]
        if len(probe_candidates) < 2:
            return route, outputs, []

        existing = {
            path[-1]
            for path in route.activated_paths
            if len(path) >= 2 and path[-2] == parent_id and path[-1] in outputs
        }
        missing = [child for child in probe_candidates if child not in existing]
        if not missing:
            return route, outputs, []

        updated_outputs = dict(outputs)
        added_paths: list[list[str]] = []
        for child_id in missing:
            if child_id not in self.forest.nodes:
                continue
            path = self.forest.path_to_root(child_id)
            if not path:
                continue
            self._run_specific_path(task, path, updated_outputs)
            added_paths.append(path)

        if not added_paths:
            return route, outputs, []

        updated_route = self._merge_route_with_paths(route, added_paths)
        updated_decisions = dict(updated_route.sibling_decisions)
        decision["probed_children"] = [
            path[-1]
            for path in updated_route.activated_paths
            if len(path) >= 2 and path[-2] == parent_id and path[-1] in probe_candidates
        ]
        updated_decisions[parent_id] = decision
        updated_route.sibling_decisions = updated_decisions

        probe_summary = {
            "parent_id": parent_id,
            "probe_candidates": probe_candidates,
            "added_children": [path[-1] for path in added_paths],
            "score_gap": decision.get("score_gap", 0.0),
        }
        return updated_route, updated_outputs, [probe_summary]

    def _run_specific_path(self, task: TaskInput, path: list[str], outputs: dict[str, Any]) -> None:
        context = task.metadata.get("context_seed", "")
        strict_contract = infer_output_contract(task.text, task.metadata)
        strict_mode = strict_contract is not None

        for branch_name in path:
            branch = self.branches.get(branch_name)
            if branch is None or not branch.is_active:
                continue

            if branch_name in outputs:
                context = self._roll_context(context, outputs[branch_name].output, strict_mode=strict_mode)
                continue

            branch_context = self._augment_context_for_branch(task, branch, context)
            branch_output = self.executor.run_branch(branch, task, task.task_type, context=branch_context)
            if not self.forest.children(branch_name):
                branch_output = self._maybe_refine_leaf_output(task, branch, branch_output)
            outputs[branch_name] = branch_output
            context = self._roll_context(context, branch_output.output, strict_mode=strict_mode)

    @staticmethod
    def _merge_route_with_paths(route: RoutingDecision, added_paths: list[list[str]]) -> RoutingDecision:
        activated_paths = [list(path) for path in route.activated_paths]
        activated_branches = list(route.activated_branches)
        for path in added_paths:
            if not path or path in activated_paths:
                continue
            activated_paths.append(path)
            for branch_name in path:
                if branch_name not in activated_branches:
                    activated_branches.append(branch_name)

        return RoutingDecision(
            task_type=route.task_type,
            activated_branches=activated_branches,
            branch_scores=dict(route.branch_scores),
            activated_paths=activated_paths,
            sibling_decisions=dict(route.sibling_decisions),
        )

    def _record_route_preferences(
        self,
        task: TaskInput,
        route: RoutingDecision,
        signal,
    ) -> list[dict[str, Any]]:
        grouped_rewards: dict[str, dict[str, float]] = {}
        for path in route.activated_paths:
            if len(path) < 2:
                continue
            parent_id = path[-2]
            child_id = path[-1]
            child_feedback = signal.branch_feedback.get(child_id)
            if child_feedback is None:
                continue
            grouped_rewards.setdefault(parent_id, {})[child_id] = float(child_feedback.reward)

        updates: list[dict[str, Any]] = []
        user_id = str(task.metadata.get("user_id", "global")).strip() or "global"
        for parent_id, reward_by_child in grouped_rewards.items():
            if len(reward_by_child) < 2:
                continue
            signal_summary = self.memory.record_sibling_probe(
                task=task,
                parent_id=parent_id,
                reward_by_child=reward_by_child,
                user_id=user_id,
                source="probe",
            )
            updates.append(
                {
                    "parent_id": parent_id,
                    "reward_by_child": {name: round(score, 4) for name, score in reward_by_child.items()},
                    "preferred_child": signal_summary.preferred_child,
                    "override_child": signal_summary.override_child,
                    "support": signal_summary.support,
                    "win_rate": round(signal_summary.win_rate, 4),
                    "expected_margin": round(signal_summary.expected_margin, 4),
                }
            )
        return updates

    def apply_feedback(
        self,
        task_id: str,
        score: float,
        accepted: bool | None = None,
        corrected_answer: str = "",
        feedback_text: str = "",
        user_id: str | None = None,
        style: str | None = None,
        verbosity: str | None = None,
        domain_preferences: list[str] | None = None,
        hard_constraints: list[str] | None = None,
    ) -> dict[str, Any]:
        record = self.memory.update_feedback(
            task_id=task_id,
            feedback_score=score,
            accepted=accepted,
            corrected_answer=corrected_answer,
            feedback_text=feedback_text,
            user_id=user_id,
        )
        if record is None:
            return {"ok": False, "reason": "task_not_found", "task_id": task_id}

        uid = user_id or record.user_id or "global"
        profile = self.memory.get_user_profile(uid)
        if any(v is not None for v in (style, verbosity, domain_preferences, hard_constraints)):
            profile = self.memory.upsert_user_profile(
                uid,
                style=style,
                verbosity=verbosity,
                domain_preferences=domain_preferences,
                hard_constraints=hard_constraints,
            )

        old_reward = record.reward_score
        blended, components = self._blend_feedback_reward(record, profile)
        record.reward_score = blended
        record.reward_components.update(components)
        record.user_id = uid

        updated_branch_rewards: dict[str, float] = {}
        path = record.activated_branches or list(record.branch_rewards.keys())
        n = len(path)
        for idx, branch_name in enumerate(path):
            prev = record.branch_rewards.get(branch_name, old_reward)
            target = blended * (0.9 ** max(0, n - 1 - idx))
            new_reward = (0.4 * prev) + (0.6 * target)
            new_reward = max(0.0, min(1.0, new_reward))
            record.branch_rewards[branch_name] = new_reward
            updated_branch_rewards[branch_name] = round(new_reward, 4)

            branch = self.branches.get(branch_name)
            if branch is None or not branch.is_active:
                continue
            branch.apply_reward(new_reward)
            delta = 0.5 * self.config.optimizer.learning_rate * (new_reward - 0.5)
            new_weight = branch.state.weight + delta
            branch.state.weight = max(self.config.optimizer.min_weight, min(self.config.optimizer.max_weight, new_weight))

        # Persist record updates.
        self.memory.persist_records()

        payload = {
            "type": "feedback",
            "task_id": task_id,
            "user_id": uid,
            "old_reward": round(old_reward, 4),
            "new_reward": round(blended, 4),
            "reward_components": components,
            "updated_branch_rewards": updated_branch_rewards,
            "profile": profile,
        }
        append_jsonl(self._event_log, payload)
        return {"ok": True, **payload}

    def _run_path(self, route: RoutingDecision, task: TaskInput) -> dict[str, Any]:
        outputs: dict[str, Any] = {}
        strict_contract = infer_output_contract(task.text, task.metadata)
        strict_mode = strict_contract is not None

        paths = route.activated_paths or ([route.activated_branches] if route.activated_branches else [])
        if not paths:
            return outputs

        for path in paths:
            context = task.metadata.get("context_seed", "")
            for branch_name in path:
                branch = self.branches.get(branch_name)
                if branch is None or not branch.is_active:
                    continue

                if branch_name in outputs:
                    context = self._roll_context(context, outputs[branch_name].output, strict_mode=strict_mode)
                    continue

                branch_context = self._augment_context_for_branch(task, branch, context)
                branch_output = self.executor.run_branch(branch, task, route.task_type, context=branch_context)
                if not self.forest.children(branch_name):
                    branch_output = self._maybe_refine_leaf_output(task, branch, branch_output)
                outputs[branch_name] = branch_output
                context = self._roll_context(context, branch_output.output, strict_mode=strict_mode)
        return outputs

    def _augment_context_for_branch(self, task: TaskInput, branch: PromptBranch, context: str) -> str:
        base_context = str(context or "").strip()
        user_id = str(task.metadata.get("user_id", "global")).strip() or "global"
        memory_hints = self.memory.execution_guidance(task, branch.name, user_id=user_id, limit=2)
        adaptive_hint = str(branch.state.metadata.get("adaptive_execution_hint", "")).strip()

        sections: list[str] = []
        if base_context:
            sections.append(base_context[-520:])

        guidance_lines: list[str] = []
        if adaptive_hint:
            guidance_lines.append(f"Adaptive branch hint: {adaptive_hint}")
        for hint in memory_hints:
            guidance_lines.append(f"Similar success: {hint}")

        if guidance_lines:
            guidance = "Execution guidance:\n" + "\n".join(f"- {line}" for line in guidance_lines)
            sections.append(guidance[:520])

        return "\n\n".join(section for section in sections if section).strip()

    def _maybe_refine_leaf_output(self, task: TaskInput, branch: PromptBranch, branch_output):
        if not self.config.execution_adaptation.enabled:
            return branch_output

        user_id = str(task.metadata.get("user_id", "global")).strip() or "global"
        playbook = self.memory.execution_playbook(
            task,
            branch.name,
            user_id=user_id,
            success_limit=self.config.execution_adaptation.max_success_examples + 1,
            failure_limit=self.config.execution_adaptation.max_failure_examples,
            min_similarity=self.config.execution_adaptation.min_similarity,
        )
        if playbook.support < self.config.execution_adaptation.min_success_support:
            return branch_output

        required_items = self._merge_coverage_items(playbook.coverage_items, self._task_priority_items(task))
        missing_before = self._missing_coverage_items(branch_output.output, required_items)
        should_refine = bool(missing_before) or bool(playbook.anti_patterns) or playbook.support > self.config.execution_adaptation.min_success_support
        if not should_refine:
            return branch_output

        original_score = self.judge.score_output(branch_output.output, task).reward
        support_output = ""
        if self.config.execution_adaptation.enable_support_pass and (
            missing_before or original_score < self.config.execution_adaptation.support_pass_reward_floor
        ):
            support_output = self._support_branch_analysis(task, branch.name, branch_output.output, playbook, missing_before)

        candidates: list[dict[str, Any]] = []
        plain_candidate = self._generate_refinement_candidate(
            task=task,
            branch_name=branch.name,
            branch_output=branch_output,
            playbook=playbook,
            required_items=required_items,
            missing_before=missing_before,
            support_output="",
        )
        if plain_candidate:
            candidates.append(plain_candidate)
        if support_output:
            support_candidate = self._generate_refinement_candidate(
                task=task,
                branch_name=branch.name,
                branch_output=branch_output,
                playbook=playbook,
                required_items=required_items,
                missing_before=missing_before,
                support_output=support_output,
            )
            if support_candidate:
                candidates.append(support_candidate)
        if not candidates:
            return branch_output

        best_candidate = max(
            candidates,
            key=lambda item: (item["score"], -len(item["missing_after"]), item["strategy"] == "support_pass"),
        )
        refined_text = str(best_candidate["text"]).strip()
        refined_score = float(best_candidate["score"])
        missing_after = list(best_candidate["missing_after"])
        improved_coverage = len(missing_after) < len(missing_before)
        gain = refined_score - original_score
        min_gain = max(0.0, self.config.execution_adaptation.min_judge_gain)

        if len(missing_after) > len(missing_before):
            return branch_output
        if gain < min_gain and not (improved_coverage and refined_score >= original_score):
            return branch_output

        refined_meta = {
            **branch_output.model_meta,
            **best_candidate["meta"],
            "adaptive_refined": True,
            "adaptive_refine_gain": round(gain, 4),
            "adaptive_refine_support": playbook.support,
            "adaptive_refine_missing_before": missing_before,
            "adaptive_refine_missing_after": missing_after,
            "adaptive_refine_strategy": best_candidate["strategy"],
        }
        return type(branch_output)(
            branch_name=branch_output.branch_name,
            prompt=branch_output.prompt,
            output=refined_text,
            task_type=branch_output.task_type,
            model_meta=refined_meta,
        )

    def _render_refinement_prompt(
        self,
        task: TaskInput,
        branch_name: str,
        draft: str,
        playbook,
        required_items: list[str],
        missing_items: list[str],
        *,
        support_output: str = "",
    ) -> str:
        coverage = required_items or playbook.coverage_items or ["none"]
        structure = playbook.structure_cues or ["keep the structure concise and easy to scan"]
        anti_patterns = playbook.anti_patterns or ["do not add meta commentary"]
        examples = playbook.success_examples[: self.config.execution_adaptation.max_success_examples]
        guidance = playbook.guidance[: self.config.execution_adaptation.max_success_examples]

        sections = [
            "You are improving a branch draft using a learned execution playbook from successful similar tasks.",
            f"Branch: {branch_name}",
            f"Task type: {task.task_type}",
            f"Task: {task.text}",
            "Current draft:",
            draft,
            "Coverage items to satisfy:",
            "\n".join(f"- {item}" for item in coverage),
            "Preferred structure cues:",
            "\n".join(f"- {item}" for item in structure),
            "Avoid these failure patterns:",
            "\n".join(f"- {item}" for item in anti_patterns),
        ]
        task_native_items = self._task_priority_items(task)
        if task_native_items:
            sections.extend(
                [
                    "Task-native signals that must remain explicit in the answer:",
                    "\n".join(f"- {item}" for item in task_native_items),
                ]
            )
        if missing_items:
            sections.extend(
                [
                    "Items missing from the current draft:",
                    "\n".join(f"- {item}" for item in missing_items),
                ]
            )
        if guidance:
            sections.extend(
                [
                    "Learned success cues:",
                    "\n".join(f"- {item}" for item in guidance),
                ]
            )
        if support_output:
            sections.extend(
                [
                    "Support branch analysis:",
                    support_output,
                ]
            )
        if examples:
            sections.extend(
                [
                    "Compact examples from successful prior outputs:",
                    "\n".join(f"- {item}" for item in examples),
                ]
            )
        sections.append(
            "Revise the draft so it better satisfies the task while preserving any correct content. "
            "Make the answer explicit, concrete, and easy to scan. "
            "Do not remove explicit task-native signals that are already present. "
            "If the task asks for confidence, end with a calibrated numeric confidence line such as confidence=0.72. "
            "Return only the revised answer."
        )
        return "\n\n".join(section for section in sections if section)

    def _generate_refinement_candidate(
        self,
        *,
        task: TaskInput,
        branch_name: str,
        branch_output,
        playbook,
        required_items: list[str],
        missing_before: list[str],
        support_output: str,
    ) -> dict[str, Any] | None:
        prompt = self._render_refinement_prompt(
            task,
            branch_name,
            branch_output.output,
            playbook,
            required_items,
            missing_before,
            support_output=support_output,
        )
        max_chars = max(400, self.config.execution_adaptation.max_prompt_chars)
        if len(prompt) > max_chars:
            prompt = prompt[:max_chars].rstrip() + "\n\nReturn only the revised answer."

        refined_text, meta = self.backend.generate(prompt, task, f"{branch_name}__adaptive_refine")
        refined_text = refined_text.strip()
        if not refined_text:
            return None

        return {
            "text": refined_text,
            "meta": meta,
            "score": self.judge.score_output(refined_text, task).reward,
            "missing_after": self._missing_coverage_items(refined_text, required_items),
            "strategy": "support_pass" if support_output else "playbook_only",
        }

    def _support_branch_analysis(
        self,
        task: TaskInput,
        primary_branch_name: str,
        draft: str,
        playbook,
        missing_items: list[str],
    ) -> str:
        support_branch_name = self._support_branch_for_task(task, primary_branch_name, missing_items)
        if not support_branch_name:
            return ""

        support_branch = self.branches.get(support_branch_name)
        if support_branch is None or not support_branch.is_active:
            return ""

        context_sections = [
            "Current draft to inspect:",
            draft,
        ]
        if missing_items:
            context_sections.extend(
                [
                    "Likely missing items:",
                    "\n".join(f"- {item}" for item in missing_items),
                ]
            )
        if playbook.guidance:
            context_sections.extend(
                [
                    "Learned success cues:",
                    "\n".join(f"- {item}" for item in playbook.guidance[:2]),
                ]
            )
        support_context = "\n\n".join(section for section in context_sections if section).strip()
        support_output = self.executor.run_branch(
            support_branch,
            task,
            task.task_type,
            context=support_context[:700],
        )
        return support_output.output.strip()

    @staticmethod
    def _support_branch_for_task(task: TaskInput, primary_branch_name: str, missing_items: list[str]) -> str:
        missing_l = " ".join(item.lower() for item in missing_items)
        text_l = task.text.lower()
        if task.task_type == "planning" or primary_branch_name.startswith("planner"):
            if any(token in (missing_l + " " + text_l) for token in ("rollback", "risk", "owner", "communication")):
                return "verification_constraint_checker"
            return "verification_consistency_auditor"
        if task.task_type == "code":
            if "review" in text_l or "checklist" in text_l:
                return "critique_failure_hunter"
            if any(token in (missing_l + " " + text_l) for token in ("consistency", "contradiction", "uncertainty", "confidence")):
                if primary_branch_name != "verification_consistency_auditor":
                    return "verification_consistency_auditor"
                return "verification_constraint_checker"
            if any(token in missing_l for token in ("tests", "rollback", "monitoring", "validation", "owner")):
                if primary_branch_name != "verification_constraint_checker":
                    return "verification_constraint_checker"
                return "critique_failure_hunter"
            if primary_branch_name == "verification_constraint_checker":
                return "verification_consistency_auditor"
            return "verification_constraint_checker"
        if task.task_type == "general":
            if any(token in text_l for token in ("consistency", "contradiction", "uncertainty", "confidence")):
                return "verification_consistency_auditor"
            if any(token in text_l for token in ("tradeoff", "decision note", "recommendation")):
                return "verification_consistency_auditor"
            return "critique_adversarial_probe"
        if task.task_type == "factual":
            return "verification_consistency_auditor"
        return ""

    @staticmethod
    def _merge_coverage_items(primary: list[str], secondary: list[str], limit: int = 8) -> list[str]:
        merged: list[str] = []
        for item in [*(primary or []), *(secondary or [])]:
            cleaned = str(item).strip()
            if not cleaned or cleaned in merged:
                continue
            merged.append(cleaned)
            if len(merged) >= max(1, limit):
                break
        return merged

    @staticmethod
    def _task_priority_items(task: TaskInput, limit: int = 8) -> list[str]:
        items: list[str] = []
        for source in (task.metadata.get("required_substrings", []), task.metadata.get("expected_keywords", [])):
            if not isinstance(source, list):
                continue
            for raw in source:
                cleaned = str(raw).strip()
                if cleaned and cleaned not in items:
                    items.append(cleaned)
                    if len(items) >= max(1, limit):
                        return items

        text_l = task.text.lower()
        heuristic_items = [
            ("consistency", "consistency"),
            ("contradiction", "contradictions"),
            ("uncertainty", "uncertainty calibration"),
            ("confidence", "confidence"),
            ("owner", "owner"),
            ("rollback", "rollback"),
            ("mitigation", "mitigation"),
            ("monitoring", "monitoring"),
            ("test", "tests"),
        ]
        for needle, item in heuristic_items:
            if needle in text_l and item not in items:
                items.append(item)
            if len(items) >= max(1, limit):
                break
        return items[: max(1, limit)]

    @staticmethod
    def _missing_coverage_items(output: str, coverage_items: list[str]) -> list[str]:
        output_l = output.lower()
        missing: list[str] = []
        for item in coverage_items:
            normalized = str(item).strip().lower()
            if not normalized:
                continue
            variants = {normalized, normalized.replace("-", " ")}
            if normalized.endswith("s"):
                variants.add(normalized[:-1])
            else:
                variants.add(f"{normalized}s")
            if any(variant and variant in output_l for variant in variants):
                continue
            missing.append(item)
        return missing

    @staticmethod
    def _candidate_leaves(route: RoutingDecision, outputs: dict[str, Any]) -> list[str]:
        leaves: list[str] = []
        if route.activated_paths:
            for path in route.activated_paths:
                if not path:
                    continue
                leaf = path[-1]
                if leaf in outputs and leaf not in leaves:
                    leaves.append(leaf)
        elif route.activated_branches:
            leaf = route.activated_branches[-1]
            if leaf in outputs:
                leaves.append(leaf)
        return leaves

    def _aggregate(
        self,
        route: RoutingDecision,
        outputs: dict[str, Any],
        numeric_scores: dict[str, float],
        candidate_branches: list[str] | None = None,
    ) -> AggregationResult:
        if not outputs:
            return AggregationResult(selected_branch="none", selected_output="", notes={"reason": "no_outputs"})

        candidates = [c for c in (candidate_branches or []) if c in outputs]
        if not candidates:
            candidates = list(outputs.keys())

        if self.config.evaluator.aggregation_strategy == "leaf_select" and candidates:
            if len(candidates) == 1:
                leaf = candidates[0]
                return AggregationResult(
                    selected_branch=leaf,
                    selected_output=outputs[leaf].output,
                    notes={"strategy": "leaf_select", "path": route.activated_branches, "candidates": candidates},
                )
            ranked = sorted(
                ((name, numeric_scores.get(name, 0.0)) for name in candidates),
                key=lambda x: x[1],
                reverse=True,
            )
            best = ranked[0][0]
            return AggregationResult(
                selected_branch=best,
                selected_output=outputs[best].output,
                notes={"strategy": "leaf_select_best_multi", "path": route.activated_branches, "ranked": ranked},
            )

        scoped_outputs = {name: outputs[name] for name in candidates}
        scoped_scores = {name: numeric_scores.get(name, 0.0) for name in candidates}
        return self.aggregator.aggregate(scoped_outputs, scoped_scores)

    def _propagate_rewards_along_path(self, route: RoutingDecision, signal, gamma: float, local_mix: float) -> None:
        paths = route.activated_paths or ([route.activated_branches] if route.activated_branches else [])
        if not paths:
            return

        selected_path: list[str] | None = None
        for path in paths:
            if path and path[-1] == signal.selected_branch:
                selected_path = path
                break
        if selected_path is None:
            selected_path = paths[0]
        if not selected_path:
            return

        leaf = selected_path[-1]
        leaf_feedback = signal.branch_feedback.get(leaf)
        leaf_reward = leaf_feedback.reward if leaf_feedback else signal.reward_score

        path_len = len(selected_path)
        for idx, branch_name in enumerate(selected_path):
            fb = signal.branch_feedback.get(branch_name)
            if fb is None:
                continue

            distance = path_len - 1 - idx
            propagated = (leaf_reward * (gamma**distance))
            blended = (local_mix * fb.reward) + ((1.0 - local_mix) * propagated)
            fb.reward = max(0.0, min(1.0, blended))
            signal.branch_feedback[branch_name] = fb

        if signal.selected_branch == leaf:
            signal.reward_score = signal.branch_feedback[leaf].reward

    def _attach_created_candidates(
        self,
        route: RoutingDecision,
        optimize_event: OptimizationEvent,
        selected_branch: str | None = None,
    ) -> None:
        for candidate_name in optimize_event.created_candidates:
            if candidate_name not in self.branches:
                continue
            if self.forest.has_node(candidate_name):
                continue

            parent_id = self.forest.root_id
            meta = self.branches[candidate_name].state.metadata
            requested_parent = str(meta.get("parent_hint", "")).strip()
            if requested_parent and self.forest.has_node(requested_parent):
                parent_id = requested_parent
            elif selected_branch and self.forest.has_node(selected_branch):
                parent_id = selected_branch
            elif route.activated_paths and route.activated_paths[0]:
                parent_id = route.activated_paths[0][-1]
            elif route.activated_branches:
                parent_id = route.activated_branches[-1]

            max_depth = self.config.optimizer.max_hierarchy_depth
            while self.forest.depth(parent_id) >= max_depth - 1 and self.forest.parent(parent_id):
                parent_id = self.forest.parent(parent_id) or self.forest.root_id

            self.forest.add_branch(
                branch_name=candidate_name,
                branch=self.branches[candidate_name],
                parent_id=parent_id,
                specialties=[route.task_type, "general"],
            )

    @staticmethod
    def _selected_path(route: RoutingDecision, selected_branch: str) -> list[str]:
        for path in route.activated_paths:
            if path and path[-1] == selected_branch:
                return list(path)
        if route.activated_paths:
            return list(route.activated_paths[0])
        if route.activated_branches:
            return list(route.activated_branches)
        return []

    @staticmethod
    def _roll_context(current_context: str, new_piece: str, max_chars: int = 800, strict_mode: bool = False) -> str:
        if strict_mode:
            snippet = PromptForestEngine._compact_context_piece(new_piece, max_chars=120)
            joined = f"{current_context}\n{snippet}".strip()
            if len(joined) <= 120:
                return joined
            return joined[-120:]

        joined = f"{current_context}\n{new_piece}".strip()
        if len(joined) <= max_chars:
            return joined
        return joined[-max_chars:]

    @staticmethod
    def _compact_context_piece(text: str, max_chars: int = 120) -> str:
        cleaned = text.replace("```", " ").replace("\n", " ").strip()
        cleaned = " ".join(cleaned.split())
        return cleaned[:max_chars].strip()

    def _mini_holdout_acceptance(
        self,
        *,
        task: TaskInput,
        branch_name: str,
        task_type: str,
        user_id: str,
        old_weight: float,
        new_weight: float,
        old_prompt: str,
        new_prompt: str,
        prompt_rewritten: bool,
    ) -> bool:
        branch = self.branches.get(branch_name)
        if branch is None:
            return True

        # Skip expensive mini-batch replay for tiny weight nudges unless a rewrite happened.
        if (not prompt_rewritten) and abs(new_weight - old_weight) < self.config.optimizer.acceptance_min_delta_for_gate:
            return True

        recent = self.memory.retrieve_similar(
            task_type=task_type,
            limit=max(self.config.optimizer.update_acceptance_window, self.config.optimizer.acceptance_minibatch_size * 2),
            user_id=user_id,
        )
        if not recent and user_id:
            recent = self.memory.retrieve_similar(
                task_type=task_type,
                limit=max(self.config.optimizer.update_acceptance_window, self.config.optimizer.acceptance_minibatch_size * 2),
                user_id=None,
            )
        if not recent:
            return True

        candidates = [r for r in reversed(recent) if branch_name in r.activated_branches or branch_name in r.branch_rewards]
        batch = candidates[: self.config.optimizer.acceptance_minibatch_size]
        if len(batch) < self.config.optimizer.acceptance_minibatch_min_samples:
            return True

        original_weight = branch.state.weight
        original_prompt = branch.state.prompt_template
        old_scores: list[float] = []
        new_scores: list[float] = []

        try:
            for record in batch:
                replay_task = TaskInput(
                    task_id=record.task_id,
                    text=record.input_text,
                    task_type=record.task_type,
                    metadata=dict(record.task_metadata or {}),
                )
                context_seed = str(replay_task.metadata.get("context_seed", ""))
                replay_context = self._augment_context_for_branch(replay_task, branch, context_seed)

                branch.state.weight = old_weight
                branch.state.prompt_template = old_prompt
                old_output = self.executor.run_branch(branch, replay_task, replay_task.task_type, context=replay_context)
                old_score = self.judge.score_output(old_output.output, replay_task).reward

                branch.state.weight = new_weight
                branch.state.prompt_template = new_prompt
                new_output = self.executor.run_branch(branch, replay_task, replay_task.task_type, context=replay_context)
                new_score = self.judge.score_output(new_output.output, replay_task).reward

                old_scores.append(old_score)
                new_scores.append(new_score)
        finally:
            branch.state.weight = original_weight
            branch.state.prompt_template = original_prompt

        if not old_scores or not new_scores:
            return True

        before = mean(old_scores)
        after = mean(new_scores)
        return after >= (before + self.config.optimizer.update_acceptance_min_gain)

    def branch_snapshot(self) -> dict[str, dict[str, Any]]:
        return self.forest.branch_snapshot()

    def routing_histogram(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for route in self._route_history:
            for branch_name in route.activated_branches:
                counts[branch_name] = counts.get(branch_name, 0) + 1
        return counts

    def openclaw_ingest(self, trajectory_event: dict[str, Any]) -> dict[str, Any]:
        """Compatibility hook for OpenClaw-style runtime events."""
        task_text = trajectory_event.get("task", "")
        task_type = trajectory_event.get("task_type", "auto")
        metadata = {
            "expected_keywords": trajectory_event.get("expected_keywords", []),
            "required_substrings": trajectory_event.get("required_checks", []),
            "trajectory": trajectory_event,
            "user_id": trajectory_event.get("user_id", "global"),
        }
        return self.run_task(text=task_text, task_type=task_type, metadata=metadata)

    def _compute_reward_components(self, task: TaskInput, output: str, llm_judge_score: float) -> dict[str, float]:
        rule_verifier_score, _ = RuleBasedReward(weight=1.0).score(output, task)
        external_verifier_score, _ = ExternalVerifierReward(weight=1.0).score(output, task)
        verifier_score = 0.5 * (rule_verifier_score + external_verifier_score)
        keyword_score, _ = KeywordReward(weight=1.0).score(output, task)
        task_specific_score, _ = TaskSpecificReward(weight=1.0).score(output, task)
        task_rules = 0.5 * (keyword_score + task_specific_score)
        return {
            "verifier": round(verifier_score, 4),
            "rule_verifier": round(rule_verifier_score, 4),
            "external_verifier": round(external_verifier_score, 4),
            "task_rules": round(task_rules, 4),
            "llm_judge": round(max(0.0, min(1.0, llm_judge_score)), 4),
        }

    def _blend_feedback_reward(self, record: MemoryRecord, profile: dict[str, Any]) -> tuple[float, dict[str, float]]:
        metadata = dict(record.task_metadata or {})
        if profile:
            metadata["user_preferences"] = profile
        task = TaskInput(
            task_id=record.task_id,
            text=record.input_text,
            task_type=record.task_type,
            metadata=metadata,
        )
        output = record.selected_output
        rule_verifier_score, _ = RuleBasedReward(weight=1.0).score(output, task)
        external_verifier_score, _ = ExternalVerifierReward(weight=1.0).score(output, task)
        verifier_score = 0.5 * (rule_verifier_score + external_verifier_score)
        keyword_score, _ = KeywordReward(weight=1.0).score(output, task)
        task_specific_score, _ = TaskSpecificReward(weight=1.0).score(output, task)
        task_rules_score = 0.5 * (keyword_score + task_specific_score)
        llm_judge_score = float(record.reward_components.get("llm_judge", record.reward_score))

        user_feedback = float(record.feedback_score if record.feedback_score is not None else 0.5)
        user_feedback = max(0.0, min(1.0, user_feedback))
        blended = (
            self.config.feedback.user_feedback_weight * user_feedback
            + self.config.feedback.verifier_weight * verifier_score
            + self.config.feedback.task_rules_weight * task_rules_score
            + self.config.feedback.llm_judge_weight * llm_judge_score
        )

        correction_alignment = 0.0
        correction_signal = user_feedback
        if record.corrected_answer:
            correction_alignment = self._correction_alignment(output, record.corrected_answer)
            if record.accepted is False:
                correction_signal = max(0.0, min(1.0, min(user_feedback, correction_alignment)))
            elif record.accepted is True:
                correction_signal = max(user_feedback, correction_alignment)
            else:
                correction_signal = 0.5 * (user_feedback + correction_alignment)

            anchor = max(0.75, min(0.95, self.config.feedback.correction_anchor_weight))
            blended = (anchor * correction_signal) + ((1.0 - anchor) * blended)

        pref_penalty = self._preference_penalty(output, profile)
        blended = max(0.0, min(1.0, blended - pref_penalty))
        components = {
            "user_feedback": round(user_feedback, 4),
            "verifier": round(verifier_score, 4),
            "rule_verifier": round(rule_verifier_score, 4),
            "external_verifier": round(external_verifier_score, 4),
            "task_rules": round(task_rules_score, 4),
            "llm_judge": round(llm_judge_score, 4),
            "correction_alignment": round(correction_alignment, 4),
            "correction_signal": round(correction_signal, 4),
            "preference_penalty": round(pref_penalty, 4),
            "blended_reward": round(blended, 4),
        }
        return blended, components

    @staticmethod
    def _correction_alignment(output: str, corrected_answer: str) -> float:
        out_tokens = {tok for tok in output.lower().split() if tok}
        corr_tokens = {tok for tok in corrected_answer.lower().split() if tok}
        if not corr_tokens:
            return 0.0
        overlap = len(out_tokens & corr_tokens)
        return overlap / len(corr_tokens)

    def _preference_penalty(self, output: str, profile: dict[str, Any]) -> float:
        if not profile:
            return 0.0
        output_l = output.lower()
        penalty = 0.0
        constraints = profile.get("hard_constraints", [])
        if isinstance(constraints, list):
            missing = [c for c in constraints if str(c).lower() not in output_l]
            penalty += 0.05 * len(missing)

        verbosity = str(profile.get("verbosity", "")).lower()
        token_count = len(output.split())
        if verbosity == "concise" and token_count > 140:
            penalty += 0.08
        elif verbosity in {"detailed", "high"} and token_count < 50:
            penalty += 0.06

        return min(self.config.feedback.preference_penalty_cap, penalty)
