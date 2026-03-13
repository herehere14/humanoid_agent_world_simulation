from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
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
from ..evaluator.judge import OutputJudge
from ..memory.store import MemoryStore
from ..rewards.modes import KeywordReward, RuleBasedReward, TaskSpecificReward
from ..router.hierarchical_router import HierarchicalRouter
from ..types import MemoryRecord, RoutingDecision, TaskInput
from ..utils.io import append_jsonl, ensure_parent
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
        user_profile = self.memory.get_user_profile(user_id)
        if user_profile:
            task_metadata["user_preferences"] = user_profile
        task = TaskInput(task_id=str(uuid4()), text=text, task_type=task_type, metadata=task_metadata)

        route = self.router.route(task, self.forest, self.memory)
        task.task_type = route.task_type
        self._route_history.append(route)

        outputs = self._run_path(route, task)
        branch_scores = self.judge.score_all(outputs, task)
        numeric_scores = {k: v.reward for k, v in branch_scores.items()}

        aggregation = self._aggregate(route, outputs, numeric_scores)
        signal = self.evaluator_agent.evaluate(
            task,
            route,
            branch_scores,
            aggregation,
            branch_outputs={k: v.output for k, v in outputs.items()},
        )
        self._propagate_rewards_along_path(route, signal, gamma=0.9, local_mix=0.55)
        reward_components = self._compute_reward_components(task, signal.selected_output, signal.reward_score)

        if adapt:
            optimize_event: OptimizationEvent = self.optimizer_agent.optimize(
                task=task,
                route=route,
                signal=signal,
                branches=self.branches,
                memory=self.memory,
            )
            self._attach_created_candidates(route, optimize_event)
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
            user_id=user_id,
            task_metadata=task.metadata,
            reward_components=reward_components,
        )
        if update_memory:
            self.memory.add(record)

        payload = {
            "task": asdict(task),
            "routing": asdict(route),
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
            "runtime": {
                "evaluator_llm_enabled": self.config.agent_runtimes.evaluator.enabled,
                "optimizer_llm_enabled": self.config.agent_runtimes.optimizer.enabled,
                "evaluator_provider": self.config.agent_runtimes.evaluator.provider,
                "optimizer_provider": self.config.agent_runtimes.optimizer.provider,
            },
            "branch_weights": {name: round(branch.state.weight, 4) for name, branch in self.branches.items()},
        }
        append_jsonl(self._event_log, payload)

        return payload

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
        outputs = {}
        context = task.metadata.get("context_seed", "")

        for branch_name in route.activated_branches:
            branch = self.branches.get(branch_name)
            if branch is None or not branch.is_active:
                continue

            branch_output = self.executor.run_branch(branch, task, route.task_type, context=context)
            outputs[branch_name] = branch_output
            context = self._roll_context(context, branch_output.output)

        return outputs

    def _aggregate(
        self,
        route: RoutingDecision,
        outputs: dict[str, Any],
        numeric_scores: dict[str, float],
    ) -> AggregationResult:
        if not outputs:
            return AggregationResult(selected_branch="none", selected_output="", notes={"reason": "no_outputs"})

        if self.config.evaluator.aggregation_strategy == "leaf_select" and route.activated_branches:
            leaf = route.activated_branches[-1]
            if leaf in outputs:
                return AggregationResult(
                    selected_branch=leaf,
                    selected_output=outputs[leaf].output,
                    notes={"strategy": "leaf_select", "path": route.activated_branches},
                )

        return self.aggregator.aggregate(outputs, numeric_scores)

    def _propagate_rewards_along_path(self, route: RoutingDecision, signal, gamma: float, local_mix: float) -> None:
        if not route.activated_branches:
            return

        leaf = route.activated_branches[-1]
        leaf_feedback = signal.branch_feedback.get(leaf)
        leaf_reward = leaf_feedback.reward if leaf_feedback else signal.reward_score

        path_len = len(route.activated_branches)
        for idx, branch_name in enumerate(route.activated_branches):
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

    def _attach_created_candidates(self, route: RoutingDecision, optimize_event: OptimizationEvent) -> None:
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
    def _roll_context(current_context: str, new_piece: str, max_chars: int = 800) -> str:
        joined = f"{current_context}\n{new_piece}".strip()
        if len(joined) <= max_chars:
            return joined
        return joined[-max_chars:]

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
        verifier_score, _ = RuleBasedReward(weight=1.0).score(output, task)
        keyword_score, _ = KeywordReward(weight=1.0).score(output, task)
        task_specific_score, _ = TaskSpecificReward(weight=1.0).score(output, task)
        task_rules = 0.5 * (keyword_score + task_specific_score)
        return {
            "verifier": round(verifier_score, 4),
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
        verifier_score, _ = RuleBasedReward(weight=1.0).score(output, task)
        keyword_score, _ = KeywordReward(weight=1.0).score(output, task)
        task_specific_score, _ = TaskSpecificReward(weight=1.0).score(output, task)
        task_rules_score = 0.5 * (keyword_score + task_specific_score)
        llm_judge_score = float(record.reward_components.get("llm_judge", record.reward_score))

        user_feedback = float(record.feedback_score if record.feedback_score is not None else 0.5)
        user_feedback = max(0.0, min(1.0, user_feedback))

        if record.corrected_answer and record.accepted is False:
            user_feedback = min(user_feedback, 0.2)
            blended = (
                self.config.feedback.correction_anchor_weight * user_feedback
                + (1.0 - self.config.feedback.correction_anchor_weight) * (0.6 * verifier_score + 0.4 * llm_judge_score)
            )
        else:
            blended = (
                self.config.feedback.user_feedback_weight * user_feedback
                + self.config.feedback.verifier_weight * verifier_score
                + self.config.feedback.task_rules_weight * task_rules_score
                + self.config.feedback.llm_judge_weight * llm_judge_score
            )

        pref_penalty = self._preference_penalty(output, profile)
        blended = max(0.0, min(1.0, blended - pref_penalty))
        components = {
            "user_feedback": round(user_feedback, 4),
            "verifier": round(verifier_score, 4),
            "task_rules": round(task_rules_score, 4),
            "llm_judge": round(llm_judge_score, 4),
            "preference_penalty": round(pref_penalty, 4),
            "blended_reward": round(blended, 4),
        }
        return blended, components

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
