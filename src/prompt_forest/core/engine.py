from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any
from uuid import uuid4

from ..agents.evaluator_agent import EvaluatorAgent
from ..agents.optimizer_agent import OptimizationEvent, OptimizerAgent
from ..aggregator.strategies import AggregationResult, Aggregator
from ..backend.base import LLMBackend
from ..backend.mock import MockLLMBackend
from ..branches.base import PromptBranch
from ..branches.hierarchical import HierarchicalPromptForest, create_default_hierarchical_forest
from ..config import EngineConfig, load_config
from ..evaluator.judge import OutputJudge
from ..memory.store import MemoryStore
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
        self.evaluator_agent = EvaluatorAgent()
        self.optimizer_agent = OptimizerAgent(self.config.optimizer)
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
        task = TaskInput(task_id=str(uuid4()), text=text, task_type=task_type, metadata=metadata or {})

        route = self.router.route(task, self.forest, self.memory)
        task.task_type = route.task_type
        self._route_history.append(route)

        outputs = self._run_path(route, task)
        branch_scores = self.judge.score_all(outputs, task)
        numeric_scores = {k: v.reward for k, v in branch_scores.items()}

        aggregation = self._aggregate(route, outputs, numeric_scores)
        signal = self.evaluator_agent.evaluate(task, route, branch_scores, aggregation)
        self._propagate_rewards_along_path(route, signal, gamma=0.9, local_mix=0.55)

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
            "branch_weights": {name: round(branch.state.weight, 4) for name, branch in self.branches.items()},
        }
        append_jsonl(self._event_log, payload)

        return payload

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
        }
        return self.run_task(text=task_text, task_type=task_type, metadata=metadata)
