from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any
from uuid import uuid4

from ..agents.evaluator_agent import EvaluatorAgent
from ..agents.optimizer_agent import OptimizationEvent, OptimizerAgent
from ..aggregator.strategies import Aggregator
from ..backend.base import LLMBackend
from ..backend.mock import MockLLMBackend
from ..branches.base import PromptBranch
from ..branches.library import create_default_branches
from ..config import EngineConfig, load_config
from ..evaluator.judge import OutputJudge
from ..memory.store import MemoryStore
from ..router.root_router import RootRouter
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

        self.branches = branches or create_default_branches()
        self.backend = backend or MockLLMBackend()
        self.executor = PromptExecutor(self.backend)

        self.memory = MemoryStore(self.config.memory, memory_path=self.artifacts_dir / "memory_records.jsonl")
        self.router = RootRouter(self.config.router)
        self.judge = OutputJudge(self.config.evaluator.reward_mode)
        self.evaluator_agent = EvaluatorAgent()
        self.optimizer_agent = OptimizerAgent(self.config.optimizer)
        self.aggregator = Aggregator(self.config.evaluator.aggregation_strategy)

        self._route_history: list[RoutingDecision] = []
        self._event_log = self.artifacts_dir / "events.jsonl"

    def run_task(self, text: str, task_type: str = "auto", metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        task = TaskInput(task_id=str(uuid4()), text=text, task_type=task_type, metadata=metadata or {})

        route = self.router.route(task, self.branches, self.memory)
        task.task_type = route.task_type
        self._route_history.append(route)

        outputs = {}
        for branch_name in route.activated_branches:
            branch = self.branches.get(branch_name)
            if branch is None or not branch.is_active:
                continue
            outputs[branch_name] = self.executor.run_branch(branch, task, route.task_type)

        branch_scores = self.judge.score_all(outputs, task)
        numeric_scores = {k: v.reward for k, v in branch_scores.items()}
        aggregation = self.aggregator.aggregate(outputs, numeric_scores)
        signal = self.evaluator_agent.evaluate(task, route, branch_scores, aggregation)

        optimize_event: OptimizationEvent = self.optimizer_agent.optimize(
            task=task,
            route=route,
            signal=signal,
            branches=self.branches,
            memory=self.memory,
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
        self.memory.add(record)

        payload = {
            "task": asdict(task),
            "routing": asdict(route),
            "branch_scores": {k: asdict(v) for k, v in branch_scores.items()},
            "evaluation_signal": {
                "reward_score": signal.reward_score,
                "confidence": signal.confidence,
                "selected_branch": signal.selected_branch,
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

    def branch_snapshot(self) -> dict[str, dict[str, Any]]:
        return {
            name: {
                "weight": round(branch.state.weight, 4),
                "status": branch.state.status.value,
                "avg_reward": round(branch.state.avg_reward(), 4),
                "trial_remaining": branch.state.trial_remaining,
                "history_len": len(branch.state.historical_rewards),
            }
            for name, branch in self.branches.items()
        }

    def routing_histogram(self) -> dict[str, int]:
        return self.router.branch_histogram(self._route_history)

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
