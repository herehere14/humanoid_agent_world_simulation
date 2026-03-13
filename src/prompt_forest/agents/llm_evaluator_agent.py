from __future__ import annotations

from typing import Any

from ..aggregator.strategies import AggregationResult
from ..config import AgentRuntimeConfig
from ..evaluator.judge import BranchScore
from ..types import BranchFeedback, EvaluationSignal, RoutingDecision, TaskInput
from .evaluator_agent import EvaluatorAgent
from .runtime_client import AgentRuntimeClient


class LLMEvaluatorAgent:
    """LLM-backed Agent 1 with deterministic fallback."""

    def __init__(self, runtime_config: AgentRuntimeConfig, fallback: EvaluatorAgent | None = None) -> None:
        self.runtime = AgentRuntimeClient(runtime_config)
        self.fallback = fallback or EvaluatorAgent()

    def evaluate(
        self,
        task: TaskInput,
        route: RoutingDecision,
        branch_scores: dict[str, BranchScore],
        aggregation: AggregationResult,
        branch_outputs: dict[str, str] | None = None,
    ) -> EvaluationSignal:
        base = self.fallback.evaluate(task, route, branch_scores, aggregation, branch_outputs=branch_outputs)
        if not self.runtime.is_enabled():
            return base

        payload = {
            "task": {
                "task_type": route.task_type,
                "text": task.text,
                "metadata": task.metadata,
            },
            "activated_branches": route.activated_branches,
            "branch_scores": {k: {"reward": v.reward, "reason": v.reason} for k, v in branch_scores.items()},
            "branch_outputs": branch_outputs or {},
            "aggregation": {
                "selected_branch": aggregation.selected_branch,
                "selected_output": aggregation.selected_output,
                "notes": aggregation.notes,
            },
        }
        system_prompt = (
            "You are Agent 1 (Evaluator/Judge) for a prompt-forest RL system.\n"
            "Return STRICT JSON only.\n"
            "Score quality and reliability of branch outputs and pick selected branch.\n"
            "Schema:\n"
            "{\n"
            '  "selected_branch": "branch_name",\n'
            '  "reward_score": 0.0,\n'
            '  "confidence": 0.0,\n'
            '  "failure_reason": "short_reason_or_empty",\n'
            '  "suggested_improvement_direction": "actionable_hint",\n'
            '  "branch_feedback": {\n'
            '    "branch_name": {\n'
            '      "reward": 0.0,\n'
            '      "confidence": 0.0,\n'
            '      "failure_reason": "short_reason_or_empty",\n'
            '      "suggested_improvement_direction": "actionable_hint"\n'
            "    }\n"
            "  }\n"
            "}\n"
            "Rules: reward/confidence in [0,1], every activated branch should appear in branch_feedback."
        )

        try:
            raw = self.runtime.generate_json(system_prompt, payload)
            return self._to_signal(raw, base, route, aggregation)
        except Exception as exc:  # pragma: no cover - runtime path depends on external APIs
            base.aggregator_notes = dict(base.aggregator_notes or {})
            base.aggregator_notes["evaluator_runtime_error"] = str(exc)
            return base

    def _to_signal(
        self,
        raw: dict[str, Any],
        base: EvaluationSignal,
        route: RoutingDecision,
        aggregation: AggregationResult,
    ) -> EvaluationSignal:
        def clamp(value: Any, default: float) -> float:
            try:
                v = float(value)
            except (TypeError, ValueError):
                return default
            return max(0.0, min(1.0, v))

        selected_branch = str(raw.get("selected_branch", base.selected_branch))
        if selected_branch not in route.activated_branches:
            selected_branch = base.selected_branch

        branch_feedback_raw = raw.get("branch_feedback", {})
        feedback: dict[str, BranchFeedback] = {}
        for branch_name in route.activated_branches:
            src = branch_feedback_raw.get(branch_name, {})
            base_fb = base.branch_feedback.get(branch_name)
            feedback[branch_name] = BranchFeedback(
                branch_name=branch_name,
                reward=clamp(src.get("reward"), base_fb.reward if base_fb else base.reward_score),
                confidence=clamp(src.get("confidence"), base_fb.confidence if base_fb else base.confidence),
                failure_reason=str(src.get("failure_reason", base_fb.failure_reason if base_fb else "")),
                suggested_improvement_direction=str(
                    src.get(
                        "suggested_improvement_direction",
                        base_fb.suggested_improvement_direction if base_fb else base.suggested_improvement_direction,
                    )
                ),
            )

        notes = dict(base.aggregator_notes or {})
        notes["evaluator_runtime"] = "llm"

        return EvaluationSignal(
            reward_score=clamp(raw.get("reward_score"), base.reward_score),
            confidence=clamp(raw.get("confidence"), base.confidence),
            selected_branch=selected_branch,
            selected_output=aggregation.selected_output,
            failure_reason=str(raw.get("failure_reason", base.failure_reason)),
            suggested_improvement_direction=str(
                raw.get("suggested_improvement_direction", base.suggested_improvement_direction)
            ),
            branch_feedback=feedback,
            aggregator_notes=notes,
        )
