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
        self._max_output_chars = 360
        self._max_reason_chars = 140
        self._max_keywords = 12
        # Conservative blend: keep deterministic judge as anchor, only trust LLM strongly at high confidence.
        self._override_conf_threshold = 0.8
        self._max_reward_blend = 0.65
        self._max_branch_blend = 0.7
        self._reward_drift_cap = 0.2
        self._branch_reward_drift_cap = 0.25

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
                "metadata": self._compact_metadata(task.metadata),
            },
            "activated_branches": route.activated_branches,
            "branch_scores": {
                k: {"reward": v.reward, "reason": self._trim(v.reason, self._max_reason_chars)}
                for k, v in branch_scores.items()
            },
            "branch_outputs": self._compact_outputs(branch_outputs or {}, route.activated_branches),
            "aggregation": {
                "selected_branch": aggregation.selected_branch,
                "selected_output": self._trim(aggregation.selected_output, self._max_output_chars),
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
            llm_signal = self._to_signal(raw, base, route, aggregation)
            if self.runtime.config.proposal_only:
                return self._proposal_only_signal(base, llm_signal, route)
            return self._stabilize_signal(base, llm_signal, route, aggregation)
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

    def _stabilize_signal(
        self,
        base: EvaluationSignal,
        llm_signal: EvaluationSignal,
        route: RoutingDecision,
        aggregation: AggregationResult,
    ) -> EvaluationSignal:
        def clamp01(value: float) -> float:
            return max(0.0, min(1.0, value))

        def blend_alpha(confidence: float, max_blend: float) -> float:
            conf = clamp01(confidence)
            if conf <= self._override_conf_threshold:
                return 0.0
            scaled = (conf - self._override_conf_threshold) / max(1e-8, 1.0 - self._override_conf_threshold)
            return clamp01(scaled) * max_blend

        selected_branch = base.selected_branch
        if llm_signal.selected_branch in route.activated_branches and llm_signal.confidence >= self._override_conf_threshold:
            selected_branch = llm_signal.selected_branch

        reward_alpha = blend_alpha(llm_signal.confidence, self._max_reward_blend)
        blended_reward = base.reward_score + reward_alpha * (llm_signal.reward_score - base.reward_score)
        reward_drift = blended_reward - base.reward_score
        reward_drift = max(-self._reward_drift_cap, min(self._reward_drift_cap, reward_drift))
        final_reward = clamp01(base.reward_score + reward_drift)

        feedback: dict[str, BranchFeedback] = {}
        for branch_name in route.activated_branches:
            base_fb = base.branch_feedback.get(branch_name)
            llm_fb = llm_signal.branch_feedback.get(branch_name)
            if base_fb is None and llm_fb is None:
                continue
            if base_fb is None:
                base_fb = BranchFeedback(
                    branch_name=branch_name,
                    reward=base.reward_score,
                    confidence=base.confidence,
                    failure_reason=base.failure_reason,
                    suggested_improvement_direction=base.suggested_improvement_direction,
                )
            if llm_fb is None:
                llm_fb = base_fb

            branch_alpha = blend_alpha(llm_fb.confidence, self._max_branch_blend)
            blended_branch_reward = base_fb.reward + branch_alpha * (llm_fb.reward - base_fb.reward)
            branch_drift = blended_branch_reward - base_fb.reward
            branch_drift = max(-self._branch_reward_drift_cap, min(self._branch_reward_drift_cap, branch_drift))
            final_branch_reward = clamp01(base_fb.reward + branch_drift)

            failure_reason = llm_fb.failure_reason if branch_alpha >= 0.25 else base_fb.failure_reason
            suggestion = (
                llm_fb.suggested_improvement_direction
                if branch_alpha >= 0.25
                else base_fb.suggested_improvement_direction
            )
            final_branch_conf = max(base_fb.confidence, llm_fb.confidence * branch_alpha)

            feedback[branch_name] = BranchFeedback(
                branch_name=branch_name,
                reward=final_branch_reward,
                confidence=clamp01(final_branch_conf),
                failure_reason=failure_reason,
                suggested_improvement_direction=suggestion,
            )

        final_conf = max(base.confidence, min(1.0, llm_signal.confidence * 0.9))
        final_failure_reason = llm_signal.failure_reason if reward_alpha >= 0.25 else base.failure_reason
        final_improvement = (
            llm_signal.suggested_improvement_direction
            if reward_alpha >= 0.25
            else base.suggested_improvement_direction
        )

        notes = dict(base.aggregator_notes or {})
        notes["evaluator_runtime"] = "llm_blended"
        notes["evaluator_reward_blend_alpha"] = round(reward_alpha, 4)
        notes["evaluator_reward_base"] = round(base.reward_score, 4)
        notes["evaluator_reward_llm"] = round(llm_signal.reward_score, 4)
        notes["evaluator_selected_branch_llm"] = llm_signal.selected_branch

        return EvaluationSignal(
            reward_score=final_reward,
            confidence=final_conf,
            selected_branch=selected_branch,
            selected_output=aggregation.selected_output,
            failure_reason=final_failure_reason,
            suggested_improvement_direction=final_improvement,
            branch_feedback=feedback,
            aggregator_notes=notes,
        )

    def _proposal_only_signal(
        self,
        base: EvaluationSignal,
        llm_signal: EvaluationSignal,
        route: RoutingDecision,
    ) -> EvaluationSignal:
        feedback: dict[str, BranchFeedback] = {}
        for branch_name in route.activated_branches:
            base_fb = base.branch_feedback.get(branch_name)
            llm_fb = llm_signal.branch_feedback.get(branch_name)
            if base_fb is None:
                continue

            use_llm_text = bool(llm_fb and llm_fb.confidence >= self._override_conf_threshold)
            feedback[branch_name] = BranchFeedback(
                branch_name=branch_name,
                reward=base_fb.reward,
                confidence=base_fb.confidence,
                failure_reason=(llm_fb.failure_reason if use_llm_text else base_fb.failure_reason) if llm_fb else base_fb.failure_reason,
                suggested_improvement_direction=(
                    llm_fb.suggested_improvement_direction if use_llm_text else base_fb.suggested_improvement_direction
                )
                if llm_fb
                else base_fb.suggested_improvement_direction,
            )

        notes = dict(base.aggregator_notes or {})
        notes["evaluator_runtime"] = "llm_proposal_only"
        notes["evaluator_selected_branch_llm"] = llm_signal.selected_branch
        notes["evaluator_reward_llm"] = round(llm_signal.reward_score, 4)

        return EvaluationSignal(
            reward_score=base.reward_score,
            confidence=base.confidence,
            selected_branch=base.selected_branch,
            selected_output=base.selected_output,
            failure_reason=base.failure_reason,
            suggested_improvement_direction=base.suggested_improvement_direction,
            branch_feedback=feedback or base.branch_feedback,
            aggregator_notes=notes,
        )

    def _compact_outputs(self, outputs: dict[str, str], activated: list[str]) -> dict[str, str]:
        compact: dict[str, str] = {}
        for name in activated:
            if name not in outputs:
                continue
            compact[name] = self._trim(outputs[name], self._max_output_chars)
        return compact

    def _compact_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        out = dict(metadata or {})
        kws = out.get("expected_keywords")
        if isinstance(kws, list):
            out["expected_keywords"] = kws[: self._max_keywords]
        return out

    @staticmethod
    def _trim(text: Any, max_chars: int) -> str:
        s = str(text)
        if len(s) <= max_chars:
            return s
        return f"{s[: max_chars - 3]}..."
