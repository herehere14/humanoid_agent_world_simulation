from __future__ import annotations

from ..aggregator.strategies import AggregationResult
from ..evaluator.judge import BranchScore
from ..types import BranchFeedback, EvaluationSignal, RoutingDecision, TaskInput


class EvaluatorAgent:
    """Agent 1: converts judge metrics into a structured optimization signal."""

    def evaluate(
        self,
        task: TaskInput,
        route: RoutingDecision,
        branch_scores: dict[str, BranchScore],
        aggregation: AggregationResult,
    ) -> EvaluationSignal:
        selected_branch = aggregation.selected_branch
        selected_score = branch_scores.get(selected_branch, BranchScore(reward=0.0, reason="missing_selected"))

        ranked_rewards = sorted((name, sc.reward) for name, sc in branch_scores.items())
        ranked_rewards_desc = sorted(ranked_rewards, key=lambda x: x[1], reverse=True)

        if len(ranked_rewards_desc) >= 2:
            margin = ranked_rewards_desc[0][1] - ranked_rewards_desc[1][1]
        elif ranked_rewards_desc:
            margin = ranked_rewards_desc[0][1]
        else:
            margin = 0.0

        confidence = max(0.05, min(0.99, 0.5 + margin))

        if selected_score.reward >= 0.75:
            failure_reason = ""
            improvement = "preserve_success_pattern"
        elif selected_score.reward >= 0.5:
            failure_reason = selected_score.reason
            improvement = self._improvement_from_reason(selected_score.reason, task.task_type)
        else:
            failure_reason = selected_score.reason or "low_reward"
            improvement = self._improvement_from_reason(failure_reason, task.task_type)

        feedback: dict[str, BranchFeedback] = {}
        for branch_name in route.activated_branches:
            score = branch_scores.get(branch_name, BranchScore(0.0, "missing_score"))
            branch_conf = max(0.05, min(0.99, 0.45 + abs(score.reward - 0.5)))
            feedback[branch_name] = BranchFeedback(
                branch_name=branch_name,
                reward=score.reward,
                confidence=branch_conf,
                failure_reason="" if score.reward >= 0.75 else score.reason,
                suggested_improvement_direction=self._improvement_from_reason(score.reason, task.task_type),
            )

        return EvaluationSignal(
            reward_score=selected_score.reward,
            confidence=confidence,
            selected_branch=selected_branch,
            selected_output=aggregation.selected_output,
            failure_reason=failure_reason,
            suggested_improvement_direction=improvement,
            branch_feedback=feedback,
            aggregator_notes=aggregation.notes,
        )

    def _improvement_from_reason(self, reason: str, task_type: str) -> str:
        reason_l = reason.lower()
        if "exact_mismatch" in reason_l:
            return "increase_verification_and_ground_truth_alignment"
        if "keyword_coverage" in reason_l:
            return "improve_keyword_coverage"
        if "rule_miss" in reason_l:
            return "add_constraint_satisfaction_checks"
        if "low_quality" in reason_l:
            return f"strengthen_{task_type}_specialization"
        return "improve_clarity_and_verification"
