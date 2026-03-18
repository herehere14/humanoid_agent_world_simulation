"""Human Mode Evaluator: coherence-based evaluation.

Agent Improvement evaluators measure task quality (correctness, coverage).
Human Mode evaluators measure *behavioral coherence* -- whether the response
is internally consistent, believable given the agent's state, and handles
internal conflicts in a psychologically realistic way.
"""

from __future__ import annotations

from typing import Any

from ...aggregator.strategies import AggregationResult
from ...evaluator.judge import BranchScore
from ...state.human_state import DriveConflict, HumanState
from ...types import BranchFeedback, EvaluationSignal, RoutingDecision, TaskInput


class HumanModeEvaluator:
    """Evaluate outputs for behavioral coherence rather than task correctness."""

    def __init__(
        self,
        coherence_weight: float = 0.30,
        consistency_weight: float = 0.25,
        believability_weight: float = 0.25,
        conflict_handling_weight: float = 0.20,
    ) -> None:
        self.coherence_w = coherence_weight
        self.consistency_w = consistency_weight
        self.believability_w = believability_weight
        self.conflict_w = conflict_handling_weight

    def evaluate(
        self,
        task: TaskInput,
        route: RoutingDecision,
        branch_scores: dict[str, BranchScore],
        aggregation: AggregationResult,
        state: HumanState,
        conflicts: list[DriveConflict],
        branch_outputs: dict[str, str] | None = None,
    ) -> EvaluationSignal:
        """Produce an evaluation signal based on behavioral coherence.

        Scoring dimensions:
          1. Coherence: Does the output reflect the currently dominant drives?
          2. Consistency: Is the response consistent with recent state history?
          3. Believability: Does the emotional tone match the internal state?
          4. Conflict handling: If conflicts exist, does the output acknowledge
             the tension rather than ignoring it?
        """
        outputs = branch_outputs or {}
        selected_output = aggregation.selected_output
        selected_branch = aggregation.selected_branch

        # 1. Coherence: dominant drives should be reflected in output
        coherence = self._score_coherence(selected_output, state, route)

        # 2. Consistency: response should feel like it follows from recent state
        consistency = self._score_consistency(selected_output, state)

        # 3. Believability: emotional tone should match state
        believability = self._score_believability(selected_output, state)

        # 4. Conflict handling: if conflicts exist, output should reflect tension
        conflict_score = self._score_conflict_handling(
            selected_output, conflicts, outputs
        )

        # Composite reward
        reward = (
            self.coherence_w * coherence
            + self.consistency_w * consistency
            + self.believability_w * believability
            + self.conflict_w * conflict_score
        )
        reward = max(0.0, min(1.0, reward))

        # Compute confidence from agreement between branches
        if len(branch_scores) >= 2:
            scores_sorted = sorted(
                branch_scores.values(), key=lambda s: s.reward, reverse=True
            )
            margin = scores_sorted[0].reward - scores_sorted[1].reward
            confidence = max(0.1, min(0.95, 0.4 + margin))
        else:
            confidence = 0.5

        # Determine failure reasons
        failure_parts = []
        if coherence < 0.4:
            failure_parts.append("drive_incoherence")
        if consistency < 0.4:
            failure_parts.append("state_inconsistency")
        if believability < 0.4:
            failure_parts.append("tone_mismatch")
        if conflict_score < 0.3 and conflicts:
            failure_parts.append("conflict_ignored")
        failure_reason = "; ".join(failure_parts) if failure_parts else ""

        improvement = ""
        if failure_reason:
            if "drive_incoherence" in failure_reason:
                improvement = "align_output_with_dominant_drives"
            elif "state_inconsistency" in failure_reason:
                improvement = "maintain_state_continuity"
            elif "tone_mismatch" in failure_reason:
                improvement = "calibrate_emotional_tone"
            elif "conflict_ignored" in failure_reason:
                improvement = "acknowledge_internal_conflict"

        # Branch feedback
        feedback: dict[str, BranchFeedback] = {}
        for branch_name in route.activated_branches:
            score = branch_scores.get(
                branch_name, BranchScore(0.5, "no_score")
            )
            feedback[branch_name] = BranchFeedback(
                branch_name=branch_name,
                reward=score.reward,
                confidence=max(0.1, min(0.9, 0.4 + abs(score.reward - 0.5))),
                failure_reason="" if score.reward >= 0.5 else score.reason,
                suggested_improvement_direction="",
            )

        return EvaluationSignal(
            reward_score=reward,
            confidence=confidence,
            selected_branch=selected_branch,
            selected_output=selected_output,
            failure_reason=failure_reason,
            suggested_improvement_direction=improvement,
            branch_feedback=feedback,
            aggregator_notes={
                "mode": "human",
                "coherence": round(coherence, 3),
                "consistency": round(consistency, 3),
                "believability": round(believability, 3),
                "conflict_handling": round(conflict_score, 3),
                "mood_valence": state.mood_valence(),
                "arousal": state.arousal_level(),
                "active_conflicts": len(conflicts),
            },
        )

    def _score_coherence(
        self, output: str, state: HumanState, route: RoutingDecision
    ) -> float:
        """Do the dominant drives show up in the output?"""
        dominant = state.dominant_drives(top_k=3)
        output_lower = output.lower()

        # Check if output reflects dominant cognitive mode
        drive_keywords: dict[str, list[str]] = {
            "curiosity": ["wonder", "interesting", "explore", "question", "what if"],
            "fear": ["risk", "danger", "careful", "worry", "concern", "threat"],
            "ambition": ["achieve", "goal", "succeed", "opportunity", "potential"],
            "empathy": ["understand", "feel", "perspective", "compassion"],
            "reflection": ["consider", "weigh", "analyze", "think", "evaluate"],
            "impulse": ["quick", "immediately", "now", "just do", "gut"],
            "stress": ["pressure", "overwhelm", "urgent", "tense"],
            "confidence": ["confident", "certain", "clear", "sure"],
            "motivation": ["driven", "excited", "eager", "want to"],
            "trust": ["trust", "rely", "believe", "faith"],
        }

        hits = 0
        checks = 0
        for drive in dominant:
            keywords = drive_keywords.get(drive, [])
            if keywords:
                checks += 1
                if any(kw in output_lower for kw in keywords):
                    hits += 1

        if checks == 0:
            return 0.6  # Neutral
        return 0.3 + 0.7 * (hits / checks)

    def _score_consistency(self, output: str, state: HumanState) -> float:
        """Is the response consistent with the current mood trajectory?"""
        mood = state.mood_valence()
        output_lower = output.lower()

        positive_markers = ["great", "excellent", "happy", "optimistic", "good"]
        negative_markers = ["unfortunately", "concern", "worry", "difficult", "problem"]

        pos_count = sum(1 for m in positive_markers if m in output_lower)
        neg_count = sum(1 for m in negative_markers if m in output_lower)

        if pos_count + neg_count == 0:
            return 0.6  # Neutral output, mild consistency

        output_valence = (pos_count - neg_count) / (pos_count + neg_count)

        # Consistency = alignment between mood and output tone
        alignment = 1.0 - abs(mood - output_valence) / 2.0
        return max(0.2, min(1.0, alignment))

    def _score_believability(self, output: str, state: HumanState) -> float:
        """Does the output feel emotionally appropriate?"""
        arousal = state.arousal_level()
        output_len = len(output)

        # High arousal should produce shorter, more urgent responses
        if arousal > 0.7:
            length_score = 1.0 if output_len < 500 else max(0.3, 1.0 - (output_len - 500) / 2000)
        elif arousal < 0.3:
            length_score = 1.0 if output_len > 200 else max(0.4, output_len / 200)
        else:
            length_score = 0.7

        # Fatigue should produce less elaborate responses
        fatigue = state.get("fatigue")
        if fatigue > 0.7 and output_len > 800:
            length_score *= 0.7

        return max(0.2, min(1.0, length_score))

    def _score_conflict_handling(
        self,
        output: str,
        conflicts: list[DriveConflict],
        branch_outputs: dict[str, str],
    ) -> float:
        """If conflicts exist, does the output acknowledge them?"""
        if not conflicts:
            return 0.8  # No conflicts to handle, good by default

        output_lower = output.lower()
        conflict_markers = [
            "however", "on the other hand", "but", "tension", "trade-off",
            "conflict", "balance", "compromise", "versus", "dilemma",
            "competing", "torn", "while", "although",
        ]

        marker_hits = sum(1 for m in conflict_markers if m in output_lower)
        # Multiple branches producing output suggests conflict was processed
        multi_branch = len(branch_outputs) >= 3

        score = 0.2
        if marker_hits > 0:
            score += min(0.5, marker_hits * 0.15)
        if multi_branch:
            score += 0.2
        if "conflict_resolver" in branch_outputs:
            score += 0.1

        return min(1.0, score)
