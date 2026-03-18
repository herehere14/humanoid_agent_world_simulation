"""Human Mode Router: conflict-aware, state-conditioned routing.

Unlike Agent Improvement routing which optimises for task performance,
Human Mode routing models how a human-like cognitive system would
allocate attention across competing internal modules.

Key differences from standard routing:
  1. State-conditioned: routing scores are modulated by current emotional/
     cognitive state variables.
  2. Conflict-aware: when opposing drives are both active, the router
     activates both sides and flags the conflict for downstream resolution.
  3. Imperfect: adds noise to simulate bounded rationality.
  4. Arousal-driven: high arousal activates fast/impulsive branches,
     low arousal favours reflective branches.
"""

from __future__ import annotations

import random
from typing import Any

from ...branches.hierarchical import HierarchicalPromptForest
from ...memory.store import MemoryStore
from ...state.human_state import HumanState, DriveConflict
from ...types import RoutingDecision, TaskInput


# Maps drives to the branches they most strongly influence.
DRIVE_TO_BRANCH: dict[str, list[str]] = {
    "reflection": ["reflective_reasoning", "working_memory", "long_term_memory"],
    "impulse": ["impulse_response"],
    "fear": ["fear_risk"],
    "curiosity": ["curiosity_exploration"],
    "empathy": ["empathy_social"],
    "self_protection": ["self_protection", "self_justification"],
    "ambition": ["ambition_reward", "long_term_goals"],
    "moral": ["moral_evaluation"],
    "emotional_regulation": ["emotional_modulation"],
    "long_term_goals": ["long_term_goals"],
}


class HumanModeRouter:
    """Conflict-aware, state-conditioned router for Human Mode."""

    def __init__(
        self,
        top_k: int = 4,
        conflict_activation: bool = True,
        noise_level: float = 0.08,
    ) -> None:
        self.top_k = top_k
        self.conflict_activation = conflict_activation
        self.noise_level = noise_level

    def route(
        self,
        task: TaskInput,
        forest: HierarchicalPromptForest,
        state: HumanState,
        memory: MemoryStore | None = None,
    ) -> tuple[RoutingDecision, list[DriveConflict]]:
        """Route a task through the cognitive forest, conditioned on state.

        Returns both the routing decision and any active conflicts that
        the downstream system should resolve.
        """
        # Step 1: Compute state-conditioned branch scores
        routing_biases = state.routing_bias()
        arousal = state.arousal_level()
        branch_scores: dict[str, float] = {}

        for branch_name, branch in forest.branches.items():
            if not branch.is_active:
                continue
            meta = branch.state.metadata
            if meta.get("category_node"):
                continue

            # Base score from weight
            score = branch.state.weight

            # Drive-based modulation
            drive = meta.get("drive", "")
            if drive:
                drive_strength = state.drive_strength(drive)
                score *= (0.5 + drive_strength)

            # Routing bias from state
            if branch_name in routing_biases:
                score += routing_biases[branch_name] * 0.3

            # Arousal modulation: high arousal favours fast branches
            speed = meta.get("speed", "medium")
            if arousal > 0.6:
                if speed in ("fast", "instant"):
                    score *= 1.2
                elif speed == "slow":
                    score *= 0.8
            elif arousal < 0.3:
                if speed == "slow":
                    score *= 1.1
                elif speed in ("fast", "instant"):
                    score *= 0.9

            # Cognitive cost penalty under fatigue
            fatigue = state.get("fatigue")
            cost = meta.get("cognitive_cost", "medium")
            if fatigue > 0.5:
                cost_penalty = {"very_low": 0, "low": 0.05, "medium": 0.1, "high": 0.2}
                score -= cost_penalty.get(cost, 0.1) * fatigue

            # Bounded rationality noise
            if self.noise_level > 0:
                score += random.gauss(0, self.noise_level)

            branch_scores[branch_name] = max(0.01, round(score, 4))

        # Step 2: Detect and handle conflicts
        conflicts = state.active_conflicts
        conflict_branches: set[str] = set()
        if self.conflict_activation and conflicts:
            for conflict in conflicts:
                # Activate branches for both sides of the conflict
                for drive in [conflict.drive_a, conflict.drive_b]:
                    if drive in DRIVE_TO_BRANCH:
                        for b in DRIVE_TO_BRANCH[drive]:
                            if b in branch_scores:
                                branch_scores[b] *= (1.0 + conflict.intensity * 0.3)
                                conflict_branches.add(b)

            # Always activate conflict resolver when conflicts exist
            if "conflict_resolver" in branch_scores:
                branch_scores["conflict_resolver"] *= 1.5

        # Step 3: Select top-K branches
        ranked = sorted(branch_scores.items(), key=lambda x: x[1], reverse=True)
        activated = [name for name, _ in ranked[:self.top_k]]

        # Ensure conflict branches are included
        for cb in conflict_branches:
            if cb not in activated and len(activated) < self.top_k + 2:
                activated.append(cb)

        # Build paths through the hierarchy
        activated_paths: list[list[str]] = []
        for branch_name in activated:
            path = forest.path_to_root(branch_name)
            if path:
                activated_paths.append(path)

        decision = RoutingDecision(
            task_type=self._classify_cognitive_context(task, state),
            activated_branches=activated,
            branch_scores=branch_scores,
            activated_paths=activated_paths,
            sibling_decisions=self._build_sibling_info(conflicts),
        )
        return decision, conflicts

    def _classify_cognitive_context(self, task: TaskInput, state: HumanState) -> str:
        """Classify the cognitive context rather than just task type.

        In Human Mode, the 'task type' reflects the cognitive situation
        rather than just the content of the query.
        """
        mood = state.mood_valence()
        arousal = state.arousal_level()

        # Check for conflict-heavy state
        if len(state.active_conflicts) >= 2:
            return "internal_conflict"

        # Check emotional extremes
        if state.get("stress") > 0.7 or state.get("fear") > 0.7:
            return "threat_response"
        if state.get("curiosity") > 0.7:
            return "exploration"
        if mood < -0.3:
            return "negative_affect"
        if mood > 0.4 and state.get("ambition") > 0.6:
            return "goal_pursuit"

        # Fall back to content-based classification
        text_lower = task.text.lower()
        if any(w in text_lower for w in ["feel", "emotion", "sad", "happy", "angry"]):
            return "emotional_processing"
        if any(w in text_lower for w in ["should", "right", "wrong", "fair", "ethical"]):
            return "moral_reasoning"
        if any(w in text_lower for w in ["plan", "goal", "future", "strategy"]):
            return "goal_pursuit"
        if any(w in text_lower for w in ["risk", "danger", "threat", "worry"]):
            return "threat_response"

        return "general_cognition"

    def _build_sibling_info(self, conflicts: list[DriveConflict]) -> dict[str, dict[str, Any]]:
        """Build sibling decision metadata from conflicts."""
        info: dict[str, dict[str, Any]] = {}
        for conflict in conflicts:
            info[f"conflict_{conflict.drive_a}_vs_{conflict.drive_b}"] = {
                "drive_a": conflict.drive_a,
                "drive_b": conflict.drive_b,
                "intensity": conflict.intensity,
                "resolution": conflict.resolution,
                "resolution_weight": conflict.resolution_weight,
            }
        return info
