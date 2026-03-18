"""Human Mode Memory: experiential, emotional memory.

Unlike Agent Improvement memory which reinforces high-performing routes,
Human Mode memory operates like human autobiographical memory:

  - Emotional events are remembered more strongly (emotional amplification).
  - Negative experiences produce avoidance biases.
  - Positive experiences produce approach biases.
  - Memories decay with recency but traumatic events decay slower.
  - Past social interactions influence trust and empathy.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from ...state.human_state import HumanState, StateSnapshot


@dataclass
class ExperientialMemory:
    """A single experiential memory entry."""

    event_id: str
    turn_index: int
    task_text: str
    task_type: str
    outcome_reward: float
    emotional_valence: float  # -1 to +1
    arousal_at_time: float
    state_snapshot: dict[str, float]
    dominant_drives: list[str]
    active_branches: list[str]
    selected_branch: str
    conflicts_present: int
    failure_reason: str
    tags: list[str] = field(default_factory=list)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class HumanModeMemory:
    """Experiential memory system for Human Mode.

    Memories are stored with emotional valence and influence future state
    and routing through bias signals.
    """

    def __init__(
        self,
        max_memories: int = 500,
        emotional_decay: float = 0.92,
        trauma_amplification: float = 1.5,
        experience_bias_strength: float = 0.4,
    ) -> None:
        self._memories: list[ExperientialMemory] = []
        self.max_memories = max_memories
        self.emotional_decay = emotional_decay
        self.trauma_amplification = trauma_amplification
        self.experience_bias_strength = experience_bias_strength

    def record(
        self,
        event_id: str,
        task: Any,
        state: HumanState,
        reward: float,
        selected_branch: str,
        active_branches: list[str],
        failure_reason: str = "",
    ) -> ExperientialMemory:
        """Record an experience into memory."""
        snap = state.snapshot()
        memory = ExperientialMemory(
            event_id=event_id,
            turn_index=state.turn_index,
            task_text=getattr(task, "text", str(task)),
            task_type=getattr(task, "task_type", "unknown"),
            outcome_reward=reward,
            emotional_valence=snap.mood_valence,
            arousal_at_time=state.arousal_level(),
            state_snapshot=dict(snap.variables),
            dominant_drives=list(snap.dominant_drives),
            active_branches=list(active_branches),
            selected_branch=selected_branch,
            conflicts_present=len(snap.active_conflicts),
            failure_reason=failure_reason,
            tags=self._auto_tag(reward, failure_reason, snap),
        )
        self._memories.append(memory)

        # Evict oldest non-traumatic memories if over capacity
        if len(self._memories) > self.max_memories:
            self._evict()

        return memory

    def recall_similar(
        self,
        task_type: str,
        state: HumanState,
        limit: int = 5,
    ) -> list[ExperientialMemory]:
        """Recall memories relevant to current situation.

        Relevance is based on:
          - Task type match
          - Emotional state similarity
          - Recency (with emotional amplification for strong memories)
        """
        if not self._memories:
            return []

        current_mood = state.mood_valence()
        current_turn = state.turn_index
        scored: list[tuple[float, ExperientialMemory]] = []

        for mem in self._memories:
            score = 0.0

            # Task type match
            if mem.task_type == task_type:
                score += 0.4

            # Emotional similarity
            mood_diff = abs(current_mood - mem.emotional_valence)
            score += 0.3 * (1.0 - min(1.0, mood_diff))

            # Recency with emotional amplification
            age = max(1, current_turn - mem.turn_index)
            recency = math.pow(self.emotional_decay, age)
            # Strong emotional events are remembered longer
            if abs(mem.emotional_valence) > 0.5:
                recency *= self.trauma_amplification
            score += 0.3 * min(1.0, recency)

            scored.append((score, mem))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [mem for _, mem in scored[:limit]]

    def experiential_bias(self, state: HumanState) -> dict[str, float]:
        """Compute routing biases from accumulated experience.

        Branches that led to positive outcomes in similar states get
        positive bias. Branches associated with negative outcomes get
        avoidance bias.
        """
        if not self._memories:
            return {}

        branch_scores: dict[str, list[float]] = {}
        current_mood = state.mood_valence()
        current_turn = state.turn_index

        for mem in self._memories[-50:]:  # Recent window
            age = max(1, current_turn - mem.turn_index)
            recency = math.pow(self.emotional_decay, age)

            # Weight by emotional relevance
            mood_sim = 1.0 - abs(current_mood - mem.emotional_valence) * 0.5
            weight = recency * mood_sim

            # Record branch outcome
            for branch in mem.active_branches:
                if branch not in branch_scores:
                    branch_scores[branch] = []
                # Positive outcome → approach, negative → avoidance
                signal = (mem.outcome_reward - 0.5) * weight
                branch_scores[branch].append(signal)

        biases: dict[str, float] = {}
        for branch, signals in branch_scores.items():
            if signals:
                avg = sum(signals) / len(signals)
                biases[branch] = round(avg * self.experience_bias_strength, 4)

        return biases

    def emotional_trajectory(self, window: int = 10) -> list[float]:
        """Return recent emotional valence trajectory."""
        recent = self._memories[-window:]
        return [m.emotional_valence for m in recent]

    def failure_patterns(self) -> dict[str, int]:
        """Count failure patterns for insight."""
        patterns: dict[str, int] = {}
        for mem in self._memories:
            if mem.failure_reason:
                patterns[mem.failure_reason] = patterns.get(mem.failure_reason, 0) + 1
        return patterns

    def state_at_turn(self, turn: int) -> dict[str, float] | None:
        """Retrieve state snapshot at a specific turn."""
        for mem in self._memories:
            if mem.turn_index == turn:
                return dict(mem.state_snapshot)
        return None

    @property
    def memory_count(self) -> int:
        return len(self._memories)

    def _auto_tag(
        self,
        reward: float,
        failure_reason: str,
        snap: StateSnapshot,
    ) -> list[str]:
        """Auto-tag a memory for future retrieval."""
        tags: list[str] = []
        if reward >= 0.8:
            tags.append("success")
        elif reward <= 0.3:
            tags.append("failure")
        if snap.mood_valence < -0.3:
            tags.append("negative_mood")
        elif snap.mood_valence > 0.3:
            tags.append("positive_mood")
        if len(snap.active_conflicts) >= 2:
            tags.append("high_conflict")
        if failure_reason:
            tags.append(f"fail:{failure_reason.split(';')[0].strip()}")
        return tags

    def _evict(self) -> None:
        """Remove oldest, least-emotional memories."""
        if len(self._memories) <= self.max_memories:
            return
        # Sort by emotional intensity (keep traumatic/peak memories longer)
        self._memories.sort(
            key=lambda m: abs(m.emotional_valence) + (0.1 if "success" in m.tags else 0),
            reverse=True,
        )
        self._memories = self._memories[:self.max_memories]
        # Re-sort by turn index for temporal ordering
        self._memories.sort(key=lambda m: m.turn_index)

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_count": len(self._memories),
            "memories": [m.to_dict() for m in self._memories[-20:]],
        }
