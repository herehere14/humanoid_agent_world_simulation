from __future__ import annotations

import random
from typing import Any

from ..types import TaskInput
from .base import LLMBackend


class DomainShiftBackend(LLMBackend):
    """Backend with deliberately shifted branch-task optima for adaptation validation."""

    def __init__(self, quality_matrix: dict[str, dict[str, float]], noise: float = 0.03, seed: int = 13) -> None:
        self.quality_matrix = quality_matrix
        self.noise = noise
        self._rng = random.Random(seed)

    def best_branch(self, task_type: str) -> str:
        scores = self.quality_matrix[task_type]
        return max(scores.items(), key=lambda x: x[1])[0]

    def generate(self, prompt: str, task: TaskInput, branch_name: str) -> tuple[str, dict[str, Any]]:
        task_type = task.task_type if task.task_type != "auto" else "general"
        scores = self.quality_matrix.get(task_type, self.quality_matrix["general"])
        quality = scores.get(branch_name, 0.4)
        quality += self._rng.uniform(-self.noise, self.noise)
        quality = max(0.01, min(0.99, quality))

        expected_keywords = task.metadata.get("expected_keywords", [])
        included_count = int(round(quality * len(expected_keywords))) if expected_keywords else 0
        included = expected_keywords[:included_count]
        evidence = "; ".join(included) if included else "limited-grounding"

        answer = (
            f"[{branch_name}] shifted-sim response: {task.text} | "
            f"key-points={evidence} | confidence={quality:.2f}"
        )
        return answer, {"quality": quality, "task_type": task_type, "branch": branch_name}


def shifted_quality_matrix() -> dict[str, dict[str, float]]:
    # Intentionally conflicts with default router affinity so adaptation is required.
    return {
        "math": {
            "analytical": 0.45,
            "planner": 0.35,
            "retrieval": 0.48,
            "critique": 0.42,
            "verification": 0.55,
            "creative": 0.9,
        },
        "planning": {
            "analytical": 0.4,
            "planner": 0.52,
            "retrieval": 0.92,
            "critique": 0.46,
            "verification": 0.44,
            "creative": 0.38,
        },
        "factual": {
            "analytical": 0.42,
            "planner": 0.89,
            "retrieval": 0.53,
            "critique": 0.4,
            "verification": 0.5,
            "creative": 0.3,
        },
        "code": {
            "analytical": 0.54,
            "planner": 0.43,
            "retrieval": 0.35,
            "critique": 0.91,
            "verification": 0.56,
            "creative": 0.33,
        },
        "creative": {
            "analytical": 0.93,
            "planner": 0.38,
            "retrieval": 0.34,
            "critique": 0.48,
            "verification": 0.29,
            "creative": 0.47,
        },
        "general": {
            "analytical": 0.45,
            "planner": 0.45,
            "retrieval": 0.45,
            "critique": 0.45,
            "verification": 0.88,
            "creative": 0.45,
        },
    }
