from __future__ import annotations

import random
from typing import Any

from ..types import TaskInput
from .base import LLMBackend


class MockLLMBackend(LLMBackend):
    """Deterministic backend with branch-task affinity to simulate adaptation effects."""

    def __init__(self, seed: int = 42) -> None:
        self._rng = random.Random(seed)
        self._quality = {
            "math": {
                "analytical": 0.88,
                "verification": 0.82,
                "planner": 0.35,
                "retrieval": 0.45,
                "critique": 0.62,
                "creative": 0.25,
            },
            "planning": {
                "planner": 0.9,
                "critique": 0.72,
                "verification": 0.68,
                "analytical": 0.6,
                "retrieval": 0.48,
                "creative": 0.55,
            },
            "factual": {
                "retrieval": 0.9,
                "verification": 0.8,
                "analytical": 0.58,
                "planner": 0.35,
                "critique": 0.56,
                "creative": 0.3,
            },
            "code": {
                "analytical": 0.86,
                "verification": 0.78,
                "planner": 0.66,
                "critique": 0.69,
                "retrieval": 0.4,
                "creative": 0.45,
            },
            "creative": {
                "creative": 0.9,
                "critique": 0.5,
                "planner": 0.55,
                "analytical": 0.42,
                "retrieval": 0.32,
                "verification": 0.25,
            },
            "general": {
                "analytical": 0.62,
                "planner": 0.62,
                "retrieval": 0.62,
                "critique": 0.62,
                "verification": 0.62,
                "creative": 0.62,
            },
        }

    def generate(self, prompt: str, task: TaskInput, branch_name: str) -> tuple[str, dict[str, Any]]:
        task_type = task.task_type if task.task_type != "auto" else "general"
        quality = self._quality_for_branch(task_type, branch_name)
        quality += self._rng.uniform(-0.07, 0.07)
        quality = max(0.01, min(0.99, quality))

        expected_keywords = task.metadata.get("expected_keywords", [])
        included_count = int(round(quality * len(expected_keywords))) if expected_keywords else 0
        included = expected_keywords[:included_count]

        if included:
            evidence = "; ".join(f"{k}" for k in included)
        else:
            evidence = "limited-grounding"

        tone = {
            "analytical": "stepwise reasoning",
            "planner": "sequenced action plan",
            "retrieval": "fact extraction",
            "critique": "failure mode analysis",
            "verification": "constraint checks",
            "creative": "novel alternatives",
        }.get(branch_name, "general synthesis")

        answer = (
            f"[{branch_name}] {tone}: {task.text} | "
            f"key-points={evidence} | confidence={quality:.2f}"
        )

        return answer, {"quality": quality, "task_type": task_type, "branch": branch_name}

    def _quality_for_branch(self, task_type: str, branch_name: str) -> float:
        per_task = self._quality.get(task_type, self._quality["general"])
        if branch_name in per_task:
            return per_task[branch_name]

        macro = branch_name.split("_")[0]
        base = per_task.get(macro, 0.5)
        tokens = set(branch_name.split("_"))

        niche_bonus = 0.0
        if task_type == "math" and {"symbolic", "solver", "constraint", "checker"} & tokens:
            niche_bonus += 0.08
        if task_type == "planning" and {"timeline", "risk", "allocator", "optimizer"} & tokens:
            niche_bonus += 0.08
        if task_type == "factual" and {"evidence", "source", "tracer", "triage"} & tokens:
            niche_bonus += 0.08
        if task_type == "code" and {"adversarial", "consistency", "checker", "auditor"} & tokens:
            niche_bonus += 0.08
        if task_type == "creative" and {"divergent", "innovator", "creative"} & tokens:
            niche_bonus += 0.08
        if task_type == "general" and {"consistency", "constraint", "auditor", "checker"} & tokens:
            niche_bonus += 0.05

        return max(0.01, min(0.99, base + niche_bonus))
