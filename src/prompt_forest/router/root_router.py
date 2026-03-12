from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Protocol

from ..branches.base import PromptBranch
from ..config import RouterConfig
from ..types import BranchStatus, RoutingDecision, TaskInput


class MemoryRoutingView(Protocol):
    def branch_success_bias(self, task_type: str) -> dict[str, float]:
        ...


class RootRouter:
    def __init__(self, config: RouterConfig, seed: int = 7) -> None:
        self.config = config
        self._rng = random.Random(seed)
        self._task_branch_affinity = {
            "math": {"analytical": 0.35, "verification": 0.25, "critique": 0.1},
            "planning": {"planner": 0.35, "critique": 0.2, "verification": 0.1},
            "factual": {"retrieval": 0.35, "verification": 0.2, "analytical": 0.1},
            "code": {"analytical": 0.3, "verification": 0.2, "planner": 0.1},
            "creative": {"creative": 0.4, "critique": 0.1, "planner": 0.1},
            "general": {"analytical": 0.1, "planner": 0.1, "retrieval": 0.1},
        }

    def classify_task_type(self, task: TaskInput) -> str:
        if task.task_type and task.task_type != "auto":
            return task.task_type

        text = task.text.lower()
        if any(k in text for k in ["proof", "calculate", "equation", "derive", "math"]):
            return "math"
        if any(k in text for k in ["plan", "roadmap", "milestone", "schedule"]):
            return "planning"
        if any(k in text for k in ["fact", "who", "when", "where", "reference", "citation"]):
            return "factual"
        if any(k in text for k in ["python", "code", "bug", "refactor", "algorithm"]):
            return "code"
        if any(k in text for k in ["creative", "story", "brainstorm", "idea"]):
            return "creative"
        return "general"

    def route(
        self,
        task: TaskInput,
        branches: dict[str, PromptBranch],
        memory: MemoryRoutingView,
    ) -> RoutingDecision:
        task_type = self.classify_task_type(task)
        history_bias = memory.branch_success_bias(task_type)
        affinity = self._task_branch_affinity.get(task_type, self._task_branch_affinity["general"])

        scores: dict[str, float] = {}
        for name, branch in branches.items():
            if branch.state.status == BranchStatus.ARCHIVED:
                continue
            base = branch.state.weight
            score = base + affinity.get(name, 0.0) + history_bias.get(name, 0.0)
            if branch.state.status == BranchStatus.CANDIDATE:
                score *= 0.85
            scores[name] = max(0.01, score)

        selected = self._select_top_with_exploration(scores)
        return RoutingDecision(task_type=task_type, activated_branches=selected, branch_scores=scores)

    def _select_top_with_exploration(self, scores: dict[str, float]) -> list[str]:
        if not scores:
            return []

        top_k = max(self.config.min_candidates, self.config.top_k)
        ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        selected = [name for name, _ in ordered[:top_k]]

        if self._rng.random() < self.config.exploration and len(ordered) > top_k:
            pool = [name for name, _ in ordered[top_k:]]
            selected[-1] = self._rng.choice(pool)

        return selected

    @staticmethod
    def normalize_scores(scores: dict[str, float]) -> dict[str, float]:
        if not scores:
            return {}
        max_v = max(scores.values())
        exps = {k: math.exp(v - max_v) for k, v in scores.items()}
        total = sum(exps.values())
        return {k: v / total for k, v in exps.items()}

    @staticmethod
    def branch_histogram(routes: list[RoutingDecision]) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for route in routes:
            for b in route.activated_branches:
                counts[b] += 1
        return dict(counts)
