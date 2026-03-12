from __future__ import annotations

import random
from typing import Protocol

from ..branches.hierarchical import HierarchicalPromptForest
from ..config import RouterConfig
from ..types import BranchStatus, RoutingDecision, TaskInput


class MemoryRoutingView(Protocol):
    def branch_success_bias(self, task_type: str) -> dict[str, float]:
        ...

    def branch_visit_counts(self, task_type: str) -> dict[str, int]:
        ...


class HierarchicalRouter:
    def __init__(self, config: RouterConfig, seed: int = 7) -> None:
        self.config = config
        self._rng = random.Random(seed)
        self._macro_affinity = {
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
        forest: HierarchicalPromptForest,
        memory: MemoryRoutingView,
    ) -> RoutingDecision:
        task_type = self.classify_task_type(task)
        history_bias = memory.branch_success_bias(task_type)
        visit_counts = memory.branch_visit_counts(task_type)

        active_path: list[str] = []
        flattened_scores: dict[str, float] = {}

        current = forest.root_id
        while True:
            children = forest.children(current)
            if not children:
                break

            scores = self._score_children(
                task_type=task_type,
                parent_id=current,
                child_ids=children,
                forest=forest,
                history_bias=history_bias,
            )
            if not scores:
                break

            selected = self._select_one(scores, visit_counts)
            active_path.append(selected)
            flattened_scores.update(scores)
            current = selected

        return RoutingDecision(task_type=task_type, activated_branches=active_path, branch_scores=flattened_scores)

    def _score_children(
        self,
        task_type: str,
        parent_id: str,
        child_ids: list[str],
        forest: HierarchicalPromptForest,
        history_bias: dict[str, float],
    ) -> dict[str, float]:
        out: dict[str, float] = {}

        for child_id in child_ids:
            branch = forest.get_branch(child_id)
            if branch.state.status == BranchStatus.ARCHIVED:
                continue

            node = forest.nodes[child_id]
            score = branch.state.weight

            if parent_id == forest.root_id:
                score += self._macro_affinity.get(task_type, {}).get(child_id, 0.0)
            else:
                if task_type in node.specialties:
                    score += 0.3
                elif "general" in node.specialties:
                    score += 0.08

            score += history_bias.get(child_id, 0.0)

            if branch.state.status == BranchStatus.CANDIDATE:
                score *= 0.85

            out[child_id] = max(0.01, score)

        return out

    def _select_one(self, scores: dict[str, float], visit_counts: dict[str, int]) -> str:
        ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        selected = ordered[0][0]

        if self._rng.random() < self.config.exploration and len(ordered) > 1:
            underexplored = [name for name in scores if visit_counts.get(name, 0) <= 1]
            if underexplored:
                selected = self._rng.choice(underexplored)
            else:
                selected = self._rng.choice([name for name, _ in ordered[1:]])

        return selected
