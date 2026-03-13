from __future__ import annotations

import math
import random
from typing import Protocol

from ..branches.hierarchical import HierarchicalPromptForest
from ..config import RouterConfig
from ..types import BranchStatus, RoutingDecision, TaskInput


class MemoryRoutingView(Protocol):
    def branch_success_bias(self, task_type: str, user_id: str | None = None) -> dict[str, float]:
        ...

    def branch_visit_counts(self, task_type: str, user_id: str | None = None) -> dict[str, int]:
        ...

    def branch_bandit_stats(self, task_type: str, user_id: str | None = None) -> dict[str, dict[str, float]]:
        ...


class HierarchicalRouter:
    def __init__(self, config: RouterConfig, seed: int = 7) -> None:
        self.config = config
        self._rng = random.Random(seed)
        self._route_calls = 0
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
        user_id = str(task.metadata.get("user_id", "global")).strip() or "global"
        history_bias = memory.branch_success_bias(task_type, user_id=user_id)
        visit_counts = memory.branch_visit_counts(task_type, user_id=user_id)
        bandit_stats = memory.branch_bandit_stats(task_type, user_id=user_id)
        bandit_total_count = sum(max(0.0, stat.get("count", 0.0)) for stat in bandit_stats.values())

        active_path: list[str] = []
        flattened_scores: dict[str, float] = {}
        exploration_rate = self._current_exploration()

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
                bandit_stats=bandit_stats,
                bandit_total_count=bandit_total_count,
            )
            if not scores:
                break

            selected = self._select_one(scores, visit_counts, exploration_rate)
            active_path.append(selected)
            flattened_scores.update(scores)
            current = selected

        self._route_calls += 1
        return RoutingDecision(task_type=task_type, activated_branches=active_path, branch_scores=flattened_scores)

    def _score_children(
        self,
        task_type: str,
        parent_id: str,
        child_ids: list[str],
        forest: HierarchicalPromptForest,
        history_bias: dict[str, float],
        bandit_stats: dict[str, dict[str, float]],
        bandit_total_count: float,
    ) -> dict[str, float]:
        out: dict[str, float] = {}

        for child_id in child_ids:
            branch = forest.get_branch(child_id)
            if branch.state.status == BranchStatus.ARCHIVED:
                continue

            affinity = self._affinity_for_child(
                task_type=task_type,
                parent_id=parent_id,
                child_id=child_id,
                forest=forest,
            )
            memory_bias = history_bias.get(child_id, 0.0)
            memory_bias = max(-self.config.memory_term_cap, min(self.config.memory_term_cap, memory_bias))

            bandit = bandit_stats.get(child_id, {})
            mean_reward = float(bandit.get("mean_reward", 0.5))
            count = max(0.0, float(bandit.get("count", 0.0)))

            shrink = count / (count + max(1e-8, self.config.bandit_shrinkage_k))
            value_term = self.config.bandit_value_coef * (mean_reward - 0.5) * shrink

            bonus_term = 0.0
            if self.config.bandit_bonus_coef > 0.0 and bandit_total_count > 0.0:
                bonus_term = self.config.bandit_bonus_coef * math.sqrt(
                    math.log1p(bandit_total_count) / (1.0 + count)
                )
                bonus_term = min(self.config.bandit_bonus_cap, bonus_term)

            score = (
                (self.config.weight_coef * branch.state.weight)
                + (self.config.affinity_coef * affinity)
                + (self.config.memory_coef * memory_bias)
                + value_term
                + bonus_term
            )

            if branch.state.status == BranchStatus.CANDIDATE:
                score *= 0.85

            out[child_id] = max(0.01, score)

        return out

    def _select_one(self, scores: dict[str, float], visit_counts: dict[str, int], exploration_rate: float) -> str:
        ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        selected = ordered[0][0]

        if self._rng.random() < exploration_rate and len(ordered) > 1:
            underexplored = [name for name in scores if visit_counts.get(name, 0) <= 1]
            if underexplored:
                selected = self._rng.choice(underexplored)
            else:
                selected = self._rng.choice([name for name, _ in ordered[1:]])

        return selected

    def _current_exploration(self) -> float:
        decayed = self.config.exploration * (self.config.exploration_decay**self._route_calls)
        return max(self.config.exploration_min, decayed)

    def _affinity_for_child(
        self,
        task_type: str,
        parent_id: str,
        child_id: str,
        forest: HierarchicalPromptForest,
    ) -> float:
        if parent_id == forest.root_id:
            return self._macro_affinity.get(task_type, {}).get(child_id, 0.0)

        node = forest.nodes[child_id]
        if task_type in node.specialties:
            return 0.3
        if "general" in node.specialties:
            return 0.08
        return 0.0
