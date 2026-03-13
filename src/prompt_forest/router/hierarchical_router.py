from __future__ import annotations

import math
import random
from typing import Protocol

from ..branches.hierarchical import HierarchicalPromptForest
from ..config import RouterConfig
from ..contracts import infer_output_contract
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
        contract_hint = infer_output_contract(task.text, task.metadata)
        beam_width = max(1, max(self.config.min_candidates, self.config.top_k))
        flattened_scores: dict[str, float] = {}
        exploration_rate = self._current_exploration()

        # Beam search over the hierarchy so we can activate multiple terminal leaves per task.
        frontier: list[tuple[str, list[str], float]] = [(forest.root_id, [], 0.0)]
        completed_paths: list[tuple[list[str], float]] = []
        max_depth = max((forest.depth(node_id) for node_id in forest.nodes), default=1) + 1

        for _ in range(max_depth):
            if not frontier:
                break

            next_frontier: list[tuple[str, list[str], float]] = []
            for node_id, path, path_score in frontier:
                children = forest.children(node_id)
                if not children:
                    completed_paths.append((path, path_score))
                    continue

                scores = self._score_children(
                    task_type=task_type,
                    parent_id=node_id,
                    child_ids=children,
                    forest=forest,
                    history_bias=history_bias,
                    bandit_stats=bandit_stats,
                    bandit_total_count=bandit_total_count,
                    exploration_rate=exploration_rate,
                    contract_hint=contract_hint,
                )
                if not scores:
                    completed_paths.append((path, path_score))
                    continue

                for child_id, score in scores.items():
                    prev = flattened_scores.get(child_id)
                    if prev is None or score > prev:
                        flattened_scores[child_id] = score

                selected_children = self._select_top_children(
                    scores=scores,
                    visit_counts=visit_counts,
                    exploration_rate=exploration_rate,
                    max_children=beam_width,
                )
                for child_id in selected_children:
                    next_frontier.append((child_id, path + [child_id], path_score + scores[child_id]))

            if not next_frontier:
                break
            frontier = self._prune_frontier(next_frontier, beam_width=beam_width)

        for node_id, path, path_score in frontier:
            if not forest.children(node_id):
                completed_paths.append((path, path_score))

        if not completed_paths and frontier:
            completed_paths = [(path, score) for _, path, score in frontier]

        ordered_paths = sorted(completed_paths, key=lambda x: x[1], reverse=True)
        activated_paths: list[list[str]] = []
        seen_leaves: set[str] = set()
        for path, _ in ordered_paths:
            if not path:
                continue
            leaf = path[-1]
            if leaf in seen_leaves:
                continue
            activated_paths.append(path)
            seen_leaves.add(leaf)
            if len(activated_paths) >= beam_width:
                break

        if contract_hint:
            contract_path = self._path_to_node(forest, contract_hint)
            if contract_path and contract_path[-1] not in seen_leaves:
                if len(activated_paths) >= beam_width:
                    activated_paths[-1] = contract_path
                else:
                    activated_paths.append(contract_path)
                seen_leaves.add(contract_path[-1])

        activated_branches: list[str] = []
        seen_branches: set[str] = set()
        for path in activated_paths:
            for branch_name in path:
                if branch_name in seen_branches:
                    continue
                seen_branches.add(branch_name)
                activated_branches.append(branch_name)

        self._route_calls += 1
        return RoutingDecision(
            task_type=task_type,
            activated_branches=activated_branches,
            branch_scores=flattened_scores,
            activated_paths=activated_paths,
        )

    @staticmethod
    def _path_to_node(forest: HierarchicalPromptForest, node_id: str) -> list[str] | None:
        if node_id not in forest.nodes:
            return None
        path: list[str] = []
        cur = node_id
        while cur and cur != forest.root_id:
            path.append(cur)
            cur = forest.parent(cur)
        if not path:
            return None
        path.reverse()
        return path

    def _score_children(
        self,
        task_type: str,
        parent_id: str,
        child_ids: list[str],
        forest: HierarchicalPromptForest,
        history_bias: dict[str, float],
        bandit_stats: dict[str, dict[str, float]],
        bandit_total_count: float,
        exploration_rate: float,
        contract_hint: str | None,
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
                contract_hint=contract_hint,
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
                dynamic_bonus = self.config.bandit_bonus_coef * (0.7 + exploration_rate)
                bonus_term = dynamic_bonus * math.sqrt(
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

    def _select_top_children(
        self,
        scores: dict[str, float],
        visit_counts: dict[str, int],
        exploration_rate: float,
        max_children: int,
    ) -> list[str]:
        ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        take = max(1, min(len(ordered), max_children))
        selected = [name for name, _ in ordered[:take]]

        if self._rng.random() < exploration_rate and len(ordered) > take:
            underexplored = [name for name in scores if visit_counts.get(name, 0) <= 1 and name not in selected]
            pool = underexplored or [name for name, _ in ordered if name not in selected]
            if pool:
                selected[-1] = self._rng.choice(pool)

        # Preserve score ordering while deduplicating any exploration replacement.
        seen: set[str] = set()
        deduped: list[str] = []
        for name in selected:
            if name in seen:
                continue
            seen.add(name)
            deduped.append(name)
        return deduped

    @staticmethod
    def _prune_frontier(frontier: list[tuple[str, list[str], float]], beam_width: int) -> list[tuple[str, list[str], float]]:
        ordered = sorted(frontier, key=lambda x: x[2], reverse=True)
        return ordered[: max(1, beam_width)]

    def _current_exploration(self) -> float:
        decayed = self.config.exploration * (self.config.exploration_decay**self._route_calls)
        return max(self.config.exploration_min, decayed)

    def _affinity_for_child(
        self,
        task_type: str,
        parent_id: str,
        child_id: str,
        forest: HierarchicalPromptForest,
        contract_hint: str | None,
    ) -> float:
        if parent_id == forest.root_id:
            base = self._macro_affinity.get(task_type, {}).get(child_id, 0.0)
            if contract_hint and child_id == "verification":
                base += 0.25
            return base

        node = forest.nodes[child_id]
        base = 0.0
        if task_type in node.specialties:
            base += 0.3
        elif "general" in node.specialties:
            base += 0.08

        if contract_hint:
            if child_id == contract_hint:
                base += 0.75
            elif contract_hint.replace("_lock", "") in child_id:
                base += 0.35
        return base
