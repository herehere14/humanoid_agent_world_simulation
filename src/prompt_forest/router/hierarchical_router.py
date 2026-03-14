from __future__ import annotations

import random
from typing import Protocol

from ..branches.hierarchical import HierarchicalPromptForest
from ..config import RouterConfig
from ..contracts import infer_output_contract
from ..types import BranchStatus, RoutingDecision, SiblingPreferenceSignal, TaskInput


class MemoryRoutingView(Protocol):
    def branch_visit_counts(self, task_type: str, user_id: str | None = None) -> dict[str, int]:
        ...

    def sibling_preference_signal(
        self,
        task: TaskInput,
        task_type: str,
        parent_id: str,
        child_ids: list[str],
        user_id: str | None = None,
    ) -> SiblingPreferenceSignal:
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
        visit_counts = memory.branch_visit_counts(task_type, user_id=user_id)
        contract_hint = infer_output_contract(task.text, task.metadata)
        beam_width = max(1, max(self.config.min_candidates, self.config.top_k))
        flattened_scores: dict[str, float] = {}
        sibling_decisions: dict[str, dict[str, object]] = {}
        exploration_rate = self._current_exploration()

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

                preference_signal = self._preference_signal_for_children(
                    task=task,
                    task_type=task_type,
                    parent_id=node_id,
                    child_ids=children,
                    forest=forest,
                    memory=memory,
                    user_id=user_id,
                )
                scores = self._score_children(
                    task_type=task_type,
                    parent_id=node_id,
                    child_ids=children,
                    forest=forest,
                    memory_scores=preference_signal.scores,
                    contract_hint=contract_hint,
                )
                if not scores:
                    completed_paths.append((path, path_score))
                    continue

                selected_by_score = max(scores.items(), key=lambda item: item[1])[0]
                decision_meta = self._sibling_decision_meta(
                    parent_id=node_id,
                    child_ids=children,
                    scores=scores,
                    signal=preference_signal,
                    selected_by_score=selected_by_score,
                )
                if decision_meta:
                    sibling_decisions[node_id] = decision_meta

                for child_id, score in scores.items():
                    prev = flattened_scores.get(child_id)
                    if prev is None or score > prev:
                        flattened_scores[child_id] = score

                selected_children = self._select_top_children(
                    scores=scores,
                    visit_counts=visit_counts,
                    exploration_rate=exploration_rate,
                    max_children=beam_width,
                    forced_first=str(decision_meta.get("override_child", "")) if decision_meta else "",
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

        ordered_paths = sorted(completed_paths, key=lambda item: item[1], reverse=True)
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
            sibling_decisions=sibling_decisions,
        )

    @staticmethod
    def _path_to_node(forest: HierarchicalPromptForest, node_id: str) -> list[str] | None:
        if node_id not in forest.nodes:
            return None
        path: list[str] = []
        cursor = node_id
        while cursor and cursor != forest.root_id:
            path.append(cursor)
            cursor = forest.parent(cursor)
        if not path:
            return None
        path.reverse()
        return path

    def _preference_signal_for_children(
        self,
        *,
        task: TaskInput,
        task_type: str,
        parent_id: str,
        child_ids: list[str],
        forest: HierarchicalPromptForest,
        memory: MemoryRoutingView,
        user_id: str,
    ) -> SiblingPreferenceSignal:
        if parent_id == forest.root_id:
            return SiblingPreferenceSignal()
        if not child_ids or not all(not forest.children(child_id) for child_id in child_ids):
            return SiblingPreferenceSignal()
        return memory.sibling_preference_signal(
            task=task,
            task_type=task_type,
            parent_id=parent_id,
            child_ids=child_ids,
            user_id=user_id,
        )

    def _score_children(
        self,
        *,
        task_type: str,
        parent_id: str,
        child_ids: list[str],
        forest: HierarchicalPromptForest,
        memory_scores: dict[str, float],
        contract_hint: str | None,
    ) -> dict[str, float]:
        out: dict[str, float] = {}

        for child_id in child_ids:
            branch = forest.get_branch(child_id)
            if branch.state.status == BranchStatus.ARCHIVED:
                continue

            effective_weight = self._effective_weight(branch)
            affinity = self._affinity_for_child(
                task_type=task_type,
                parent_id=parent_id,
                child_id=child_id,
                forest=forest,
                contract_hint=contract_hint,
            )
            lock_adjustment = self._lock_adjustment(child_id, contract_hint)
            memory_term = self.config.memory_coef * memory_scores.get(child_id, 0.0)

            score = (
                (self.config.weight_coef * effective_weight)
                + (self.config.affinity_coef * affinity)
                + memory_term
                + lock_adjustment
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
        forced_first: str = "",
    ) -> list[str]:
        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        take = max(1, min(len(ordered), max_children))
        selected = [name for name, _ in ordered[:take]]

        if forced_first and forced_first in scores:
            if forced_first in selected:
                selected.remove(forced_first)
            selected.insert(0, forced_first)
            selected = selected[:take]

        if not forced_first and self._rng.random() < exploration_rate and len(ordered) > take:
            underexplored = [name for name in scores if visit_counts.get(name, 0) <= 1 and name not in selected]
            pool = underexplored or [name for name, _ in ordered if name not in selected]
            if pool:
                selected[-1] = self._rng.choice(pool)

        seen: set[str] = set()
        deduped: list[str] = []
        for name in selected:
            if name in seen:
                continue
            seen.add(name)
            deduped.append(name)
        return deduped

    def _sibling_decision_meta(
        self,
        *,
        parent_id: str,
        child_ids: list[str],
        scores: dict[str, float],
        signal: SiblingPreferenceSignal,
        selected_by_score: str,
    ) -> dict[str, object]:
        if len(child_ids) < 2:
            return {}

        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        probe_candidates: list[str] = []
        score_gap = 1.0
        if len(ordered) >= 2:
            score_gap = max(0.0, ordered[0][1] - ordered[1][1])
            if score_gap <= self.config.sibling_probe_score_gap:
                probe_candidates = [child_id for child_id, _ in ordered[: self.config.sibling_probe_top_n]]

        override_child = ""
        if signal.preferred_child and signal.preferred_child in scores:
            if (
                signal.support >= self.config.route_override_min_support
                and signal.win_rate >= self.config.route_override_min_win_rate
                and signal.expected_margin >= self.config.route_override_min_margin
            ):
                override_child = signal.preferred_child

        return {
            "parent_id": parent_id,
            "selected_by_score": selected_by_score,
            "selected_child": override_child or selected_by_score,
            "memory_scores": dict(signal.scores),
            "preferred_child": signal.preferred_child,
            "override_child": override_child,
            "support": signal.support,
            "win_rate": round(signal.win_rate, 4),
            "expected_margin": round(signal.expected_margin, 4),
            "score_gap": round(score_gap, 4),
            "probe_candidates": probe_candidates,
            "child_scores": {child_id: round(scores.get(child_id, 0.0), 4) for child_id in child_ids if child_id in scores},
            "details": dict(signal.details),
        }

    def _effective_weight(self, branch) -> float:
        base_weight = float(branch.state.metadata.get("base_weight", branch.state.weight))
        learned_delta = branch.state.weight - base_weight
        if abs(learned_delta) < 1e-9:
            return branch.state.weight

        support = len(branch.state.historical_rewards)
        if support < max(0, self.config.learned_weight_min_support):
            return base_weight

        k = max(1e-8, self.config.learned_weight_support_k)
        shrink = support / (support + k)
        return base_weight + (learned_delta * shrink)

    @staticmethod
    def _prune_frontier(frontier: list[tuple[str, list[str], float]], beam_width: int) -> list[tuple[str, list[str], float]]:
        ordered = sorted(frontier, key=lambda item: item[2], reverse=True)
        return ordered[: max(1, beam_width)]

    def _current_exploration(self) -> float:
        decayed = self.config.exploration * (self.config.exploration_decay**self._route_calls)
        return max(self.config.exploration_min, decayed)

    def _affinity_for_child(
        self,
        *,
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

    @staticmethod
    def _lock_adjustment(child_id: str, contract_hint: str | None) -> float:
        if not child_id.endswith("_lock"):
            return 0.0
        if contract_hint == child_id:
            return 0.0
        if contract_hint:
            return -0.18
        return -0.12
