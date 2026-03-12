from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..types import BranchOutput


@dataclass
class AggregationResult:
    selected_branch: str
    selected_output: str
    notes: dict[str, Any]


class Aggregator:
    def __init__(self, strategy: str = "judge_select") -> None:
        self.strategy = strategy

    def aggregate(self, outputs: dict[str, BranchOutput], branch_rewards: dict[str, float]) -> AggregationResult:
        if not outputs:
            return AggregationResult("none", "", {"reason": "no_outputs"})

        if self.strategy in {"best_of_n", "judge_select"}:
            return self._best_of_n(outputs, branch_rewards)
        if self.strategy == "weighted_voting":
            return self._weighted_vote(outputs, branch_rewards)
        if self.strategy == "merge_and_refine":
            return self._merge_and_refine(outputs, branch_rewards)
        return self._best_of_n(outputs, branch_rewards)

    def _best_of_n(self, outputs: dict[str, BranchOutput], branch_rewards: dict[str, float]) -> AggregationResult:
        ranked = sorted(branch_rewards.items(), key=lambda x: x[1], reverse=True)
        selected_branch = ranked[0][0]
        return AggregationResult(
            selected_branch=selected_branch,
            selected_output=outputs[selected_branch].output,
            notes={"ranked": ranked},
        )

    def _weighted_vote(self, outputs: dict[str, BranchOutput], branch_rewards: dict[str, float]) -> AggregationResult:
        ranked = sorted(branch_rewards.items(), key=lambda x: x[1], reverse=True)
        leader = ranked[0][0]
        top_two = [name for name, _ in ranked[:2]]
        merged = "\n".join(outputs[name].output for name in top_two)
        return AggregationResult(
            selected_branch=leader,
            selected_output=merged,
            notes={"top_two": top_two, "strategy": "weighted_voting"},
        )

    def _merge_and_refine(self, outputs: dict[str, BranchOutput], branch_rewards: dict[str, float]) -> AggregationResult:
        ranked = sorted(branch_rewards.items(), key=lambda x: x[1], reverse=True)
        names = [name for name, _ in ranked[:3]]
        parts = [outputs[n].output for n in names]
        merged = "Merged candidate:\n" + "\n".join(f"- {p}" for p in parts)
        return AggregationResult(
            selected_branch=names[0],
            selected_output=merged,
            notes={"merged_from": names, "strategy": "merge_and_refine"},
        )
