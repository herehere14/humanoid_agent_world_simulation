from __future__ import annotations

from dataclasses import dataclass

from ..rewards.modes import ExactMatchReward, HybridReward, KeywordReward, RuleBasedReward
from ..types import BranchOutput, TaskInput


@dataclass
class BranchScore:
    reward: float
    reason: str


class OutputJudge:
    def __init__(self, reward_mode: str = "hybrid") -> None:
        self.reward_mode = reward_mode
        self._reward_fn = self._build_reward(reward_mode)

    def _build_reward(self, reward_mode: str):
        if reward_mode == "exact":
            return ExactMatchReward(weight=1.0)
        if reward_mode == "rule":
            return RuleBasedReward(weight=1.0)
        if reward_mode == "keyword":
            return KeywordReward(weight=1.0)
        return HybridReward(
            exact=ExactMatchReward(weight=0.3),
            keyword=KeywordReward(weight=0.5),
            rule=RuleBasedReward(weight=0.2),
        )

    def score_output(self, output: str, task: TaskInput) -> BranchScore:
        reward, reason = self._reward_fn.score(output, task)
        return BranchScore(reward=max(0.0, min(1.0, reward)), reason=reason)

    def score_all(self, branch_outputs: dict[str, BranchOutput], task: TaskInput) -> dict[str, BranchScore]:
        out: dict[str, BranchScore] = {}
        for branch_name, branch_output in branch_outputs.items():
            out[branch_name] = self.score_output(branch_output.output, task)
        return out
