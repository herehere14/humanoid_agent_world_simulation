from __future__ import annotations

from dataclasses import dataclass

from ..types import BranchState, BranchStatus


@dataclass
class PromptBranch:
    state: BranchState

    def render_prompt(self, task_text: str, task_type: str) -> str:
        return self.state.prompt_template.format(task=task_text, task_type=task_type)

    @property
    def name(self) -> str:
        return self.state.name

    @property
    def is_active(self) -> bool:
        return self.state.status in {BranchStatus.ACTIVE, BranchStatus.CANDIDATE}

    def apply_reward(self, reward: float) -> None:
        self.state.historical_rewards.append(reward)

    def rewrite_prompt(self, rewrite_hint: str, max_variants: int = 5) -> None:
        self.state.rewrite_history.append(self.state.prompt_template)
        addon = f"\n\nExtra directive: {rewrite_hint}. Focus on concrete, verifiable outputs."
        if addon not in self.state.prompt_template:
            self.state.prompt_template = self.state.prompt_template + addon

        if len(self.state.rewrite_history) > max_variants:
            self.state.rewrite_history = self.state.rewrite_history[-max_variants:]

    def rollback_prompt(self) -> bool:
        if not self.state.rewrite_history:
            return False
        self.state.prompt_template = self.state.rewrite_history.pop()
        return True
