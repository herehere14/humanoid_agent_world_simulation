from __future__ import annotations

from ..backend.base import LLMBackend
from ..branches.base import PromptBranch
from ..types import BranchOutput, TaskInput


class PromptExecutor:
    def __init__(self, backend: LLMBackend) -> None:
        self.backend = backend

    def run_branch(self, branch: PromptBranch, task: TaskInput, task_type: str, context: str = "") -> BranchOutput:
        prompt = branch.render_prompt(task.text, task_type, context=context)
        output, meta = self.backend.generate(prompt, task, branch.name)
        return BranchOutput(
            branch_name=branch.name,
            prompt=prompt,
            output=output,
            task_type=task_type,
            model_meta=meta,
        )
