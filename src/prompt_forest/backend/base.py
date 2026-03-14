from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from ..types import TaskInput


class LLMBackend(ABC):
    @abstractmethod
    def generate(self, prompt: str, task: TaskInput, branch_name: str) -> tuple[str, dict[str, Any]]:
        raise NotImplementedError

    def generate_stream(
        self,
        prompt: str,
        task: TaskInput,
        branch_name: str,
        *,
        on_delta: Callable[[str], None] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        text, meta = self.generate(prompt, task, branch_name)
        if on_delta and text:
            on_delta(text)
        return text, meta
