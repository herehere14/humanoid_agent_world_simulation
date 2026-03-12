from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..types import TaskInput


class LLMBackend(ABC):
    @abstractmethod
    def generate(self, prompt: str, task: TaskInput, branch_name: str) -> tuple[str, dict[str, Any]]:
        raise NotImplementedError
