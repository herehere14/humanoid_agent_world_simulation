from __future__ import annotations

from typing import Any

from ..branches.base import PromptBranch
from ..config import AgentRuntimeConfig
from ..types import EvaluationSignal, RoutingDecision, TaskInput
from .runtime_client import AgentRuntimeClient


class LLMOptimizerAdvisor:
    """Optional LLM advisor for Agent 2. Produces bounded local suggestions."""

    def __init__(self, runtime_config: AgentRuntimeConfig) -> None:
        self.runtime = AgentRuntimeClient(runtime_config)
        self._max_context_chars = 260
        self._max_keywords = 12

    def is_enabled(self) -> bool:
        return self.runtime.is_enabled()

    def advise(
        self,
        task: TaskInput,
        route: RoutingDecision,
        signal: EvaluationSignal,
        branches: dict[str, PromptBranch],
    ) -> dict[str, Any]:
        if not self.is_enabled():
            return {}

        payload = {
            "task": {
                "task_type": route.task_type,
                "text": self._trim(task.text, self._max_context_chars),
                "metadata": self._compact_metadata(task.metadata),
            },
            "activated_branches": route.activated_branches,
            "selected_branch": signal.selected_branch,
            "reward_score": signal.reward_score,
            "failure_reason": self._trim(signal.failure_reason, self._max_context_chars),
            "branch_feedback": {
                name: {
                    "reward": fb.reward,
                    "confidence": fb.confidence,
                    "failure_reason": self._trim(fb.failure_reason, self._max_context_chars),
                    "suggested_improvement_direction": self._trim(
                        fb.suggested_improvement_direction, self._max_context_chars
                    ),
                }
                for name, fb in signal.branch_feedback.items()
            },
            "branch_state_on_path": {
                name: {
                    "weight": branches[name].state.weight,
                    "status": branches[name].state.status.value,
                    "avg_reward": branches[name].state.avg_reward(),
                    "purpose": self._trim(branches[name].state.purpose, self._max_context_chars),
                }
                for name in route.activated_branches
                if name in branches
            },
        }
        system_prompt = (
            "You are Agent 2 (Optimizer Advisor) for a hierarchical prompt-forest.\n"
            "Return STRICT JSON only with bounded local suggestions.\n"
            "Never propose global rewrites. Focus only on activated branches and at most two candidate branch proposals.\n"
            "Schema:\n"
            "{\n"
            '  "branch_directives": [\n'
            "    {\n"
            '      "branch_name": "name_on_path",\n'
            '      "extra_weight_delta": -0.05,\n'
            '      "rewrite_hint": "short actionable hint or empty",\n'
            '      "confidence": 0.0\n'
            "    }\n"
            "  ],\n"
            '  "candidate_proposals": [\n'
            "    {\n"
            '      "base_name": "new_capability_name",\n'
            '      "capability_tag": "short_tag",\n'
            '      "purpose": "one sentence",\n'
            '      "prompt_template": "must include {task_type}, {task}, {context}",\n'
            '      "parent_hint": "activated_branch_or_empty"\n'
            "    }\n"
            "  ]\n"
            "}\n"
            "Constraints: extra_weight_delta must be between -0.1 and 0.1."
        )
        return self.runtime.generate_json(system_prompt, payload)

    def _compact_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        out = dict(metadata or {})
        kws = out.get("expected_keywords")
        if isinstance(kws, list):
            out["expected_keywords"] = kws[: self._max_keywords]
        return out

    @staticmethod
    def _trim(text: Any, max_chars: int) -> str:
        s = str(text)
        if len(s) <= max_chars:
            return s
        return f"{s[: max_chars - 3]}..."
