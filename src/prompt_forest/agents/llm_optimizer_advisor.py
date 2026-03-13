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
                "text": task.text,
                "metadata": task.metadata,
            },
            "activated_branches": route.activated_branches,
            "selected_branch": signal.selected_branch,
            "reward_score": signal.reward_score,
            "failure_reason": signal.failure_reason,
            "branch_feedback": {
                name: {
                    "reward": fb.reward,
                    "confidence": fb.confidence,
                    "failure_reason": fb.failure_reason,
                    "suggested_improvement_direction": fb.suggested_improvement_direction,
                }
                for name, fb in signal.branch_feedback.items()
            },
            "branch_state_on_path": {
                name: {
                    "weight": branches[name].state.weight,
                    "status": branches[name].state.status.value,
                    "avg_reward": branches[name].state.avg_reward(),
                    "purpose": branches[name].state.purpose,
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
