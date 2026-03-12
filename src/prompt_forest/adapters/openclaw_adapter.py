from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from ..core.engine import PromptForestEngine


@dataclass
class OpenClawTrajectory:
    episode_id: str
    task: str
    task_type: str = "auto"
    tool_events: list[dict[str, Any]] = field(default_factory=list)
    branch_activity: list[str] = field(default_factory=list)
    outputs: dict[str, str] = field(default_factory=dict)
    expected_keywords: list[str] = field(default_factory=list)
    required_checks: list[str] = field(default_factory=list)
    judge_signals: dict[str, Any] = field(default_factory=dict)


class OpenClawAdapter:
    """Adapter that lets an OpenClaw-style runtime call into prompt-forest adaptation."""

    def __init__(self, engine: PromptForestEngine) -> None:
        self.engine = engine

    def process_trajectory(self, trajectory: OpenClawTrajectory) -> dict[str, Any]:
        event = asdict(trajectory)
        result = self.engine.openclaw_ingest(event)
        return {
            "episode_id": trajectory.episode_id,
            "selected_branch": result["evaluation_signal"]["selected_branch"],
            "reward": result["evaluation_signal"]["reward_score"],
            "failure_reason": result["evaluation_signal"]["failure_reason"],
            "routing": result["routing"],
            "optimization": result["optimization"],
        }
