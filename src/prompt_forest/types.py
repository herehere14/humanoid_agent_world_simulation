from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class BranchStatus(str, Enum):
    ACTIVE = "active"
    CANDIDATE = "candidate"
    ARCHIVED = "archived"


@dataclass
class TaskInput:
    task_id: str
    text: str
    task_type: str
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class BranchState:
    name: str
    purpose: str
    prompt_template: str
    weight: float = 1.0
    status: BranchStatus = BranchStatus.ACTIVE
    historical_rewards: list[float] = field(default_factory=list)
    rewrite_history: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    trial_remaining: int = 0

    def avg_reward(self) -> float:
        if not self.historical_rewards:
            return 0.0
        return sum(self.historical_rewards) / len(self.historical_rewards)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        return payload


@dataclass
class BranchOutput:
    branch_name: str
    prompt: str
    output: str
    task_type: str
    model_meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class RoutingDecision:
    task_type: str
    activated_branches: list[str]
    branch_scores: dict[str, float]


@dataclass
class BranchFeedback:
    branch_name: str
    reward: float
    confidence: float
    failure_reason: str
    suggested_improvement_direction: str


@dataclass
class EvaluationSignal:
    reward_score: float
    confidence: float
    selected_branch: str
    selected_output: str
    failure_reason: str
    suggested_improvement_direction: str
    branch_feedback: dict[str, BranchFeedback]
    aggregator_notes: dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryRecord:
    task_id: str
    task_type: str
    input_text: str
    activated_branches: list[str]
    branch_outputs: dict[str, str]
    selected_branch: str
    selected_output: str
    reward_score: float
    failure_reason: str
    confidence: float
    useful_patterns: list[str]
    branch_rewards: dict[str, float] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentSummary:
    episodes: int
    average_reward: float
    reward_by_round: list[float]
    branch_weight_trajectory: dict[str, list[float]]
    route_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
