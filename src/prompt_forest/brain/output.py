from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class BrainConflictSignal:
    """Structured representation of an active internal conflict."""

    name: str
    drive_a: str
    drive_b: str
    intensity: float
    resolution: str
    dominant_drive: str
    resolution_weight: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BrainControlSignals:
    """Portable control surface that an external agent can act on."""

    approach_drive: float
    avoidance_drive: float
    exploration_drive: float
    switch_pressure: float
    persistence_drive: float
    self_protection: float
    social_openness: float
    cognitive_effort: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass
class BrainActionTendencies:
    """Domain-agnostic behavioral tendencies inferred from the current brain state."""

    act: float
    inhibit: float
    explore: float
    exploit: float
    reflect: float
    react: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass
class BrainOutput:
    """Reusable brain readout produced by the Prompt Forest cognitive layer."""

    regime: str
    state: dict[str, float]
    dominant_drives: list[str]
    branch_activations: dict[str, float]
    active_branches: list[str]
    conflicts: list[BrainConflictSignal]
    control_signals: BrainControlSignals
    action_tendencies: BrainActionTendencies
    memory_biases: dict[str, float] = field(default_factory=dict)
    state_summary: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["conflicts"] = [conflict.to_dict() for conflict in self.conflicts]
        payload["control_signals"] = self.control_signals.to_dict()
        payload["action_tendencies"] = self.action_tendencies.to_dict()
        return payload
