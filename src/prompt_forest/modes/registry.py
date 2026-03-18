"""Operating mode registry and configuration.

The system supports two distinct operating modes that share the same core
engine but differ in branch taxonomy, routing priorities, evaluation criteria,
and state management.

- ``agent_improvement``: Optimize task performance through adaptive routing.
- ``human_mode``: Simulate structured human-like cognition and affect.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class OperatingMode(str, Enum):
    AGENT_IMPROVEMENT = "agent_improvement"
    HUMAN_MODE = "human_mode"


@dataclass
class HumanModeConfig:
    """Configuration specific to Human Mode."""

    # State dynamics
    state_decay_rate: float = 0.05
    state_momentum: float = 0.7
    stress_amplification: float = 1.2
    emotional_inertia: float = 0.6

    # Conflict resolution
    conflict_threshold: float = 0.15
    conflict_resolution_strategy: str = "weighted_compromise"
    max_active_conflicts: int = 3

    # Drive weights (baseline influence of each drive category)
    drive_weights: dict[str, float] = field(default_factory=lambda: {
        "reasoning": 1.0,
        "impulse": 0.6,
        "fear": 0.7,
        "curiosity": 0.8,
        "empathy": 0.7,
        "self_protection": 0.6,
        "ambition": 0.7,
        "moral": 0.8,
        "long_term_goals": 0.9,
        "emotional_regulation": 0.8,
    })

    # Evaluation weights for behavioral coherence scoring
    coherence_weight: float = 0.3
    consistency_weight: float = 0.25
    believability_weight: float = 0.25
    conflict_handling_weight: float = 0.2

    # Memory influence
    emotional_memory_decay: float = 0.92
    experience_bias_strength: float = 0.4
    trauma_amplification: float = 1.5

    # Imperfect reasoning simulation
    noise_level: float = 0.08
    bounded_rationality_cap: float = 0.85


@dataclass
class AgentImprovementConfig:
    """Configuration specific to Agent Improvement Mode.

    This is largely a passthrough since the existing EngineConfig already
    covers agent improvement, but it provides a place for mode-specific
    extensions without polluting the shared config.
    """

    # Whether to use aggressive optimization
    aggressive_adaptation: bool = True
    # Performance tracking window
    performance_window: int = 20
    # Minimum improvement threshold to report gain
    min_improvement_threshold: float = 0.02


@dataclass
class ModeConfig:
    """Top-level mode configuration that wraps mode-specific settings."""

    mode: OperatingMode = OperatingMode.AGENT_IMPROVEMENT
    agent_improvement: AgentImprovementConfig = field(
        default_factory=AgentImprovementConfig
    )
    human_mode: HumanModeConfig = field(default_factory=HumanModeConfig)

    @property
    def is_human_mode(self) -> bool:
        return self.mode == OperatingMode.HUMAN_MODE

    @property
    def is_agent_mode(self) -> bool:
        return self.mode == OperatingMode.AGENT_IMPROVEMENT


def get_mode_config(
    mode: str | OperatingMode = "agent_improvement",
    overrides: dict[str, Any] | None = None,
) -> ModeConfig:
    """Create a ModeConfig from a mode string and optional overrides."""
    if isinstance(mode, str):
        mode = OperatingMode(mode)
    cfg = ModeConfig(mode=mode)
    if overrides:
        mode_key = "human_mode" if cfg.is_human_mode else "agent_improvement"
        sub = getattr(cfg, mode_key)
        for k, v in overrides.items():
            if hasattr(sub, k):
                setattr(sub, k, v)
    return cfg
