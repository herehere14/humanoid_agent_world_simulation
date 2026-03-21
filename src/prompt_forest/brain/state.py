from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class BrainState:
    """Compact latent-state representation exported by the Prompt Forest brain."""

    variables: dict[str, float]
    dominant_drives: list[str]
    mood_valence: float
    arousal: float
    turn_index: int
    regime: str
    derived: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
