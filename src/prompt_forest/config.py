from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .utils.io import read_json


@dataclass
class RouterConfig:
    top_k: int = 3
    exploration: float = 0.15
    min_candidates: int = 2


@dataclass
class OptimizerConfig:
    learning_rate: float = 0.1
    weight_decay: float = 0.03
    min_weight: float = 0.1
    max_weight: float = 3.0
    prompt_rewrite_threshold: float = 0.45
    max_prompt_variants: int = 5
    candidate_trial_episodes: int = 6
    candidate_promote_threshold: float = 0.6
    candidate_failure_trigger: int = 4


@dataclass
class MemoryConfig:
    max_records: int = 5000
    similarity_window: int = 100
    bias_scale: float = 0.6


@dataclass
class EvaluatorConfig:
    aggregation_strategy: str = "judge_select"
    reward_mode: str = "hybrid"


@dataclass
class EngineConfig:
    router: RouterConfig = field(default_factory=RouterConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    evaluator: EvaluatorConfig = field(default_factory=EvaluatorConfig)
    artifacts_dir: str = "artifacts"



def _merge_dataclass(instance: Any, payload: dict[str, Any]) -> Any:
    for key, value in payload.items():
        if not hasattr(instance, key):
            continue
        current = getattr(instance, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _merge_dataclass(current, value)
        else:
            setattr(instance, key, value)
    return instance


def load_config(path: str | Path | None = None) -> EngineConfig:
    cfg = EngineConfig()
    if path is None:
        return cfg
    data = read_json(Path(path))
    return _merge_dataclass(cfg, data)
