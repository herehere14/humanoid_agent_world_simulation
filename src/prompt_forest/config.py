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
    weight_coef: float = 1.0
    affinity_coef: float = 0.6
    memory_coef: float = 0.2
    memory_term_cap: float = 0.15
    exploration_min: float = 0.03
    exploration_decay: float = 0.997


@dataclass
class OptimizerConfig:
    learning_rate: float = 0.1
    weight_decay: float = 0.03
    advantage_baseline_beta: float = 0.1
    min_weight: float = 0.1
    max_weight: float = 3.0
    prompt_rewrite_threshold: float = 0.45
    max_prompt_variants: int = 5
    candidate_trial_episodes: int = 10
    candidate_promote_threshold: float = 0.6
    candidate_neutral_band: float = 0.05
    candidate_extension_episodes: int = 3
    candidate_max_extensions: int = 1
    candidate_failure_trigger: int = 4
    max_active_branches: int = 24
    max_active_candidates: int = 4
    candidate_initial_weight: float = 0.6
    max_hierarchy_depth: int = 4
    candidate_spawn_per_event: int = 1
    candidate_max_children_per_parent: int = 8
    advisor_weight_delta_cap: float = 0.02
    advisor_rewrite_confidence_threshold: float = 0.8
    rewrite_cooldown_episodes: int = 8
    rewrite_failure_streak_trigger: int = 3
    update_acceptance_window: int = 8
    update_acceptance_min_gain: float = 0.002
    candidate_parent_win_rate_threshold: float = 0.55
    candidate_parent_min_comparisons: int = 3


@dataclass
class MemoryConfig:
    max_records: int = 5000
    similarity_window: int = 100
    bias_scale: float = 0.6
    bias_cap: float = 0.15
    shrinkage_k: float = 20.0
    recency_decay: float = 0.98
    user_bias_mix: float = 0.7


@dataclass
class FeedbackConfig:
    user_feedback_weight: float = 0.5
    verifier_weight: float = 0.25
    task_rules_weight: float = 0.15
    llm_judge_weight: float = 0.1
    correction_anchor_weight: float = 0.7
    preference_penalty_cap: float = 0.25


@dataclass
class EvaluatorConfig:
    aggregation_strategy: str = "leaf_select"
    reward_mode: str = "hybrid"


@dataclass
class AgentRuntimeConfig:
    enabled: bool = False
    provider: str = "openai_compatible"
    model: str = "gpt-4.1-mini"
    api_key_env: str = "OPENAI_API_KEY"
    base_url: str = "https://api.openai.com/v1"
    temperature: float = 0.0
    seed: int | None = 42
    max_output_tokens: int = 800
    timeout_seconds: int = 45


@dataclass
class AgentRuntimesConfig:
    evaluator: AgentRuntimeConfig = field(default_factory=AgentRuntimeConfig)
    optimizer: AgentRuntimeConfig = field(default_factory=AgentRuntimeConfig)


@dataclass
class EngineConfig:
    router: RouterConfig = field(default_factory=RouterConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    feedback: FeedbackConfig = field(default_factory=FeedbackConfig)
    evaluator: EvaluatorConfig = field(default_factory=EvaluatorConfig)
    agent_runtimes: AgentRuntimesConfig = field(default_factory=AgentRuntimesConfig)
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
