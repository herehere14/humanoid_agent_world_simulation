from __future__ import annotations

import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

from ..backend.simulated import DomainShiftBackend, shifted_quality_matrix
from ..config import EngineConfig, load_config
from ..core.engine import PromptForestEngine
from ..types import BranchStatus, TaskInput
from ..utils.io import write_json


@dataclass
class PolicyRunMetrics:
    train_rewards: list[float]
    holdout_rewards: list[float]
    train_task_types: list[str]
    holdout_task_types: list[str]
    holdout_user_ids: list[str]
    selected_train: list[str]
    selected_holdout: list[str]
    optimal_train: list[str]
    optimal_holdout: list[str]

    @property
    def avg_train_reward(self) -> float:
        return mean(self.train_rewards) if self.train_rewards else 0.0

    @property
    def avg_holdout_reward(self) -> float:
        return mean(self.holdout_rewards) if self.holdout_rewards else 0.0

    @property
    def first_quarter_train_reward(self) -> float:
        n = max(1, len(self.train_rewards) // 4)
        return mean(self.train_rewards[:n])

    @property
    def last_quarter_train_reward(self) -> float:
        n = max(1, len(self.train_rewards) // 4)
        return mean(self.train_rewards[-n:])

    @property
    def train_reward_trend(self) -> float:
        return self.last_quarter_train_reward - self.first_quarter_train_reward

    @property
    def train_hit_rate(self) -> float:
        if not self.selected_train:
            return 0.0
        hits = sum(1 for s, o in zip(self.selected_train, self.optimal_train) if s == o)
        return hits / len(self.selected_train)

    @property
    def holdout_hit_rate(self) -> float:
        if not self.selected_holdout:
            return 0.0
        hits = sum(1 for s, o in zip(self.selected_holdout, self.optimal_holdout) if s == o)
        return hits / len(self.selected_holdout)


@dataclass
class TrialResult:
    seed: int
    policy_metrics: dict[str, PolicyRunMetrics]


class RLLearningValidator:
    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir)
        self.config_path = self.root_dir / "configs" / "default.json"

    def run(
        self,
        seeds: list[int] | None = None,
        episodes_per_seed: int = 240,
        policies: list[str] | None = None,
        config_patch: dict[str, float] | None = None,
    ) -> dict:
        seeds = seeds or [11, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        policies = policies or ["full", "frozen", "memory_only", "weight_only", "exploration_only"]

        all_results: list[TrialResult] = []
        for seed in seeds:
            train_tasks = self._build_mixed_task_stream(seed=seed, episodes=episodes_per_seed, holdout=False)
            holdout_tasks = self._build_mixed_task_stream(seed=seed + 10000, episodes=max(80, episodes_per_seed // 3), holdout=True)

            policy_metrics: dict[str, PolicyRunMetrics] = {}
            for policy in policies:
                policy_metrics[policy] = self._run_policy(
                    train_tasks=train_tasks,
                    holdout_tasks=holdout_tasks,
                    seed=seed,
                    policy=policy,
                    config_patch=config_patch,
                )
            all_results.append(TrialResult(seed=seed, policy_metrics=policy_metrics))

        policy_aggregate: dict[str, dict[str, float]] = {}
        policy_task_breakdown: dict[str, dict[str, float]] = {}
        policy_user_breakdown: dict[str, dict[str, float]] = {}
        policy_branch_concentration: dict[str, float] = {}
        for policy in policies:
            runs = [trial.policy_metrics[policy] for trial in all_results]
            policy_aggregate[policy] = {
                "mean_train_reward": round(mean(r.avg_train_reward for r in runs), 4),
                "mean_holdout_reward": round(mean(r.avg_holdout_reward for r in runs), 4),
                "mean_train_reward_trend": round(mean(r.train_reward_trend for r in runs), 4),
                "mean_train_hit_rate": round(mean(r.train_hit_rate for r in runs), 4),
                "mean_holdout_hit_rate": round(mean(r.holdout_hit_rate for r in runs), 4),
            }
            policy_task_breakdown[policy] = self._mean_holdout_reward_by_task(runs)
            policy_user_breakdown[policy] = self._mean_holdout_reward_by_user(runs)
            policy_branch_concentration[policy] = round(mean(self._hhi(r.selected_holdout) for r in runs), 4)

        gains_vs_frozen: dict[str, dict[str, float | list[float]]] = {}
        frozen_runs = [trial.policy_metrics["frozen"] for trial in all_results]
        for policy in policies:
            if policy == "frozen":
                continue
            current_runs = [trial.policy_metrics[policy] for trial in all_results]
            holdout_gains = [c.avg_holdout_reward - f.avg_holdout_reward for c, f in zip(current_runs, frozen_runs)]
            train_gains = [c.avg_train_reward - f.avg_train_reward for c, f in zip(current_runs, frozen_runs)]
            gains_vs_frozen[policy] = {
                "mean_train_reward_gain": round(mean(train_gains), 4),
                "mean_holdout_reward_gain": round(mean(holdout_gains), 4),
                "holdout_gain_win_rate": round(sum(1 for g in holdout_gains if g > 0) / max(1, len(holdout_gains)), 4),
                "holdout_gain_bootstrap_ci95": self._bootstrap_ci95(holdout_gains, n_boot=2000),
            }

        full_minus_frozen = [
            trial.policy_metrics["full"].avg_holdout_reward - trial.policy_metrics["frozen"].avg_holdout_reward
            for trial in all_results
        ]
        full_minus_memory = [
            trial.policy_metrics["full"].avg_holdout_reward - trial.policy_metrics["memory_only"].avg_holdout_reward
            for trial in all_results
        ]

        report = {
            "seeds": seeds,
            "episodes_per_seed": episodes_per_seed,
            "policies": policies,
            "evaluation_design": {
                "train_stream": "mixed non-stationary distribution with phase drift",
                "holdout_stream": "unseen prompts/keywords; no adaptation during holdout scoring",
            },
            "aggregate": {
                "policy_metrics": policy_aggregate,
                "policy_holdout_by_task": policy_task_breakdown,
                "policy_holdout_by_user": policy_user_breakdown,
                "policy_branch_concentration_hhi": policy_branch_concentration,
                "gains_vs_frozen": gains_vs_frozen,
                "full_minus_frozen_holdout": round(mean(full_minus_frozen), 4),
                "full_minus_memory_only_holdout": round(mean(full_minus_memory), 4),
                "full_over_frozen_holdout_win_rate": round(
                    sum(1 for g in full_minus_frozen if g > 0) / max(1, len(full_minus_frozen)), 4
                ),
                "full_over_memory_only_holdout_win_rate": round(
                    sum(1 for g in full_minus_memory if g > 0) / max(1, len(full_minus_memory)), 4
                ),
            },
            "per_trial": [
                {
                    "seed": trial.seed,
                    "policy_metrics": {
                        policy: {
                            "avg_train_reward": round(metrics.avg_train_reward, 4),
                            "avg_holdout_reward": round(metrics.avg_holdout_reward, 4),
                            "holdout_by_task": self._holdout_reward_by_task(metrics),
                            "train_reward_trend": round(metrics.train_reward_trend, 4),
                            "train_hit_rate": round(metrics.train_hit_rate, 4),
                            "holdout_hit_rate": round(metrics.holdout_hit_rate, 4),
                            "selected_branch_hhi": round(self._hhi(metrics.selected_holdout), 4),
                        }
                        for policy, metrics in trial.policy_metrics.items()
                    },
                    "gains_vs_frozen": {
                        policy: {
                            "train_reward_gain": round(
                                trial.policy_metrics[policy].avg_train_reward - trial.policy_metrics["frozen"].avg_train_reward,
                                4,
                            ),
                            "holdout_reward_gain": round(
                                trial.policy_metrics[policy].avg_holdout_reward - trial.policy_metrics["frozen"].avg_holdout_reward,
                                4,
                            ),
                        }
                        for policy in policies
                        if policy != "frozen"
                    },
                }
                for trial in all_results
            ],
        }

        out_dir = self.root_dir / "artifacts"
        out_dir.mkdir(parents=True, exist_ok=True)
        write_json(out_dir / "rl_validation_report.json", report)
        return report

    def _build_mixed_task_stream(self, seed: int, episodes: int, holdout: bool) -> list[TaskInput]:
        rng = random.Random(seed)

        train_templates = {
            "math": [
                "Solve a symbolic math reasoning task with constraints and verification.",
                "Compute a constrained derivation and explain the consistency checks.",
            ],
            "planning": [
                "Create a practical execution plan under changing deadlines.",
                "Draft a phased roadmap with milestones and risk controls.",
            ],
            "factual": [
                "Provide factual explanation with grounded points and caveats.",
                "Summarize evidence-backed facts and confidence limitations.",
            ],
            "code": [
                "Analyze and improve a code snippet with bug checks.",
                "Refactor an algorithmic routine and include correctness checks.",
            ],
            "creative": [
                "Produce novel but constrained solution ideas with tradeoffs.",
                "Generate diverse concepts while preserving practical constraints.",
            ],
            "general": [
                "Deliver a robust answer that passes verification.",
                "Provide a balanced response with explicit validation checks.",
            ],
        }

        holdout_templates = {
            "math": [
                "Unseen math prompt: perform algebraic reasoning under hidden shift assumptions.",
                "Holdout derivation challenge: solve and validate a transformed symbolic objective.",
            ],
            "planning": [
                "Unseen planning prompt: restructure delivery after an abrupt requirement drift.",
                "Holdout roadmap challenge: reprioritize milestones with late-stage uncertainty.",
            ],
            "factual": [
                "Unseen factual prompt: resolve conflicting evidence and state confidence bounds.",
                "Holdout grounding challenge: reconcile source disagreement with explicit caveats.",
            ],
            "code": [
                "Unseen code prompt: diagnose regression risk in a modified implementation.",
                "Holdout engineering challenge: identify failure modes after API changes.",
            ],
            "creative": [
                "Unseen creative prompt: generate high-novelty options under strict constraints.",
                "Holdout ideation challenge: diversify solutions while preserving feasibility.",
            ],
            "general": [
                "Unseen general prompt: synthesize an answer robust to distribution shift.",
                "Holdout synthesis challenge: provide calibrated response under ambiguous context.",
            ],
        }

        # Non-stationary phases: distribution drifts over time.
        phases = [
            [("planning", 0.3), ("factual", 0.28), ("general", 0.22), ("math", 0.1), ("code", 0.06), ("creative", 0.04)],
            [("code", 0.3), ("math", 0.26), ("factual", 0.18), ("planning", 0.14), ("general", 0.08), ("creative", 0.04)],
            [("creative", 0.32), ("general", 0.25), ("planning", 0.17), ("code", 0.12), ("math", 0.09), ("factual", 0.05)],
        ]

        tasks: list[TaskInput] = []
        for i in range(episodes):
            phase_idx = min(2, int((i / max(1, episodes)) * 3))
            task_type = self._weighted_choice(phases[phase_idx], rng)
            templates = holdout_templates if holdout else train_templates
            text = rng.choice(templates[task_type])
            keyword_prefix = "holdout_kw" if holdout else "train_kw"
            expected_keywords = [f"{keyword_prefix}_{task_type}_{j}" for j in range(14)]
            required_substrings = ["confidence", "key-points"]

            tasks.append(
                TaskInput(
                    task_id=f"{'hold' if holdout else 'train'}-{seed}-{i}",
                    text=text,
                    task_type=task_type,
                    metadata={
                        "expected_keywords": expected_keywords,
                        "required_substrings": required_substrings,
                        "distribution_phase": phase_idx,
                        "split": "holdout" if holdout else "train",
                        "user_id": rng.choice(["user_alpha", "user_beta", "user_gamma"]),
                    },
                )
            )
        return tasks

    def _weighted_choice(self, choices: list[tuple[str, float]], rng: random.Random) -> str:
        threshold = rng.random()
        cumulative = 0.0
        for name, weight in choices:
            cumulative += weight
            if threshold <= cumulative:
                return name
        return choices[-1][0]

    def _run_policy(
        self,
        train_tasks: list[TaskInput],
        holdout_tasks: list[TaskInput],
        seed: int,
        policy: str,
        config_patch: dict[str, float] | None = None,
    ) -> PolicyRunMetrics:
        cfg = load_config(self.config_path)
        self._configure_for_validation(cfg, policy, config_patch=config_patch)

        run_dir = self.root_dir / "artifacts" / "validation_runs" / f"{policy}_seed_{seed}"
        if run_dir.exists():
            shutil.rmtree(run_dir)
        cfg.artifacts_dir = str(run_dir)

        backend = DomainShiftBackend(shifted_quality_matrix(), noise=0.03, seed=seed)
        engine = PromptForestEngine(config=cfg, backend=backend)

        adapt_train, memory_train = self._policy_flags(policy)

        train_rewards: list[float] = []
        train_task_types: list[str] = []
        selected_train: list[str] = []
        optimal_train: list[str] = []

        for task in train_tasks:
            result = engine.run_task_controlled(
                text=task.text,
                task_type=task.task_type,
                metadata=task.metadata,
                adapt=adapt_train,
                update_memory=memory_train,
            )
            train_rewards.append(result["evaluation_signal"]["reward_score"])
            train_task_types.append(task.task_type)
            selected_train.append(result["evaluation_signal"]["selected_branch"])
            optimal_train.append(backend.best_branch(task.task_type, candidates=self._active_terminal_nodes(engine)))

        holdout_rewards: list[float] = []
        holdout_task_types: list[str] = []
        holdout_user_ids: list[str] = []
        selected_holdout: list[str] = []
        optimal_holdout: list[str] = []

        for task in holdout_tasks:
            result = engine.run_task_controlled(
                text=task.text,
                task_type=task.task_type,
                metadata=task.metadata,
                adapt=False,
                update_memory=False,
            )
            holdout_rewards.append(result["evaluation_signal"]["reward_score"])
            holdout_task_types.append(task.task_type)
            holdout_user_ids.append(str(task.metadata.get("user_id", "global")))
            selected_holdout.append(result["evaluation_signal"]["selected_branch"])
            optimal_holdout.append(backend.best_branch(task.task_type, candidates=self._active_terminal_nodes(engine)))

        return PolicyRunMetrics(
            train_rewards=train_rewards,
            holdout_rewards=holdout_rewards,
            train_task_types=train_task_types,
            holdout_task_types=holdout_task_types,
            holdout_user_ids=holdout_user_ids,
            selected_train=selected_train,
            selected_holdout=selected_holdout,
            optimal_train=optimal_train,
            optimal_holdout=optimal_holdout,
        )

    def _configure_for_validation(
        self,
        cfg: EngineConfig,
        policy: str,
        config_patch: dict[str, float] | None = None,
    ) -> None:
        cfg.router.top_k = 1
        cfg.router.min_candidates = 1

        # Use fixed validation defaults to avoid benchmark drift from config mutations.
        cfg.memory.bias_scale = 0.4
        cfg.memory.bias_cap = 0.15
        cfg.memory.shrinkage_k = 20.0
        cfg.memory.recency_decay = 0.97
        cfg.memory.user_bias_mix = 0.0

        cfg.router.weight_coef = 1.0
        cfg.router.affinity_coef = 0.6
        cfg.router.memory_coef = 0.25
        cfg.router.memory_term_cap = 0.15

        cfg.optimizer.learning_rate = 0.1
        cfg.optimizer.weight_decay = 0.03
        cfg.optimizer.advantage_baseline_beta = 0.12
        cfg.optimizer.candidate_trial_episodes = 10
        cfg.optimizer.candidate_failure_trigger = 999
        cfg.optimizer.max_active_candidates = 0
        # Keep API-runtime evaluations conservative; LLM noise can otherwise amplify unstable updates.
        llm_runtime_on = cfg.agent_runtimes.evaluator.enabled or cfg.agent_runtimes.optimizer.enabled
        if llm_runtime_on:
            cfg.optimizer.rewrite_cooldown_episodes = max(cfg.optimizer.rewrite_cooldown_episodes, 6)
            cfg.optimizer.rewrite_failure_streak_trigger = max(cfg.optimizer.rewrite_failure_streak_trigger, 3)
            cfg.optimizer.update_acceptance_min_gain = max(cfg.optimizer.update_acceptance_min_gain, 0.001)
        else:
            cfg.optimizer.rewrite_cooldown_episodes = 1
            cfg.optimizer.rewrite_failure_streak_trigger = 1
            cfg.optimizer.update_acceptance_min_gain = -1.0

        if policy == "full":
            if llm_runtime_on:
                # API-runtime mode: avoid short-run overreaction from noisy judge/advisor signals.
                cfg.router.exploration = 0.03
                cfg.router.exploration_min = 0.01
                cfg.router.exploration_decay = 0.997
                cfg.router.memory_coef = 0.08
                cfg.memory.bias_scale = 0.35
                cfg.optimizer.learning_rate = 0.07
            else:
                cfg.router.exploration = 0.08
                cfg.router.exploration_min = 0.03
                cfg.router.exploration_decay = 0.995
                cfg.router.memory_coef = 0.12
                cfg.memory.bias_scale = 0.4
                cfg.optimizer.learning_rate = 0.1
            if config_patch:
                self._apply_config_patch(cfg, config_patch)
            return

        if policy == "frozen":
            cfg.router.exploration = 0.02
            cfg.router.memory_coef = 0.0
            cfg.memory.bias_scale = 0.2
            if config_patch:
                self._apply_config_patch(cfg, config_patch)
            return

        if policy == "memory_only":
            cfg.router.exploration = 0.08
            cfg.router.memory_coef = 0.3
            cfg.memory.bias_scale = 0.5
            if config_patch:
                self._apply_config_patch(cfg, config_patch)
            return

        if policy == "weight_only":
            cfg.router.exploration = 0.12
            cfg.router.memory_coef = 0.0
            cfg.memory.bias_scale = 0.2
            if config_patch:
                self._apply_config_patch(cfg, config_patch)
            return

        if policy == "exploration_only":
            cfg.router.exploration = 0.35
            cfg.router.exploration_min = 0.12
            cfg.router.exploration_decay = 0.998
            cfg.router.memory_coef = 0.0
            cfg.memory.bias_scale = 0.2
            if config_patch:
                self._apply_config_patch(cfg, config_patch)
            return

        raise ValueError(f"Unknown policy: {policy}")

    def _apply_config_patch(self, cfg: EngineConfig, config_patch: dict[str, float]) -> None:
        for key, value in config_patch.items():
            if "." not in key:
                continue
            section, field = key.split(".", 1)
            target = getattr(cfg, section, None)
            if target is None or not hasattr(target, field):
                continue
            setattr(target, field, value)

    def _policy_flags(self, policy: str) -> tuple[bool, bool]:
        if policy == "full":
            return True, True
        if policy == "frozen":
            return False, False
        if policy == "memory_only":
            return False, True
        if policy == "weight_only":
            return True, False
        if policy == "exploration_only":
            return False, False
        raise ValueError(f"Unknown policy: {policy}")

    def _active_terminal_nodes(self, engine: PromptForestEngine) -> list[str]:
        terminals: list[str] = []
        for node_id, branch in engine.branches.items():
            if branch.state.status == BranchStatus.ARCHIVED:
                continue
            children = [
                c
                for c in engine.forest.children(node_id)
                if c in engine.branches and engine.branches[c].state.status != BranchStatus.ARCHIVED
            ]
            if not children:
                terminals.append(node_id)
        return terminals

    def _bootstrap_ci95(self, samples: list[float], n_boot: int = 1000) -> list[float]:
        rng = random.Random(1234)
        if not samples:
            return [0.0, 0.0]
        means: list[float] = []
        n = len(samples)
        for _ in range(n_boot):
            draw = [samples[rng.randrange(0, n)] for _ in range(n)]
            means.append(mean(draw))
        means.sort()
        lo = means[int(0.025 * len(means))]
        hi = means[int(0.975 * len(means))]
        return [round(lo, 4), round(hi, 4)]

    def _mean_holdout_reward_by_task(self, runs: list[PolicyRunMetrics]) -> dict[str, float]:
        merged: dict[str, list[float]] = {}
        for run in runs:
            local = self._holdout_reward_by_task(run)
            for task_type, val in local.items():
                merged.setdefault(task_type, []).append(val)
        return {task_type: round(mean(vals), 4) for task_type, vals in sorted(merged.items())}

    def _mean_holdout_reward_by_user(self, runs: list[PolicyRunMetrics]) -> dict[str, float]:
        merged: dict[str, list[float]] = {}
        for run in runs:
            local = self._holdout_reward_by_user(run)
            for user_id, val in local.items():
                merged.setdefault(user_id, []).append(val)
        return {user_id: round(mean(vals), 4) for user_id, vals in sorted(merged.items())}

    def _holdout_reward_by_task(self, run: PolicyRunMetrics) -> dict[str, float]:
        grouped: dict[str, list[float]] = {}
        for task_type, reward in zip(run.holdout_task_types, run.holdout_rewards):
            grouped.setdefault(task_type, []).append(reward)
        return {task_type: round(mean(vals), 4) for task_type, vals in sorted(grouped.items())}

    def _holdout_reward_by_user(self, run: PolicyRunMetrics) -> dict[str, float]:
        grouped: dict[str, list[float]] = {}
        for user_id, reward in zip(run.holdout_user_ids, run.holdout_rewards):
            grouped.setdefault(user_id, []).append(reward)
        return {user_id: round(mean(vals), 4) for user_id, vals in sorted(grouped.items())}

    def _hhi(self, selections: list[str]) -> float:
        if not selections:
            return 0.0
        counts: dict[str, int] = {}
        for branch in selections:
            counts[branch] = counts.get(branch, 0) + 1
        n = len(selections)
        return sum((count / n) ** 2 for count in counts.values())
