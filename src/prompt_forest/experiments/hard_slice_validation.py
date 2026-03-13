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
class PolicyRun:
    train_rewards: list[float]
    holdout_rewards: list[float]
    selected_holdout: list[str]
    optimal_holdout: list[str]

    @property
    def avg_train_reward(self) -> float:
        return mean(self.train_rewards) if self.train_rewards else 0.0

    @property
    def avg_holdout_reward(self) -> float:
        return mean(self.holdout_rewards) if self.holdout_rewards else 0.0

    @property
    def holdout_hit_rate(self) -> float:
        if not self.selected_holdout:
            return 0.0
        hits = sum(1 for s, o in zip(self.selected_holdout, self.optimal_holdout) if s == o)
        return hits / len(self.selected_holdout)


class HardSliceValidator:
    """Failure-heavy benchmark emphasizing verifier-grounded correctness."""

    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir)
        self.config_path = self.root_dir / "configs" / "default.json"

    def run(
        self,
        seeds: list[int] | None = None,
        episodes_per_seed: int = 220,
        policies: list[str] | None = None,
        policy_patches: dict[str, dict[str, float | int | bool]] | None = None,
        start_mode: str = "default",
        simulate_oracle_feedback: bool = False,
    ) -> dict:
        seeds = seeds or [11, 17, 19, 23, 29, 31, 37, 41]
        policies = policies or ["full", "frozen", "memory_only"]

        per_seed: list[dict] = []
        aggregate_by_policy: dict[str, list[PolicyRun]] = {p: [] for p in policies}

        for seed in seeds:
            train_tasks = self._build_hard_stream(seed=seed, episodes=episodes_per_seed, holdout=False)
            holdout_tasks = self._build_hard_stream(seed=seed + 10000, episodes=max(100, episodes_per_seed // 2), holdout=True)
            seed_row: dict[str, dict[str, float]] = {}

            for policy in policies:
                run = self._run_policy(
                    seed=seed,
                    policy=policy,
                    train_tasks=train_tasks,
                    holdout_tasks=holdout_tasks,
                    policy_patch=(policy_patches or {}).get(policy),
                    start_mode=start_mode,
                    simulate_oracle_feedback=simulate_oracle_feedback,
                )
                aggregate_by_policy[policy].append(run)
                seed_row[policy] = {
                    "avg_train_reward": round(run.avg_train_reward, 4),
                    "avg_holdout_reward": round(run.avg_holdout_reward, 4),
                    "holdout_hit_rate": round(run.holdout_hit_rate, 4),
                }

            full_holdout = seed_row["full"]["avg_holdout_reward"]
            frozen_holdout = seed_row["frozen"]["avg_holdout_reward"]
            seed_row["gains"] = {
                "full_minus_frozen_holdout": round(full_holdout - frozen_holdout, 4),
                "full_over_frozen_relative_gain": round(
                    (full_holdout - frozen_holdout) / max(1e-8, abs(frozen_holdout)),
                    4,
                ),
            }
            per_seed.append({"seed": seed, **seed_row})

        policy_metrics: dict[str, dict[str, float]] = {}
        for policy in policies:
            runs = aggregate_by_policy[policy]
            policy_metrics[policy] = {
                "mean_train_reward": round(mean(r.avg_train_reward for r in runs), 4),
                "mean_holdout_reward": round(mean(r.avg_holdout_reward for r in runs), 4),
                "mean_holdout_hit_rate": round(mean(r.holdout_hit_rate for r in runs), 4),
            }

        full_holdout = policy_metrics["full"]["mean_holdout_reward"]
        frozen_holdout = policy_metrics["frozen"]["mean_holdout_reward"]
        memory_holdout = policy_metrics["memory_only"]["mean_holdout_reward"]

        full_minus_frozen = full_holdout - frozen_holdout
        full_minus_memory = full_holdout - memory_holdout
        full_rel = full_minus_frozen / max(1e-8, abs(frozen_holdout))

        seed_rel = [row["gains"]["full_over_frozen_relative_gain"] for row in per_seed]
        win_rate = sum(1 for g in seed_rel if g > 0.0) / max(1, len(seed_rel))

        report = {
            "seeds": seeds,
            "episodes_per_seed": episodes_per_seed,
            "policies": policies,
            "evaluation_design": {
                "train": "hard-slice non-stationary distribution with strict verifier specs",
                "holdout": "unseen hard prompts + stricter verifier requirements; no adaptation during scoring",
                "reward_mode": "hybrid_verifier",
                "simulate_oracle_feedback": simulate_oracle_feedback,
            },
            "aggregate": {
                "policy_metrics": policy_metrics,
                "full_minus_frozen_holdout": round(full_minus_frozen, 4),
                "full_minus_memory_only_holdout": round(full_minus_memory, 4),
                "full_over_frozen_relative_gain": round(full_rel, 4),
                "full_over_frozen_holdout_win_rate": round(win_rate, 4),
                "target_relative_gain": 0.2,
                "passes_target_relative_gain": full_rel >= 0.2,
            },
            "per_seed": per_seed,
        }

        out = self.root_dir / "artifacts" / "hard_slice_validation_report.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        write_json(out, report)
        return report

    def _run_policy(
        self,
        seed: int,
        policy: str,
        train_tasks: list[TaskInput],
        holdout_tasks: list[TaskInput],
        policy_patch: dict[str, float | int | bool] | None = None,
        start_mode: str = "default",
        simulate_oracle_feedback: bool = False,
    ) -> PolicyRun:
        cfg = load_config(self.config_path)
        self._configure_policy(cfg, policy)
        if policy_patch:
            self._apply_config_patch(cfg, policy_patch)
        run_dir = self.root_dir / "artifacts" / "hard_slice_runs" / f"{policy}_seed_{seed}"
        if run_dir.exists():
            shutil.rmtree(run_dir)
        cfg.artifacts_dir = str(run_dir)

        backend = DomainShiftBackend(shifted_quality_matrix(), noise=0.04, seed=seed)
        engine = PromptForestEngine(config=cfg, backend=backend)
        if start_mode == "anti_prior":
            self._initialize_hard_slice_weights(engine)

        adapt_train, memory_train = self._policy_flags(policy)

        train_rewards: list[float] = []
        for task in train_tasks:
            result = engine.run_task_controlled(
                text=task.text,
                task_type=task.task_type,
                metadata=task.metadata,
                adapt=adapt_train,
                update_memory=memory_train,
            )
            train_rewards.append(result["evaluation_signal"]["reward_score"])
            if simulate_oracle_feedback and policy == "full":
                optimal = backend.best_branch(task.task_type, candidates=self._active_terminal_nodes(engine))
                selected = result["evaluation_signal"]["selected_branch"]
                accepted = selected == optimal
                corrected = ""
                if not accepted:
                    corrected = f"Prefer branch strategy: {optimal}"
                engine.apply_feedback(
                    task_id=result["task"]["task_id"],
                    score=1.0 if accepted else 0.0,
                    accepted=accepted,
                    corrected_answer=corrected,
                    feedback_text="oracle_feedback",
                    user_id=str(task.metadata.get("user_id", "global")),
                )

        selected_holdout: list[str] = []
        optimal_holdout: list[str] = []
        holdout_rewards: list[float] = []
        for task in holdout_tasks:
            result = engine.run_task_controlled(
                text=task.text,
                task_type=task.task_type,
                metadata=task.metadata,
                adapt=False,
                update_memory=False,
            )
            holdout_rewards.append(result["evaluation_signal"]["reward_score"])
            selected_holdout.append(result["evaluation_signal"]["selected_branch"])
            optimal_holdout.append(backend.best_branch(task.task_type, candidates=self._active_terminal_nodes(engine)))

        return PolicyRun(
            train_rewards=train_rewards,
            holdout_rewards=holdout_rewards,
            selected_holdout=selected_holdout,
            optimal_holdout=optimal_holdout,
        )

    @staticmethod
    def _initialize_hard_slice_weights(engine: PromptForestEngine) -> None:
        # Shared anti-prior: intentionally misalign initial routing with shifted optima.
        macro_targets = {
            "analytical": 1.8,
            "verification": 1.6,
            "planner": 0.45,
            "retrieval": 0.45,
            "critique": 0.45,
            "creative": 0.35,
        }
        for name, weight in macro_targets.items():
            branch = engine.branches.get(name)
            if branch is not None:
                branch.state.weight = weight

        # Keep children conservative so adaptation must discover and upweight useful leaves.
        for node_id, branch in engine.branches.items():
            if node_id in macro_targets:
                continue
            branch.state.weight = min(branch.state.weight, 0.55)

    def _configure_policy(self, cfg: EngineConfig, policy: str) -> None:
        cfg.agent_runtimes.evaluator.enabled = False
        cfg.agent_runtimes.optimizer.enabled = False
        cfg.evaluator.reward_mode = "hybrid_verifier"
        cfg.composer.enabled = False
        cfg.router.top_k = 1
        cfg.router.min_candidates = 1
        cfg.router.weight_coef = 1.0
        cfg.router.affinity_coef = 0.6
        cfg.router.memory_coef = 0.14
        cfg.router.memory_term_cap = 0.15
        cfg.router.bandit_value_coef = 0.0
        cfg.router.bandit_bonus_coef = 0.0
        cfg.memory.bias_scale = 0.42
        cfg.memory.user_bias_mix = 0.0

        cfg.optimizer.learning_rate = 0.1
        cfg.optimizer.weight_decay = 0.03
        cfg.optimizer.advantage_baseline_beta = 0.12
        cfg.optimizer.branch_advantage_mix = 0.1
        cfg.optimizer.branch_baseline_beta = 0.05
        cfg.optimizer.candidate_failure_trigger = 4
        cfg.optimizer.candidate_trial_episodes = 10
        cfg.optimizer.max_active_candidates = 0
        cfg.optimizer.update_acceptance_min_gain = -1.0
        cfg.optimizer.rewrite_cooldown_episodes = 1
        cfg.optimizer.rewrite_failure_streak_trigger = 1

        if policy == "full":
            cfg.router.exploration = 0.16
            cfg.router.exploration_min = 0.01
            cfg.router.exploration_decay = 0.995
            cfg.router.memory_coef = 0.18
            cfg.optimizer.learning_rate = 0.14
            cfg.optimizer.weight_decay = 0.02
            return
        if policy == "frozen":
            cfg.router.exploration = 0.02
            cfg.router.memory_coef = 0.0
            cfg.memory.bias_scale = 0.2
            return
        if policy == "memory_only":
            cfg.router.exploration = 0.08
            cfg.router.memory_coef = 0.28
            cfg.memory.bias_scale = 0.5
            return
        raise ValueError(f"Unknown policy: {policy}")

    @staticmethod
    def _apply_config_patch(cfg: EngineConfig, patch: dict[str, float | int | bool]) -> None:
        for key, value in patch.items():
            if "." not in key:
                continue
            section, field = key.split(".", 1)
            target = getattr(cfg, section, None)
            if target is None or not hasattr(target, field):
                continue
            setattr(target, field, value)

    def _build_hard_stream(self, seed: int, episodes: int, holdout: bool) -> list[TaskInput]:
        rng = random.Random(seed)
        tasks: list[TaskInput] = []
        phases = [
            [("planning", 0.3), ("factual", 0.25), ("general", 0.2), ("math", 0.15), ("code", 0.07), ("creative", 0.03)],
            [("code", 0.3), ("math", 0.26), ("factual", 0.2), ("planning", 0.12), ("general", 0.08), ("creative", 0.04)],
            [("creative", 0.32), ("general", 0.24), ("planning", 0.16), ("code", 0.12), ("math", 0.1), ("factual", 0.06)],
        ]

        templates = {
            "math": "Hard-slice math task: solve under strict constraints and verify consistency.",
            "planning": "Hard-slice planning task: produce phased execution under risk and deadline drift.",
            "factual": "Hard-slice factual task: reconcile conflicting evidence with source-confidence calibration.",
            "code": "Hard-slice code task: diagnose regression risk and produce test-backed fix plan.",
            "creative": "Hard-slice creative task: generate diverse options while preserving hard constraints.",
            "general": "Hard-slice synthesis task: provide robust answer with explicit verification checklist.",
        }
        requirements = {
            "math": ["equation", "constraint", "derive", "confidence"],
            "planning": ["plan", "timeline", "risk", "confidence"],
            "factual": ["evidence", "source", "grounded", "confidence"],
            "code": ["bug", "test", "refactor", "confidence"],
            "creative": ["novel", "option", "constraint", "confidence"],
            "general": ["verification", "check", "key-points", "confidence"],
        }

        for i in range(episodes):
            phase = min(2, int((i / max(1, episodes)) * 3))
            task_type = self._weighted_choice(phases[phase], rng)
            hard = requirements[task_type]

            suffix = "holdout" if holdout else "train"
            expected_keywords = list(hard) + [f"{suffix}_kw_{task_type}_{j}" for j in range(14)]
            required_substrings = ["confidence", "key-points", hard[0]]
            verifier_spec = {
                "must_include": hard[:3] + ["confidence"],
                "must_exclude": ["hallucinated", "fabricated"],
                "regex_must_match": [r"confidence=0\\.[0-9]+"],
                "min_token_count": 18 if holdout else 14,
                "confidence_range": [0.58, 0.99] if holdout else [0.5, 0.99],
            }
            text = templates[task_type]
            if holdout:
                text = f"{text} Unseen variant #{i} with stricter validation."

            tasks.append(
                TaskInput(
                    task_id=f"{'hard_hold' if holdout else 'hard_train'}-{seed}-{i}",
                    text=text,
                    task_type=task_type,
                    metadata={
                        "expected_keywords": expected_keywords,
                        "required_substrings": required_substrings,
                        "verifier_spec": verifier_spec,
                        "split": "holdout" if holdout else "train",
                        "distribution_phase": phase,
                        "user_id": rng.choice(["user_alpha", "user_beta", "user_gamma"]),
                    },
                )
            )
        return tasks

    @staticmethod
    def _weighted_choice(choices: list[tuple[str, float]], rng: random.Random) -> str:
        threshold = rng.random()
        cumulative = 0.0
        for name, weight in choices:
            cumulative += weight
            if threshold <= cumulative:
                return name
        return choices[-1][0]

    @staticmethod
    def _policy_flags(policy: str) -> tuple[bool, bool]:
        if policy == "full":
            return True, True
        if policy == "frozen":
            return False, False
        if policy == "memory_only":
            return False, True
        raise ValueError(f"Unknown policy: {policy}")

    @staticmethod
    def _active_terminal_nodes(engine: PromptForestEngine) -> list[str]:
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
