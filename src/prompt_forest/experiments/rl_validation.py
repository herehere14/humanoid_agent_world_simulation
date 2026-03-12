from __future__ import annotations

import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

from ..backend.simulated import DomainShiftBackend, shifted_quality_matrix
from ..branches.library import create_default_branches
from ..config import load_config
from ..core.engine import PromptForestEngine
from ..types import TaskInput
from ..utils.io import write_json


@dataclass
class PolicyRunMetrics:
    rewards: list[float]
    selected_branches: list[str]
    optimal_branches: list[str]

    @property
    def avg_reward(self) -> float:
        return mean(self.rewards) if self.rewards else 0.0

    @property
    def first_quarter_reward(self) -> float:
        n = max(1, len(self.rewards) // 4)
        return mean(self.rewards[:n])

    @property
    def last_quarter_reward(self) -> float:
        n = max(1, len(self.rewards) // 4)
        return mean(self.rewards[-n:])

    @property
    def overall_optimal_hit_rate(self) -> float:
        if not self.selected_branches:
            return 0.0
        hits = sum(1 for s, o in zip(self.selected_branches, self.optimal_branches) if s == o)
        return hits / len(self.selected_branches)

    @property
    def first_quarter_hit_rate(self) -> float:
        n = max(1, len(self.selected_branches) // 4)
        hits = sum(1 for s, o in zip(self.selected_branches[:n], self.optimal_branches[:n]) if s == o)
        return hits / n

    @property
    def last_quarter_hit_rate(self) -> float:
        n = max(1, len(self.selected_branches) // 4)
        hits = sum(1 for s, o in zip(self.selected_branches[-n:], self.optimal_branches[-n:]) if s == o)
        return hits / n


@dataclass
class TrialResult:
    seed: int
    task_type: str
    adaptive: PolicyRunMetrics
    frozen: PolicyRunMetrics

    @property
    def avg_reward_gain(self) -> float:
        return self.adaptive.avg_reward - self.frozen.avg_reward

    @property
    def adaptive_reward_trend(self) -> float:
        return self.adaptive.last_quarter_reward - self.adaptive.first_quarter_reward

    @property
    def adaptive_hit_trend(self) -> float:
        return self.adaptive.last_quarter_hit_rate - self.adaptive.first_quarter_hit_rate


class RLLearningValidator:
    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir)
        self.config_path = self.root_dir / "configs" / "default.json"

    def run(
        self,
        seeds: list[int] | None = None,
        task_types: list[str] | None = None,
        episodes_per_seed: int = 240,
    ) -> dict:
        seeds = seeds or [11, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        task_types = task_types or ["math", "planning", "factual", "code", "creative", "general"]
        all_results: list[TrialResult] = []

        for seed in seeds:
            for task_type in task_types:
                tasks = self._build_task_stream(seed=seed, episodes=episodes_per_seed, task_type=task_type)
                adaptive = self._run_policy(tasks=tasks, seed=seed, adaptive=True)
                frozen = self._run_policy(tasks=tasks, seed=seed, adaptive=False)
                all_results.append(TrialResult(seed=seed, task_type=task_type, adaptive=adaptive, frozen=frozen))

        reward_gains = [r.avg_reward_gain for r in all_results]
        reward_trends = [r.adaptive_reward_trend for r in all_results]
        hit_trends = [r.adaptive_hit_trend for r in all_results]

        report = {
            "seeds": seeds,
            "task_types": task_types,
            "episodes_per_seed": episodes_per_seed,
            "num_trials": len(all_results),
            "aggregate": {
                "mean_adaptive_reward": round(mean([r.adaptive.avg_reward for r in all_results]), 4),
                "mean_frozen_reward": round(mean([r.frozen.avg_reward for r in all_results]), 4),
                "mean_reward_gain": round(mean(reward_gains), 4),
                "reward_gain_win_rate": round(sum(1 for g in reward_gains if g > 0) / len(reward_gains), 4),
                "mean_adaptive_reward_trend": round(mean(reward_trends), 4),
                "mean_adaptive_optimal_hit_trend": round(mean(hit_trends), 4),
                "reward_gain_bootstrap_ci95": self._bootstrap_ci95(reward_gains, n_boot=2000),
            },
            "per_trial": [
                {
                    "seed": r.seed,
                    "task_type": r.task_type,
                    "adaptive_avg_reward": round(r.adaptive.avg_reward, 4),
                    "frozen_avg_reward": round(r.frozen.avg_reward, 4),
                    "reward_gain": round(r.avg_reward_gain, 4),
                    "adaptive_first_quarter_reward": round(r.adaptive.first_quarter_reward, 4),
                    "adaptive_last_quarter_reward": round(r.adaptive.last_quarter_reward, 4),
                    "adaptive_reward_trend": round(r.adaptive_reward_trend, 4),
                    "adaptive_first_quarter_hit_rate": round(r.adaptive.first_quarter_hit_rate, 4),
                    "adaptive_last_quarter_hit_rate": round(r.adaptive.last_quarter_hit_rate, 4),
                    "adaptive_hit_trend": round(r.adaptive_hit_trend, 4),
                }
                for r in all_results
            ],
        }

        out_dir = self.root_dir / "artifacts"
        out_dir.mkdir(parents=True, exist_ok=True)
        write_json(out_dir / "rl_validation_report.json", report)
        return report

    def _build_task_stream(self, seed: int, episodes: int, task_type: str) -> list[TaskInput]:
        text_map = {
            "math": "Solve a symbolic math reasoning task with constraints.",
            "planning": "Create a practical execution plan under deadlines.",
            "factual": "Provide factual explanation with grounded points.",
            "code": "Analyze and improve a code snippet with bug checks.",
            "creative": "Produce novel but constrained solution ideas.",
            "general": "Deliver a robust answer that passes verification.",
        }

        tasks: list[TaskInput] = []
        for i in range(episodes):
            keywords = [f"kw_{task_type}_{j}" for j in range(12)]
            tasks.append(
                TaskInput(
                    task_id=f"seed-{seed}-{task_type}-ep-{i}",
                    text=text_map[task_type],
                    task_type=task_type,
                    metadata={
                        "expected_keywords": keywords,
                        "required_substrings": ["confidence"],
                    },
                )
            )
        return tasks

    def _run_policy(self, tasks: list[TaskInput], seed: int, adaptive: bool) -> PolicyRunMetrics:
        cfg = load_config(self.config_path)
        cfg.router.top_k = 1
        cfg.router.min_candidates = 1
        cfg.router.exploration = 0.3 if adaptive else 0.02
        cfg.optimizer.learning_rate = 0.24
        cfg.optimizer.weight_decay = 0.02
        cfg.memory.bias_scale = 2.0

        mode = "adaptive" if adaptive else "frozen"
        task_type = tasks[0].task_type if tasks else "unknown"
        run_dir = self.root_dir / "artifacts" / "validation_runs" / f"{mode}_seed_{seed}_{task_type}"
        if run_dir.exists():
            shutil.rmtree(run_dir)
        cfg.artifacts_dir = str(run_dir)

        backend = DomainShiftBackend(shifted_quality_matrix(), noise=0.03, seed=seed)
        branches = create_default_branches()
        for branch in branches.values():
            branch.state.weight = 1.0
        engine = PromptForestEngine(config=cfg, backend=backend, branches=branches)

        rewards: list[float] = []
        selected: list[str] = []
        optimal: list[str] = []

        for task in tasks:
            result = engine.run_task_controlled(
                text=task.text,
                task_type=task.task_type,
                metadata=task.metadata,
                adapt=adaptive,
                update_memory=adaptive,
            )
            rewards.append(result["evaluation_signal"]["reward_score"])
            selected.append(result["evaluation_signal"]["selected_branch"])
            optimal.append(backend.best_branch(task.task_type))

        return PolicyRunMetrics(rewards=rewards, selected_branches=selected, optimal_branches=optimal)

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
