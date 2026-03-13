from __future__ import annotations

import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev

from ..backend.simulated import DomainShiftBackend, shifted_quality_matrix
from ..config import load_config
from ..core.engine import PromptForestEngine
from ..types import BranchStatus, TaskInput
from ..utils.io import write_json


@dataclass
class PolicyMetrics:
    rewards: list[float]
    selected: list[str]
    optimal: list[str]
    path_lengths: list[int]

    @property
    def avg_reward(self) -> float:
        return mean(self.rewards) if self.rewards else 0.0

    @property
    def reward_std(self) -> float:
        return pstdev(self.rewards) if len(self.rewards) >= 2 else 0.0

    @property
    def high_reward_rate(self) -> float:
        if not self.rewards:
            return 0.0
        return sum(1 for r in self.rewards if r >= 0.75) / len(self.rewards)

    @property
    def optimal_hit_rate(self) -> float:
        if not self.selected:
            return 0.0
        hits = sum(1 for s, o in zip(self.selected, self.optimal) if s == o)
        return hits / len(self.selected)

    @property
    def avg_path_length(self) -> float:
        return mean(self.path_lengths) if self.path_lengths else 0.0

    @property
    def early_reward(self) -> float:
        n = max(1, len(self.rewards) // 4)
        return mean(self.rewards[:n])

    @property
    def late_reward(self) -> float:
        n = max(1, len(self.rewards) // 4)
        return mean(self.rewards[-n:])

    @property
    def late_minus_early(self) -> float:
        return self.late_reward - self.early_reward


class DetailedHierarchicalValidator:
    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir)
        self.config_path = self.root_dir / "configs" / "default.json"

    def run(
        self,
        seeds: list[int] | None = None,
        episodes_per_seed: int = 240,
        task_types: list[str] | None = None,
    ) -> dict:
        seeds = seeds or [11, 17, 19, 23, 29]
        task_types = task_types or ["math", "planning", "factual", "code", "creative", "general"]

        adaptive_runs: list[PolicyMetrics] = []
        frozen_runs: list[PolicyMetrics] = []
        macro_only_runs: list[PolicyMetrics] = []
        no_verification_runs: list[PolicyMetrics] = []
        no_retrieval_runs: list[PolicyMetrics] = []

        per_seed: list[dict] = []
        branch_usage: Counter[str] = Counter()

        for seed in seeds:
            for task_type in task_types:
                tasks = self._build_task_stream(seed=seed, episodes=episodes_per_seed, task_type=task_type)

                adaptive = self._run_policy(tasks=tasks, seed=seed, adaptive=True, ablation="none")
                frozen = self._run_policy(tasks=tasks, seed=seed, adaptive=False, ablation="none")
                macro_only = self._run_policy(tasks=tasks, seed=seed, adaptive=True, ablation="macro_only")
                no_ver = self._run_policy(tasks=tasks, seed=seed, adaptive=True, ablation="no_verification_subtree")
                no_ret = self._run_policy(tasks=tasks, seed=seed, adaptive=True, ablation="no_retrieval_subtree")

                adaptive_runs.append(adaptive)
                frozen_runs.append(frozen)
                macro_only_runs.append(macro_only)
                no_verification_runs.append(no_ver)
                no_retrieval_runs.append(no_ret)

                branch_usage.update(adaptive.selected)

                per_seed.append(
                    {
                        "seed": seed,
                        "task_type": task_type,
                        "adaptive_avg_reward": round(adaptive.avg_reward, 4),
                        "frozen_avg_reward": round(frozen.avg_reward, 4),
                        "reward_gain": round(adaptive.avg_reward - frozen.avg_reward, 4),
                        "adaptive_hit_rate": round(adaptive.optimal_hit_rate, 4),
                        "frozen_hit_rate": round(frozen.optimal_hit_rate, 4),
                        "hit_gain": round(adaptive.optimal_hit_rate - frozen.optimal_hit_rate, 4),
                        "adaptive_trend": round(adaptive.late_minus_early, 4),
                        "macro_only_avg_reward": round(macro_only.avg_reward, 4),
                        "no_verification_avg_reward": round(no_ver.avg_reward, 4),
                        "no_retrieval_avg_reward": round(no_ret.avg_reward, 4),
                    }
                )

        branch_inventory = self._branch_inventory()

        learning = {
            "mean_adaptive_reward": round(mean(m.avg_reward for m in adaptive_runs), 4),
            "mean_frozen_reward": round(mean(m.avg_reward for m in frozen_runs), 4),
            "mean_reward_gain": round(mean(a.avg_reward - f.avg_reward for a, f in zip(adaptive_runs, frozen_runs)), 4),
            "reward_gain_win_rate": round(
                sum(1 for a, f in zip(adaptive_runs, frozen_runs) if a.avg_reward > f.avg_reward) / len(adaptive_runs), 4
            ),
            "mean_adaptive_optimal_hit_rate": round(mean(m.optimal_hit_rate for m in adaptive_runs), 4),
            "mean_frozen_optimal_hit_rate": round(mean(m.optimal_hit_rate for m in frozen_runs), 4),
            "mean_optimal_hit_gain": round(
                mean(a.optimal_hit_rate - f.optimal_hit_rate for a, f in zip(adaptive_runs, frozen_runs)), 4
            ),
            "mean_adaptive_high_reward_rate": round(mean(m.high_reward_rate for m in adaptive_runs), 4),
            "mean_frozen_high_reward_rate": round(mean(m.high_reward_rate for m in frozen_runs), 4),
            "mean_late_minus_early": round(mean(m.late_minus_early for m in adaptive_runs), 4),
            "mean_adaptive_path_length": round(mean(m.avg_path_length for m in adaptive_runs), 4),
            "mean_frozen_path_length": round(mean(m.avg_path_length for m in frozen_runs), 4),
            "mean_adaptive_reward_std": round(mean(m.reward_std for m in adaptive_runs), 4),
            "mean_frozen_reward_std": round(mean(m.reward_std for m in frozen_runs), 4),
        }

        branch_effect = {
            "full_minus_macro_only": round(
                mean(a.avg_reward - b.avg_reward for a, b in zip(adaptive_runs, macro_only_runs)), 4
            ),
            "full_minus_no_verification": round(
                mean(a.avg_reward - b.avg_reward for a, b in zip(adaptive_runs, no_verification_runs)), 4
            ),
            "full_minus_no_retrieval": round(
                mean(a.avg_reward - b.avg_reward for a, b in zip(adaptive_runs, no_retrieval_runs)), 4
            ),
        }

        top_selected = [{"branch": b, "count": c} for b, c in branch_usage.most_common(12)]

        report = {
            "seeds": seeds,
            "episodes_per_seed": episodes_per_seed,
            "task_types": task_types,
            "branch_inventory": branch_inventory,
            "learning": learning,
            "branch_effect": branch_effect,
            "growth": self._growth_probe(seeds=seeds, episodes=episodes_per_seed),
            "top_selected_leaves": top_selected,
            "per_seed": per_seed,
        }

        out_path = self.root_dir / "artifacts" / "detailed_validation_report.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(out_path, report)
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
                    metadata={"expected_keywords": keywords, "required_substrings": ["confidence"]},
                )
            )
        return tasks

    def _run_policy(self, tasks: list[TaskInput], seed: int, adaptive: bool, ablation: str) -> PolicyMetrics:
        cfg = load_config(self.config_path)
        cfg.agent_runtimes.evaluator.enabled = False
        cfg.agent_runtimes.optimizer.enabled = False
        cfg.router.top_k = 1
        cfg.router.min_candidates = 1
        cfg.router.exploration = 0.12 if adaptive else 0.02
        cfg.router.memory_coef = 0.2
        cfg.memory.bias_scale = max(0.2, min(0.6, cfg.memory.bias_scale))
        cfg.memory.user_bias_mix = 0.0
        cfg.optimizer.learning_rate = 0.08
        cfg.optimizer.weight_decay = 0.03
        cfg.optimizer.rewrite_cooldown_episodes = 1
        cfg.optimizer.rewrite_failure_streak_trigger = 1
        cfg.optimizer.update_acceptance_min_gain = -1.0
        cfg.optimizer.candidate_failure_trigger = 999
        cfg.optimizer.max_active_candidates = 0
        cfg.composer.enabled = False

        mode = "adaptive" if adaptive else "frozen"
        task_key = tasks[0].task_type if tasks else "unknown"
        run_dir = self.root_dir / "artifacts" / "detailed_validation_runs" / f"{mode}_{ablation}_seed_{seed}_{task_key}"
        if run_dir.exists():
            shutil.rmtree(run_dir)
        cfg.artifacts_dir = str(run_dir)

        backend = DomainShiftBackend(shifted_quality_matrix(), noise=0.03, seed=seed)
        engine = PromptForestEngine(config=cfg, backend=backend)
        self._apply_ablation(engine, ablation)

        rewards: list[float] = []
        selected: list[str] = []
        optimal: list[str] = []
        path_lengths: list[int] = []

        terminal_nodes = self._active_terminal_nodes(engine)

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
            optimal.append(backend.best_branch(task.task_type, candidates=terminal_nodes))
            path_lengths.append(len(result["routing"]["activated_branches"]))

        return PolicyMetrics(rewards=rewards, selected=selected, optimal=optimal, path_lengths=path_lengths)

    def _apply_ablation(self, engine: PromptForestEngine, ablation: str) -> None:
        if ablation == "none":
            return

        if ablation == "macro_only":
            for node_id, node in engine.forest.nodes.items():
                if node_id in engine.branches and node.depth >= 2:
                    engine.branches[node_id].state.status = BranchStatus.ARCHIVED
            return

        if ablation == "no_verification_subtree":
            self._archive_subtree(engine, "verification")
            return

        if ablation == "no_retrieval_subtree":
            self._archive_subtree(engine, "retrieval")
            return

        raise ValueError(f"Unknown ablation: {ablation}")

    def _archive_subtree(self, engine: PromptForestEngine, root: str) -> None:
        stack = [root]
        while stack:
            node_id = stack.pop()
            if node_id in engine.branches:
                engine.branches[node_id].state.status = BranchStatus.ARCHIVED
            stack.extend(engine.forest.children(node_id))

    def _active_terminal_nodes(self, engine: PromptForestEngine) -> list[str]:
        terminals: list[str] = []
        for node_id in engine.branches:
            branch = engine.branches[node_id]
            if branch.state.status == BranchStatus.ARCHIVED:
                continue
            children = [c for c in engine.forest.children(node_id) if engine.branches.get(c) and engine.branches[c].state.status != BranchStatus.ARCHIVED]
            if not children:
                terminals.append(node_id)
        return terminals

    def _branch_inventory(self) -> dict:
        engine = PromptForestEngine(config=load_config(self.config_path))
        snapshot = engine.branch_snapshot()

        macro = [name for name, meta in snapshot.items() if meta["depth"] == 1]
        niche = [name for name, meta in snapshot.items() if meta["depth"] >= 2]

        return {
            "total_trainable_branches": len(snapshot),
            "macro_branch_count": len(macro),
            "niche_sub_branch_count": len(niche),
            "macro_branches": sorted(macro),
            "niche_sub_branches": sorted(niche),
        }

    def _growth_probe(self, seeds: list[int], episodes: int) -> dict:
        created_counts: list[int] = []
        depth_gains: list[int] = []
        reward_trends: list[float] = []

        for seed in seeds:
            cfg = load_config(self.config_path)
            cfg.router.top_k = 1
            cfg.router.min_candidates = 1
            cfg.router.exploration = 0.35
            cfg.router.memory_coef = 0.2
            cfg.optimizer.learning_rate = 0.08
            cfg.optimizer.weight_decay = 0.03
            cfg.optimizer.advantage_baseline_beta = 0.1
            cfg.optimizer.rewrite_cooldown_episodes = 1
            cfg.optimizer.rewrite_failure_streak_trigger = 1
            cfg.optimizer.update_acceptance_min_gain = -1.0
            cfg.optimizer.candidate_failure_trigger = 2
            cfg.optimizer.candidate_trial_episodes = 4
            cfg.optimizer.max_active_candidates = 6
            cfg.optimizer.max_active_branches = 40
            cfg.memory.bias_scale = 0.5
            cfg.composer.enabled = False

            run_dir = self.root_dir / "artifacts" / "growth_probe_runs" / f"seed_{seed}"
            if run_dir.exists():
                shutil.rmtree(run_dir)
            cfg.artifacts_dir = str(run_dir)

            backend = DomainShiftBackend(shifted_quality_matrix(), noise=0.03, seed=seed)
            engine = PromptForestEngine(config=cfg, backend=backend)

            rewards: list[float] = []
            base_branch_count = len(engine.branch_snapshot())
            base_max_depth = max(meta["depth"] for meta in engine.branch_snapshot().values())

            for i in range(episodes):
                task_type = ["planning", "factual", "code", "creative", "general", "math"][i % 6]
                metadata = {
                    "expected_keywords": [f"kw_{task_type}_{j}" for j in range(20)],
                    "required_substrings": ["never_present_token_zzz"],  # induce misses and candidate proposals
                }
                result = engine.run_task(
                    text=f"High-complexity {task_type} task requiring missing capability fill-in",
                    task_type=task_type,
                    metadata=metadata,
                )
                rewards.append(result["evaluation_signal"]["reward_score"])

            snap = engine.branch_snapshot()
            created_counts.append(len(snap) - base_branch_count)
            depth_gains.append(max(meta["depth"] for meta in snap.values()) - base_max_depth)
            n = max(1, len(rewards) // 3)
            reward_trends.append(mean(rewards[-n:]) - mean(rewards[:n]))

        return {
            "mean_new_branches_created": round(mean(created_counts), 4),
            "mean_depth_gain": round(mean(depth_gains), 4),
            "mean_growth_reward_trend": round(mean(reward_trends), 4),
        }
