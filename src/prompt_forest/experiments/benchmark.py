from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from statistics import mean
from typing import Any

from ..backend.mock import MockLLMBackend
from ..backend.simulated import DomainShiftBackend
from ..core.engine import PromptForestEngine
from ..types import ExperimentSummary
from ..utils.io import read_json, write_json


class BenchmarkRunner:
    def __init__(self, engine: PromptForestEngine) -> None:
        self.engine = engine

    def run(self, dataset_path: str | Path, rounds: int = 3) -> ExperimentSummary:
        dataset = read_json(Path(dataset_path)).get("tasks", [])
        train_stream = self._build_nonstationary_train_stream(dataset=dataset, rounds=rounds)
        holdout_stream = self._build_holdout_stream(dataset=dataset)

        adaptive_engine = self.engine
        frozen_engine = self._fresh_engine_for_baseline()

        adaptive_train = self._run_stream(train_stream, adaptive_engine, adapt=True, update_memory=True)
        frozen_train = self._run_stream(train_stream, frozen_engine, adapt=False, update_memory=False)

        adaptive_holdout = self._run_stream(holdout_stream, adaptive_engine, adapt=False, update_memory=False)
        frozen_holdout = self._run_stream(holdout_stream, frozen_engine, adapt=False, update_memory=False)

        summary = adaptive_train
        write_json(self.engine.artifacts_dir / "benchmark_summary.json", summary.to_dict())
        write_json(
            self.engine.artifacts_dir / "benchmark_comparison.json",
            {
                "train": {
                    "adaptive": adaptive_train.to_dict(),
                    "frozen": frozen_train.to_dict(),
                    "average_reward_gain": round(adaptive_train.average_reward - frozen_train.average_reward, 4),
                },
                "holdout": {
                    "adaptive_average_reward": adaptive_holdout.average_reward,
                    "frozen_average_reward": frozen_holdout.average_reward,
                    "average_reward_gain": round(adaptive_holdout.average_reward - frozen_holdout.average_reward, 4),
                },
                "evaluation_design": {
                    "train": "non-stationary stream with task-type drift by phase",
                    "holdout": "unseen prompt variants and keyword sets; no adaptation during scoring",
                },
            },
        )
        return summary

    @staticmethod
    def save_run_report(path: str | Path, payload: dict[str, Any]) -> None:
        write_json(Path(path), payload)

    def _run_stream(
        self,
        tasks: list[dict[str, Any]],
        engine: PromptForestEngine,
        adapt: bool,
        update_memory: bool,
    ) -> ExperimentSummary:
        rewards: list[float] = []
        branch_weight_trajectory: dict[str, list[float]] = {name: [] for name in engine.branches.keys()}
        route_counts: dict[str, int] = {}

        for task in tasks:
            result = engine.run_task_controlled(
                text=task["text"],
                task_type=task.get("task_type", "auto"),
                metadata=task.get("metadata", {}),
                adapt=adapt,
                update_memory=update_memory,
            )
            rewards.append(result["evaluation_signal"]["reward_score"])

        snapshot = engine.branch_snapshot()
        for name, info in snapshot.items():
            branch_weight_trajectory.setdefault(name, []).append(info["weight"])
        route_counts = engine.routing_histogram()

        return ExperimentSummary(
            episodes=len(tasks),
            average_reward=round(mean(rewards), 4) if rewards else 0.0,
            reward_by_round=[round(mean(rewards), 4)] if rewards else [0.0],
            branch_weight_trajectory=branch_weight_trajectory,
            route_counts=route_counts,
        )

    def _build_nonstationary_train_stream(self, dataset: list[dict[str, Any]], rounds: int) -> list[dict[str, Any]]:
        if not dataset:
            return []

        phase_priority = [
            ["planning", "factual", "general", "math", "code", "creative"],
            ["code", "math", "factual", "planning", "general", "creative"],
            ["creative", "general", "planning", "code", "math", "factual"],
        ]

        by_type: dict[str, list[dict[str, Any]]] = {}
        for task in dataset:
            by_type.setdefault(task.get("task_type", "general"), []).append(task)

        stream: list[dict[str, Any]] = []
        for round_idx in range(max(1, rounds)):
            order = phase_priority[round_idx % len(phase_priority)]
            for task_type in order:
                for task in by_type.get(task_type, []):
                    stream.append(deepcopy(task))
            # Include any unseen task types from the dataset at the end of each phase.
            for task in dataset:
                task_type = task.get("task_type", "general")
                if task_type not in order:
                    stream.append(deepcopy(task))

        return stream

    def _build_holdout_stream(self, dataset: list[dict[str, Any]]) -> list[dict[str, Any]]:
        holdout: list[dict[str, Any]] = []
        for idx, task in enumerate(dataset):
            meta = deepcopy(task.get("metadata", {}))
            ttype = task.get("task_type", "general")
            meta["expected_keywords"] = [f"holdout_kw_{ttype}_{j}" for j in range(12)]
            meta["required_substrings"] = ["confidence", "key-points"]
            meta["split"] = "holdout"
            holdout.append(
                {
                    "text": f"{task['text']} Unseen distribution-shift variant #{idx}.",
                    "task_type": ttype,
                    "metadata": meta,
                }
            )
        return holdout

    def _fresh_engine_for_baseline(self) -> PromptForestEngine:
        cfg = deepcopy(self.engine.config)
        cfg.artifacts_dir = str(self.engine.artifacts_dir / "frozen_baseline_run")
        backend = self._clone_backend()
        return PromptForestEngine(config=cfg, backend=backend)

    def _clone_backend(self):
        current = self.engine.backend
        if isinstance(current, DomainShiftBackend):
            return DomainShiftBackend(
                quality_matrix=deepcopy(current.quality_matrix),
                noise=current.noise,
                seed=13,
            )
        return MockLLMBackend()
