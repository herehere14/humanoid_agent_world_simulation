from __future__ import annotations

from pathlib import Path
from statistics import mean
from typing import Any

from ..core.engine import PromptForestEngine
from ..types import ExperimentSummary
from ..utils.io import read_json, write_json


class BenchmarkRunner:
    def __init__(self, engine: PromptForestEngine) -> None:
        self.engine = engine

    def run(self, dataset_path: str | Path, rounds: int = 3) -> ExperimentSummary:
        dataset = read_json(Path(dataset_path)).get("tasks", [])

        reward_by_round: list[float] = []
        branch_weight_trajectory: dict[str, list[float]] = {name: [] for name in self.engine.branches.keys()}
        route_counts: dict[str, int] = {}

        total_episodes = 0

        for _ in range(rounds):
            round_rewards: list[float] = []
            for task in dataset:
                result = self.engine.run_task(
                    text=task["text"],
                    task_type=task.get("task_type", "auto"),
                    metadata=task.get("metadata", {}),
                )
                total_episodes += 1
                round_rewards.append(result["evaluation_signal"]["reward_score"])

            reward_by_round.append(round(mean(round_rewards), 4) if round_rewards else 0.0)

            snapshot = self.engine.branch_snapshot()
            for name, info in snapshot.items():
                branch_weight_trajectory.setdefault(name, []).append(info["weight"])

            route_counts = self.engine.routing_histogram()

        summary = ExperimentSummary(
            episodes=total_episodes,
            average_reward=round(mean(reward_by_round), 4) if reward_by_round else 0.0,
            reward_by_round=reward_by_round,
            branch_weight_trajectory=branch_weight_trajectory,
            route_counts=route_counts,
        )

        output_path = self.engine.artifacts_dir / "benchmark_summary.json"
        write_json(output_path, summary.to_dict())
        return summary

    @staticmethod
    def save_run_report(path: str | Path, payload: dict[str, Any]) -> None:
        write_json(Path(path), payload)
