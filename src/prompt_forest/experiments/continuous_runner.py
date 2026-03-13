from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from ..experiments.auto_improve import AutoImprover
from ..utils.io import read_json, write_json


@dataclass
class CycleMetrics:
    cycle: int
    objective: float
    full_minus_frozen: float
    full_minus_memory_only: float
    full_win_rate_vs_frozen: float
    full_ci95_low: float
    fairness_gap: float
    tests_passed: bool


class ContinuousImprover:
    """Long-running improvement loop with regression gates and rollback."""

    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir)
        self.config_path = self.root_dir / "configs" / "default.json"
        self.auto = AutoImprover(self.root_dir)

    def run(
        self,
        max_cycles: int = 6,
        rounds_per_cycle: int = 2,
        candidates_per_round: int = 6,
        seeds: list[int] | None = None,
        episodes_per_seed: int = 180,
        final_eval_seeds: list[int] | None = None,
        final_eval_episodes: int = 220,
        run_tests_each_cycle: bool = True,
        sleep_seconds: int = 0,
        early_stop_patience: int = 2,
        apply_best: bool = True,
    ) -> dict:
        seeds = seeds or [11, 17, 19, 23, 29, 31, 37, 41]
        final_eval_seeds = final_eval_seeds or [11, 17, 19, 23, 29, 31, 37, 41]

        out_dir = self.root_dir / "artifacts" / "continuous_improve"
        out_dir.mkdir(parents=True, exist_ok=True)

        best_config = read_json(self.config_path)
        best_score = float("-inf")
        best_cycle = 0
        no_improve = 0

        cycle_results: list[dict] = []

        for cycle in range(1, max_cycles + 1):
            snapshot_before = read_json(self.config_path)

            auto_report = self.auto.run(
                rounds=rounds_per_cycle,
                candidates_per_round=candidates_per_round,
                seeds=seeds,
                episodes_per_seed=episodes_per_seed,
                final_eval_seeds=final_eval_seeds,
                final_eval_episodes=final_eval_episodes,
                apply_best=apply_best,
            )

            best_candidate = auto_report["best_candidate"]
            tests_passed = True
            if run_tests_each_cycle:
                tests_passed = self._run_regression_gates()

            metrics = CycleMetrics(
                cycle=cycle,
                objective=float(best_candidate["objective"]),
                full_minus_frozen=float(best_candidate["full_minus_frozen"]),
                full_minus_memory_only=float(best_candidate["full_minus_memory_only"]),
                full_win_rate_vs_frozen=float(best_candidate["full_win_rate_vs_frozen"]),
                full_ci95_low=float(best_candidate.get("full_ci95_low", -1.0)),
                fairness_gap=float(best_candidate["fairness_gap"]),
                tests_passed=tests_passed,
            )

            accepted = tests_passed and self._accept(metrics, best_score)
            if accepted:
                best_score = metrics.objective
                best_cycle = cycle
                if apply_best:
                    best_config = read_json(self.config_path)
                no_improve = 0
            else:
                # Roll back config if gates fail or objective regresses.
                if apply_best:
                    write_json(self.config_path, best_config if best_cycle else snapshot_before)
                no_improve += 1

            payload = {
                "cycle": cycle,
                "accepted": accepted,
                "metrics": {
                    "objective": round(metrics.objective, 6),
                    "full_minus_frozen": round(metrics.full_minus_frozen, 6),
                    "full_minus_memory_only": round(metrics.full_minus_memory_only, 6),
                    "full_win_rate_vs_frozen": round(metrics.full_win_rate_vs_frozen, 6),
                    "full_ci95_low": round(metrics.full_ci95_low, 6),
                    "fairness_gap": round(metrics.fairness_gap, 6),
                    "tests_passed": metrics.tests_passed,
                },
                "auto_report": auto_report,
            }
            cycle_results.append(payload)
            write_json(out_dir / f"cycle_{cycle}_report.json", payload)

            if no_improve >= early_stop_patience:
                break
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

        summary = {
            "cycles_run": len(cycle_results),
            "best_cycle": best_cycle,
            "best_objective": round(best_score, 6) if best_cycle else None,
            "early_stopped": no_improve >= early_stop_patience,
            "cycle_results": cycle_results,
        }
        write_json(out_dir / "continuous_improve_summary.json", summary)
        return summary

    def _run_regression_gates(self) -> bool:
        cmd = [
            "pytest",
            "-q",
            "tests/test_learning_dynamics.py",
            "tests/test_router_memory.py",
            "tests/test_optimizer.py",
            "tests/test_detailed_validation.py",
        ]
        res = subprocess.run(cmd, cwd=self.root_dir, check=False, capture_output=True, text=True)
        return res.returncode == 0

    @staticmethod
    def _accept(metrics: CycleMetrics, current_best: float) -> bool:
        if metrics.full_minus_frozen <= 0.0:
            return False
        if metrics.full_minus_memory_only <= 0.0:
            return False
        if metrics.full_win_rate_vs_frozen < 0.5:
            return False
        if metrics.full_ci95_low < 0.0:
            return False
        if metrics.fairness_gap > 0.3:
            return False
        return metrics.objective > current_best + 1e-6
