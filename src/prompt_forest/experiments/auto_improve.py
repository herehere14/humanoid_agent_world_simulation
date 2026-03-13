from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from statistics import pstdev

from ..experiments.rl_validation import RLLearningValidator
from ..utils.io import read_json, write_json


@dataclass
class CandidateResult:
    patch: dict[str, float]
    objective: float
    full_holdout: float
    full_minus_frozen: float
    full_minus_memory_only: float
    full_win_rate_vs_frozen: float
    full_ci95_low: float
    fairness_gap: float
    gain_std: float
    branch_concentration_gap: float
    report_path: str
    ci_gate_passed: bool


class AutoImprover:
    """Multi-round config search with conservative anti-bias objective."""

    def __init__(self, root_dir: str | Path, seed: int = 20260313) -> None:
        self.root_dir = Path(root_dir)
        self.config_path = self.root_dir / "configs" / "default.json"
        self.validator = RLLearningValidator(self.root_dir)
        self._rng = random.Random(seed)

        self.search_space: dict[str, tuple[float, float]] = {
            "router.memory_coef": (0.1, 0.3),
            "router.exploration": (0.08, 0.18),
            "router.exploration_decay": (0.994, 0.999),
            "memory.bias_scale": (0.2, 0.6),
            "memory.recency_decay": (0.94, 0.99),
            "memory.shrinkage_k": (12.0, 40.0),
            "optimizer.learning_rate": (0.06, 0.14),
            "optimizer.weight_decay": (0.02, 0.05),
            "optimizer.advantage_baseline_beta": (0.08, 0.2),
        }

    def run(
        self,
        rounds: int = 3,
        candidates_per_round: int = 12,
        seeds: list[int] | None = None,
        episodes_per_seed: int = 140,
        final_eval_seeds: list[int] | None = None,
        final_eval_episodes: int = 220,
        apply_best: bool = True,
    ) -> dict:
        seeds = seeds or [11, 17, 19, 23, 29, 31]
        final_eval_seeds = final_eval_seeds or [11, 17, 19, 23, 29, 31, 37, 41]

        out_dir = self.root_dir / "artifacts" / "auto_improve"
        out_dir.mkdir(parents=True, exist_ok=True)

        incumbent_patch = self._default_patch_from_config()
        incumbent_eval = self._evaluate_patch(
            patch=incumbent_patch,
            seeds=seeds,
            episodes_per_seed=episodes_per_seed,
            report_path=out_dir / "baseline_report.json",
        )

        rounds_payload: list[dict] = []
        best = incumbent_eval

        for round_idx in range(1, rounds + 1):
            candidates = [incumbent_patch]
            while len(candidates) < candidates_per_round:
                mode = "exploit" if self._rng.random() < 0.65 else "explore"
                candidate = self._sample_patch(center=incumbent_patch, mode=mode)
                candidates.append(candidate)

            evaluated: list[CandidateResult] = []
            for cand_idx, patch in enumerate(candidates):
                report_path = out_dir / f"round_{round_idx}_candidate_{cand_idx}.json"
                result = self._evaluate_patch(
                    patch=patch,
                    seeds=seeds,
                    episodes_per_seed=episodes_per_seed,
                    report_path=report_path,
                )
                evaluated.append(result)

            round_best = max(evaluated, key=lambda x: x.objective)
            improved = round_best.objective > best.objective + 1e-6
            if improved and round_best.ci_gate_passed:
                best = round_best
                incumbent_patch = dict(round_best.patch)

            rounds_payload.append(
                {
                    "round": round_idx,
                    "improved": improved,
                    "best_objective": round(best.objective, 6),
                    "round_best": self._result_to_dict(round_best),
                    "incumbent": self._result_to_dict(best),
                }
            )

            write_json(out_dir / f"round_{round_idx}_summary.json", rounds_payload[-1])

        if apply_best:
            self._apply_patch_to_default_config(best.patch)

        final_report = self.validator.run(
            seeds=final_eval_seeds,
            episodes_per_seed=final_eval_episodes,
            config_patch=best.patch,
        )
        write_json(out_dir / "final_validation_report.json", final_report)

        payload = {
            "rounds": rounds,
            "candidates_per_round": candidates_per_round,
            "search_seeds": seeds,
            "search_episodes_per_seed": episodes_per_seed,
            "final_eval_seeds": final_eval_seeds,
            "final_eval_episodes": final_eval_episodes,
            "best_candidate": self._result_to_dict(best),
            "round_summaries": rounds_payload,
            "final_validation_path": str(out_dir / "final_validation_report.json"),
            "applied_to_default_config": apply_best,
        }
        write_json(out_dir / "auto_improve_summary.json", payload)
        return payload

    def _evaluate_patch(
        self,
        patch: dict[str, float],
        seeds: list[int],
        episodes_per_seed: int,
        report_path: Path,
    ) -> CandidateResult:
        report = self.validator.run(seeds=seeds, episodes_per_seed=episodes_per_seed, config_patch=patch)
        write_json(report_path, report)

        aggregate = report["aggregate"]
        policy_metrics = aggregate["policy_metrics"]
        full_holdout = float(policy_metrics["full"]["mean_holdout_reward"])
        full_minus_frozen = float(aggregate["full_minus_frozen_holdout"])
        full_minus_memory = float(aggregate["full_minus_memory_only_holdout"])
        full_win_rate_vs_frozen = float(aggregate["full_over_frozen_holdout_win_rate"])
        full_ci95_low = float(aggregate["gains_vs_frozen"]["full"]["holdout_gain_bootstrap_ci95"][0])

        by_task_full = aggregate["policy_holdout_by_task"]["full"]
        fairness_gap = max(by_task_full.values()) - min(by_task_full.values()) if by_task_full else 0.0

        gains = [row["gains_vs_frozen"]["full"]["holdout_reward_gain"] for row in report["per_trial"]]
        gain_std = pstdev(gains) if len(gains) >= 2 else 0.0

        hhi = aggregate["policy_branch_concentration_hhi"]
        branch_concentration_gap = float(hhi["full"] - hhi["frozen"])

        objective = self._objective(
            full_holdout=full_holdout,
            full_minus_frozen=full_minus_frozen,
            full_minus_memory=full_minus_memory,
            full_win_rate_vs_frozen=full_win_rate_vs_frozen,
            full_ci95_low=full_ci95_low,
            fairness_gap=fairness_gap,
            gain_std=gain_std,
            branch_concentration_gap=branch_concentration_gap,
        )

        return CandidateResult(
            patch=patch,
            objective=objective,
            full_holdout=full_holdout,
            full_minus_frozen=full_minus_frozen,
            full_minus_memory_only=full_minus_memory,
            full_win_rate_vs_frozen=full_win_rate_vs_frozen,
            full_ci95_low=full_ci95_low,
            fairness_gap=fairness_gap,
            gain_std=gain_std,
            branch_concentration_gap=branch_concentration_gap,
            report_path=str(report_path),
            ci_gate_passed=(full_ci95_low >= 0.0),
        )

    def _objective(
        self,
        full_holdout: float,
        full_minus_frozen: float,
        full_minus_memory: float,
        full_win_rate_vs_frozen: float,
        full_ci95_low: float,
        fairness_gap: float,
        gain_std: float,
        branch_concentration_gap: float,
    ) -> float:
        score = full_holdout
        score += 0.9 * full_minus_frozen
        score += 0.7 * full_minus_memory
        score += 0.25 * full_win_rate_vs_frozen
        score -= 0.35 * fairness_gap
        score -= 0.25 * gain_std

        # Penalize branch collapse to avoid route-selection bias.
        if branch_concentration_gap > 0.04:
            score -= 0.3 * (branch_concentration_gap - 0.04)

        # Hard penalties for failing core policy goals.
        if full_minus_frozen <= 0.0:
            score -= 0.8
        if full_minus_memory <= 0.0:
            score -= 0.6
        if full_win_rate_vs_frozen < 0.6:
            score -= 0.6 * (0.6 - full_win_rate_vs_frozen)
        if full_ci95_low < 0.0:
            score -= 0.35 * abs(full_ci95_low)

        return score

    def _default_patch_from_config(self) -> dict[str, float]:
        config = read_json(self.config_path)
        patch: dict[str, float] = {}
        for key in self.search_space:
            section, field = key.split(".", 1)
            patch[key] = float(config.get(section, {}).get(field))
        return patch

    def _sample_patch(self, center: dict[str, float], mode: str) -> dict[str, float]:
        patch: dict[str, float] = {}
        for key, (low, high) in self.search_space.items():
            span = high - low
            base = center[key]
            if mode == "exploit":
                sigma = 0.12 * span
                sampled = self._rng.gauss(base, sigma)
            else:
                sampled = self._rng.uniform(low, high)
            patch[key] = round(max(low, min(high, sampled)), 6)

        # Keep memory coefficient conservative by construction.
        patch["router.memory_coef"] = round(min(patch["router.memory_coef"], 0.32), 6)
        patch["memory.bias_scale"] = round(min(max(patch["memory.bias_scale"], 0.2), 0.6), 6)
        return patch

    def _apply_patch_to_default_config(self, patch: dict[str, float]) -> None:
        config = read_json(self.config_path)
        for key, value in patch.items():
            section, field = key.split(".", 1)
            if section not in config:
                config[section] = {}
            config[section][field] = value
        write_json(self.config_path, config)

    @staticmethod
    def _result_to_dict(result: CandidateResult) -> dict:
        return {
            "patch": result.patch,
            "objective": round(result.objective, 6),
            "full_holdout": round(result.full_holdout, 6),
            "full_minus_frozen": round(result.full_minus_frozen, 6),
            "full_minus_memory_only": round(result.full_minus_memory_only, 6),
            "full_win_rate_vs_frozen": round(result.full_win_rate_vs_frozen, 6),
            "full_ci95_low": round(result.full_ci95_low, 6),
            "fairness_gap": round(result.fairness_gap, 6),
            "gain_std": round(result.gain_std, 6),
            "branch_concentration_gap": round(result.branch_concentration_gap, 6),
            "report_path": result.report_path,
            "ci_gate_passed": result.ci_gate_passed,
        }
