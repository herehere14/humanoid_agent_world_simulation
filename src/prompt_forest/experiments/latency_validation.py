from __future__ import annotations

import json
import shutil
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from time import perf_counter
from typing import Any

from ..backend.base import LLMBackend
from ..backend.mock import MockLLMBackend
from ..backend.openai_chat import OpenAIChatBackend
from ..config import apply_latency_profile, load_config
from ..core.engine import PromptForestEngine
from ..utils.io import read_json, write_json


@dataclass
class RunRecord:
    phase: str
    round_idx: int
    policy: str
    task_id: str
    task_type: str
    latency_ms: float
    reward: float
    selected_branch: str
    path: list[str]
    total_ms: float
    route_ms: float
    execute_ms: float
    probe_ms: float
    compose_ms: float
    evaluate_ms: float
    optimize_ms: float
    primary_backend_calls: int
    evaluator_runtime_calls: int
    optimizer_runtime_calls: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "round_idx": self.round_idx,
            "policy": self.policy,
            "task_id": self.task_id,
            "task_type": self.task_type,
            "latency_ms": round(self.latency_ms, 3),
            "reward": round(self.reward, 4),
            "selected_branch": self.selected_branch,
            "path": list(self.path),
            "timings": {
                "total_ms": round(self.total_ms, 3),
                "route_ms": round(self.route_ms, 3),
                "execute_ms": round(self.execute_ms, 3),
                "probe_ms": round(self.probe_ms, 3),
                "compose_ms": round(self.compose_ms, 3),
                "evaluate_ms": round(self.evaluate_ms, 3),
                "optimize_ms": round(self.optimize_ms, 3),
            },
            "calls": {
                "primary_backend_calls": self.primary_backend_calls,
                "evaluator_runtime_calls": self.evaluator_runtime_calls,
                "optimizer_runtime_calls": self.optimizer_runtime_calls,
            },
        }


class LatencyValidator:
    """Compare full vs fast profiles with order-balanced, fresh-engine runs."""

    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir)
        self.config_path = self.root_dir / "configs" / "default.json"

    def run(
        self,
        *,
        dataset_path: str | Path,
        rounds: int = 3,
        model: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        base_url: str = "https://api.openai.com/v1",
        temperature: float = 0.2,
        max_output_tokens: int = 700,
        api_mode: str = "chat_completions",
        reasoning_effort: str | None = None,
        output_subdir: str = "latency_validation",
        report_prefix: str = "latency_validation_report",
    ) -> dict[str, Any]:
        tasks = self._load_tasks(Path(dataset_path))
        if not tasks:
            raise RuntimeError("Latency validation dataset is empty.")

        out_dir = self.root_dir / "artifacts" / output_subdir
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        records: list[RunRecord] = []
        for round_idx in range(max(1, rounds)):
            cold_order = ["full", "fast"] if round_idx % 2 == 0 else ["fast", "full"]
            for task_idx, task in enumerate(tasks):
                for policy in cold_order if task_idx % 2 == 0 else list(reversed(cold_order)):
                    records.append(
                        self._run_cold_task(
                            task=task,
                            policy=policy,
                            round_idx=round_idx,
                            model=model,
                            api_key_env=api_key_env,
                            base_url=base_url,
                            temperature=temperature,
                            max_output_tokens=max_output_tokens,
                            api_mode=api_mode,
                            reasoning_effort=reasoning_effort,
                            artifacts_dir=out_dir / "cold" / f"round_{round_idx}" / policy / task["task_id"],
                        )
                    )

            session_order = ["full", "fast"] if round_idx % 2 == 0 else ["fast", "full"]
            for policy in session_order:
                records.extend(
                    self._run_session_round(
                        tasks=tasks,
                        policy=policy,
                        round_idx=round_idx,
                        model=model,
                        api_key_env=api_key_env,
                        base_url=base_url,
                        temperature=temperature,
                        max_output_tokens=max_output_tokens,
                        api_mode=api_mode,
                        reasoning_effort=reasoning_effort,
                        artifacts_dir=out_dir / "session" / f"round_{round_idx}" / policy,
                    )
                )

        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "dataset_path": str(Path(dataset_path)),
            "mode": "live_openai" if model else "mock_backend",
            "models": {
                "generation_model": model or "mock",
                "api_key_env": api_key_env if model else "",
                "base_url": base_url if model else "",
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
                "api_mode": api_mode,
                "reasoning_effort": reasoning_effort or "",
            },
            "settings": {
                "rounds": max(1, rounds),
                "task_count": len(tasks),
                "target_latency_seconds": 7,
                "profiles": {
                    "full": "default adaptive execution path",
                    "fast": "single-path, no composer, no refinement, no API-backed evaluator/optimizer",
                },
                "anti_bias_controls": [
                    "cold-turn runs use fresh engines and fresh backends for every task",
                    "policy execution order alternates by round and task index",
                    "both profiles use the same dataset, same backend family, and same deterministic seeds when available",
                    "session runs start from fresh engines each round so cross-round memory does not leak",
                ],
            },
            "aggregate": self._aggregate_records(records),
            "records": [record.to_dict() for record in records],
            "limitations": [
                "Fast-profile gains on real API latency can only be fully verified in live mode with a valid API key.",
                "Mock mode is unbiased for call-count and control-flow comparisons, but not a substitute for live network latency.",
            ],
        }

        json_path = out_dir / f"{report_prefix}.json"
        md_path = out_dir / f"{report_prefix}.md"
        write_json(json_path, report)
        md_path.write_text(self._render_markdown_report(report), encoding="utf-8")
        report["report_paths"] = {"json": str(json_path), "markdown": str(md_path)}
        write_json(json_path, report)
        return report

    def _run_cold_task(
        self,
        *,
        task: dict[str, Any],
        policy: str,
        round_idx: int,
        model: str | None,
        api_key_env: str,
        base_url: str,
        temperature: float,
        max_output_tokens: int,
        api_mode: str,
        reasoning_effort: str | None,
        artifacts_dir: Path,
    ) -> RunRecord:
        engine = self._build_engine(
            policy=policy,
            model=model,
            api_key_env=api_key_env,
            base_url=base_url,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            api_mode=api_mode,
            reasoning_effort=reasoning_effort,
            artifacts_dir=artifacts_dir,
            seed=1000 + round_idx,
        )
        return self._run_record(engine, task, policy=policy, round_idx=round_idx, phase="cold")

    def _run_session_round(
        self,
        *,
        tasks: list[dict[str, Any]],
        policy: str,
        round_idx: int,
        model: str | None,
        api_key_env: str,
        base_url: str,
        temperature: float,
        max_output_tokens: int,
        api_mode: str,
        reasoning_effort: str | None,
        artifacts_dir: Path,
    ) -> list[RunRecord]:
        engine = self._build_engine(
            policy=policy,
            model=model,
            api_key_env=api_key_env,
            base_url=base_url,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            api_mode=api_mode,
            reasoning_effort=reasoning_effort,
            artifacts_dir=artifacts_dir,
            seed=5000 + round_idx,
        )
        return [self._run_record(engine, task, policy=policy, round_idx=round_idx, phase="session") for task in tasks]

    def _build_engine(
        self,
        *,
        policy: str,
        model: str | None,
        api_key_env: str,
        base_url: str,
        temperature: float,
        max_output_tokens: int,
        api_mode: str,
        reasoning_effort: str | None,
        artifacts_dir: Path,
        seed: int,
    ) -> PromptForestEngine:
        cfg = load_config(self.config_path)
        if policy == "fast":
            cfg = apply_latency_profile(cfg, "fast")
        cfg.artifacts_dir = str(artifacts_dir)
        backend = self._make_backend(
            model=model,
            api_key_env=api_key_env,
            base_url=base_url,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            api_mode=api_mode,
            reasoning_effort=reasoning_effort,
            seed=seed,
        )
        return PromptForestEngine(config=cfg, backend=backend)

    @staticmethod
    def _make_backend(
        *,
        model: str | None,
        api_key_env: str,
        base_url: str,
        temperature: float,
        max_output_tokens: int,
        api_mode: str,
        reasoning_effort: str | None,
        seed: int,
    ) -> LLMBackend:
        if model:
            return OpenAIChatBackend(
                model=model,
                api_key_env=api_key_env,
                base_url=base_url,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                api_mode=api_mode,
                reasoning_effort=reasoning_effort,
                seed=seed,
            )
        return MockLLMBackend(seed=seed)

    @staticmethod
    def _load_tasks(path: Path) -> list[dict[str, Any]]:
        payload = read_json(path)
        if "tasks" in payload:
            return [
                {
                    "task_id": f"task_{idx}",
                    "text": item["text"],
                    "task_type": item.get("task_type", "auto"),
                    "metadata": deepcopy(item.get("metadata", {})),
                }
                for idx, item in enumerate(payload.get("tasks", []), start=1)
            ]

        tasks: list[dict[str, Any]] = []
        for section in ("train", "holdout"):
            for idx, item in enumerate(payload.get(section, []), start=1):
                tasks.append(
                    {
                        "task_id": str(item.get("task_id", f"{section}_{idx}")),
                        "text": item["text"],
                        "task_type": item.get("task_type", "auto"),
                        "metadata": deepcopy(item.get("metadata", {})),
                    }
                )
        return tasks

    @staticmethod
    def _run_record(
        engine: PromptForestEngine,
        task: dict[str, Any],
        *,
        policy: str,
        round_idx: int,
        phase: str,
    ) -> RunRecord:
        metadata = deepcopy(task.get("metadata", {}))
        started = perf_counter()
        result = engine.run_task(text=task["text"], task_type=task.get("task_type", "auto"), metadata=metadata)
        latency_ms = (perf_counter() - started) * 1000.0
        timings = dict(result.get("timings", {}) or {})
        signal = dict(result.get("evaluation_signal", {}) or {})
        return RunRecord(
            phase=phase,
            round_idx=round_idx,
            policy=policy,
            task_id=str(task.get("task_id", "")),
            task_type=str(task.get("task_type", "auto")),
            latency_ms=latency_ms,
            reward=float(signal.get("reward_score", 0.0) or 0.0),
            selected_branch=str(signal.get("selected_branch", "")),
            path=list(result.get("selected_path", []) or []),
            total_ms=float(timings.get("total_ms", 0.0) or 0.0),
            route_ms=float(timings.get("route_ms", 0.0) or 0.0),
            execute_ms=float(timings.get("execute_ms", 0.0) or 0.0),
            probe_ms=float(timings.get("probe_ms", 0.0) or 0.0),
            compose_ms=float(timings.get("compose_ms", 0.0) or 0.0),
            evaluate_ms=float(timings.get("evaluate_ms", 0.0) or 0.0),
            optimize_ms=float(timings.get("optimize_ms", 0.0) or 0.0),
            primary_backend_calls=int(timings.get("primary_backend_calls", 0) or 0),
            evaluator_runtime_calls=int(timings.get("evaluator_runtime_calls", 0) or 0),
            optimizer_runtime_calls=int(timings.get("optimizer_runtime_calls", 0) or 0),
        )

    def _aggregate_records(self, records: list[RunRecord]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for phase in ("cold", "session"):
            phase_records = [record for record in records if record.phase == phase]
            out[phase] = {
                policy: self._summarize_policy([record for record in phase_records if record.policy == policy])
                for policy in ("full", "fast")
            }
            out[phase]["delta_fast_minus_full"] = self._delta_summary(
                out[phase]["full"],
                out[phase]["fast"],
            )
        return out

    @staticmethod
    def _summarize_policy(records: list[RunRecord]) -> dict[str, Any]:
        if not records:
            return {}
        latencies = [record.latency_ms for record in records]
        rewards = [record.reward for record in records]
        totals = [record.total_ms for record in records]
        route = [record.route_ms for record in records]
        execute = [record.execute_ms for record in records]
        probe = [record.probe_ms for record in records]
        compose = [record.compose_ms for record in records]
        evaluate = [record.evaluate_ms for record in records]
        optimize = [record.optimize_ms for record in records]
        primary_calls = [record.primary_backend_calls for record in records]
        evaluator_calls = [record.evaluator_runtime_calls for record in records]
        optimizer_calls = [record.optimizer_runtime_calls for record in records]
        under_target = [value for value in latencies if value <= 7000.0]
        sorted_latencies = sorted(latencies)
        p95_index = max(0, min(len(sorted_latencies) - 1, int(round((len(sorted_latencies) - 1) * 0.95))))
        return {
            "runs": len(records),
            "mean_latency_ms": round(mean(latencies), 3),
            "median_latency_ms": round(median(latencies), 3),
            "p95_latency_ms": round(sorted_latencies[p95_index], 3),
            "turns_under_7s_pct": round((len(under_target) / len(records)) * 100.0, 2),
            "mean_reward": round(mean(rewards), 4),
            "mean_total_ms_reported": round(mean(totals), 3),
            "mean_phase_ms": {
                "route_ms": round(mean(route), 3),
                "execute_ms": round(mean(execute), 3),
                "probe_ms": round(mean(probe), 3),
                "compose_ms": round(mean(compose), 3),
                "evaluate_ms": round(mean(evaluate), 3),
                "optimize_ms": round(mean(optimize), 3),
            },
            "mean_calls": {
                "primary_backend_calls": round(mean(primary_calls), 3),
                "evaluator_runtime_calls": round(mean(evaluator_calls), 3),
                "optimizer_runtime_calls": round(mean(optimizer_calls), 3),
            },
        }

    @staticmethod
    def _delta_summary(full: dict[str, Any], fast: dict[str, Any]) -> dict[str, Any]:
        if not full or not fast:
            return {}
        return {
            "mean_latency_ms": round(float(fast["mean_latency_ms"]) - float(full["mean_latency_ms"]), 3),
            "mean_reward": round(float(fast["mean_reward"]) - float(full["mean_reward"]), 4),
            "turns_under_7s_pct": round(float(fast["turns_under_7s_pct"]) - float(full["turns_under_7s_pct"]), 2),
            "primary_backend_calls": round(
                float(fast["mean_calls"]["primary_backend_calls"]) - float(full["mean_calls"]["primary_backend_calls"]),
                3,
            ),
            "evaluator_runtime_calls": round(
                float(fast["mean_calls"]["evaluator_runtime_calls"]) - float(full["mean_calls"]["evaluator_runtime_calls"]),
                3,
            ),
            "optimizer_runtime_calls": round(
                float(fast["mean_calls"]["optimizer_runtime_calls"]) - float(full["mean_calls"]["optimizer_runtime_calls"]),
                3,
            ),
        }

    @staticmethod
    def _render_markdown_report(report: dict[str, Any]) -> str:
        cold = report["aggregate"]["cold"]
        session = report["aggregate"]["session"]
        lines = [
            "# Latency Validation Report",
            "",
            f"- Generated at: `{report['generated_at']}`",
            f"- Dataset: `{report['dataset_path']}`",
            f"- Backend mode: `{report['mode']}`",
            f"- Generation model: `{report['models']['generation_model']}`",
            f"- Task count: `{report['settings']['task_count']}`",
            f"- Rounds: `{report['settings']['rounds']}`",
            "",
            "## Cold Turn Summary",
            "",
            f"- Full mean latency: `{cold['full']['mean_latency_ms']}` ms",
            f"- Fast mean latency: `{cold['fast']['mean_latency_ms']}` ms",
            f"- Fast minus full latency delta: `{cold['delta_fast_minus_full']['mean_latency_ms']}` ms",
            f"- Full mean reward: `{cold['full']['mean_reward']}`",
            f"- Fast mean reward: `{cold['fast']['mean_reward']}`",
            f"- Fast minus full reward delta: `{cold['delta_fast_minus_full']['mean_reward']}`",
            f"- Full mean primary calls: `{cold['full']['mean_calls']['primary_backend_calls']}`",
            f"- Fast mean primary calls: `{cold['fast']['mean_calls']['primary_backend_calls']}`",
            "",
            "## Session Summary",
            "",
            f"- Full mean latency: `{session['full']['mean_latency_ms']}` ms",
            f"- Fast mean latency: `{session['fast']['mean_latency_ms']}` ms",
            f"- Fast minus full latency delta: `{session['delta_fast_minus_full']['mean_latency_ms']}` ms",
            f"- Full mean reward: `{session['full']['mean_reward']}`",
            f"- Fast mean reward: `{session['fast']['mean_reward']}`",
            f"- Fast minus full reward delta: `{session['delta_fast_minus_full']['mean_reward']}`",
            "",
            "## Anti-Bias Controls",
            "",
        ]
        lines.extend(f"- {item}" for item in report["settings"]["anti_bias_controls"])
        lines.extend(["", "## Limitations", ""])
        lines.extend(f"- {item}" for item in report["limitations"])
        return "\n".join(lines)
