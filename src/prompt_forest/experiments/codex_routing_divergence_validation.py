from __future__ import annotations

import shutil
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from ..backend.openai_chat import OpenAIChatBackend
from ..config import load_config
from ..core.engine import PromptForestEngine
from ..contracts import infer_output_contract
from ..types import RoutingDecision, TaskInput
from ..utils.io import read_json, write_json
from .live_ablation_validation import LiveAblationValidator
from .live_model_validation import PairwiseJudgement


class CodexRoutingDivergenceValidator(LiveAblationValidator):
    """Focused adaptive-vs-frozen benchmark for sibling-routing divergence on Codex."""

    def run(
        self,
        *,
        dataset_path: str | Path,
        model: str = "gpt-5.3-codex",
        judge_model: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        base_url: str = "https://api.openai.com/v1",
        temperature: float = 0.2,
        max_output_tokens: int = 1200,
        api_mode: str = "responses",
        reasoning_effort: str | None = "medium",
        judge_api_mode: str | None = "chat_completions",
        judge_reasoning_effort: str | None = None,
        judge_temperature: float = 0.0,
        judge_max_output_tokens: int = 280,
        train_rounds: int = 2,
        use_agent_runtimes: bool = False,
        train_candidates_per_task: int = 2,
        output_subdir: str = "codex_routing_divergence_benchmark",
        report_prefix: str = "codex_routing_divergence_report",
    ) -> dict[str, Any]:
        judge_model = judge_model or "gpt-4.1-mini"
        judge_api_mode, judge_reasoning_effort = self._resolve_judge_backend_options(
            model=model,
            judge_model=judge_model,
            api_mode=api_mode,
            reasoning_effort=reasoning_effort,
            judge_api_mode=judge_api_mode,
            judge_reasoning_effort=judge_reasoning_effort,
        )

        raw_dataset = read_json(Path(dataset_path))
        train_specs = list(raw_dataset.get("train", []))
        holdout_specs = list(raw_dataset.get("holdout", []))
        if not train_specs or not holdout_specs:
            raise RuntimeError("Codex divergence dataset must include non-empty train and holdout sections.")

        out_dir = self.root_dir / "artifacts" / output_subdir
        if out_dir.exists():
            shutil.rmtree(out_dir)

        policy_defs = [self._policy_definition("full_adaptive"), self._policy_definition("frozen")]
        engines: dict[str, PromptForestEngine] = {}
        backends: dict[str, OpenAIChatBackend] = {}
        for policy in policy_defs:
            cfg = load_config(self.config_path)
            self._configure_live_eval(cfg, use_agent_runtimes=use_agent_runtimes)
            cfg.composer.enabled = False
            cfg.artifacts_dir = str(out_dir / policy.name)

            backend = OpenAIChatBackend(
                model=model,
                api_key_env=api_key_env,
                base_url=base_url,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                api_mode=api_mode,
                reasoning_effort=reasoning_effort,
                seed=42,
            )
            engines[policy.name] = PromptForestEngine(config=cfg, backend=backend)
            backends[policy.name] = backend

        judge_backend = OpenAIChatBackend(
            model=judge_model,
            api_key_env=api_key_env,
            base_url=base_url,
            temperature=judge_temperature,
            max_output_tokens=judge_max_output_tokens,
            api_mode=judge_api_mode,
            reasoning_effort=judge_reasoning_effort,
            seed=7,
            system_prompt=(
                "You are an impartial evaluator. Return valid JSON only. "
                "Prefer correctness and explicit instruction-following over style."
            ),
        )

        train_tasks = self._expand_train_tasks(train_specs, rounds=max(1, train_rounds))
        holdout_tasks = [self._task_from_spec(deepcopy(spec), split="holdout") for spec in holdout_specs]
        policy_runs: dict[str, dict[str, list[dict[str, Any]]]] = {
            policy.name: {"train": [], "holdout": []} for policy in policy_defs
        }

        for task in train_tasks:
            for policy in policy_defs:
                policy_runs[policy.name]["train"].append(
                    self._run_forced_engine_task(
                        engines[policy.name],
                        task,
                        adapt=policy.adapt_train,
                        update_memory=policy.update_memory_train,
                        discovery_mode=True,
                        max_children=train_candidates_per_task,
                    )
                )

        holdout_judgements: list[PairwiseJudgement] = []
        for task in holdout_tasks:
            adaptive_result = self._run_forced_engine_task(
                engines["full_adaptive"],
                task,
                adapt=False,
                update_memory=False,
                discovery_mode=False,
                max_children=1,
            )
            frozen_result = self._run_forced_engine_task(
                engines["frozen"],
                task,
                adapt=False,
                update_memory=False,
                discovery_mode=False,
                max_children=1,
            )
            policy_runs["full_adaptive"]["holdout"].append(adaptive_result)
            policy_runs["frozen"]["holdout"].append(frozen_result)
            holdout_judgements.append(
                self._judge_pair(
                    judge_backend,
                    task,
                    "full_adaptive",
                    adaptive_result,
                    "frozen",
                    frozen_result,
                )
            )

        aggregate = {
            split: {
                policy.name: self._summarize_policy_runs(policy_runs[policy.name][split])
                for policy in policy_defs
            }
            for split in ("train", "holdout")
        }
        pairwise_summary = self._pairwise_summary(holdout_judgements).get("full_adaptive__vs__frozen", {})
        divergence = self._holdout_divergence_summary(
            policy_runs["full_adaptive"]["holdout"],
            policy_runs["frozen"]["holdout"],
        )
        aspect_summary = self._aspect_summary(
            policy_runs["full_adaptive"]["holdout"],
            policy_runs["frozen"]["holdout"],
        )
        weight_deltas = self._weight_delta_summary(
            engines["full_adaptive"].branch_snapshot(),
            engines["frozen"].branch_snapshot(),
        )

        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "dataset_path": str(Path(dataset_path)),
            "models": {
                "generation_model": model,
                "judge_model": judge_model,
                "api_key_env": api_key_env,
                "base_url": base_url,
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
                "api_mode": api_mode,
                "reasoning_effort": reasoning_effort or "",
                "judge_api_mode": judge_api_mode,
                "judge_reasoning_effort": judge_reasoning_effort or "",
                "judge_temperature": judge_temperature,
                "judge_max_output_tokens": judge_max_output_tokens,
            },
            "settings": {
                "train_rounds": max(1, train_rounds),
                "use_agent_runtimes": use_agent_runtimes,
                "train_candidates_per_task": train_candidates_per_task,
                "holdout_candidates_per_task": 1,
                "composer_enabled": next(iter(engines.values())).config.composer.enabled,
                "reward_mode": next(iter(engines.values())).config.evaluator.reward_mode,
            },
            "dataset_summary": self._dataset_summary(train_specs, holdout_specs, train_rounds=max(1, train_rounds)),
            "aggregate": aggregate,
            "comparisons": {
                "adaptive_vs_frozen": {
                    "objective_reward_gain": round(
                        aggregate["holdout"]["full_adaptive"]["mean_objective_reward"]
                        - aggregate["holdout"]["frozen"]["mean_objective_reward"],
                        4,
                    ),
                    "pairwise": pairwise_summary,
                    "branch_divergence": divergence,
                    "aspect_summary": aspect_summary,
                    "weight_delta_summary": weight_deltas,
                },
                "pairwise_summary": {"full_adaptive__vs__frozen": pairwise_summary},
            },
            "pairwise_holdout": [self._pairwise_to_dict(item) for item in holdout_judgements],
            "per_policy": {
                policy.name: {
                    "train": policy_runs[policy.name]["train"],
                    "holdout": policy_runs[policy.name]["holdout"],
                    "branch_snapshot": engines[policy.name].branch_snapshot(),
                    "memory": engines[policy.name].memory.stats(),
                    "routing_histogram": engines[policy.name].routing_histogram(),
                    "backend_usage_total": backends[policy.name].usage_summary(),
                    "evaluator_runtime_usage_total": engines[policy.name].evaluator_agent.runtime.usage_summary(),
                    "optimizer_runtime_usage_total": engines[policy.name].optimizer_agent._advisor.runtime.usage_summary(),  # noqa: SLF001
                }
                for policy in policy_defs
            },
            "judge_backend_usage_total": judge_backend.usage_summary(),
            "limitations": [
                "This benchmark fixes the parent branch per task so it measures sibling adaptation rather than full-tree macro routing.",
                "Training evaluates multiple sibling leaves per task to let adaptive update from the best-performing leaf under that parent.",
                "Holdout executes only the policy-chosen sibling leaf, using current learned weights and sibling memory under the forced parent.",
                "Pairwise judging uses an LLM judge and should be read together with the objective verifier score.",
            ],
        }

        json_path = out_dir / f"{report_prefix}.json"
        md_path = out_dir / f"{report_prefix}.md"
        write_json(json_path, report)
        md_path.write_text(self._render_markdown_report(report), encoding="utf-8")
        report["report_paths"] = {"json": str(json_path), "markdown": str(md_path)}
        write_json(json_path, report)
        return report

    def _run_forced_engine_task(
        self,
        engine: PromptForestEngine,
        task: TaskInput,
        *,
        adapt: bool,
        update_memory: bool,
        discovery_mode: bool,
        max_children: int,
    ) -> dict[str, Any]:
        parent_id, child_ids = self._forced_parent_and_children(task)
        backend_start = len(engine.backend.call_log())  # type: ignore[attr-defined]
        eval_start = len(engine.evaluator_agent.runtime.call_log())
        opt_start = len(engine.optimizer_agent._advisor.runtime.call_log())  # noqa: SLF001

        if discovery_mode:
            selected_children, child_scores, decision_meta = self._rank_forced_children(
                engine=engine,
                task=task,
                parent_id=parent_id,
                child_ids=child_ids,
                max_children=max_children,
            )
        else:
            selected_child, child_scores, decision_meta = self._choose_forced_child(
                engine=engine,
                task=task,
                parent_id=parent_id,
                child_ids=child_ids,
            )
            selected_children = [selected_child]

        route = self._forced_route(
            task=task,
            parent_id=parent_id,
            child_ids=selected_children,
            child_scores=child_scores,
            decision_meta=decision_meta,
        )
        original_route = engine.router.route
        engine.router.route = lambda *_args, **_kwargs: route  # type: ignore[assignment]
        try:
            result = engine.run_task_controlled(
                text=task.text,
                task_type=task.task_type,
                metadata=dict(task.metadata),
                adapt=adapt,
                update_memory=update_memory,
            )
        finally:
            engine.router.route = original_route  # type: ignore[assignment]

        output = str(result["evaluation_signal"]["selected_output"])
        metrics = self._objective_metrics(task, output)
        return {
            "task_id": task.task_id,
            "split": str(task.metadata.get("split", "")),
            "task_type": task.task_type,
            "aspect": str(task.metadata.get("aspect", task.task_type)),
            "text": task.text,
            "selected_output": output,
            "selected_branch": result["evaluation_signal"]["selected_branch"],
            "activated_branches": result["routing"].get("activated_branches", []),
            "activated_paths": result["routing"].get("activated_paths", []),
            "forced_parent": parent_id,
            "forced_children": list(child_ids),
            "internal_reward_score": round(float(result["evaluation_signal"]["reward_score"]), 4),
            "internal_confidence": round(float(result["evaluation_signal"]["confidence"]), 4),
            "failure_reason": result["evaluation_signal"].get("failure_reason", ""),
            "optimizer": self._compact_optimizer(result.get("optimization", {})),
            "objective_metrics": metrics,
            "backend_usage": self._summarize_calls(engine.backend.call_log()[backend_start:]),  # type: ignore[attr-defined]
            "evaluator_runtime_usage": self._summarize_calls(engine.evaluator_agent.runtime.call_log()[eval_start:]),
            "optimizer_runtime_usage": self._summarize_calls(
                engine.optimizer_agent._advisor.runtime.call_log()[opt_start:]  # noqa: SLF001
            ),
        }

    @staticmethod
    def _forced_parent_and_children(task: TaskInput) -> tuple[str, list[str]]:
        parent_id = str(task.metadata.get("forced_parent", "")).strip()
        child_ids = [str(item).strip() for item in task.metadata.get("forced_children", []) if str(item).strip()]
        if parent_id and child_ids:
            return parent_id, child_ids

        aspect = str(task.metadata.get("aspect", "")).strip().lower()
        if aspect.startswith("planning_"):
            return "planner", ["planner_timeline_optimizer", "planner_risk_allocator"]
        if aspect.startswith("general_") or aspect.startswith("code_"):
            return "verification", ["verification_constraint_checker", "verification_consistency_auditor"]
        raise RuntimeError(f"Unable to infer forced_parent/forced_children for task: {task.task_id}")

    def _rank_forced_children(
        self,
        *,
        engine: PromptForestEngine,
        task: TaskInput,
        parent_id: str,
        child_ids: list[str],
        max_children: int,
    ) -> tuple[list[str], dict[str, float], dict[str, object]]:
        scores, decision_meta = self._forced_child_scores(
            engine=engine,
            task=task,
            parent_id=parent_id,
            child_ids=child_ids,
        )
        selected = engine.router._select_top_children(  # noqa: SLF001
            scores=scores,
            visit_counts=engine.memory.branch_visit_counts(task.task_type, user_id=str(task.metadata.get("user_id", "global")).strip() or "global"),
            exploration_rate=0.0,
            max_children=max_children,
            forced_first=str(decision_meta.get("override_child", "")),
        )
        return selected, scores, decision_meta

    def _choose_forced_child(
        self,
        *,
        engine: PromptForestEngine,
        task: TaskInput,
        parent_id: str,
        child_ids: list[str],
    ) -> tuple[str, dict[str, float], dict[str, object]]:
        scores, decision_meta = self._forced_child_scores(
            engine=engine,
            task=task,
            parent_id=parent_id,
            child_ids=child_ids,
        )
        override_child = str(decision_meta.get("override_child", ""))
        if override_child and override_child in scores:
            return override_child, scores, decision_meta
        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        if not ordered:
            raise RuntimeError(f"No forced-child scores available for task: {task.task_id}")
        return ordered[0][0], scores, decision_meta

    def _forced_child_scores(
        self,
        *,
        engine: PromptForestEngine,
        task: TaskInput,
        parent_id: str,
        child_ids: list[str],
    ) -> tuple[dict[str, float], dict[str, object]]:
        user_id = str(task.metadata.get("user_id", "global")).strip() or "global"
        contract_hint = infer_output_contract(task.text, task.metadata)
        preference_signal = engine.router._preference_signal_for_children(  # noqa: SLF001
            task=task,
            task_type=task.task_type,
            parent_id=parent_id,
            child_ids=child_ids,
            forest=engine.forest,
            memory=engine.memory,
            user_id=user_id,
        )
        scores = engine.router._score_children(  # noqa: SLF001
            task_type=task.task_type,
            parent_id=parent_id,
            child_ids=child_ids,
            forest=engine.forest,
            memory_scores=preference_signal.scores,
            contract_hint=contract_hint,
        )
        selected_by_score = max(scores.items(), key=lambda item: item[1])[0] if scores else ""
        decision_meta = engine.router._sibling_decision_meta(  # noqa: SLF001
            parent_id=parent_id,
            child_ids=child_ids,
            scores=scores,
            signal=preference_signal,
            selected_by_score=selected_by_score,
        )
        return scores, decision_meta

    @staticmethod
    def _forced_route(
        *,
        task: TaskInput,
        parent_id: str,
        child_ids: list[str],
        child_scores: dict[str, float],
        decision_meta: dict[str, object] | None = None,
    ) -> RoutingDecision:
        activated_paths = [[parent_id, child_id] for child_id in child_ids]
        activated_branches = [parent_id]
        activated_branches.extend(child_id for child_id in child_ids if child_id not in activated_branches)
        flattened_scores = dict(child_scores)
        flattened_scores[parent_id] = max(child_scores.values(), default=1.0)
        return RoutingDecision(
            task_type=task.task_type,
            activated_branches=activated_branches,
            branch_scores=flattened_scores,
            activated_paths=activated_paths,
            sibling_decisions={parent_id: dict(decision_meta or {})},
        )

    @staticmethod
    def _holdout_divergence_summary(
        adaptive_runs: list[dict[str, Any]],
        frozen_runs: list[dict[str, Any]],
    ) -> dict[str, Any]:
        frozen_by_task = {str(item["task_id"]): item for item in frozen_runs}
        aligned: list[dict[str, Any]] = []
        for adaptive in adaptive_runs:
            task_id = str(adaptive["task_id"])
            frozen = frozen_by_task.get(task_id)
            if not frozen:
                continue
            adaptive_reward = float(adaptive["objective_metrics"]["hybrid_verifier_reward"])
            frozen_reward = float(frozen["objective_metrics"]["hybrid_verifier_reward"])
            branch_changed = str(adaptive["selected_branch"]) != str(frozen["selected_branch"])
            path_changed = adaptive.get("activated_paths", []) != frozen.get("activated_paths", [])
            aligned.append(
                {
                    "task_id": task_id,
                    "task_type": str(adaptive["task_type"]),
                    "aspect": str(adaptive["aspect"]),
                    "adaptive_branch": str(adaptive["selected_branch"]),
                    "frozen_branch": str(frozen["selected_branch"]),
                    "adaptive_reward": round(adaptive_reward, 4),
                    "frozen_reward": round(frozen_reward, 4),
                    "reward_delta": round(adaptive_reward - frozen_reward, 4),
                    "branch_changed": branch_changed,
                    "path_changed": path_changed,
                }
            )

        divergent = [item for item in aligned if item["branch_changed"]]
        path_divergent = [item for item in aligned if item["path_changed"]]
        divergent_deltas = [item["reward_delta"] for item in divergent]
        non_divergent_deltas = [item["reward_delta"] for item in aligned if not item["branch_changed"]]
        return {
            "holdout_tasks": len(aligned),
            "selected_branch_diff_count": len(divergent),
            "selected_branch_diff_rate": round(len(divergent) / max(1, len(aligned)), 4),
            "activated_path_diff_count": len(path_divergent),
            "activated_path_diff_rate": round(len(path_divergent) / max(1, len(aligned)), 4),
            "mean_reward_delta_on_divergent": round(mean(divergent_deltas), 4) if divergent_deltas else 0.0,
            "mean_reward_delta_on_non_divergent": round(mean(non_divergent_deltas), 4) if non_divergent_deltas else 0.0,
            "divergent_tasks": divergent,
        }

    @staticmethod
    def _aspect_summary(
        adaptive_runs: list[dict[str, Any]],
        frozen_runs: list[dict[str, Any]],
    ) -> dict[str, Any]:
        frozen_by_task = {str(item["task_id"]): item for item in frozen_runs}
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for adaptive in adaptive_runs:
            task_id = str(adaptive["task_id"])
            frozen = frozen_by_task.get(task_id)
            if not frozen:
                continue
            aspect = str(adaptive["aspect"])
            buckets[aspect].append(
                {
                    "adaptive_branch": str(adaptive["selected_branch"]),
                    "frozen_branch": str(frozen["selected_branch"]),
                    "adaptive_reward": float(adaptive["objective_metrics"]["hybrid_verifier_reward"]),
                    "frozen_reward": float(frozen["objective_metrics"]["hybrid_verifier_reward"]),
                }
            )

        out: dict[str, Any] = {}
        for aspect, rows in sorted(buckets.items()):
            adaptive_rewards = [row["adaptive_reward"] for row in rows]
            frozen_rewards = [row["frozen_reward"] for row in rows]
            out[aspect] = {
                "n": len(rows),
                "adaptive_mean_reward": round(mean(adaptive_rewards), 4),
                "frozen_mean_reward": round(mean(frozen_rewards), 4),
                "adaptive_minus_frozen": round(mean(adaptive_rewards) - mean(frozen_rewards), 4),
                "branch_diff_count": sum(1 for row in rows if row["adaptive_branch"] != row["frozen_branch"]),
                "adaptive_branch_counts": dict(Counter(row["adaptive_branch"] for row in rows)),
                "frozen_branch_counts": dict(Counter(row["frozen_branch"] for row in rows)),
            }
        return out

    @staticmethod
    def _weight_delta_summary(
        adaptive_snapshot: dict[str, dict[str, Any]],
        frozen_snapshot: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for branch_name, adaptive in adaptive_snapshot.items():
            frozen = frozen_snapshot.get(branch_name, {})
            adaptive_weight = adaptive.get("weight")
            frozen_weight = frozen.get("weight")
            if adaptive_weight is None or frozen_weight is None:
                continue
            delta = round(float(adaptive_weight) - float(frozen_weight), 4)
            if abs(delta) < 1e-6:
                continue
            rows.append(
                {
                    "branch_name": branch_name,
                    "adaptive_weight": round(float(adaptive_weight), 4),
                    "frozen_weight": round(float(frozen_weight), 4),
                    "delta": delta,
                }
            )
        rows.sort(key=lambda item: abs(item["delta"]), reverse=True)
        return rows[:12]

    def _render_markdown_report(self, report: dict[str, Any]) -> str:
        aggregate = report["aggregate"]
        comparison = report["comparisons"]["adaptive_vs_frozen"]
        divergence = comparison["branch_divergence"]
        pairwise = comparison["pairwise"]

        lines = [
            "# Codex Routing Divergence Benchmark",
            "",
            f"- Generated at: `{report['generated_at']}`",
            f"- Generation model: `{report['models']['generation_model']}`",
            f"- Judge model: `{report['models']['judge_model']}`",
            f"- Dataset: `{report['dataset_path']}`",
            f"- Train sibling candidates per task: `{report['settings']['train_candidates_per_task']}`",
            f"- Holdout sibling candidates per task: `{report['settings']['holdout_candidates_per_task']}`",
            "",
            "## Aggregate Metrics",
            "",
            "| Split | Policy | Mean objective reward | Mean path length | Branch HHI |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
        for split in ("train", "holdout"):
            for policy in ("full_adaptive", "frozen"):
                item = aggregate[split][policy]
                lines.append(
                    f"| {split} | {policy} | {item['mean_objective_reward']:.4f} | "
                    f"{item['mean_path_length']:.4f} | {item['selected_branch_hhi']:.4f} |"
                )

        lines.extend(
            [
                "",
                "## Divergence Summary",
                "",
                f"- Holdout objective gain (adaptive - frozen): `{comparison['objective_reward_gain']}`",
                f"- Selected-branch divergence: `{divergence['selected_branch_diff_count']}/{divergence['holdout_tasks']}`",
                f"- Activated-path divergence: `{divergence['activated_path_diff_count']}/{divergence['holdout_tasks']}`",
                f"- Mean reward delta on divergent tasks: `{divergence['mean_reward_delta_on_divergent']}`",
                f"- Mean reward delta on non-divergent tasks: `{divergence['mean_reward_delta_on_non_divergent']}`",
                "",
                "## Pairwise Judge",
                "",
                f"- Adaptive wins: `{pairwise.get('left_wins', 0)}`",
                f"- Frozen wins: `{pairwise.get('right_wins', 0)}`",
                f"- Ties: `{pairwise.get('ties', 0)}`",
                f"- Mean score adaptive: `{pairwise.get('mean_score_left', 0.0)}`",
                f"- Mean score frozen: `{pairwise.get('mean_score_right', 0.0)}`",
                "",
                "## By Aspect",
                "",
                "| Aspect | n | Adaptive mean | Frozen mean | Delta | Branch diffs | Adaptive branches | Frozen branches |",
                "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
            ]
        )
        for aspect, item in comparison["aspect_summary"].items():
            lines.append(
                f"| {aspect} | {item['n']} | {item['adaptive_mean_reward']:.4f} | "
                f"{item['frozen_mean_reward']:.4f} | {item['adaptive_minus_frozen']:.4f} | "
                f"{item['branch_diff_count']} | {item['adaptive_branch_counts']} | {item['frozen_branch_counts']} |"
            )

        lines.extend(
            [
                "",
                "## Divergent Holdout Tasks",
                "",
                "| Task | Aspect | Adaptive branch | Frozen branch | Adaptive | Frozen | Delta |",
                "| --- | --- | --- | --- | ---: | ---: | ---: |",
            ]
        )
        for item in divergence["divergent_tasks"]:
            lines.append(
                f"| {item['task_id']} | {item['aspect']} | {item['adaptive_branch']} | {item['frozen_branch']} | "
                f"{item['adaptive_reward']:.4f} | {item['frozen_reward']:.4f} | {item['reward_delta']:.4f} |"
            )
        if not divergence["divergent_tasks"]:
            lines.append("| none | n/a | n/a | n/a | 0.0000 | 0.0000 | 0.0000 |")

        lines.extend(
            [
                "",
                "## Top Learned Weight Deltas",
                "",
                "| Branch | Adaptive weight | Frozen weight | Delta |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for item in comparison["weight_delta_summary"]:
            lines.append(
                f"| {item['branch_name']} | {item['adaptive_weight']:.4f} | {item['frozen_weight']:.4f} | {item['delta']:.4f} |"
            )
        if not comparison["weight_delta_summary"]:
            lines.append("| none | 0.0000 | 0.0000 | 0.0000 |")

        lines.extend(
            [
                "",
                "## Usage",
                "",
                f"- adaptive_full backend usage: `{report['per_policy']['full_adaptive']['backend_usage_total']}`",
                f"- frozen backend usage: `{report['per_policy']['frozen']['backend_usage_total']}`",
                f"- judge backend usage: `{report['judge_backend_usage_total']}`",
                "",
                "## Notes",
                "",
            ]
        )
        for note in report.get("limitations", []):
            lines.append(f"- {note}")
        lines.append("")
        return "\n".join(lines)
