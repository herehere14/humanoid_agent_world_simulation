from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
import shutil
from typing import Any

from ..backend.openai_chat import OpenAIChatBackend
from ..config import load_config
from ..core.engine import PromptForestEngine
from ..utils.io import read_json, write_json
from .live_model_validation import LiveModelValidator, PairwiseJudgement


@dataclass(frozen=True)
class PolicyDefinition:
    name: str
    adapt_train: bool
    update_memory_train: bool
    description: str


class LiveAblationValidator(LiveModelValidator):
    """Run live ablations across frozen, memory-only, weight-only, and full adaptive policies."""

    def run(
        self,
        *,
        dataset_path: str | Path,
        model: str = "gpt-4.1-mini",
        judge_model: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        base_url: str = "https://api.openai.com/v1",
        temperature: float = 0.2,
        max_output_tokens: int = 700,
        train_rounds: int = 2,
        use_agent_runtimes: bool = False,
        policies: list[str] | None = None,
    ) -> dict[str, Any]:
        judge_model = judge_model or model
        policy_names = policies or ["full_adaptive", "frozen", "memory_only", "weight_only"]
        policy_defs = [self._policy_definition(name) for name in policy_names]

        raw_dataset = read_json(Path(dataset_path))
        train_specs = list(raw_dataset.get("train", []))
        holdout_specs = list(raw_dataset.get("holdout", []))
        if not train_specs or not holdout_specs:
            raise RuntimeError("Live evaluation dataset must include non-empty train and holdout sections.")

        out_dir = self.root_dir / "artifacts" / "live_model_ablation_validation"
        if out_dir.exists():
            shutil.rmtree(out_dir)
        engines: dict[str, PromptForestEngine] = {}
        backends: dict[str, OpenAIChatBackend] = {}

        for policy in policy_defs:
            cfg = load_config(self.config_path)
            self._configure_live_eval(cfg, use_agent_runtimes=use_agent_runtimes)
            cfg.artifacts_dir = str(out_dir / policy.name)

            backend = OpenAIChatBackend(
                model=model,
                api_key_env=api_key_env,
                base_url=base_url,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                seed=42,
            )
            engines[policy.name] = PromptForestEngine(config=cfg, backend=backend)
            backends[policy.name] = backend

        judge_backend = OpenAIChatBackend(
            model=judge_model,
            api_key_env=api_key_env,
            base_url=base_url,
            temperature=0.0,
            max_output_tokens=500,
            seed=7,
            system_prompt=(
                "You are an impartial evaluator. Return valid JSON only. "
                "Prefer correctness and explicit instruction-following over style."
            ),
        )

        train_tasks = self._expand_train_tasks(train_specs, rounds=max(1, train_rounds))
        holdout_tasks = [self._task_from_spec(spec, split="holdout") for spec in holdout_specs]
        policy_runs: dict[str, dict[str, list[dict[str, Any]]]] = {
            policy.name: {"train": [], "holdout": []} for policy in policy_defs
        }

        for task in train_tasks:
            for policy in policy_defs:
                policy_runs[policy.name]["train"].append(
                    self._run_engine_task(
                        engines[policy.name],
                        task,
                        adapt=policy.adapt_train,
                        update_memory=policy.update_memory_train,
                    )
                )

        holdout_judgements: list[PairwiseJudgement] = []
        for task in holdout_tasks:
            holdout_results: dict[str, dict[str, Any]] = {}
            for policy in policy_defs:
                result = self._run_engine_task(
                    engines[policy.name],
                    task,
                    adapt=False,
                    update_memory=False,
                )
                holdout_results[policy.name] = result
                policy_runs[policy.name]["holdout"].append(result)

            for left_name, right_name in combinations(policy_names, 2):
                holdout_judgements.append(
                    self._judge_pair(
                        judge_backend,
                        task,
                        left_name,
                        holdout_results[left_name],
                        right_name,
                        holdout_results[right_name],
                    )
                )

        aggregate = {
            split: {
                policy.name: self._summarize_policy_runs(policy_runs[policy.name][split])
                for policy in policy_defs
            }
            for split in ("train", "holdout")
        }
        comparisons = self._build_ablation_comparisons(policy_defs, aggregate, holdout_judgements)

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
            },
            "settings": {
                "train_rounds": max(1, train_rounds),
                "use_agent_runtimes": use_agent_runtimes,
                "policy_order": policy_names,
                "router_top_k": next(iter(engines.values())).config.router.top_k,
                "router_min_candidates": next(iter(engines.values())).config.router.min_candidates,
                "composer_enabled": next(iter(engines.values())).config.composer.enabled,
                "reward_mode": next(iter(engines.values())).config.evaluator.reward_mode,
            },
            "policies": {
                policy.name: {
                    "adapt_train": policy.adapt_train,
                    "update_memory_train": policy.update_memory_train,
                    "description": policy.description,
                }
                for policy in policy_defs
            },
            "dataset_summary": self._dataset_summary(train_specs, holdout_specs, train_rounds=max(1, train_rounds)),
            "aggregate": aggregate,
            "comparisons": comparisons,
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
                "This evaluation uses a real closed-source model for generation and judging, but the task set is curated rather than benchmark-standard.",
                "Explicit human feedback loops were not exercised because no real user labels were available.",
                "Pairwise judging uses an LLM judge, so those results should be read alongside the objective heuristic metrics.",
            ],
        }

        json_path = out_dir / "live_model_ablation_report.json"
        md_path = out_dir / "live_model_ablation_report.md"
        write_json(json_path, report)
        md_path.write_text(self._render_markdown_report(report), encoding="utf-8")
        report["report_paths"] = {"json": str(json_path), "markdown": str(md_path)}
        write_json(json_path, report)
        return report

    def _policy_definition(self, name: str) -> PolicyDefinition:
        normalized = name.strip().lower()
        if normalized == "full_adaptive":
            return PolicyDefinition(
                name="full_adaptive",
                adapt_train=True,
                update_memory_train=True,
                description="Weight updates and memory updates both enabled during training.",
            )
        if normalized == "frozen":
            return PolicyDefinition(
                name="frozen",
                adapt_train=False,
                update_memory_train=False,
                description="No weight updates and no memory updates during training.",
            )
        if normalized == "memory_only":
            return PolicyDefinition(
                name="memory_only",
                adapt_train=False,
                update_memory_train=True,
                description="Memory updates enabled; optimizer disabled.",
            )
        if normalized == "weight_only":
            return PolicyDefinition(
                name="weight_only",
                adapt_train=True,
                update_memory_train=False,
                description="Optimizer enabled; memory updates disabled.",
            )
        raise ValueError(f"Unknown live ablation policy: {name}")

    def _build_ablation_comparisons(
        self,
        policy_defs: list[PolicyDefinition],
        aggregate: dict[str, dict[str, dict[str, Any]]],
        judgements: list[PairwiseJudgement],
    ) -> dict[str, Any]:
        policy_names = [policy.name for policy in policy_defs]
        holdout = aggregate["holdout"]
        pair_summary = self._pairwise_summary(judgements)

        objective_delta_matrix: dict[str, dict[str, float]] = {}
        for left_name in policy_names:
            objective_delta_matrix[left_name] = {}
            for right_name in policy_names:
                objective_delta_matrix[left_name][right_name] = round(
                    holdout[left_name]["mean_objective_reward"] - holdout[right_name]["mean_objective_reward"],
                    4,
                )

        ranking = [
            {"policy": name, "mean_objective_reward": holdout[name]["mean_objective_reward"]}
            for name in sorted(policy_names, key=lambda item: holdout[item]["mean_objective_reward"], reverse=True)
        ]

        versus_frozen: dict[str, Any] = {}
        if "frozen" in holdout:
            for name in policy_names:
                if name == "frozen":
                    continue
                key = f"{name}__vs__frozen" if f"{name}__vs__frozen" in pair_summary else f"frozen__vs__{name}"
                summary = pair_summary.get(key, {})
                if key.startswith("frozen__vs__") and summary:
                    summary = {
                        **summary,
                        "left_policy": "frozen",
                        "right_policy": name,
                    }
                versus_frozen[name] = {
                    "objective_reward_gain": round(
                        holdout[name]["mean_objective_reward"] - holdout["frozen"]["mean_objective_reward"], 4
                    ),
                    "contract_pass_delta": self._delta(
                        holdout[name].get("contract_pass_rate"),
                        holdout["frozen"].get("contract_pass_rate"),
                    ),
                    "pairwise": summary,
                }

        return {
            "holdout_ranking": ranking,
            "objective_delta_matrix": objective_delta_matrix,
            "versus_frozen": versus_frozen,
            "pairwise_summary": pair_summary,
        }

    def _render_markdown_report(self, report: dict[str, Any]) -> str:
        aggregate = report["aggregate"]
        policy_order = report["settings"]["policy_order"]
        lines = [
            "# Live Model Ablation Report",
            "",
            f"- Generated at: `{report['generated_at']}`",
            f"- Generation model: `{report['models']['generation_model']}`",
            f"- Judge model: `{report['models']['judge_model']}`",
            f"- Dataset: `{report['dataset_path']}`",
            "",
            "## Policies",
            "",
        ]
        for name in policy_order:
            policy = report["policies"][name]
            lines.append(
                f"- `{name}`: adapt_train=`{policy['adapt_train']}`, "
                f"update_memory_train=`{policy['update_memory_train']}`. {policy['description']}"
            )

        lines.extend(
            [
                "",
                "## Aggregate Metrics",
                "",
                "| Split | Policy | Mean objective reward | Contract pass rate | Mean path length | Branch HHI |",
                "| --- | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for split in ("train", "holdout"):
            for name in policy_order:
                item = aggregate[split][name]
                lines.append(
                    f"| {split} | {name} | {item['mean_objective_reward']:.4f} | "
                    f"{self._fmt_optional(item.get('contract_pass_rate'))} | "
                    f"{item['mean_path_length']:.4f} | {item['selected_branch_hhi']:.4f} |"
                )

        lines.extend(
            [
                "",
                "## Holdout Ranking",
                "",
            ]
        )
        for idx, item in enumerate(report["comparisons"]["holdout_ranking"], start=1):
            lines.append(f"{idx}. `{item['policy']}`: `{item['mean_objective_reward']:.4f}`")

        lines.extend(
            [
                "",
                "## Versus Frozen",
                "",
            ]
        )
        for name, item in report["comparisons"]["versus_frozen"].items():
            lines.append(
                f"- `{name}` vs `frozen`: objective delta `{item['objective_reward_gain']}`, "
                f"contract delta `{item['contract_pass_delta']}`"
            )

        lines.extend(
            [
                "",
                "## Pairwise Judge Summary",
                "",
            ]
        )
        for key, value in report["comparisons"]["pairwise_summary"].items():
            lines.append(
                f"- `{key}`: left win rate `{value['left_win_rate']}`, "
                f"right win rate `{value['right_win_rate']}`, ties `{value['ties']}/{value['n']}`"
            )
        lines.append("")
        return "\n".join(lines)
