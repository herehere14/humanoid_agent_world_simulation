from __future__ import annotations

import json
import random
import shutil
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from ..agents.runtime_client import AgentRuntimeClient
from ..backend.openai_chat import OpenAIChatBackend
from ..config import AgentRuntimeConfig, EngineConfig, load_config
from ..contracts import evaluate_output_contract, infer_output_contract
from ..core.engine import PromptForestEngine
from ..evaluator.judge import OutputJudge
from ..rewards.modes import ExactMatchReward, KeywordReward, RuleBasedReward, TaskSpecificReward
from ..rewards.verifiers import ExternalVerifierReward
from ..types import TaskInput
from ..utils.io import read_json, write_json


@dataclass
class PairwiseJudgement:
    task_id: str
    split: str
    task_type: str
    left_policy: str
    right_policy: str
    winner: str
    score_left: float
    score_right: float
    rationale: str
    raw_response: dict[str, Any]


class LiveModelValidator:
    """Run real closed-source model evaluation against adaptive and frozen policies."""

    def __init__(self, root_dir: str | Path, seed: int = 20260314) -> None:
        self.root_dir = Path(root_dir)
        self.config_path = self.root_dir / "configs" / "default.json"
        self._rng = random.Random(seed)
        self._objective_judge = OutputJudge("hybrid_verifier")

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
        use_agent_runtimes: bool = True,
    ) -> dict[str, Any]:
        judge_model = judge_model or model
        raw_dataset = read_json(Path(dataset_path))
        train_specs = list(raw_dataset.get("train", []))
        holdout_specs = list(raw_dataset.get("holdout", []))
        if not train_specs or not holdout_specs:
            raise RuntimeError("Live evaluation dataset must include non-empty train and holdout sections.")

        adaptive_cfg = load_config(self.config_path)
        frozen_cfg = load_config(self.config_path)
        self._configure_live_eval(adaptive_cfg, use_agent_runtimes=use_agent_runtimes)
        self._configure_live_eval(frozen_cfg, use_agent_runtimes=use_agent_runtimes)

        out_dir = self.root_dir / "artifacts" / "live_model_validation"
        if out_dir.exists():
            shutil.rmtree(out_dir)
        adaptive_cfg.artifacts_dir = str(out_dir / "adaptive")
        frozen_cfg.artifacts_dir = str(out_dir / "frozen")

        adaptive_backend = OpenAIChatBackend(
            model=model,
            api_key_env=api_key_env,
            base_url=base_url,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            seed=42,
        )
        frozen_backend = OpenAIChatBackend(
            model=model,
            api_key_env=api_key_env,
            base_url=base_url,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            seed=42,
        )
        direct_backend = OpenAIChatBackend(
            model=model,
            api_key_env=api_key_env,
            base_url=base_url,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            seed=42,
        )
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

        adaptive_engine = PromptForestEngine(config=adaptive_cfg, backend=adaptive_backend)
        frozen_engine = PromptForestEngine(config=frozen_cfg, backend=frozen_backend)

        train_tasks = self._expand_train_tasks(train_specs, rounds=max(1, train_rounds))
        holdout_tasks = [self._task_from_spec(spec, split="holdout") for spec in holdout_specs]

        policy_runs: dict[str, dict[str, list[dict[str, Any]]]] = {
            "adaptive_full": {"train": [], "holdout": []},
            "frozen_forest": {"train": [], "holdout": []},
            "direct_model": {"train": [], "holdout": []},
        }

        for task in train_tasks:
            policy_runs["adaptive_full"]["train"].append(self._run_engine_task(adaptive_engine, task, adapt=True, update_memory=True))
            policy_runs["frozen_forest"]["train"].append(self._run_engine_task(frozen_engine, task, adapt=False, update_memory=False))
            policy_runs["direct_model"]["train"].append(self._run_direct_task(direct_backend, task))

        holdout_judgements: list[PairwiseJudgement] = []
        for task in holdout_tasks:
            adaptive_result = self._run_engine_task(adaptive_engine, task, adapt=False, update_memory=False)
            frozen_result = self._run_engine_task(frozen_engine, task, adapt=False, update_memory=False)
            direct_result = self._run_direct_task(direct_backend, task)

            policy_runs["adaptive_full"]["holdout"].append(adaptive_result)
            policy_runs["frozen_forest"]["holdout"].append(frozen_result)
            policy_runs["direct_model"]["holdout"].append(direct_result)

            holdout_judgements.extend(
                [
                    self._judge_pair(judge_backend, task, "adaptive_full", adaptive_result, "frozen_forest", frozen_result),
                    self._judge_pair(judge_backend, task, "adaptive_full", adaptive_result, "direct_model", direct_result),
                    self._judge_pair(judge_backend, task, "frozen_forest", frozen_result, "direct_model", direct_result),
                ]
            )

        aggregate = {
            split: {
                policy: self._summarize_policy_runs(policy_runs[policy][split])
                for policy in ("adaptive_full", "frozen_forest", "direct_model")
            }
            for split in ("train", "holdout")
        }
        comparisons = self._build_comparisons(aggregate, holdout_judgements)

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
                "router_top_k": adaptive_engine.config.router.top_k,
                "router_min_candidates": adaptive_engine.config.router.min_candidates,
                "composer_enabled": adaptive_engine.config.composer.enabled,
                "reward_mode": adaptive_engine.config.evaluator.reward_mode,
            },
            "dataset_summary": self._dataset_summary(train_specs, holdout_specs, train_rounds=max(1, train_rounds)),
            "aggregate": aggregate,
            "comparisons": comparisons,
            "pairwise_holdout": [self._pairwise_to_dict(item) for item in holdout_judgements],
            "per_policy": {
                "adaptive_full": {
                    "train": policy_runs["adaptive_full"]["train"],
                    "holdout": policy_runs["adaptive_full"]["holdout"],
                    "branch_snapshot": adaptive_engine.branch_snapshot(),
                    "memory": adaptive_engine.memory.stats(),
                    "routing_histogram": adaptive_engine.routing_histogram(),
                    "backend_usage_total": adaptive_backend.usage_summary(),
                    "evaluator_runtime_usage_total": adaptive_engine.evaluator_agent.runtime.usage_summary(),
                    "optimizer_runtime_usage_total": adaptive_engine.optimizer_agent._advisor.runtime.usage_summary(),  # noqa: SLF001
                },
                "frozen_forest": {
                    "train": policy_runs["frozen_forest"]["train"],
                    "holdout": policy_runs["frozen_forest"]["holdout"],
                    "branch_snapshot": frozen_engine.branch_snapshot(),
                    "memory": frozen_engine.memory.stats(),
                    "routing_histogram": frozen_engine.routing_histogram(),
                    "backend_usage_total": frozen_backend.usage_summary(),
                    "evaluator_runtime_usage_total": frozen_engine.evaluator_agent.runtime.usage_summary(),
                    "optimizer_runtime_usage_total": frozen_engine.optimizer_agent._advisor.runtime.usage_summary(),  # noqa: SLF001
                },
                "direct_model": {
                    "train": policy_runs["direct_model"]["train"],
                    "holdout": policy_runs["direct_model"]["holdout"],
                    "backend_usage_total": direct_backend.usage_summary(),
                },
            },
            "judge_backend_usage_total": judge_backend.usage_summary(),
            "limitations": [
                "This evaluation uses a real closed-source model for generation and judging, but the task set is curated rather than benchmark-standard.",
                "Explicit human feedback loops were not exercised because no real user labels were available.",
                "Pairwise judging uses an LLM judge, so those results should be read alongside the objective heuristic metrics.",
            ],
        }

        json_path = out_dir / "live_model_validation_report.json"
        md_path = out_dir / "live_model_validation_report.md"
        write_json(json_path, report)
        md_path.write_text(self._render_markdown_report(report), encoding="utf-8")
        report["report_paths"] = {"json": str(json_path), "markdown": str(md_path)}
        write_json(json_path, report)
        return report

    def _configure_live_eval(self, cfg: EngineConfig, *, use_agent_runtimes: bool) -> None:
        cfg.router.top_k = 1
        cfg.router.min_candidates = 1
        cfg.evaluator.reward_mode = "hybrid_verifier"
        cfg.agent_runtimes.evaluator.enabled = use_agent_runtimes
        cfg.agent_runtimes.optimizer.enabled = use_agent_runtimes
        cfg.memory.user_bias_mix = 0.0
        cfg.optimizer.max_active_candidates = min(cfg.optimizer.max_active_candidates, 4)
        cfg.optimizer.max_active_branches = min(cfg.optimizer.max_active_branches, 36)

    def _expand_train_tasks(self, specs: list[dict[str, Any]], rounds: int) -> list[TaskInput]:
        tasks: list[TaskInput] = []
        for round_idx in range(rounds):
            for spec in specs:
                cloned = deepcopy(spec)
                task = self._task_from_spec(cloned, split="train")
                task.task_id = f"{task.task_id}::round_{round_idx + 1}"
                task.metadata["round_index"] = round_idx + 1
                tasks.append(task)
        return tasks

    def _task_from_spec(self, spec: dict[str, Any], *, split: str) -> TaskInput:
        metadata = dict(spec.get("metadata", {}))
        metadata["split"] = split
        metadata.setdefault("user_id", spec.get("user_id", "eval_user"))
        return TaskInput(
            task_id=str(spec.get("task_id")),
            text=str(spec.get("text")),
            task_type=str(spec.get("task_type", "general")),
            metadata=metadata,
        )

    def _run_engine_task(
        self,
        engine: PromptForestEngine,
        task: TaskInput,
        *,
        adapt: bool,
        update_memory: bool,
    ) -> dict[str, Any]:
        backend_start = len(engine.backend.call_log())  # type: ignore[attr-defined]
        eval_start = len(engine.evaluator_agent.runtime.call_log())
        opt_start = len(engine.optimizer_agent._advisor.runtime.call_log())  # noqa: SLF001
        result = engine.run_task_controlled(
            text=task.text,
            task_type=task.task_type,
            metadata=dict(task.metadata),
            adapt=adapt,
            update_memory=update_memory,
        )
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

    def _run_direct_task(self, backend: OpenAIChatBackend, task: TaskInput) -> dict[str, Any]:
        start = len(backend.call_log())
        prompt = self._render_direct_prompt(task)
        output, meta = backend.generate(prompt, task, branch_name="direct_model")
        metrics = self._objective_metrics(task, output)
        return {
            "task_id": task.task_id,
            "split": str(task.metadata.get("split", "")),
            "task_type": task.task_type,
            "aspect": str(task.metadata.get("aspect", task.task_type)),
            "text": task.text,
            "selected_output": output,
            "selected_branch": "direct_model",
            "activated_branches": ["direct_model"],
            "activated_paths": [["direct_model"]],
            "internal_reward_score": None,
            "internal_confidence": None,
            "failure_reason": "",
            "optimizer": {},
            "objective_metrics": metrics,
            "backend_usage": self._summarize_calls(backend.call_log()[start:]),
            "model_meta": meta,
        }

    def _render_direct_prompt(self, task: TaskInput) -> str:
        return (
            "You are answering a user task directly without any routing or helper branches.\n"
            f"Task type: {task.task_type}\n"
            f"Task: {task.text}\n"
            "Follow any explicit formatting requirements in the task exactly. "
            "Be correct, concise when appropriate, and include confidence if the task asks for it."
        )

    def _objective_metrics(self, task: TaskInput, output: str) -> dict[str, Any]:
        overall = self._objective_judge.score_output(output, task)
        exact, exact_reason = ExactMatchReward(weight=1.0).score(output, task)
        keyword, keyword_reason = KeywordReward(weight=1.0).score(output, task)
        rule, rule_reason = RuleBasedReward(weight=1.0).score(output, task)
        task_specific, task_reason = TaskSpecificReward(weight=1.0).score(output, task)
        external, external_reason = ExternalVerifierReward(weight=1.0).score(output, task)
        contract = infer_output_contract(task.text, task.metadata)
        contract_pass = True
        contract_reason = "no_contract"
        if contract:
            contract_pass, contract_reason = evaluate_output_contract(output, contract, task.text)

        return {
            "hybrid_verifier_reward": round(overall.reward, 4),
            "hybrid_verifier_reason": overall.reason,
            "exact_score": round(exact, 4),
            "exact_reason": exact_reason,
            "keyword_score": round(keyword, 4),
            "keyword_reason": keyword_reason,
            "rule_score": round(rule, 4),
            "rule_reason": rule_reason,
            "task_specific_score": round(task_specific, 4),
            "task_specific_reason": task_reason,
            "external_verifier_score": round(external, 4),
            "external_verifier_reason": external_reason,
            "contract": contract or "",
            "contract_pass": contract_pass,
            "contract_reason": contract_reason,
            "output_word_count": len(output.split()),
        }

    def _judge_pair(
        self,
        judge_backend: OpenAIChatBackend,
        task: TaskInput,
        left_policy: str,
        left_result: dict[str, Any],
        right_policy: str,
        right_result: dict[str, Any],
    ) -> PairwiseJudgement:
        swapped = self._rng.random() < 0.5
        first_policy, second_policy = (right_policy, left_policy) if swapped else (left_policy, right_policy)
        first_result, second_result = (right_result, left_result) if swapped else (left_result, right_result)

        prompt = self._render_pairwise_prompt(task, first_policy, first_result, second_policy, second_result)
        raw_text, _ = judge_backend.generate(prompt, task, branch_name=f"judge::{left_policy}::{right_policy}")
        try:
            parsed = AgentRuntimeClient._parse_json_text(raw_text)
        except Exception:
            parsed = {"winner": "tie", "score_a": 5.0, "score_b": 5.0, "rationale": raw_text.strip()}

        winner = str(parsed.get("winner", "tie")).strip().lower()
        score_a = self._clamp_score(parsed.get("score_a", 5.0))
        score_b = self._clamp_score(parsed.get("score_b", 5.0))
        rationale = str(parsed.get("rationale", "")).strip()

        if winner not in {"a", "b", "tie"}:
            winner = "tie"

        if swapped:
            mapped_winner = {"a": right_policy, "b": left_policy, "tie": "tie"}[winner]
            left_score, right_score = score_b, score_a
        else:
            mapped_winner = {"a": left_policy, "b": right_policy, "tie": "tie"}[winner]
            left_score, right_score = score_a, score_b

        return PairwiseJudgement(
            task_id=task.task_id,
            split=str(task.metadata.get("split", "")),
            task_type=task.task_type,
            left_policy=left_policy,
            right_policy=right_policy,
            winner=mapped_winner,
            score_left=left_score,
            score_right=right_score,
            rationale=rationale,
            raw_response=parsed,
        )

    def _render_pairwise_prompt(
        self,
        task: TaskInput,
        policy_a: str,
        result_a: dict[str, Any],
        policy_b: str,
        result_b: dict[str, Any],
    ) -> str:
        rubric = task.metadata.get("evaluation_rubric", [])
        required = task.metadata.get("required_substrings", [])
        reference = str(task.metadata.get("reference_answer", "")).strip()
        return (
            "Evaluate two candidate answers for the same task.\n"
            "Return JSON only with keys winner, score_a, score_b, rationale.\n"
            "winner must be A, B, or tie. Scores are 0-10.\n"
            "Primary criteria: correctness, explicit constraint satisfaction, completeness, calibration, usefulness.\n"
            f"Task type: {task.task_type}\n"
            f"Task: {task.text}\n"
            f"Required constraints: {required}\n"
            f"Rubric: {rubric}\n"
            f"Reference answer: {reference or 'none'}\n"
            f"Candidate A ({policy_a}):\n{result_a['selected_output']}\n\n"
            f"Candidate B ({policy_b}):\n{result_b['selected_output']}\n"
        )

    def _summarize_policy_runs(self, runs: list[dict[str, Any]]) -> dict[str, Any]:
        if not runs:
            return {}

        objective_rewards = [float(item["objective_metrics"]["hybrid_verifier_reward"]) for item in runs]
        contract_tasks = [item for item in runs if item["objective_metrics"].get("contract")]
        contract_pass_rate = (
            sum(1 for item in contract_tasks if item["objective_metrics"].get("contract_pass")) / len(contract_tasks)
            if contract_tasks
            else None
        )
        by_task_type: dict[str, list[float]] = defaultdict(list)
        selected_branches: list[str] = []
        path_lengths: list[int] = []
        optimizer_counts = Counter()
        for item in runs:
            by_task_type[item["task_type"]].append(float(item["objective_metrics"]["hybrid_verifier_reward"]))
            selected_branches.append(str(item["selected_branch"]))
            path_lengths.append(len(item.get("activated_branches", [])))
            optimizer = item.get("optimizer", {})
            if optimizer:
                optimizer_counts["rewritten_prompts"] += len(optimizer.get("rewritten_prompts", []))
                optimizer_counts["created_candidates"] += len(optimizer.get("created_candidates", []))
                optimizer_counts["promoted_candidates"] += len(optimizer.get("promoted_candidates", []))
                optimizer_counts["archived_candidates"] += len(optimizer.get("archived_candidates", []))

        first_n = max(1, len(objective_rewards) // 3)
        backend_usage = self._combine_usage([item.get("backend_usage", {}) for item in runs])
        evaluator_usage = self._combine_usage([item.get("evaluator_runtime_usage", {}) for item in runs])
        optimizer_usage = self._combine_usage([item.get("optimizer_runtime_usage", {}) for item in runs])
        return {
            "episodes": len(runs),
            "mean_objective_reward": round(mean(objective_rewards), 4),
            "first_third_mean_reward": round(mean(objective_rewards[:first_n]), 4),
            "last_third_mean_reward": round(mean(objective_rewards[-first_n:]), 4),
            "last_minus_first_third": round(mean(objective_rewards[-first_n:]) - mean(objective_rewards[:first_n]), 4),
            "contract_pass_rate": round(contract_pass_rate, 4) if contract_pass_rate is not None else None,
            "mean_path_length": round(mean(path_lengths), 4),
            "selected_branch_hhi": round(self._hhi(selected_branches), 4),
            "selected_branch_counts": dict(Counter(selected_branches).most_common(12)),
            "objective_reward_by_task_type": {
                task_type: round(mean(vals), 4) for task_type, vals in sorted(by_task_type.items())
            },
            "optimizer_event_counts": dict(optimizer_counts),
            "backend_usage": backend_usage,
            "evaluator_runtime_usage": evaluator_usage,
            "optimizer_runtime_usage": optimizer_usage,
        }

    def _build_comparisons(
        self,
        aggregate: dict[str, dict[str, dict[str, Any]]],
        judgements: list[PairwiseJudgement],
    ) -> dict[str, Any]:
        holdout = aggregate["holdout"]
        adaptive = holdout["adaptive_full"]
        frozen = holdout["frozen_forest"]
        direct = holdout["direct_model"]

        pair_summary = self._pairwise_summary(judgements)
        return {
            "adaptive_vs_frozen": {
                "objective_reward_gain": round(
                    adaptive["mean_objective_reward"] - frozen["mean_objective_reward"], 4
                ),
                "contract_pass_delta": self._delta(adaptive.get("contract_pass_rate"), frozen.get("contract_pass_rate")),
                "pairwise": pair_summary.get("adaptive_full__vs__frozen_forest", {}),
            },
            "adaptive_vs_direct": {
                "objective_reward_gain": round(
                    adaptive["mean_objective_reward"] - direct["mean_objective_reward"], 4
                ),
                "contract_pass_delta": self._delta(adaptive.get("contract_pass_rate"), direct.get("contract_pass_rate")),
                "pairwise": pair_summary.get("adaptive_full__vs__direct_model", {}),
            },
            "frozen_vs_direct": {
                "objective_reward_gain": round(
                    frozen["mean_objective_reward"] - direct["mean_objective_reward"], 4
                ),
                "contract_pass_delta": self._delta(frozen.get("contract_pass_rate"), direct.get("contract_pass_rate")),
                "pairwise": pair_summary.get("frozen_forest__vs__direct_model", {}),
            },
            "pairwise_summary": pair_summary,
        }

    def _pairwise_summary(self, judgements: list[PairwiseJudgement]) -> dict[str, Any]:
        grouped: dict[str, list[PairwiseJudgement]] = defaultdict(list)
        for item in judgements:
            key = f"{item.left_policy}__vs__{item.right_policy}"
            grouped[key].append(item)

        out: dict[str, Any] = {}
        for key, items in grouped.items():
            wins_left = sum(1 for item in items if item.winner == item.left_policy)
            wins_right = sum(1 for item in items if item.winner == item.right_policy)
            ties = sum(1 for item in items if item.winner == "tie")
            by_type: dict[str, dict[str, int]] = defaultdict(lambda: {"left": 0, "right": 0, "tie": 0})
            for item in items:
                bucket = by_type[item.task_type]
                if item.winner == item.left_policy:
                    bucket["left"] += 1
                elif item.winner == item.right_policy:
                    bucket["right"] += 1
                else:
                    bucket["tie"] += 1
            out[key] = {
                "n": len(items),
                "left_wins": wins_left,
                "right_wins": wins_right,
                "ties": ties,
                "left_win_rate": round(wins_left / max(1, len(items)), 4),
                "right_win_rate": round(wins_right / max(1, len(items)), 4),
                "mean_score_left": round(mean(item.score_left for item in items), 4),
                "mean_score_right": round(mean(item.score_right for item in items), 4),
                "by_task_type": by_type,
            }
        return out

    def _dataset_summary(
        self,
        train_specs: list[dict[str, Any]],
        holdout_specs: list[dict[str, Any]],
        *,
        train_rounds: int,
    ) -> dict[str, Any]:
        def counts(specs: list[dict[str, Any]]) -> dict[str, int]:
            return dict(Counter(str(item.get("task_type", "general")) for item in specs))

        contract_train = sum(1 for item in train_specs if infer_output_contract(str(item.get("text", "")), item.get("metadata", {})))
        contract_holdout = sum(1 for item in holdout_specs if infer_output_contract(str(item.get("text", "")), item.get("metadata", {})))
        return {
            "train_tasks_per_round": len(train_specs),
            "train_rounds": train_rounds,
            "expanded_train_tasks": len(train_specs) * train_rounds,
            "holdout_tasks": len(holdout_specs),
            "train_task_types": counts(train_specs),
            "holdout_task_types": counts(holdout_specs),
            "train_contract_tasks_per_round": contract_train,
            "holdout_contract_tasks": contract_holdout,
        }

    def _render_markdown_report(self, report: dict[str, Any]) -> str:
        aggregate = report["aggregate"]
        comparisons = report["comparisons"]
        lines = [
            "# Live Model Validation Report",
            "",
            f"- Generated at: `{report['generated_at']}`",
            f"- Generation model: `{report['models']['generation_model']}`",
            f"- Judge model: `{report['models']['judge_model']}`",
            f"- Dataset: `{report['dataset_path']}`",
            "",
            "## Dataset",
            "",
            f"- Train tasks per round: `{report['dataset_summary']['train_tasks_per_round']}`",
            f"- Train rounds: `{report['dataset_summary']['train_rounds']}`",
            f"- Expanded train tasks: `{report['dataset_summary']['expanded_train_tasks']}`",
            f"- Holdout tasks: `{report['dataset_summary']['holdout_tasks']}`",
            "",
            "## Aggregate Metrics",
            "",
            "| Split | Policy | Mean objective reward | Contract pass rate | Mean path length | Branch HHI |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
        for split in ("train", "holdout"):
            for policy in ("adaptive_full", "frozen_forest", "direct_model"):
                item = aggregate[split][policy]
                lines.append(
                    f"| {split} | {policy} | {item['mean_objective_reward']:.4f} | "
                    f"{self._fmt_optional(item.get('contract_pass_rate'))} | "
                    f"{item['mean_path_length']:.4f} | {item['selected_branch_hhi']:.4f} |"
                )

        lines.extend(
            [
                "",
                "## Holdout Comparisons",
                "",
                f"- Adaptive vs frozen objective gain: `{comparisons['adaptive_vs_frozen']['objective_reward_gain']}`",
                f"- Adaptive vs direct objective gain: `{comparisons['adaptive_vs_direct']['objective_reward_gain']}`",
                f"- Frozen vs direct objective gain: `{comparisons['frozen_vs_direct']['objective_reward_gain']}`",
                "",
                "## Pairwise Judge Summary",
                "",
            ]
        )
        for key, value in comparisons["pairwise_summary"].items():
            lines.append(
                f"- `{key}`: left win rate `{value['left_win_rate']}`, "
                f"right win rate `{value['right_win_rate']}`, ties `{value['ties']}/{value['n']}`"
            )

        lines.extend(
            [
                "",
                "## Notes",
                "",
            ]
        )
        for note in report.get("limitations", []):
            lines.append(f"- {note}")
        lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _compact_optimizer(optimization: dict[str, Any]) -> dict[str, Any]:
        if not optimization:
            return {}
        return {
            "rewritten_prompts": optimization.get("rewritten_prompts", []),
            "promoted_candidates": optimization.get("promoted_candidates", []),
            "archived_candidates": optimization.get("archived_candidates", []),
            "created_candidates": optimization.get("created_candidates", []),
            "advisor_used": optimization.get("advisor_used", False),
            "advisor_error": optimization.get("advisor_error", ""),
        }

    @staticmethod
    def _summarize_calls(calls: list[dict[str, Any]]) -> dict[str, Any]:
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        total_latency_ms = 0.0
        ok_calls = 0
        error_calls = 0
        for item in calls:
            usage = item.get("usage", {}) or {}
            prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
            completion_tokens += int(usage.get("completion_tokens", 0) or 0)
            total_tokens += int(usage.get("total_tokens", 0) or 0)
            total_latency_ms += float(item.get("latency_ms", 0.0) or 0.0)
            if item.get("ok", False):
                ok_calls += 1
            else:
                error_calls += 1
        return {
            "call_count": len(calls),
            "ok_calls": ok_calls,
            "error_calls": error_calls,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "total_latency_ms": round(total_latency_ms, 3),
            "mean_latency_ms": round(total_latency_ms / max(1, len(calls)), 3),
        }

    @staticmethod
    def _combine_usage(items: list[dict[str, Any]]) -> dict[str, Any]:
        out = {
            "call_count": 0,
            "ok_calls": 0,
            "error_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "total_latency_ms": 0.0,
            "mean_latency_ms": 0.0,
        }
        for item in items:
            out["call_count"] += int(item.get("call_count", 0) or 0)
            out["ok_calls"] += int(item.get("ok_calls", 0) or 0)
            out["error_calls"] += int(item.get("error_calls", 0) or 0)
            out["prompt_tokens"] += int(item.get("prompt_tokens", 0) or 0)
            out["completion_tokens"] += int(item.get("completion_tokens", 0) or 0)
            out["total_tokens"] += int(item.get("total_tokens", 0) or 0)
            out["total_latency_ms"] += float(item.get("total_latency_ms", 0.0) or 0.0)
        out["total_latency_ms"] = round(out["total_latency_ms"], 3)
        out["mean_latency_ms"] = round(out["total_latency_ms"] / max(1, out["call_count"]), 3)
        return out

    @staticmethod
    def _pairwise_to_dict(item: PairwiseJudgement) -> dict[str, Any]:
        return {
            "task_id": item.task_id,
            "split": item.split,
            "task_type": item.task_type,
            "left_policy": item.left_policy,
            "right_policy": item.right_policy,
            "winner": item.winner,
            "score_left": round(item.score_left, 4),
            "score_right": round(item.score_right, 4),
            "rationale": item.rationale,
            "raw_response": item.raw_response,
        }

    @staticmethod
    def _delta(left: float | None, right: float | None) -> float | None:
        if left is None or right is None:
            return None
        return round(left - right, 4)

    @staticmethod
    def _hhi(items: list[str]) -> float:
        if not items:
            return 0.0
        counts = Counter(items)
        n = len(items)
        return sum((count / n) ** 2 for count in counts.values())

    @staticmethod
    def _clamp_score(value: Any) -> float:
        try:
            score = float(value)
        except (TypeError, ValueError):
            return 5.0
        return max(0.0, min(10.0, score))

    @staticmethod
    def _fmt_optional(value: float | None) -> str:
        if value is None:
            return "n/a"
        return f"{value:.4f}"
