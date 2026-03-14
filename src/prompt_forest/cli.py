from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from .backend.base import LLMBackend
from .backend.mock import MockLLMBackend
from .backend.openai_chat import OpenAIChatBackend
from .config import load_config
from .contracts import evaluate_output_contract, infer_output_contract
from .core.engine import PromptForestEngine
from .experiments.auto_improve import AutoImprover
from .experiments.benchmark import BenchmarkRunner
from .experiments.continuous_runner import ContinuousImprover
from .experiments.detailed_validation import DetailedHierarchicalValidator
from .experiments.hard_slice_validation import HardSliceValidator
from .experiments.live_ablation_validation import LiveAblationValidator
from .experiments.live_model_validation import LiveModelValidator
from .experiments.rl_validation import RLLearningValidator
from .observability.trace import format_comparison_trace, format_turn_trace
from .rewards.modes import ExactMatchReward, KeywordReward, RuleBasedReward, TaskSpecificReward
from .rewards.verifiers import ExternalVerifierReward
from .types import TaskInput
from .utils.io import read_json, read_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prompt-Forest adaptive agent CLI")
    parser.add_argument("--config", type=str, default="configs/default.json", help="Path to config JSON")

    sub = parser.add_subparsers(dest="command", required=True)

    task_cmd = sub.add_parser("run-task", help="Run one task through the prompt-forest")
    task_cmd.add_argument("--task", type=str, required=True)
    task_cmd.add_argument("--task-type", type=str, default="auto")
    task_cmd.add_argument("--user-id", type=str, default="global")
    task_cmd.add_argument("--metadata", type=str, default="{}", help="JSON string metadata")
    task_cmd.add_argument(
        "--visibility",
        type=str,
        default="minimal",
        choices=["minimal", "eval", "opt", "full"],
        help="Human-readable trace level for evaluator/optimizer internals",
    )
    task_cmd.add_argument("--top-branches", type=int, default=5, help="How many routing scores to print in trace mode")
    task_cmd.add_argument(
        "--compare-base",
        action="store_true",
        help="Also run the same task through the raw base model and print a real-time comparison",
    )

    chat_cmd = sub.add_parser("chat", help="Interactive chat with the adaptive RL prompt-forest agent")
    chat_cmd.add_argument("--task-type", type=str, default="auto")
    chat_cmd.add_argument("--user-id", type=str, default="global")
    chat_cmd.add_argument("--show-route", action="store_true", help="Show activated route and reward after each turn")
    chat_cmd.add_argument(
        "--visibility",
        type=str,
        default="full",
        choices=["minimal", "eval", "opt", "full"],
        help="Trace verbosity for evaluator and optimizer details per turn",
    )
    chat_cmd.add_argument("--top-branches", type=int, default=5)
    chat_cmd.add_argument(
        "--compare-base",
        action="store_true",
        help="Show a direct base-model answer alongside the adaptive system on every turn",
    )

    bench_cmd = sub.add_parser("benchmark", help="Run benchmark dataset")
    bench_cmd.add_argument("--dataset", type=str, default="examples/demo_tasks.json")
    bench_cmd.add_argument("--rounds", type=int, default=4)

    val_cmd = sub.add_parser("rl-validate", help="Run adaptive-vs-frozen RL learning validation")
    val_cmd.add_argument("--episodes", type=int, default=240)
    val_cmd.add_argument("--seeds", type=str, default="11,17,19,23,29,31,37,41,43,47")
    val_cmd.add_argument(
        "--start-mode",
        type=str,
        default="anti_prior",
        choices=["default", "anti_prior"],
        help="Initialization regime for validation runs",
    )
    val_cmd.add_argument(
        "--oracle-feedback",
        action="store_true",
        help="Inject oracle correction feedback during full-policy training",
    )

    detail_cmd = sub.add_parser("detailed-validate", help="Run detailed hierarchical learning + ablation validation")
    detail_cmd.add_argument("--episodes", type=int, default=240)
    detail_cmd.add_argument("--seeds", type=str, default="11,17,19,23,29")

    hard_cmd = sub.add_parser("hard-validate", help="Run hard-slice verifier-grounded adaptive validation")
    hard_cmd.add_argument("--episodes", type=int, default=220)
    hard_cmd.add_argument("--seeds", type=str, default="11,17,19,23,29,31,37,41")
    hard_cmd.add_argument("--oracle-feedback", action="store_true", help="Simulate oracle correction feedback during full-policy training")

    live_cmd = sub.add_parser("live-validate", help="Run real-model validation against adaptive, frozen, and direct baselines")
    live_cmd.add_argument("--dataset", type=str, default="examples/live_eval_tasks.json")
    live_cmd.add_argument("--model", type=str, default="gpt-4.1-mini")
    live_cmd.add_argument("--judge-model", type=str, default="")
    live_cmd.add_argument("--api-key-env", type=str, default="OPENAI_API_KEY")
    live_cmd.add_argument("--base-url", type=str, default="https://api.openai.com/v1")
    live_cmd.add_argument("--temperature", type=float, default=0.2)
    live_cmd.add_argument("--max-output-tokens", type=int, default=700)
    live_cmd.add_argument("--api-mode", type=str, default="chat_completions", choices=["chat_completions", "responses"])
    live_cmd.add_argument("--reasoning-effort", type=str, default="")
    live_cmd.add_argument("--judge-api-mode", type=str, default="")
    live_cmd.add_argument("--judge-reasoning-effort", type=str, default="")
    live_cmd.add_argument("--judge-temperature", type=float, default=0.0)
    live_cmd.add_argument("--judge-max-output-tokens", type=int, default=500)
    live_cmd.add_argument("--train-rounds", type=int, default=2)
    live_cmd.add_argument("--output-subdir", type=str, default="live_model_validation")
    live_cmd.add_argument("--report-prefix", type=str, default="live_model_validation_report")
    live_cmd.add_argument("--disable-agent-runtimes", action="store_true")

    ablate_cmd = sub.add_parser("live-ablate", help="Run live ablation for frozen, memory-only, weight-only, and full adaptive policies")
    ablate_cmd.add_argument("--dataset", type=str, default="examples/live_eval_tasks.json")
    ablate_cmd.add_argument("--model", type=str, default="gpt-4.1-mini")
    ablate_cmd.add_argument("--judge-model", type=str, default="")
    ablate_cmd.add_argument("--api-key-env", type=str, default="OPENAI_API_KEY")
    ablate_cmd.add_argument("--base-url", type=str, default="https://api.openai.com/v1")
    ablate_cmd.add_argument("--temperature", type=float, default=0.2)
    ablate_cmd.add_argument("--max-output-tokens", type=int, default=700)
    ablate_cmd.add_argument("--api-mode", type=str, default="chat_completions", choices=["chat_completions", "responses"])
    ablate_cmd.add_argument("--reasoning-effort", type=str, default="")
    ablate_cmd.add_argument("--judge-api-mode", type=str, default="")
    ablate_cmd.add_argument("--judge-reasoning-effort", type=str, default="")
    ablate_cmd.add_argument("--judge-temperature", type=float, default=0.0)
    ablate_cmd.add_argument("--judge-max-output-tokens", type=int, default=500)
    ablate_cmd.add_argument("--train-rounds", type=int, default=2)
    ablate_cmd.add_argument("--output-subdir", type=str, default="live_model_ablation_validation")
    ablate_cmd.add_argument("--report-prefix", type=str, default="live_model_ablation_report")
    ablate_cmd.add_argument("--disable-agent-runtimes", action="store_true")

    improve_cmd = sub.add_parser("auto-improve", help="Run multi-round config tuning with anti-bias objective")
    improve_cmd.add_argument("--rounds", type=int, default=3)
    improve_cmd.add_argument("--candidates", type=int, default=12)
    improve_cmd.add_argument("--episodes", type=int, default=140)
    improve_cmd.add_argument("--final-episodes", type=int, default=220)
    improve_cmd.add_argument("--seeds", type=str, default="3,5,7,11")
    improve_cmd.add_argument("--final-seeds", type=str, default="11,17,19,23,29,31,37,41")
    improve_cmd.add_argument("--no-apply", action="store_true", help="Do not write best patch into configs/default.json")

    continuous_cmd = sub.add_parser("continuous-improve", help="Run repeated auto-improve cycles with test gates")
    continuous_cmd.add_argument("--cycles", type=int, default=6)
    continuous_cmd.add_argument("--rounds-per-cycle", type=int, default=2)
    continuous_cmd.add_argument("--candidates", type=int, default=6)
    continuous_cmd.add_argument("--episodes", type=int, default=180)
    continuous_cmd.add_argument("--final-episodes", type=int, default=220)
    continuous_cmd.add_argument("--seeds", type=str, default="11,17,19,23,29,31,37,41")
    continuous_cmd.add_argument("--final-seeds", type=str, default="11,17,19,23,29,31,37,41")
    continuous_cmd.add_argument("--sleep-seconds", type=int, default=0)
    continuous_cmd.add_argument("--patience", type=int, default=2)
    continuous_cmd.add_argument("--skip-tests", action="store_true")
    continuous_cmd.add_argument("--no-apply", action="store_true", help="Evaluate cycles without writing configs/default.json")

    oc_cmd = sub.add_parser("openclaw-event", help="Process OpenClaw-style trajectory event JSON")
    oc_cmd.add_argument("--event-file", type=str, required=True)

    fb_cmd = sub.add_parser("feedback", help="Apply explicit user feedback to a previous task")
    fb_cmd.add_argument("--task-id", type=str, required=True)
    fb_cmd.add_argument("--score", type=float, required=True, help="Feedback score in [0,1] or [1,5]")
    fb_cmd.add_argument("--accepted", action="store_true", help="Mark previous answer as accepted")
    fb_cmd.add_argument("--rejected", action="store_true", help="Mark previous answer as rejected")
    fb_cmd.add_argument("--corrected-answer", type=str, default="")
    fb_cmd.add_argument("--feedback-text", type=str, default="")
    fb_cmd.add_argument("--user-id", type=str, default="")
    fb_cmd.add_argument("--style", type=str, default=None)
    fb_cmd.add_argument("--verbosity", type=str, default=None)
    fb_cmd.add_argument("--domain-preferences", type=str, default="", help="Comma-separated domains")
    fb_cmd.add_argument("--hard-constraints", type=str, default="", help="Comma-separated required constraints")

    inspect_cmd = sub.add_parser("inspect-events", help="Inspect recent evaluator/optimizer traces from events log")
    inspect_cmd.add_argument("--limit", type=int, default=10)
    inspect_cmd.add_argument(
        "--visibility",
        type=str,
        default="full",
        choices=["minimal", "eval", "opt", "full"],
    )
    inspect_cmd.add_argument("--top-branches", type=int, default=5)

    sub.add_parser("state", help="Show branch state and memory stats")

    return parser


def _chat_metadata_from_text(text: str, user_id: str) -> dict:
    tokens = re.findall(r"[a-zA-Z]{4,}", text.lower())
    deduped: list[str] = []
    for token in tokens:
        if token not in deduped:
            deduped.append(token)
        if len(deduped) >= 10:
            break
    return {
        "expected_keywords": deduped,
        "required_substrings": ["confidence"],
        "user_id": user_id,
    }


def _clone_backend_for_compare(backend: LLMBackend) -> LLMBackend:
    if isinstance(backend, OpenAIChatBackend):
        return OpenAIChatBackend(
            model=backend.model,
            api_key_env=backend.api_key_env,
            base_url=backend.base_url,
            temperature=backend.temperature,
            max_output_tokens=backend.max_output_tokens,
            timeout_seconds=backend.timeout_seconds,
            max_retries=backend.max_retries,
            retry_backoff_seconds=backend.retry_backoff_seconds,
            api_mode=backend.api_mode,
            reasoning_effort=backend.reasoning_effort,
            seed=backend.seed,
            system_prompt=backend.system_prompt,
        )
    if isinstance(backend, MockLLMBackend):
        return MockLLMBackend(seed=getattr(backend, "seed", 42))
    return backend


def _task_from_payload(payload: dict) -> TaskInput:
    return TaskInput(
        task_id=str(payload.get("task_id", "compare")),
        text=str(payload.get("text", "")),
        task_type=str(payload.get("task_type", "general")),
        metadata=dict(payload.get("metadata", {}) or {}),
    )


def _render_direct_prompt(task: TaskInput) -> str:
    return (
        "You are answering a user task directly without any routing or helper branches.\n"
        f"Task type: {task.task_type}\n"
        f"Task: {task.text}\n"
        "Follow any explicit formatting requirements in the task exactly. "
        "Be correct, concise when appropriate, and include confidence if the task asks for it."
    )


def _objective_metrics_for_output(engine: PromptForestEngine, task: TaskInput, output: str) -> dict[str, object]:
    overall = engine.judge.score_output(output, task)
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


def _base_model_comparison(
    engine: PromptForestEngine,
    result: dict[str, object],
    *,
    backend: LLMBackend | None = None,
) -> dict[str, object]:
    task = _task_from_payload(dict(result.get("task", {}) or {}))
    compare_backend = backend or _clone_backend_for_compare(engine.backend)
    base_output, base_meta = compare_backend.generate(_render_direct_prompt(task), task, branch_name="base_model_direct")
    base_metrics = _objective_metrics_for_output(engine, task, base_output)

    adaptive_output = str(dict(result.get("evaluation_signal", {}) or {}).get("selected_output", ""))
    adaptive_metrics = _objective_metrics_for_output(engine, task, adaptive_output)
    delta = round(
        float(adaptive_metrics["hybrid_verifier_reward"]) - float(base_metrics["hybrid_verifier_reward"]),
        4,
    )
    if delta > 0:
        winner = "adaptive_system"
    elif delta < 0:
        winner = "base_model"
    else:
        winner = "tie"

    return {
        "adaptive_system": {
            "selected_branch": str(dict(result.get("evaluation_signal", {}) or {}).get("selected_branch", "none")),
            "selected_output": adaptive_output,
            "objective_metrics": adaptive_metrics,
            "reward_components": dict(result.get("reward_components", {}) or {}),
        },
        "base_model": {
            "selected_branch": "base_model_direct",
            "selected_output": base_output,
            "objective_metrics": base_metrics,
            "model_meta": base_meta,
        },
        "delta": {
            "hybrid_verifier_reward": delta,
        },
        "winner": winner,
    }


def _run_chat_loop(
    engine: PromptForestEngine,
    default_task_type: str,
    user_id: str,
    show_route: bool,
    visibility: str,
    top_branches: int,
    compare_base: bool,
) -> None:
    task_type = default_task_type
    compare_enabled = compare_base
    compare_backend = _clone_backend_for_compare(engine.backend) if compare_enabled else None
    print("Interactive RL chat started. Commands: /exit, /type <task_type>, /auto, /compare on|off, /visibility <level>")
    while True:
        try:
            text = input("you> ").strip()
        except EOFError:
            print()
            break
        except KeyboardInterrupt:
            print()
            break

        if not text:
            continue
        if text in {"/exit", "exit", "quit"}:
            break
        if text.startswith("/type "):
            task_type = text.split(" ", 1)[1].strip() or "auto"
            print(f"task_type set to: {task_type}")
            continue
        if text == "/auto":
            task_type = "auto"
            print("task_type set to: auto")
            continue
        if text.startswith("/compare "):
            arg = text.split(" ", 1)[1].strip().lower()
            compare_enabled = arg in {"on", "true", "1", "yes"}
            compare_backend = _clone_backend_for_compare(engine.backend) if compare_enabled else None
            print(f"base comparison {'enabled' if compare_enabled else 'disabled'}")
            continue
        if text.startswith("/visibility "):
            arg = text.split(" ", 1)[1].strip().lower()
            if arg in {"minimal", "eval", "opt", "full"}:
                visibility = arg
                print(f"visibility set to: {visibility}")
            else:
                print("visibility must be one of: minimal, eval, opt, full")
            continue

        metadata = _chat_metadata_from_text(text, user_id=user_id)
        result = engine.run_task(text=text, task_type=task_type, metadata=metadata)
        comparison = _base_model_comparison(engine, result, backend=compare_backend) if compare_enabled else None
        signal = result["evaluation_signal"]
        if comparison:
            print(f"base> {comparison['base_model']['selected_output']}")
        print(f"adaptive> {signal['selected_output']}")
        if visibility != "minimal":
            print(format_turn_trace(result, visibility=visibility, top_branches=top_branches))
        elif show_route:
            path = " -> ".join(result["routing"]["activated_branches"])
            print(f"[route] {path}")
            print(
                f"[reward] {signal['reward_score']:.3f} "
                f"selected={signal['selected_branch']} reason={signal['failure_reason'] or 'ok'}"
            )
        if comparison:
            print(format_comparison_trace(comparison))


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path if config_path.exists() else None)
    engine = PromptForestEngine(config=config)

    if args.command == "run-task":
        metadata = json.loads(args.metadata)
        metadata["user_id"] = args.user_id
        result = engine.run_task(text=args.task, task_type=args.task_type, metadata=metadata)
        if args.compare_base:
            result["base_model_comparison"] = _base_model_comparison(engine, result)
        if args.visibility != "minimal":
            print(format_turn_trace(result, visibility=args.visibility, top_branches=args.top_branches))
            print()
        if args.compare_base:
            print(format_comparison_trace(result["base_model_comparison"]))
            print()
        print(json.dumps(result, indent=2))
        return

    if args.command == "chat":
        _run_chat_loop(
            engine,
            default_task_type=args.task_type,
            user_id=args.user_id,
            show_route=args.show_route,
            visibility=args.visibility,
            top_branches=args.top_branches,
            compare_base=args.compare_base,
        )
        return

    if args.command == "benchmark":
        runner = BenchmarkRunner(engine)
        summary = runner.run(dataset_path=args.dataset, rounds=args.rounds)
        print(json.dumps(summary.to_dict(), indent=2))
        return

    if args.command == "rl-validate":
        seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
        validator = RLLearningValidator(Path.cwd())
        report = validator.run(
            seeds=seeds,
            episodes_per_seed=args.episodes,
            start_mode=args.start_mode,
            simulate_oracle_feedback=args.oracle_feedback,
        )
        print(json.dumps(report, indent=2))
        return

    if args.command == "detailed-validate":
        seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
        validator = DetailedHierarchicalValidator(Path.cwd())
        report = validator.run(seeds=seeds, episodes_per_seed=args.episodes)
        print(json.dumps(report, indent=2))
        return

    if args.command == "hard-validate":
        seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
        validator = HardSliceValidator(Path.cwd())
        report = validator.run(
            seeds=seeds,
            episodes_per_seed=args.episodes,
            simulate_oracle_feedback=args.oracle_feedback,
        )
        print(json.dumps(report, indent=2))
        return

    if args.command == "live-validate":
        validator = LiveModelValidator(Path.cwd())
        report = validator.run(
            dataset_path=args.dataset,
            model=args.model,
            judge_model=args.judge_model or None,
            api_key_env=args.api_key_env,
            base_url=args.base_url,
            temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
            api_mode=args.api_mode,
            reasoning_effort=args.reasoning_effort or None,
            judge_api_mode=args.judge_api_mode or None,
            judge_reasoning_effort=args.judge_reasoning_effort or None,
            judge_temperature=args.judge_temperature,
            judge_max_output_tokens=args.judge_max_output_tokens,
            train_rounds=args.train_rounds,
            use_agent_runtimes=not args.disable_agent_runtimes,
            output_subdir=args.output_subdir,
            report_prefix=args.report_prefix,
        )
        print(json.dumps(report, indent=2))
        return

    if args.command == "live-ablate":
        validator = LiveAblationValidator(Path.cwd())
        report = validator.run(
            dataset_path=args.dataset,
            model=args.model,
            judge_model=args.judge_model or None,
            api_key_env=args.api_key_env,
            base_url=args.base_url,
            temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
            api_mode=args.api_mode,
            reasoning_effort=args.reasoning_effort or None,
            judge_api_mode=args.judge_api_mode or None,
            judge_reasoning_effort=args.judge_reasoning_effort or None,
            judge_temperature=args.judge_temperature,
            judge_max_output_tokens=args.judge_max_output_tokens,
            train_rounds=args.train_rounds,
            use_agent_runtimes=not args.disable_agent_runtimes,
            output_subdir=args.output_subdir,
            report_prefix=args.report_prefix,
        )
        print(json.dumps(report, indent=2))
        return

    if args.command == "auto-improve":
        seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
        final_seeds = [int(x.strip()) for x in args.final_seeds.split(",") if x.strip()]
        improver = AutoImprover(Path.cwd())
        report = improver.run(
            rounds=args.rounds,
            candidates_per_round=args.candidates,
            seeds=seeds,
            episodes_per_seed=args.episodes,
            final_eval_seeds=final_seeds,
            final_eval_episodes=args.final_episodes,
            apply_best=not args.no_apply,
        )
        print(json.dumps(report, indent=2))
        return

    if args.command == "continuous-improve":
        seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
        final_seeds = [int(x.strip()) for x in args.final_seeds.split(",") if x.strip()]
        runner = ContinuousImprover(Path.cwd())
        report = runner.run(
            max_cycles=args.cycles,
            rounds_per_cycle=args.rounds_per_cycle,
            candidates_per_round=args.candidates,
            seeds=seeds,
            episodes_per_seed=args.episodes,
            final_eval_seeds=final_seeds,
            final_eval_episodes=args.final_episodes,
            run_tests_each_cycle=not args.skip_tests,
            sleep_seconds=args.sleep_seconds,
            early_stop_patience=args.patience,
            apply_best=not args.no_apply,
        )
        print(json.dumps(report, indent=2))
        return

    if args.command == "openclaw-event":
        event = read_json(Path(args.event_file))
        result = engine.openclaw_ingest(event)
        print(json.dumps(result, indent=2))
        return

    if args.command == "feedback":
        accepted_value = None
        if args.accepted:
            accepted_value = True
        elif args.rejected:
            accepted_value = False
        domains = [x.strip() for x in args.domain_preferences.split(",") if x.strip()]
        constraints = [x.strip() for x in args.hard_constraints.split(",") if x.strip()]
        result = engine.apply_feedback(
            task_id=args.task_id,
            score=args.score,
            accepted=accepted_value,
            corrected_answer=args.corrected_answer,
            feedback_text=args.feedback_text,
            user_id=args.user_id or None,
            style=args.style,
            verbosity=args.verbosity,
            domain_preferences=domains if domains else None,
            hard_constraints=constraints if constraints else None,
        )
        print(json.dumps(result, indent=2))
        return

    if args.command == "inspect-events":
        events = read_jsonl(engine.artifacts_dir / "events.jsonl")
        if not events:
            print("No events logged yet. Run tasks first.")
            return

        limit = max(1, args.limit)
        start = max(0, len(events) - limit)
        for idx, event in enumerate(events[start:], start=start + 1):
            print(f"=== event #{idx} ===")
            print(format_turn_trace(event, visibility=args.visibility, top_branches=args.top_branches))
            if idx < len(events):
                print()
        return

    if args.command == "state":
        payload = {
            "branches": engine.branch_snapshot(),
            "memory": engine.memory.stats(),
            "user_profiles": int(engine.memory.stats().get("user_profiles", 0)),
            "routing_histogram": engine.routing_histogram(),
            "runtime": {
                "evaluator_llm_enabled": engine.config.agent_runtimes.evaluator.enabled,
                "optimizer_llm_enabled": engine.config.agent_runtimes.optimizer.enabled,
                "evaluator_provider": engine.config.agent_runtimes.evaluator.provider,
                "optimizer_provider": engine.config.agent_runtimes.optimizer.provider,
            },
        }
        print(json.dumps(payload, indent=2))
        return


if __name__ == "__main__":
    main()
