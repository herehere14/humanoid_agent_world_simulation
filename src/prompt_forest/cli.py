from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from .config import load_config
from .core.engine import PromptForestEngine
from .experiments.auto_improve import AutoImprover
from .experiments.benchmark import BenchmarkRunner
from .experiments.continuous_runner import ContinuousImprover
from .experiments.detailed_validation import DetailedHierarchicalValidator
from .experiments.rl_validation import RLLearningValidator
from .utils.io import read_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prompt-Forest adaptive agent CLI")
    parser.add_argument("--config", type=str, default="configs/default.json", help="Path to config JSON")

    sub = parser.add_subparsers(dest="command", required=True)

    task_cmd = sub.add_parser("run-task", help="Run one task through the prompt-forest")
    task_cmd.add_argument("--task", type=str, required=True)
    task_cmd.add_argument("--task-type", type=str, default="auto")
    task_cmd.add_argument("--metadata", type=str, default="{}", help="JSON string metadata")

    chat_cmd = sub.add_parser("chat", help="Interactive chat with the adaptive RL prompt-forest agent")
    chat_cmd.add_argument("--task-type", type=str, default="auto")
    chat_cmd.add_argument("--show-route", action="store_true", help="Show activated route and reward after each turn")

    bench_cmd = sub.add_parser("benchmark", help="Run benchmark dataset")
    bench_cmd.add_argument("--dataset", type=str, default="examples/demo_tasks.json")
    bench_cmd.add_argument("--rounds", type=int, default=4)

    val_cmd = sub.add_parser("rl-validate", help="Run adaptive-vs-frozen RL learning validation")
    val_cmd.add_argument("--episodes", type=int, default=240)
    val_cmd.add_argument("--seeds", type=str, default="11,17,19,23,29,31,37,41,43,47")

    detail_cmd = sub.add_parser("detailed-validate", help="Run detailed hierarchical learning + ablation validation")
    detail_cmd.add_argument("--episodes", type=int, default=240)
    detail_cmd.add_argument("--seeds", type=str, default="11,17,19,23,29")

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

    sub.add_parser("state", help="Show branch state and memory stats")

    return parser


def _chat_metadata_from_text(text: str) -> dict:
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
    }


def _run_chat_loop(engine: PromptForestEngine, default_task_type: str, show_route: bool) -> None:
    task_type = default_task_type
    print("Interactive RL chat started. Commands: /exit, /type <task_type>, /auto")
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

        metadata = _chat_metadata_from_text(text)
        result = engine.run_task(text=text, task_type=task_type, metadata=metadata)
        signal = result["evaluation_signal"]
        print(f"agent> {signal['selected_output']}")
        if show_route:
            path = " -> ".join(result["routing"]["activated_branches"])
            print(f"[route] {path}")
            print(
                f"[reward] {signal['reward_score']:.3f} "
                f"selected={signal['selected_branch']} reason={signal['failure_reason'] or 'ok'}"
            )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path if config_path.exists() else None)
    engine = PromptForestEngine(config=config)

    if args.command == "run-task":
        metadata = json.loads(args.metadata)
        result = engine.run_task(text=args.task, task_type=args.task_type, metadata=metadata)
        print(json.dumps(result, indent=2))
        return

    if args.command == "chat":
        _run_chat_loop(engine, default_task_type=args.task_type, show_route=args.show_route)
        return

    if args.command == "benchmark":
        runner = BenchmarkRunner(engine)
        summary = runner.run(dataset_path=args.dataset, rounds=args.rounds)
        print(json.dumps(summary.to_dict(), indent=2))
        return

    if args.command == "rl-validate":
        seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
        validator = RLLearningValidator(Path.cwd())
        report = validator.run(seeds=seeds, episodes_per_seed=args.episodes)
        print(json.dumps(report, indent=2))
        return

    if args.command == "detailed-validate":
        seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
        validator = DetailedHierarchicalValidator(Path.cwd())
        report = validator.run(seeds=seeds, episodes_per_seed=args.episodes)
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

    if args.command == "state":
        payload = {
            "branches": engine.branch_snapshot(),
            "memory": engine.memory.stats(),
            "routing_histogram": engine.routing_histogram(),
        }
        print(json.dumps(payload, indent=2))
        return


if __name__ == "__main__":
    main()
