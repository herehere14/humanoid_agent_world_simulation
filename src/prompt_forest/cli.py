from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .core.engine import PromptForestEngine
from .experiments.benchmark import BenchmarkRunner
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

    bench_cmd = sub.add_parser("benchmark", help="Run benchmark dataset")
    bench_cmd.add_argument("--dataset", type=str, default="examples/demo_tasks.json")
    bench_cmd.add_argument("--rounds", type=int, default=4)

    val_cmd = sub.add_parser("rl-validate", help="Run adaptive-vs-frozen RL learning validation")
    val_cmd.add_argument("--episodes", type=int, default=240)
    val_cmd.add_argument("--seeds", type=str, default="11,17,19,23,29,31,37,41,43,47")

    oc_cmd = sub.add_parser("openclaw-event", help="Process OpenClaw-style trajectory event JSON")
    oc_cmd.add_argument("--event-file", type=str, required=True)

    sub.add_parser("state", help="Show branch state and memory stats")

    return parser


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
