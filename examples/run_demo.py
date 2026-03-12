from __future__ import annotations

import json
from pathlib import Path
import shutil

from prompt_forest.config import load_config
from prompt_forest.core.engine import PromptForestEngine
from prompt_forest.experiments.benchmark import BenchmarkRunner


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    config_path = root / "configs" / "default.json"
    dataset = root / "examples" / "demo_tasks.json"

    cfg = load_config(config_path)
    sandbox_dir = root / "artifacts" / "demo_sandbox"
    if sandbox_dir.exists():
        shutil.rmtree(sandbox_dir)
    cfg.artifacts_dir = str(sandbox_dir)
    engine = PromptForestEngine(config=cfg)
    runner = BenchmarkRunner(engine)
    summary = runner.run(dataset_path=dataset, rounds=6)

    output = {
        "summary": summary.to_dict(),
        "final_branches": engine.branch_snapshot(),
        "memory": engine.memory.stats(),
    }

    report = root / "artifacts" / "demo_report.json"
    report.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
