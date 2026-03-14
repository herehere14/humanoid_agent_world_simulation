from __future__ import annotations

import json
from pathlib import Path

from prompt_forest.experiments.live_model_validation import LiveModelValidator


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    validator = LiveModelValidator(root)
    report = validator.run(
        dataset_path=root / "examples" / "live_eval_tasks.json",
        model="gpt-4.1-mini",
        judge_model="gpt-4.1-mini",
        api_key_env="OPENAI_API_KEY",
        train_rounds=2,
        use_agent_runtimes=True,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
