from __future__ import annotations

import json
from pathlib import Path

from prompt_forest.experiments.continuous_runner import ContinuousImprover


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    runner = ContinuousImprover(root)
    report = runner.run(
        max_cycles=4,
        rounds_per_cycle=2,
        candidates_per_round=6,
        seeds=[11, 17, 19, 23, 29, 31, 37, 41],
        episodes_per_seed=180,
        final_eval_seeds=[11, 17, 19, 23, 29, 31, 37, 41],
        final_eval_episodes=220,
        run_tests_each_cycle=True,
        sleep_seconds=0,
        early_stop_patience=2,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
