from __future__ import annotations

import json
from pathlib import Path

from prompt_forest.experiments.auto_improve import AutoImprover


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    improver = AutoImprover(root)
    report = improver.run(
        rounds=3,
        candidates_per_round=10,
        seeds=[3, 5, 7, 11],
        episodes_per_seed=140,
        final_eval_seeds=[11, 17, 19, 23, 29, 31, 37, 41],
        final_eval_episodes=220,
        apply_best=True,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
