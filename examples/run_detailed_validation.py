from __future__ import annotations

import json
from pathlib import Path

from prompt_forest.experiments.detailed_validation import DetailedHierarchicalValidator


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    validator = DetailedHierarchicalValidator(root)
    report = validator.run(seeds=[11, 17, 19, 23, 29], episodes_per_seed=240)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
