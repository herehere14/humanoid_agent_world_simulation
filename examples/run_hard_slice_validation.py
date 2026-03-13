from __future__ import annotations

import json
from pathlib import Path

from prompt_forest.experiments.hard_slice_validation import HardSliceValidator


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    validator = HardSliceValidator(root)
    report = validator.run(seeds=[11, 17, 19, 23, 29, 31, 37, 41], episodes_per_seed=220)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
