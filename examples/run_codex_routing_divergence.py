from __future__ import annotations

import json
from pathlib import Path

from prompt_forest.experiments.codex_routing_divergence_validation import CodexRoutingDivergenceValidator


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    validator = CodexRoutingDivergenceValidator(root)
    report = validator.run(
        dataset_path=root / "examples" / "codex_routing_divergence_tasks.json",
    )
    print(json.dumps(report["report_paths"], indent=2))


if __name__ == "__main__":
    main()
