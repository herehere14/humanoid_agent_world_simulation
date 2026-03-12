from __future__ import annotations

import json
from pathlib import Path

from prompt_forest.config import load_config
from prompt_forest.core.engine import PromptForestEngine


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs" / "default.json")
    cfg.artifacts_dir = str(root / "artifacts" / "candidate_sandbox")
    cfg.optimizer.candidate_failure_trigger = 2
    cfg.router.top_k = 2
    engine = PromptForestEngine(config=cfg)

    created = []
    promoted = []
    archived = []

    metadata = {
        "expected_keywords": ["alpha", "beta", "gamma", "delta"],
        "required_substrings": ["never_present_token_zzz"],
    }

    for i in range(10):
        result = engine.run_task(
            text=f"General reliability task {i}: produce robust answer",
            task_type="general",
            metadata=metadata,
        )
        opt = result["optimization"]
        created.extend(opt["created_candidates"])
        promoted.extend(opt["promoted_candidates"])
        archived.extend(opt["archived_candidates"])

    payload = {
        "created_candidates": created,
        "promoted_candidates": promoted,
        "archived_candidates": archived,
        "final_snapshot": engine.branch_snapshot(),
        "memory": engine.memory.stats(),
    }

    output_path = root / "artifacts" / "candidate_trial_report.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
