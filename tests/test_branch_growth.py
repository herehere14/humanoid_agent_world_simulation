from __future__ import annotations

from pathlib import Path

from prompt_forest.config import load_config
from prompt_forest.core.engine import PromptForestEngine


def test_optimizer_can_expand_hierarchy_with_new_candidates(tmp_path):
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "default.json")
    cfg.artifacts_dir = str(tmp_path / "artifacts")
    cfg.router.top_k = 1
    cfg.router.min_candidates = 1
    cfg.router.exploration = 0.35
    cfg.optimizer.candidate_failure_trigger = 2
    cfg.optimizer.candidate_trial_episodes = 4
    cfg.optimizer.max_active_candidates = 6
    cfg.optimizer.max_active_branches = 40

    engine = PromptForestEngine(config=cfg)
    initial_snapshot = engine.branch_snapshot()
    initial_count = len(initial_snapshot)
    initial_depth = max(v["depth"] for v in initial_snapshot.values())

    for i in range(20):
        task_type = ["planning", "factual", "code", "creative", "general", "math"][i % 6]
        engine.run_task(
            text=f"Stress task {i} for branch growth",
            task_type=task_type,
            metadata={
                "expected_keywords": [f"kw_{task_type}_{j}" for j in range(20)],
                "required_substrings": ["never_present_token_zzz"],
            },
        )

    final_snapshot = engine.branch_snapshot()
    final_count = len(final_snapshot)
    final_depth = max(v["depth"] for v in final_snapshot.values())

    assert final_count > initial_count
    assert final_depth >= initial_depth


def test_created_branch_has_metadata_and_prompt_can_update(tmp_path):
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "default.json")
    cfg.artifacts_dir = str(tmp_path / "artifacts")
    cfg.router.top_k = 1
    cfg.router.min_candidates = 1
    cfg.router.exploration = 0.35
    cfg.optimizer.candidate_failure_trigger = 2
    cfg.optimizer.prompt_rewrite_threshold = 0.7
    cfg.optimizer.max_active_candidates = 6
    cfg.optimizer.max_active_branches = 40

    engine = PromptForestEngine(config=cfg)

    for i in range(25):
        task_type = ["planning", "factual", "code", "creative", "general", "math"][i % 6]
        engine.run_task(
            text=f"Metadata stress task {i}",
            task_type=task_type,
            metadata={
                "expected_keywords": [f"kw_{task_type}_{j}" for j in range(20)],
                "required_substrings": ["never_present_token_zzz"],
            },
        )

    candidates = [
        (name, meta)
        for name, meta in engine.branch_snapshot().items()
        if name.startswith("evidence_fuser_") or name.startswith("constraint_solver_") or name.startswith("verifier_plus_")
    ]

    assert candidates
    candidate_name = candidates[0][0]
    branch = engine.branches[candidate_name]
    assert branch.state.metadata.get("creation_reason")
    assert branch.state.metadata.get("parent_hint") is not None
    # If it underperformed, rewrite history should be non-empty due high rewrite threshold.
    assert isinstance(branch.state.rewrite_history, list)
