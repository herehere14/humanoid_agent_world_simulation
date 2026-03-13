from __future__ import annotations

from prompt_forest.config import EngineConfig
from prompt_forest.core.engine import PromptForestEngine


def test_router_activates_multiple_paths_when_top_k_gt_1():
    cfg = EngineConfig()
    cfg.router.top_k = 3
    cfg.router.min_candidates = 2
    cfg.router.exploration = 0.0
    cfg.composer.enabled = False

    engine = PromptForestEngine(config=cfg)
    result = engine.run_task_controlled(
        text="Output ONLY CSV lines in the same order: A,fast B,slow",
        task_type="general",
        metadata={"expected_keywords": ["A,fast", "B,slow"], "required_substrings": ["A,fast"]},
        adapt=False,
        update_memory=False,
    )

    paths = result["routing"].get("activated_paths", [])
    assert len(paths) >= 2
    assert all(len(path) >= 2 for path in paths)


def test_strict_tasks_limit_context_roll_size():
    short = PromptForestEngine._roll_context("", "x" * 600, strict_mode=True)
    long = PromptForestEngine._roll_context("", "x" * 600, strict_mode=False)
    assert len(short) <= 120
    assert len(long) > len(short)


def test_contract_leaf_is_forced_into_activated_paths():
    cfg = EngineConfig()
    cfg.router.top_k = 2
    cfg.router.min_candidates = 2
    cfg.router.exploration = 0.0
    engine = PromptForestEngine(config=cfg)

    result = engine.run_task_controlled(
        text="Output bullets only. Exactly 4 bullets.",
        task_type="general",
        metadata={"output_contract": "bullet_lock"},
        adapt=False,
        update_memory=False,
    )
    leaves = [path[-1] for path in result["routing"]["activated_paths"] if path]
    assert "bullet_lock" in leaves


def test_strict_contract_disables_composer_and_preserves_lock_format():
    cfg = EngineConfig()
    cfg.composer.enabled = True
    engine = PromptForestEngine(config=cfg)

    result = engine.run_task_controlled(
        text="Each bullet must start with '- '. Output bullets only. Exactly 4 bullets.",
        task_type="general",
        metadata={"output_contract": "bullet_lock"},
        adapt=False,
        update_memory=False,
    )
    output = result["evaluation_signal"]["selected_output"].strip()
    lines = [line for line in output.splitlines() if line.strip()]

    assert result["composer"] == {}
    assert len(lines) == 4
    assert all(line.startswith("- ") for line in lines)
