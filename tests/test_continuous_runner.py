from __future__ import annotations

from pathlib import Path

from prompt_forest.experiments.continuous_runner import ContinuousImprover


def test_continuous_runner_smoke_cycle():
    root = Path(__file__).resolve().parents[1]
    runner = ContinuousImprover(root)
    report = runner.run(
        max_cycles=1,
        rounds_per_cycle=1,
        candidates_per_round=2,
        seeds=[3],
        episodes_per_seed=40,
        final_eval_seeds=[3],
        final_eval_episodes=50,
        run_tests_each_cycle=False,
        sleep_seconds=0,
        early_stop_patience=1,
        apply_best=False,
    )

    assert report["cycles_run"] == 1
    assert isinstance(report["cycle_results"], list)
    assert report["cycle_results"]
