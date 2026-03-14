from __future__ import annotations

from pathlib import Path

from prompt_forest.experiments.live_model_validation import LiveModelValidator, PairwiseJudgement


def test_dataset_summary_counts_contract_tasks():
    root = Path(__file__).resolve().parents[1]
    validator = LiveModelValidator(root)
    dataset = validator._dataset_summary(  # noqa: SLF001
        train_specs=[
            {"task_type": "math", "text": "Solve x", "metadata": {}},
            {"task_type": "general", "text": "Respond ONLY as minified JSON", "metadata": {"output_contract": "json_lock"}},
        ],
        holdout_specs=[
            {"task_type": "general", "text": "Output bullets only", "metadata": {"output_contract": "bullet_lock"}},
        ],
        train_rounds=2,
    )

    assert dataset["expanded_train_tasks"] == 4
    assert dataset["train_contract_tasks_per_round"] == 1
    assert dataset["holdout_contract_tasks"] == 1


def test_pairwise_summary_counts_wins_and_ties():
    root = Path(__file__).resolve().parents[1]
    validator = LiveModelValidator(root)
    summary = validator._pairwise_summary(  # noqa: SLF001
        [
            PairwiseJudgement("t1", "holdout", "math", "adaptive_full", "frozen_forest", "adaptive_full", 8.0, 6.0, "", {}),
            PairwiseJudgement("t2", "holdout", "math", "adaptive_full", "frozen_forest", "tie", 7.0, 7.0, "", {}),
            PairwiseJudgement("t3", "holdout", "code", "adaptive_full", "frozen_forest", "frozen_forest", 5.0, 8.0, "", {}),
        ]
    )

    item = summary["adaptive_full__vs__frozen_forest"]
    assert item["n"] == 3
    assert item["left_wins"] == 1
    assert item["right_wins"] == 1
    assert item["ties"] == 1


def test_judge_backend_options_follow_generation_for_same_model():
    api_mode, reasoning = LiveModelValidator._resolve_judge_backend_options(  # noqa: SLF001
        model="gpt-5.3-codex",
        judge_model="gpt-5.3-codex",
        api_mode="responses",
        reasoning_effort="medium",
        judge_api_mode=None,
        judge_reasoning_effort=None,
    )

    assert api_mode == "responses"
    assert reasoning == "medium"


def test_judge_backend_options_default_to_chat_for_different_model():
    api_mode, reasoning = LiveModelValidator._resolve_judge_backend_options(  # noqa: SLF001
        model="gpt-5.3-codex",
        judge_model="gpt-4.1-mini",
        api_mode="responses",
        reasoning_effort="medium",
        judge_api_mode=None,
        judge_reasoning_effort=None,
    )

    assert api_mode == "chat_completions"
    assert reasoning is None
