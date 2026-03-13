from __future__ import annotations

from prompt_forest.contracts import evaluate_output_contract, infer_output_contract
from prompt_forest.evaluator.judge import OutputJudge
from prompt_forest.types import TaskInput


def test_infer_output_contract_from_prompt_text():
    assert infer_output_contract("Respond ONLY as minified JSON with keys result and confidence.") == "json_lock"
    assert infer_output_contract("Output ONLY CSV lines in order, no header.") == "csv_lock"
    assert infer_output_contract("Return exactly: FIX: ... TESTS: ...") == "code_patch_lock"
    assert infer_output_contract("Give exactly 4 bullet points. Output bullets only.") == "bullet_lock"


def test_json_contract_rejects_extra_text():
    passed, reason = evaluate_output_contract('{"a":1}\\nextra', "json_lock")
    assert passed is False
    assert reason == "not_pure_json_object"


def test_judge_applies_contract_rejection_gate():
    judge = OutputJudge("hybrid_verifier")
    task = TaskInput(
        task_id="1",
        text="Respond ONLY as minified JSON with keys answer and confidence.",
        task_type="general",
        metadata={
            "expected_keywords": ["answer", "confidence"],
            "required_substrings": ["answer", "confidence"],
        },
    )
    score = judge.score_output("answer: 42; confidence: 0.9", task)
    assert score.reward == 0.0
    assert "contract_reject:json_lock" in score.reason
