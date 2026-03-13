from __future__ import annotations

from prompt_forest.rewards.verifiers import ExternalVerifierReward
from prompt_forest.types import TaskInput


def test_external_verifier_spec_rewards_required_signals():
    verifier = ExternalVerifierReward(weight=1.0)
    task = TaskInput(
        task_id="v1",
        text="hard task",
        task_type="code",
        metadata={
            "verifier_spec": {
                "must_include": ["bug", "test", "confidence"],
                "must_exclude": ["hallucinated"],
                "regex_must_match": [r"confidence=0\\.[0-9]+"],
            }
        },
    )

    strong = "bug fix plan with test strategy and confidence=0.88"
    weak = "generic answer confidence uncertain"

    strong_score, _ = verifier.score(strong, task)
    weak_score, _ = verifier.score(weak, task)

    assert strong_score > weak_score
