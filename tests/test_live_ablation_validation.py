from __future__ import annotations

from pathlib import Path

from prompt_forest.experiments.live_ablation_validation import LiveAblationValidator


def test_policy_definition_flags():
    root = Path(__file__).resolve().parents[1]
    validator = LiveAblationValidator(root)

    frozen = validator._policy_definition("frozen")  # noqa: SLF001
    memory_only = validator._policy_definition("memory_only")  # noqa: SLF001
    weight_only = validator._policy_definition("weight_only")  # noqa: SLF001
    full = validator._policy_definition("full_adaptive")  # noqa: SLF001

    assert (frozen.adapt_train, frozen.update_memory_train) == (False, False)
    assert (memory_only.adapt_train, memory_only.update_memory_train) == (False, True)
    assert (weight_only.adapt_train, weight_only.update_memory_train) == (True, False)
    assert (full.adapt_train, full.update_memory_train) == (True, True)


def test_ablation_comparisons_include_delta_matrix_and_vs_frozen():
    root = Path(__file__).resolve().parents[1]
    validator = LiveAblationValidator(root)
    policies = [
        validator._policy_definition("full_adaptive"),  # noqa: SLF001
        validator._policy_definition("frozen"),  # noqa: SLF001
        validator._policy_definition("memory_only"),  # noqa: SLF001
        validator._policy_definition("weight_only"),  # noqa: SLF001
    ]
    aggregate = {
        "holdout": {
            "full_adaptive": {"mean_objective_reward": 0.7, "contract_pass_rate": 1.0},
            "frozen": {"mean_objective_reward": 0.69, "contract_pass_rate": 1.0},
            "memory_only": {"mean_objective_reward": 0.68, "contract_pass_rate": 1.0},
            "weight_only": {"mean_objective_reward": 0.66, "contract_pass_rate": 0.75},
        }
    }
    comparisons = validator._build_ablation_comparisons(policies, aggregate, [])  # noqa: SLF001

    assert comparisons["holdout_ranking"][0]["policy"] == "full_adaptive"
    assert comparisons["objective_delta_matrix"]["full_adaptive"]["frozen"] == 0.01
    assert comparisons["versus_frozen"]["memory_only"]["objective_reward_gain"] == -0.01
    assert comparisons["versus_frozen"]["weight_only"]["contract_pass_delta"] == -0.25
