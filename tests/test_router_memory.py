from __future__ import annotations

from prompt_forest.config import MemoryConfig
from prompt_forest.memory.store import MemoryStore
from prompt_forest.types import MemoryRecord


def test_memory_bias_reflects_success(tmp_path):
    memory = MemoryStore(MemoryConfig(max_records=100, similarity_window=50), memory_path=tmp_path / "m.jsonl")

    for i in range(5):
        memory.add(
            MemoryRecord(
                task_id=f"t{i}",
                task_type="math",
                input_text="calc",
                activated_branches=["analytical"],
                branch_outputs={"analytical": "ok"},
                selected_branch="analytical",
                selected_output="ok",
                reward_score=0.9,
                failure_reason="",
                confidence=0.8,
                useful_patterns=[],
            )
        )

    bias = memory.branch_success_bias("math")
    assert bias["analytical"] > 0
