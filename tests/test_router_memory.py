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


def test_memory_bias_is_capped_and_shrunk_for_small_samples(tmp_path):
    memory = MemoryStore(
        MemoryConfig(max_records=100, similarity_window=50, bias_scale=0.6, bias_cap=0.15, shrinkage_k=20.0),
        memory_path=tmp_path / "m.jsonl",
    )
    memory.add(
        MemoryRecord(
            task_id="one",
            task_type="math",
            input_text="calc",
            activated_branches=["verification"],
            branch_outputs={"verification": "ok"},
            selected_branch="verification",
            selected_output="ok",
            reward_score=1.0,
            failure_reason="",
            confidence=0.8,
            useful_patterns=[],
        )
    )

    bias = memory.branch_success_bias("math")
    assert 0.0 < bias["verification"] < 0.15


def test_memory_recency_decay_prioritizes_recent_records(tmp_path):
    memory = MemoryStore(
        MemoryConfig(max_records=100, similarity_window=50, bias_scale=0.6, recency_decay=0.9),
        memory_path=tmp_path / "m.jsonl",
    )

    for i in range(8):
        memory.add(
            MemoryRecord(
                task_id=f"old-{i}",
                task_type="planning",
                input_text="plan",
                activated_branches=["planner"],
                branch_outputs={"planner": "ok"},
                selected_branch="planner",
                selected_output="ok",
                reward_score=0.95,
                failure_reason="",
                confidence=0.8,
                useful_patterns=[],
            )
        )
    for i in range(2):
        memory.add(
            MemoryRecord(
                task_id=f"recent-{i}",
                task_type="planning",
                input_text="plan",
                activated_branches=["planner"],
                branch_outputs={"planner": "ok"},
                selected_branch="planner",
                selected_output="ok",
                reward_score=0.1,
                failure_reason="",
                confidence=0.8,
                useful_patterns=[],
            )
        )

    bias = memory.branch_success_bias("planning")
    # Un-decayed average would be strongly positive; recency decay should pull it downward.
    assert bias["planner"] < 0.1


def test_memory_bandit_stats_track_branch_quality(tmp_path):
    memory = MemoryStore(
        MemoryConfig(max_records=100, similarity_window=50, recency_decay=0.95),
        memory_path=tmp_path / "m.jsonl",
    )

    for i in range(6):
        memory.add(
            MemoryRecord(
                task_id=f"a-{i}",
                task_type="code",
                input_text="fix bug",
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
    for i in range(6):
        memory.add(
            MemoryRecord(
                task_id=f"r-{i}",
                task_type="code",
                input_text="fix bug",
                activated_branches=["retrieval"],
                branch_outputs={"retrieval": "weak"},
                selected_branch="retrieval",
                selected_output="weak",
                reward_score=0.3,
                failure_reason="",
                confidence=0.8,
                useful_patterns=[],
            )
        )

    stats = memory.branch_bandit_stats("code")
    assert "analytical" in stats
    assert "retrieval" in stats
    assert stats["analytical"]["count"] > 0.0
    assert stats["retrieval"]["count"] > 0.0
    assert stats["analytical"]["mean_reward"] > stats["retrieval"]["mean_reward"]


def test_branch_reward_moments_capture_variance_and_mean(tmp_path):
    memory = MemoryStore(
        MemoryConfig(max_records=100, similarity_window=50, recency_decay=1.0),
        memory_path=tmp_path / "m.jsonl",
    )

    rewards = [0.9, 0.1, 0.8, 0.2]
    for i, reward in enumerate(rewards):
        memory.add(
            MemoryRecord(
                task_id=f"v-{i}",
                task_type="math",
                input_text="x",
                activated_branches=["analytical"],
                branch_outputs={"analytical": "x"},
                selected_branch="analytical",
                selected_output="x",
                reward_score=reward,
                failure_reason="",
                confidence=0.7,
                useful_patterns=[],
            )
        )

    moments = memory.branch_reward_moments("analytical", "math", limit=20)
    assert moments["count"] > 0.0
    assert 0.45 <= moments["mean"] <= 0.55
    assert moments["variance"] > 0.05
