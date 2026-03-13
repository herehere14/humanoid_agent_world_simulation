from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..config import MemoryConfig
from ..types import MemoryRecord
from ..utils.io import append_jsonl, read_jsonl


class MemoryStore:
    def __init__(self, config: MemoryConfig, memory_path: str | Path | None = None) -> None:
        self.config = config
        self._records: list[MemoryRecord] = []
        self.memory_path = Path(memory_path) if memory_path else None
        if self.memory_path and self.memory_path.exists():
            self._load()

    def add(self, record: MemoryRecord) -> None:
        self._records.append(record)
        if len(self._records) > self.config.max_records:
            self._records = self._records[-self.config.max_records :]

        if self.memory_path:
            append_jsonl(self.memory_path, record.to_dict())

    def _load(self) -> None:
        assert self.memory_path is not None
        raw = read_jsonl(self.memory_path)
        self._records = [MemoryRecord(**item) for item in raw][-self.config.max_records :]

    def recent(self, limit: int = 20) -> list[MemoryRecord]:
        return self._records[-limit:]

    def retrieve_similar(self, task_type: str, limit: int | None = None) -> list[MemoryRecord]:
        n = limit or self.config.similarity_window
        out = [r for r in reversed(self._records) if r.task_type == task_type]
        return list(reversed(out[:n]))

    def branch_success_bias(self, task_type: str) -> dict[str, float]:
        records = self.retrieve_similar(task_type, limit=self.config.similarity_window)
        if not records:
            return {}

        weighted_sum: dict[str, float] = defaultdict(float)
        weight_sum: dict[str, float] = defaultdict(float)
        sample_count: dict[str, int] = defaultdict(int)

        n_records = len(records)
        for idx, record in enumerate(records):
            age = n_records - 1 - idx
            recency_weight = self.config.recency_decay**age
            if record.branch_rewards:
                for branch, reward in record.branch_rewards.items():
                    weighted_sum[branch] += reward * recency_weight
                    weight_sum[branch] += recency_weight
                    sample_count[branch] += 1
            else:
                for branch in record.activated_branches:
                    weighted_sum[branch] += record.reward_score * recency_weight
                    weight_sum[branch] += recency_weight
                    sample_count[branch] += 1

        bias: dict[str, float] = {}
        for branch, total in weighted_sum.items():
            avg = total / max(1e-8, weight_sum[branch])
            raw_bias = (avg - 0.5) * self.config.bias_scale

            # Conservative shrinkage keeps low-sample memory from dominating routing.
            n = sample_count[branch]
            shrink = n / (n + self.config.shrinkage_k)
            shrunk = raw_bias * shrink

            capped = max(-self.config.bias_cap, min(self.config.bias_cap, shrunk))
            bias[branch] = capped
        return bias

    def repeated_failures(self, min_count: int = 3) -> dict[str, int]:
        fails = [r.failure_reason for r in self.recent(200) if r.reward_score < 0.45 and r.failure_reason]
        counts = Counter(fails)
        return {k: v for k, v in counts.items() if v >= min_count}

    def branch_failure_counts(self, branch_name: str, threshold: float = 0.45) -> int:
        count = 0
        for r in self._records:
            if branch_name in r.activated_branches and r.reward_score < threshold:
                count += 1
        return count

    def branch_visit_counts(self, task_type: str) -> dict[str, int]:
        counts: Counter[str] = Counter()
        for r in self.retrieve_similar(task_type, limit=self.config.similarity_window):
            for branch_name in r.activated_branches:
                counts[branch_name] += 1
        return dict(counts)

    def useful_patterns(self, task_type: str) -> list[str]:
        records = self.retrieve_similar(task_type, limit=50)
        patterns: Counter[str] = Counter()
        for r in records:
            if r.branch_rewards:
                for b, reward in r.branch_rewards.items():
                    if reward >= 0.7:
                        patterns[f"task={task_type}|branch={b}"] += 1
            elif r.reward_score >= 0.7:
                for b in r.activated_branches:
                    patterns[f"task={task_type}|branch={b}"] += 1
        return [p for p, _ in patterns.most_common(5)]

    def stats(self) -> dict[str, Any]:
        if not self._records:
            return {"records": 0, "avg_reward": 0.0}
        avg_reward = sum(r.reward_score for r in self._records) / len(self._records)
        task_counts = Counter(r.task_type for r in self._records)
        return {
            "records": len(self._records),
            "avg_reward": round(avg_reward, 4),
            "task_distribution": dict(task_counts),
        }

    def dump_records(self) -> list[dict[str, Any]]:
        return [asdict(r) for r in self._records]
