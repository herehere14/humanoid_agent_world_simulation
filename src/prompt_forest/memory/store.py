from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..config import MemoryConfig
from ..types import MemoryRecord
from ..utils.io import append_jsonl, read_json, read_jsonl, write_json, write_jsonl


class MemoryStore:
    def __init__(self, config: MemoryConfig, memory_path: str | Path | None = None) -> None:
        self.config = config
        self._records: list[MemoryRecord] = []
        self.memory_path = Path(memory_path) if memory_path else None
        self.user_profiles_path = self.memory_path.parent / "user_profiles.json" if self.memory_path else None
        self._user_profiles: dict[str, dict[str, Any]] = {}
        if self.memory_path and self.memory_path.exists():
            self._load()
        if self.user_profiles_path and self.user_profiles_path.exists():
            self._user_profiles = read_json(self.user_profiles_path)

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

    def retrieve_similar(self, task_type: str, limit: int | None = None, user_id: str | None = None) -> list[MemoryRecord]:
        n = limit or self.config.similarity_window
        out = [r for r in reversed(self._records) if r.task_type == task_type and (not user_id or r.user_id == user_id)]
        return list(reversed(out[:n]))

    def branch_success_bias(self, task_type: str, user_id: str | None = None) -> dict[str, float]:
        global_records = self.retrieve_similar(task_type, limit=self.config.similarity_window)
        user_records = self.retrieve_similar(task_type, limit=self.config.similarity_window, user_id=user_id) if user_id else []
        if not global_records and not user_records:
            return {}

        global_bias = self._compute_bias(global_records)
        if not user_records:
            return global_bias

        user_bias = self._compute_bias(user_records)
        user_mix = max(0.0, min(1.0, self.config.user_bias_mix))
        out = dict(global_bias)
        all_branches = set(global_bias) | set(user_bias)
        for branch in all_branches:
            gb = global_bias.get(branch, 0.0)
            ub = user_bias.get(branch, 0.0)
            out[branch] = ((1.0 - user_mix) * gb) + (user_mix * ub)
        return out

    def _compute_bias(self, records: list[MemoryRecord]) -> dict[str, float]:
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
        fails: list[str] = []
        for record in self.recent(200):
            if not record.failure_reason:
                continue
            reason = record.failure_reason.lower()
            if record.reward_score < 0.45 or "rule_miss" in reason or "low_quality" in reason:
                fails.append(self._canonical_failure_reason(record.failure_reason))
        counts = Counter(fails)
        return {k: v for k, v in counts.items() if v >= min_count}

    @staticmethod
    def _canonical_failure_reason(reason: str) -> str:
        reason_l = reason.lower()
        parts: list[str] = []
        if "rule_miss" in reason_l:
            parts.append("rule_miss")
        if "keyword_coverage" in reason_l:
            parts.append("keyword_coverage")
        if "low_quality" in reason_l:
            parts.append("low_quality")
        if "medium_quality" in reason_l:
            parts.append("medium_quality")
        if not parts:
            token = reason_l.split("|", 1)[0].strip()
            return token or "unknown_failure"
        return "|".join(parts)

    def branch_failure_counts(self, branch_name: str, threshold: float = 0.45) -> int:
        count = 0
        for r in self._records:
            if branch_name in r.activated_branches and r.reward_score < threshold:
                count += 1
        return count

    def branch_visit_counts(self, task_type: str, user_id: str | None = None) -> dict[str, int]:
        counts: Counter[str] = Counter()
        records = self.retrieve_similar(task_type, limit=self.config.similarity_window, user_id=user_id)
        if not records:
            records = self.retrieve_similar(task_type, limit=self.config.similarity_window, user_id=None)
        for r in records:
            for branch_name in r.activated_branches:
                counts[branch_name] += 1
        return dict(counts)

    def branch_bandit_stats(self, task_type: str, user_id: str | None = None) -> dict[str, dict[str, float]]:
        global_records = self.retrieve_similar(task_type, limit=self.config.similarity_window, user_id=None)
        user_records = self.retrieve_similar(task_type, limit=self.config.similarity_window, user_id=user_id) if user_id else []
        if not global_records and not user_records:
            return {}

        global_stats = self._compute_bandit_stats(global_records)
        if not user_records:
            return global_stats

        user_stats = self._compute_bandit_stats(user_records)
        mix = max(0.0, min(1.0, self.config.user_bias_mix))
        out: dict[str, dict[str, float]] = {}
        all_branches = set(global_stats) | set(user_stats)
        for branch_name in all_branches:
            gs = global_stats.get(branch_name, {"mean_reward": 0.5, "count": 0.0})
            us = user_stats.get(branch_name, {"mean_reward": 0.5, "count": 0.0})
            mean_reward = ((1.0 - mix) * gs["mean_reward"]) + (mix * us["mean_reward"])
            count = ((1.0 - mix) * gs["count"]) + (mix * us["count"])
            if count <= 0.0:
                continue
            out[branch_name] = {"mean_reward": mean_reward, "count": count}
        return out

    def _compute_bandit_stats(self, records: list[MemoryRecord]) -> dict[str, dict[str, float]]:
        if not records:
            return {}

        reward_sum: dict[str, float] = defaultdict(float)
        weight_sum: dict[str, float] = defaultdict(float)

        n_records = len(records)
        for idx, record in enumerate(records):
            age = n_records - 1 - idx
            recency_weight = self.config.recency_decay**age
            if record.branch_rewards:
                for branch_name, reward in record.branch_rewards.items():
                    reward_sum[branch_name] += reward * recency_weight
                    weight_sum[branch_name] += recency_weight
            else:
                for branch_name in record.activated_branches:
                    reward_sum[branch_name] += record.reward_score * recency_weight
                    weight_sum[branch_name] += recency_weight

        out: dict[str, dict[str, float]] = {}
        for branch_name, total in reward_sum.items():
            count = weight_sum[branch_name]
            if count <= 0.0:
                continue
            out[branch_name] = {"mean_reward": total / count, "count": count}
        return out

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

    def persist_records(self) -> None:
        if self.memory_path:
            write_jsonl(self.memory_path, [r.to_dict() for r in self._records[-self.config.max_records :]])

    def update_feedback(
        self,
        task_id: str,
        feedback_score: float,
        accepted: bool | None = None,
        corrected_answer: str = "",
        feedback_text: str = "",
        user_id: str | None = None,
    ) -> MemoryRecord | None:
        target: MemoryRecord | None = None
        for record in reversed(self._records):
            if record.task_id != task_id:
                continue
            target = record
            break
        if target is None:
            return None

        score = self._normalize_feedback_score(feedback_score)
        target.feedback_score = score
        if accepted is not None:
            target.accepted = accepted
        if corrected_answer:
            target.corrected_answer = corrected_answer
        if feedback_text:
            target.feedback_text = feedback_text
        if user_id:
            target.user_id = user_id

        if self.memory_path:
            write_jsonl(self.memory_path, [r.to_dict() for r in self._records[-self.config.max_records :]])
        return target

    @staticmethod
    def _normalize_feedback_score(score: float) -> float:
        if score > 1.0 and score <= 5.0:
            score = score / 5.0
        return max(0.0, min(1.0, float(score)))

    def branch_expected_reward(
        self,
        branch_name: str,
        task_type: str,
        user_id: str | None = None,
        limit: int = 12,
    ) -> float | None:
        records = self.retrieve_similar(task_type, limit=limit, user_id=user_id)
        if not records and user_id:
            records = self.retrieve_similar(task_type, limit=limit, user_id=None)
        if not records:
            return None
        vals: list[float] = []
        n_records = len(records)
        for idx, record in enumerate(records):
            age = n_records - 1 - idx
            recency_weight = self.config.recency_decay**age
            if branch_name in record.branch_rewards:
                vals.append(record.branch_rewards[branch_name] * recency_weight)
            elif branch_name in record.activated_branches:
                vals.append(record.reward_score * recency_weight)
        if not vals:
            return None
        return sum(vals) / len(vals)

    def branch_reward_moments(
        self,
        branch_name: str,
        task_type: str,
        user_id: str | None = None,
        limit: int = 24,
    ) -> dict[str, float]:
        records = self.retrieve_similar(task_type, limit=limit, user_id=user_id)
        if not records and user_id:
            records = self.retrieve_similar(task_type, limit=limit, user_id=None)
        if not records:
            return {"count": 0.0, "mean": 0.5, "variance": 0.0}

        weighted_sum = 0.0
        weight_sum = 0.0
        weighted_sq_sum = 0.0

        n_records = len(records)
        for idx, record in enumerate(records):
            reward: float | None = None
            if branch_name in record.branch_rewards:
                reward = float(record.branch_rewards[branch_name])
            elif branch_name in record.activated_branches:
                reward = float(record.reward_score)
            if reward is None:
                continue

            age = n_records - 1 - idx
            recency_weight = self.config.recency_decay**age
            weighted_sum += reward * recency_weight
            weighted_sq_sum += (reward**2) * recency_weight
            weight_sum += recency_weight

        if weight_sum <= 0.0:
            return {"count": 0.0, "mean": 0.5, "variance": 0.0}

        mean_reward = weighted_sum / weight_sum
        second_moment = weighted_sq_sum / weight_sum
        variance = max(0.0, second_moment - (mean_reward**2))
        return {
            "count": weight_sum,
            "mean": mean_reward,
            "variance": variance,
        }

    def get_user_profile(self, user_id: str | None) -> dict[str, Any]:
        if not user_id:
            return {}
        return dict(self._user_profiles.get(user_id, {}))

    def upsert_user_profile(
        self,
        user_id: str,
        *,
        style: str | None = None,
        verbosity: str | None = None,
        domain_preferences: list[str] | None = None,
        hard_constraints: list[str] | None = None,
    ) -> dict[str, Any]:
        profile = dict(self._user_profiles.get(user_id, {}))
        if style is not None:
            profile["style"] = style
        if verbosity is not None:
            profile["verbosity"] = verbosity
        if domain_preferences is not None:
            profile["domain_preferences"] = list(domain_preferences)
        if hard_constraints is not None:
            profile["hard_constraints"] = list(hard_constraints)
        self._user_profiles[user_id] = profile
        if self.user_profiles_path:
            write_json(self.user_profiles_path, self._user_profiles)
        return profile
