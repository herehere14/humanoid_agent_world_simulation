from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

from ..config import MemoryConfig
from ..contracts import infer_output_contract
from ..types import MemoryRecord, SiblingPreferenceSignal, TaskInput
from ..utils.io import append_jsonl, read_json, read_jsonl, write_json, write_jsonl

_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "briefly",
    "by",
    "can",
    "confidence",
    "create",
    "end",
    "exactly",
    "for",
    "from",
    "give",
    "include",
    "into",
    "line",
    "make",
    "new",
    "next",
    "no",
    "of",
    "on",
    "one",
    "only",
    "or",
    "output",
    "respond",
    "return",
    "short",
    "show",
    "that",
    "the",
    "their",
    "this",
    "today",
    "with",
    "write",
}


@dataclass(frozen=True)
class RoutingContext:
    parent_id: str
    task_type: str
    aspect: str
    contract_hint: str
    lexical_terms: tuple[str, ...]
    pattern_tags: tuple[str, ...]

    def key(self) -> str:
        terms = ",".join(self.lexical_terms) or "none"
        tags = ",".join(self.pattern_tags) or "none"
        aspect = self.aspect or "none"
        contract = self.contract_hint or "none"
        return (
            f"parent={self.parent_id}|task={self.task_type}|aspect={aspect}|contract={contract}|"
            f"terms={terms}|tags={tags}"
        )

    def similarity(self, other: "RoutingContext") -> float:
        if self.parent_id != other.parent_id or self.task_type != other.task_type:
            return 0.0

        if self.contract_hint != other.contract_hint:
            if self.contract_hint or other.contract_hint:
                return 0.0

        score = 0.2
        if self.aspect and self.aspect == other.aspect:
            score += 0.1
        if self.contract_hint and self.contract_hint == other.contract_hint:
            score += 0.2

        score += 0.35 * _jaccard(self.lexical_terms, other.lexical_terms)
        score += 0.15 * _jaccard(self.pattern_tags, other.pattern_tags)
        return min(1.0, score)


@dataclass(frozen=True)
class ExecutionPlaybook:
    support: int
    coverage_items: list[str] = field(default_factory=list)
    structure_cues: list[str] = field(default_factory=list)
    anti_patterns: list[str] = field(default_factory=list)
    success_examples: list[str] = field(default_factory=list)
    guidance: list[str] = field(default_factory=list)
    case_summaries: list[str] = field(default_factory=list)
    pattern_summaries: list[str] = field(default_factory=list)
    recommended_flows: list[str] = field(default_factory=list)


@dataclass
class SiblingPreferenceRecord:
    task_id: str
    task_type: str
    parent_id: str
    child_ids: list[str]
    input_text: str
    task_metadata: dict[str, Any]
    reward_by_child: dict[str, float]
    winning_child: str = ""
    losing_child: str = ""
    margin: float = 0.0
    routing_context_key: str = ""
    user_id: str = "global"
    active: bool = False
    source: str = "probe"
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def pair_key(self) -> tuple[str, str]:
        children = sorted(str(child).strip() for child in self.child_ids if str(child).strip())
        if len(children) < 2:
            return ("", "")
        return children[0], children[1]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class UserProfileStore:
    def __init__(self, path: Path | None) -> None:
        self.path = path
        self._profiles: dict[str, dict[str, Any]] = {}
        if self.path and self.path.exists():
            self._profiles = read_json(self.path)

    def get(self, user_id: str | None) -> dict[str, Any]:
        if not user_id:
            return {}
        return dict(self._profiles.get(user_id, {}))

    def upsert(
        self,
        user_id: str,
        *,
        style: str | None = None,
        verbosity: str | None = None,
        domain_preferences: list[str] | None = None,
        hard_constraints: list[str] | None = None,
    ) -> dict[str, Any]:
        profile = dict(self._profiles.get(user_id, {}))
        if style is not None:
            profile["style"] = style
        if verbosity is not None:
            profile["verbosity"] = verbosity
        if domain_preferences is not None:
            profile["domain_preferences"] = list(domain_preferences)
        if hard_constraints is not None:
            profile["hard_constraints"] = list(hard_constraints)
        self._profiles[user_id] = profile
        if self.path:
            write_json(self.path, self._profiles)
        return profile

    @property
    def count(self) -> int:
        return len(self._profiles)


class MemoryStore:
    def __init__(self, config: MemoryConfig, memory_path: str | Path | None = None) -> None:
        self.config = config
        self._records: list[MemoryRecord] = []
        self._sibling_preferences: list[SiblingPreferenceRecord] = []
        self.memory_path = Path(memory_path) if memory_path else None
        user_profiles_path = self.memory_path.parent / "user_profiles.json" if self.memory_path else None
        self.preference_path = self.memory_path.parent / "sibling_preferences.jsonl" if self.memory_path else None
        self._profiles = UserProfileStore(user_profiles_path)
        if self.memory_path and self.memory_path.exists():
            self._load()
        if self.preference_path and self.preference_path.exists():
            self._load_preferences()

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

    def _load_preferences(self) -> None:
        assert self.preference_path is not None
        raw = read_jsonl(self.preference_path)
        self._sibling_preferences = [SiblingPreferenceRecord(**item) for item in raw][-self.config.max_records :]

    def recent(self, limit: int = 20) -> list[MemoryRecord]:
        return self._records[-limit:]

    def retrieve_similar(self, task_type: str, limit: int | None = None, user_id: str | None = None) -> list[MemoryRecord]:
        n = limit or self.config.similarity_window
        out = [r for r in reversed(self._records) if r.task_type == task_type and (not user_id or r.user_id == user_id)]
        return list(reversed(out[:n]))

    def routing_context_key_for_task(self, task: TaskInput, parent_id: str) -> str:
        return self._build_routing_context(task.text, task.task_type, task.metadata, parent_id).key()

    def sibling_preference_signal(
        self,
        task: TaskInput,
        task_type: str,
        parent_id: str,
        child_ids: list[str],
        user_id: str | None = None,
    ) -> SiblingPreferenceSignal:
        if not parent_id or len(child_ids) < 2:
            return SiblingPreferenceSignal()

        context = self._build_routing_context(task.text, task_type, task.metadata, parent_id)
        if user_id:
            user_signal = self._aggregate_preference_signal(
                context=context,
                parent_id=parent_id,
                child_ids=child_ids,
                user_id=user_id,
                active_only=True,
            )
            if user_signal.support > 0:
                return user_signal
        return self._aggregate_preference_signal(
            context=context,
            parent_id=parent_id,
            child_ids=child_ids,
            user_id=None,
            active_only=True,
        )

    def sibling_routing_scores(
        self,
        task: TaskInput,
        task_type: str,
        parent_id: str,
        child_ids: list[str],
        user_id: str | None = None,
    ) -> dict[str, float]:
        return self.sibling_preference_signal(
            task=task,
            task_type=task_type,
            parent_id=parent_id,
            child_ids=child_ids,
            user_id=user_id,
        ).scores

    def record_sibling_probe(
        self,
        *,
        task: TaskInput,
        parent_id: str,
        reward_by_child: dict[str, float],
        user_id: str | None = None,
        source: str = "probe",
    ) -> SiblingPreferenceSignal:
        if not parent_id or len(reward_by_child) < 2:
            return SiblingPreferenceSignal()

        cleaned_rewards = {
            str(child).strip(): float(reward)
            for child, reward in reward_by_child.items()
            if str(child).strip()
        }
        if len(cleaned_rewards) < 2:
            return SiblingPreferenceSignal()

        context = self._build_routing_context(task.text, task.task_type, task.metadata, parent_id)
        stored_any = False
        for left, right in combinations(sorted(cleaned_rewards), 2):
            reward_left = cleaned_rewards[left]
            reward_right = cleaned_rewards[right]
            diff = reward_left - reward_right
            winning_child = ""
            losing_child = ""
            if abs(diff) >= self.config.routing_pair_tie_margin:
                if diff > 0.0:
                    winning_child, losing_child = left, right
                else:
                    winning_child, losing_child = right, left

            pref = SiblingPreferenceRecord(
                task_id=task.task_id,
                task_type=task.task_type,
                parent_id=parent_id,
                child_ids=[left, right],
                input_text=task.text,
                task_metadata=dict(task.metadata or {}),
                reward_by_child={left: reward_left, right: reward_right},
                winning_child=winning_child,
                losing_child=losing_child,
                margin=abs(diff),
                routing_context_key=context.key(),
                user_id=user_id or str(task.metadata.get("user_id", "global")).strip() or "global",
                active=False,
                source=source,
            )
            self._sibling_preferences.append(pref)
            if len(self._sibling_preferences) > self.config.max_records:
                self._sibling_preferences = self._sibling_preferences[-self.config.max_records :]
            if self.preference_path:
                append_jsonl(self.preference_path, pref.to_dict())
            self._maybe_promote_preference_cluster(
                context=context,
                parent_id=parent_id,
                pair_key=pref.pair_key(),
                user_id=pref.user_id,
            )
            stored_any = True

        if not stored_any:
            return SiblingPreferenceSignal()

        return self.sibling_preference_signal(
            task=task,
            task_type=task.task_type,
            parent_id=parent_id,
            child_ids=list(cleaned_rewards),
            user_id=user_id or str(task.metadata.get("user_id", "global")).strip() or "global",
        )

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
            n = sample_count[branch]
            shrink = n / (n + self.config.shrinkage_k)
            capped = max(-self.config.bias_cap, min(self.config.bias_cap, raw_bias * shrink))
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
        for record in self._records:
            if branch_name in record.activated_branches and record.reward_score < threshold:
                count += 1
        return count

    def branch_visit_counts(self, task_type: str, user_id: str | None = None) -> dict[str, int]:
        counts: Counter[str] = Counter()
        records = self.retrieve_similar(task_type, limit=self.config.similarity_window, user_id=user_id)
        if not records and user_id:
            records = self.retrieve_similar(task_type, limit=self.config.similarity_window, user_id=None)
        for record in records:
            for branch_name in record.activated_branches:
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
        for record in records:
            if record.reward_score < 0.7:
                continue
            branch_name = record.selected_branch or (record.selected_path[-1] if record.selected_path else "")
            if branch_name:
                patterns[f"task={task_type}|leaf={branch_name}"] += 1
        return [pattern for pattern, _ in patterns.most_common(5)]

    def execution_guidance(
        self,
        task: TaskInput,
        branch_name: str,
        *,
        user_id: str | None = None,
        limit: int = 2,
    ) -> list[str]:
        hints = self._execution_guidance(task=task, branch_name=branch_name, user_id=user_id, limit=limit)
        if hints or not user_id:
            return hints
        return self._execution_guidance(task=task, branch_name=branch_name, user_id=None, limit=limit)

    def execution_playbook(
        self,
        task: TaskInput,
        branch_name: str,
        *,
        user_id: str | None = None,
        success_limit: int = 3,
        failure_limit: int = 2,
        min_similarity: float | None = None,
    ) -> ExecutionPlaybook:
        scored = self._scored_execution_records(
            task=task,
            branch_name=branch_name,
            user_id=user_id,
            min_similarity=min_similarity,
        )
        if not scored and user_id:
            scored = self._scored_execution_records(
                task=task,
                branch_name=branch_name,
                user_id=None,
                min_similarity=min_similarity,
            )
        if not scored:
            return ExecutionPlaybook(support=0)

        successes = [record for _, record in scored if record.reward_score >= self.config.execution_hint_min_reward]
        failures = [record for _, record in scored if record.reward_score < self.config.execution_hint_min_reward]
        return self._build_execution_playbook(
            task=task,
            successes=successes[: max(1, success_limit)],
            failures=failures[: max(0, failure_limit)],
        )

    def stats(self) -> dict[str, Any]:
        if not self._records:
            return {
                "records": 0,
                "avg_reward": 0.0,
                "route_contexts": 0,
                "user_profiles": self._profiles.count,
                "active_sibling_preferences": sum(1 for pref in self._sibling_preferences if pref.active),
                "shadow_sibling_preferences": sum(1 for pref in self._sibling_preferences if not pref.active),
            }
        avg_reward = sum(record.reward_score for record in self._records) / len(self._records)
        task_counts = Counter(record.task_type for record in self._records)
        route_contexts = {record.routing_context_key for record in self._records if record.routing_context_key}
        return {
            "records": len(self._records),
            "avg_reward": round(avg_reward, 4),
            "task_distribution": dict(task_counts),
            "route_contexts": len(route_contexts),
            "user_profiles": self._profiles.count,
            "active_sibling_preferences": sum(1 for pref in self._sibling_preferences if pref.active),
            "shadow_sibling_preferences": sum(1 for pref in self._sibling_preferences if not pref.active),
        }

    def dump_records(self) -> list[dict[str, Any]]:
        return [asdict(record) for record in self._records]

    def persist_records(self) -> None:
        if self.memory_path:
            write_jsonl(self.memory_path, [record.to_dict() for record in self._records[-self.config.max_records :]])
        if self.preference_path:
            write_jsonl(
                self.preference_path,
                [record.to_dict() for record in self._sibling_preferences[-self.config.max_records :]],
            )

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
            write_jsonl(self.memory_path, [record.to_dict() for record in self._records[-self.config.max_records :]])
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
        return self._profiles.get(user_id)

    def upsert_user_profile(
        self,
        user_id: str,
        *,
        style: str | None = None,
        verbosity: str | None = None,
        domain_preferences: list[str] | None = None,
        hard_constraints: list[str] | None = None,
    ) -> dict[str, Any]:
        return self._profiles.upsert(
            user_id,
            style=style,
            verbosity=verbosity,
            domain_preferences=domain_preferences,
            hard_constraints=hard_constraints,
        )

    def _aggregate_preference_signal(
        self,
        *,
        context: RoutingContext,
        parent_id: str,
        child_ids: list[str],
        user_id: str | None,
        active_only: bool,
    ) -> SiblingPreferenceSignal:
        child_set = {str(child).strip() for child in child_ids if str(child).strip()}
        if len(child_set) < 2:
            return SiblingPreferenceSignal()

        pair_stats: dict[tuple[str, str], dict[str, Any]] = {}
        recent_prefs = self._sibling_preferences[-self.config.similarity_window :]
        for age, record in enumerate(reversed(recent_prefs)):
            if record.parent_id != parent_id:
                continue
            if active_only and not record.active:
                continue
            if user_id and record.user_id != user_id:
                continue

            pair_key = record.pair_key()
            if not pair_key[0] or not set(pair_key).issubset(child_set):
                continue

            other_context = self._build_routing_context(
                record.input_text,
                record.task_type,
                record.task_metadata,
                parent_id,
            )
            similarity = context.similarity(other_context)
            if similarity < self.config.routing_min_similarity:
                continue

            recency_weight = self.config.recency_decay**age
            weight = max(1e-6, similarity * recency_weight)
            stats = pair_stats.setdefault(
                pair_key,
                {
                    "support": 0,
                    "wins": Counter(),
                    "weighted_support": 0.0,
                    "weighted_margin_sum": 0.0,
                    "weighted_signed_margin_sum": 0.0,
                    "weighted_signed_margin_abs_sum": defaultdict(float),
                    "child_support": Counter(),
                },
            )
            stats["support"] += 1
            stats["weighted_support"] += weight
            stats["weighted_margin_sum"] += record.margin * weight
            for child_id in pair_key:
                stats["child_support"][child_id] += 1

            if record.winning_child and record.losing_child:
                stats["wins"][record.winning_child] += 1
                signed_margin = record.margin if record.winning_child == pair_key[0] else -record.margin
                stats["weighted_signed_margin_sum"] += signed_margin * weight
                stats["weighted_signed_margin_abs_sum"][record.winning_child] += record.margin * weight

        if not pair_stats:
            return SiblingPreferenceSignal()

        scores = defaultdict(float)
        child_support: Counter[str] = Counter()
        child_wins: Counter[str] = Counter()
        child_margin_sum: defaultdict[str, float] = defaultdict(float)
        pair_summaries: dict[str, dict[str, Any]] = {}
        total_support = 0

        for pair_key, stats in pair_stats.items():
            support = int(stats["support"])
            if support < self.config.routing_min_support:
                continue

            total_support += support
            first, second = pair_key
            wins_first = int(stats["wins"].get(first, 0))
            wins_second = int(stats["wins"].get(second, 0))
            weighted_support = float(stats["weighted_support"])
            if weighted_support <= 0.0:
                continue

            signed_margin = float(stats["weighted_signed_margin_sum"]) / weighted_support
            if abs(signed_margin) < self.config.routing_margin:
                continue

            if signed_margin > 0.0:
                preferred, other = first, second
                win_count = wins_first
            else:
                preferred, other = second, first
                win_count = wins_second

            win_rate = win_count / max(1, support)
            expected_margin = abs(signed_margin)
            shrink = support / (support + self.config.routing_shrinkage_k)
            pair_strength = (((win_rate - 0.5) * 2.0) + (expected_margin * 2.0)) * shrink

            scores[preferred] += pair_strength
            scores[other] -= pair_strength
            child_support[preferred] += support
            child_support[other] += support
            child_wins[preferred] += win_count
            child_margin_sum[preferred] += expected_margin * support

            pair_summaries[f"{first}__vs__{second}"] = {
                "pair": [first, second],
                "preferred_child": preferred,
                "support": support,
                "win_rate": round(win_rate, 4),
                "expected_margin": round(expected_margin, 4),
            }

        if not scores:
            return SiblingPreferenceSignal()

        normalized_scores: dict[str, float] = {}
        for child_id, raw in scores.items():
            support = child_support.get(child_id, 0)
            if support <= 0:
                continue
            shrink = support / (support + self.config.routing_shrinkage_k)
            normalized_scores[child_id] = max(-1.0, min(1.0, raw * shrink))

        if not normalized_scores:
            return SiblingPreferenceSignal()

        preferred_child = max(normalized_scores.items(), key=lambda item: item[1])[0]
        preferred_support = int(child_support.get(preferred_child, 0))
        preferred_win_rate = child_wins.get(preferred_child, 0) / max(1, preferred_support)
        preferred_margin = child_margin_sum.get(preferred_child, 0.0) / max(1, preferred_support)
        return SiblingPreferenceSignal(
            scores=normalized_scores,
            preferred_child=preferred_child,
            override_child="",
            support=preferred_support,
            win_rate=preferred_win_rate,
            expected_margin=preferred_margin,
            details={"pairs": pair_summaries, "active_only": active_only, "total_support": total_support},
        )

    def _matching_preference_records(
        self,
        *,
        context: RoutingContext,
        parent_id: str,
        pair_key: tuple[str, str],
        user_id: str | None,
        active: bool | None,
    ) -> list[SiblingPreferenceRecord]:
        out: list[SiblingPreferenceRecord] = []
        recent_prefs = self._sibling_preferences[-self.config.similarity_window :]
        for record in recent_prefs:
            if record.parent_id != parent_id:
                continue
            if pair_key[0] and record.pair_key() != pair_key:
                continue
            if user_id and record.user_id != user_id:
                continue
            if active is not None and bool(record.active) != bool(active):
                continue

            other_context = self._build_routing_context(
                record.input_text,
                record.task_type,
                record.task_metadata,
                parent_id,
            )
            if context.similarity(other_context) < self.config.routing_min_similarity:
                continue
            out.append(record)
        return out

    def _maybe_promote_preference_cluster(
        self,
        *,
        context: RoutingContext,
        parent_id: str,
        pair_key: tuple[str, str],
        user_id: str | None,
    ) -> None:
        if not pair_key[0]:
            return

        cluster = self._matching_preference_records(
            context=context,
            parent_id=parent_id,
            pair_key=pair_key,
            user_id=user_id,
            active=False,
        )
        if len(cluster) < max(self.config.routing_promotion_min_support, self.config.routing_pair_replay_min_samples):
            return

        wins = Counter(pref.winning_child for pref in cluster if pref.winning_child)
        if not wins:
            return

        preferred_child, preferred_wins = wins.most_common(1)[0]
        support = len(cluster)
        win_rate = preferred_wins / max(1, support)
        preferred_margins = [pref.margin for pref in cluster if pref.winning_child == preferred_child]
        expected_margin = sum(preferred_margins) / max(1, len(preferred_margins))

        if (
            win_rate < self.config.routing_promotion_min_win_rate
            or expected_margin < self.config.routing_promotion_min_margin
        ):
            return
        if not self._preference_replay_accepts(
            context=context,
            parent_id=parent_id,
            preferred_child=preferred_child,
            other_child=pair_key[1] if pair_key[0] == preferred_child else pair_key[0],
            user_id=user_id,
        ):
            return

        updated = False
        for pref in cluster:
            if pref.active:
                continue
            pref.active = True
            updated = True
        if updated and self.preference_path:
            write_jsonl(
                self.preference_path,
                [record.to_dict() for record in self._sibling_preferences[-self.config.max_records :]],
            )

    def _preference_replay_accepts(
        self,
        *,
        context: RoutingContext,
        parent_id: str,
        preferred_child: str,
        other_child: str,
        user_id: str | None,
    ) -> bool:
        if not preferred_child or not other_child:
            return False

        matched = 0
        preferred_wins = 0
        signed_margin_sum = 0.0
        recent_records = self._records[-self.config.routing_pair_replay_window :]
        for record in reversed(recent_records):
            if user_id and record.user_id != user_id:
                continue
            if preferred_child not in record.branch_rewards or other_child not in record.branch_rewards:
                continue

            record_leaf = self._selected_leaf_under_parent(record, parent_id)
            if record_leaf not in {preferred_child, other_child}:
                continue

            other_context = self._build_routing_context(
                record.input_text,
                record.task_type,
                record.task_metadata,
                parent_id,
            )
            if context.similarity(other_context) < self.config.routing_min_similarity:
                continue

            matched += 1
            preferred_reward = float(record.branch_rewards.get(preferred_child, 0.0))
            other_reward = float(record.branch_rewards.get(other_child, 0.0))
            diff = preferred_reward - other_reward
            signed_margin_sum += diff
            if diff >= self.config.routing_pair_tie_margin:
                preferred_wins += 1

        if matched < self.config.routing_pair_replay_min_samples:
            return False

        win_rate = preferred_wins / max(1, matched)
        expected_margin = signed_margin_sum / max(1, matched)
        return (
            win_rate >= self.config.routing_promotion_min_win_rate
            and expected_margin >= self.config.routing_promotion_min_margin
        )

    def _execution_guidance(
        self,
        *,
        task: TaskInput,
        branch_name: str,
        user_id: str | None,
        limit: int,
    ) -> list[str]:
        scored = self._scored_execution_records(task=task, branch_name=branch_name, user_id=user_id)

        if not scored:
            return []

        hints: list[str] = []
        seen: set[str] = set()
        for _, record in scored:
            if record.reward_score < self.config.execution_hint_min_reward:
                continue
            hint = self._summarize_execution_record(record)
            if not hint or hint in seen:
                continue
            seen.add(hint)
            hints.append(hint)
            if len(hints) >= max(1, limit):
                break
        return hints

    @staticmethod
    def _summarize_execution_record(record: MemoryRecord) -> str:
        output = record.selected_output
        output_l = output.lower()
        directives: list[str] = []
        required = [str(x).strip() for x in record.task_metadata.get("required_substrings", []) if str(x).strip()]
        if required:
            directives.append(f"explicitly include {', '.join(required[:4])}")
        if output.count("|") >= 8:
            directives.append("use a compact table for structured fields")
        if "##" in output or "#" in output:
            directives.append("organize with short section headings")
        if re.search(r"\b(phase|week|day|timeline)\b", output_l):
            directives.append("sequence the answer by time blocks")
        if "correctness" in output_l:
            directives.append("make correctness risks explicit")
        if "test" in output_l or "validation" in output_l:
            directives.append("call out tests or validation gaps")
        if "tradeoff" in output_l or "tradeoffs" in output_l:
            directives.append("spell out tradeoffs explicitly")
        if "recommendation" in output_l:
            directives.append("finish with a direct recommendation")
        if "owner" in output_l:
            directives.append("assign explicit owners")
        if "risk" in output_l:
            directives.append("call out risks directly")
        if "rollback" in output_l or "fallback" in output_l:
            directives.append("state rollback or fallback")
        if "confidence" in output_l:
            directives.append("end with calibrated confidence")

        excerpt = MemoryStore._compact_output_excerpt(output)
        if excerpt:
            directives.append(f'style cue: "{excerpt}"')
        return ". ".join(directives[:5])

    def _build_execution_playbook(
        self,
        *,
        task: TaskInput,
        successes: list[MemoryRecord],
        failures: list[MemoryRecord],
    ) -> ExecutionPlaybook:
        if not successes:
            return ExecutionPlaybook(support=0)

        coverage = Counter()
        structure = Counter()
        anti = Counter()
        pattern_counts = Counter()
        flow_counts = Counter()
        guidance: list[str] = []
        examples: list[str] = []
        case_summaries: list[str] = []
        seen_examples: set[str] = set()
        seen_cases: set[str] = set()

        current_items = _task_checklist_items(task.text)
        task_contract = infer_output_contract(task.text, task.metadata) or str(task.metadata.get("output_contract", "")).strip().lower()
        for record in successes:
            for item in _task_checklist_items(record.input_text):
                coverage[item] += 1
            for cue in _structure_cues(record.selected_output, record.input_text):
                structure[cue] += 1
            for tag in _pattern_tags(record.input_text, task_contract):
                pattern_counts[_describe_pattern_tag(tag)] += 1
            flow = _flow_signature(record.selected_output, record.input_text)
            if flow:
                flow_counts[flow] += 1

            hint = self._summarize_execution_record(record)
            if hint and hint not in guidance:
                guidance.append(hint)

            excerpt = self._compact_output_excerpt(record.selected_output, max_chars=140)
            if excerpt and excerpt not in seen_examples:
                seen_examples.add(excerpt)
                examples.append(excerpt)

            case_line = self._case_memory_line(record)
            if case_line and case_line not in seen_cases:
                seen_cases.add(case_line)
                case_summaries.append(case_line)

        for record in failures:
            record_items = _task_checklist_items(record.input_text)
            for item in record_items:
                if _contains_concept(record.selected_output, item):
                    continue
                anti[f"do not omit {item}"] += 1

        boosted_coverage = list(current_items)
        for item, _ in coverage.most_common(6):
            if item not in boosted_coverage:
                boosted_coverage.append(item)

        return ExecutionPlaybook(
            support=len(successes),
            coverage_items=boosted_coverage[:6],
            structure_cues=[cue for cue, _ in structure.most_common(4)],
            anti_patterns=[cue for cue, _ in anti.most_common(3)],
            success_examples=examples[:2],
            guidance=guidance[:3],
            case_summaries=case_summaries[:3],
            pattern_summaries=[item for item, _ in pattern_counts.most_common(4)],
            recommended_flows=[item for item, _ in flow_counts.most_common(3)],
        )

    @staticmethod
    def _compact_output_excerpt(text: str, max_chars: int = 160) -> str:
        cleaned = re.sub(r"\s+", " ", text.replace("composer-fusion:", " ").strip())
        if len(cleaned) <= max_chars:
            return cleaned
        return cleaned[:max_chars].rstrip() + "..."

    @classmethod
    def _case_memory_line(cls, record: MemoryRecord) -> str:
        meta = dict(record.task_metadata or {})
        contract = infer_output_contract(record.input_text, meta) or str(meta.get("output_contract", "")).strip().lower()
        tags = [_describe_pattern_tag(tag) for tag in _pattern_tags(record.input_text, contract)]
        flow = _flow_signature(record.selected_output, record.input_text)
        excerpt = cls._compact_output_excerpt(record.selected_output, max_chars=110)
        details: list[str] = []
        if tags:
            details.append("best_for=" + ", ".join(tags[:3]))
        if flow:
            details.append(f"flow={flow}")
        if excerpt:
            details.append(f'example="{excerpt}"')
        return " | ".join(details[:3])

    def _scored_execution_records(
        self,
        *,
        task: TaskInput,
        branch_name: str,
        user_id: str | None,
        min_similarity: float | None = None,
    ) -> list[tuple[float, MemoryRecord]]:
        task_meta = dict(task.metadata or {})
        task_contract = infer_output_contract(task.text, task_meta) or str(task_meta.get("output_contract", "")).strip().lower()
        task_aspect = str(task_meta.get("aspect", "")).strip().lower()
        task_terms = _lexical_terms(task.text)
        task_tags = _pattern_tags(task.text, task_contract)
        task_items = _task_checklist_items(task.text)

        scored: list[tuple[float, MemoryRecord]] = []
        recent_records = self._records[-self.config.similarity_window :]
        for age, record in enumerate(reversed(recent_records)):
            if user_id and record.user_id != user_id:
                continue
            if record.selected_branch != branch_name or record.task_type != task.task_type:
                continue

            record_meta = dict(record.task_metadata or {})
            record_contract = infer_output_contract(record.input_text, record_meta) or str(record_meta.get("output_contract", "")).strip().lower()
            if task_contract != record_contract:
                if task_contract or record_contract:
                    continue

            record_aspect = str(record_meta.get("aspect", "")).strip().lower()
            record_items = _task_checklist_items(record.input_text)
            term_overlap = _jaccard(task_terms, _lexical_terms(record.input_text))
            tag_overlap = _jaccard(task_tags, _pattern_tags(record.input_text, record_contract))
            item_overlap = _jaccard(task_items, record_items)

            similarity = 0.2 + (0.35 * term_overlap) + (0.2 * tag_overlap) + (0.15 * item_overlap)
            if task_aspect and task_aspect == record_aspect:
                similarity += 0.1
            similarity += max(0.0, record.reward_score - 0.7) * 0.25
            threshold = max(
                self.config.routing_min_similarity if min_similarity is None else float(min_similarity),
                0.0,
            )
            if similarity < threshold:
                continue

            recency_weight = self.config.recency_decay**age
            scored.append((similarity * recency_weight, record))

        scored.sort(key=lambda item: item[0], reverse=True)
        return scored

    @staticmethod
    def _selected_leaf_under_parent(record: MemoryRecord, parent_id: str) -> str:
        path = list(record.selected_path or [])
        if len(path) >= 2 and path[-2] == parent_id:
            return path[-1]
        return ""

    @staticmethod
    def _build_routing_context(
        text: str,
        task_type: str,
        metadata: dict[str, Any] | None,
        parent_id: str,
    ) -> RoutingContext:
        meta = dict(metadata or {})
        aspect = str(meta.get("aspect", "")).strip().lower()
        contract_hint = infer_output_contract(text, meta) or str(meta.get("output_contract", "")).strip().lower()
        lexical_terms = _lexical_terms(text)
        pattern_tags = _pattern_tags(text, contract_hint)
        return RoutingContext(
            parent_id=parent_id,
            task_type=task_type,
            aspect=aspect,
            contract_hint=contract_hint,
            lexical_terms=lexical_terms,
            pattern_tags=pattern_tags,
        )


def _lexical_terms(text: str, limit: int = 8) -> tuple[str, ...]:
    tokens = re.findall(r"[a-z]{3,}", text.lower())
    counts = Counter(token for token in tokens if token not in _STOP_WORDS)
    top_terms = [token for token, _ in counts.most_common(limit)]
    return tuple(sorted(top_terms))


def _pattern_tags(text: str, contract_hint: str) -> tuple[str, ...]:
    text_l = text.lower()
    tags: set[str] = set()
    if contract_hint:
        tags.add(contract_hint)
    if "risk register" in text_l or ("mitigation" in text_l and "owner" in text_l):
        tags.add("risk_register")
    if "incident response" in text_l or "recovery plan" in text_l or "failed production deploy" in text_l:
        tags.add("recovery")
    if "review checklist" in text_l or "pull-request review" in text_l or ("review" in text_l and "checklist" in text_l):
        tags.add("review_checklist")
    if "timeline" in text_l or "schedule" in text_l or "48-hour" in text_l or "30-day" in text_l or "next-7-days" in text_l:
        tags.add("timeline")
    if "tradeoff" in text_l or "recommendation" in text_l or "decision note" in text_l:
        tags.add("recommendation")
    if "rollback" in text_l:
        tags.add("rollback")
    if "owner" in text_l or "owners" in text_l:
        tags.add("owner")
    if "risk" in text_l or "risks" in text_l:
        tags.add("risk")
    if "json" in text_l:
        tags.add("json")
    if "csv" in text_l:
        tags.add("csv")
    if "bullet" in text_l:
        tags.add("bullet")
    if "fix:" in text_l or "tests:" in text_l:
        tags.add("code_patch")
    return tuple(sorted(tags))


def _describe_pattern_tag(tag: str) -> str:
    mapping = {
        "risk_register": "use a risk-register frame with owners, mitigations, and rollback",
        "recovery": "treat the task like incident recovery with mitigation and fallback",
        "review_checklist": "use an audit/checklist frame with explicit pass-fail criteria",
        "timeline": "sequence the answer across phases, checkpoints, or time blocks",
        "recommendation": "end with a direct recommendation after the analysis",
        "rollback": "make rollback or fallback steps explicit",
        "owner": "assign explicit owners for each action",
        "risk": "call out risks directly instead of leaving them implicit",
        "json": "preserve a strict JSON-style contract",
        "csv": "preserve a strict CSV-style contract",
        "bullet": "prefer short bullets over dense prose",
        "code_patch": "ground the answer in code-specific fixes, tests, and risks",
    }
    return mapping.get(tag, tag.replace("_", " "))


def _jaccard(left: tuple[str, ...], right: tuple[str, ...]) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


def _task_checklist_items(text: str, limit: int = 6) -> tuple[str, ...]:
    text_l = re.sub(r"\s+", " ", text.lower())
    fragments: list[str] = []
    for pattern in (
        r"\binclude\s+([^.;]+)",
        r"\bcovering\s+([^.;]+)",
        r"\bcover\s+([^.;]+)",
        r"\bexplain\s+([^.;]+)",
        r"\baudit\s+[^.]*?\b(?:for|when)\s+([^.;]+)",
        r"\bwith\s+([^.;]+)",
    ):
        match = re.search(pattern, text_l)
        if not match:
            continue
        fragment = match.group(1).strip()
        if fragment:
            fragments.append(fragment)

    raw_items: list[str] = []
    for fragment in fragments:
        normalized = fragment.replace(" and ", ", ")
        for item in normalized.split(","):
            cleaned = item.strip(" .:-")
            cleaned = re.sub(r"^(clear|explicit|one|two|three|practical|concise|brief|short)\s+", "", cleaned)
            if not cleaned:
                continue
            if len(cleaned) <= 2:
                continue
            raw_items.append(cleaned)

    deduped: list[str] = []
    for item in raw_items:
        if item in deduped:
            continue
        deduped.append(item)
        if len(deduped) >= max(1, limit):
            break
    return tuple(deduped)


def _structure_cues(output: str, task_text: str) -> tuple[str, ...]:
    output_l = output.lower()
    task_l = task_text.lower()
    cues: list[str] = []
    if output.count("|") >= 6:
        cues.append("use a compact table when several fields must be tracked")
    if re.search(r"(^|\n)\s*[-*]\s+", output):
        cues.append("use short checklist bullets")
    if re.search(r"(^|\n)\s*\d+\.\s+", output):
        cues.append("use numbered steps")
    if "##" in output or "#" in output:
        cues.append("organize with short section headings")
    if re.search(r"\b(phase|day|week|timeline)\b", output_l):
        cues.append("sequence the answer by time blocks")
    if "recommendation" in output_l or "recommend" in task_l:
        cues.append("end with a direct recommendation")
    if "confidence" in output_l or "confidence" in task_l:
        cues.append("finish with calibrated confidence")
    return tuple(cues[:4])


def _flow_signature(output: str, task_text: str) -> str:
    combined = f"{task_text}\n{output}".lower()
    ordered_steps = [
        ("timeline", ("timeline", "phase", "day ", "week ", "checkpoint", "schedule")),
        ("owners", ("owner", "owners", "owner:")),
        ("risks", ("risk", "risks")),
        ("mitigation", ("mitigation", "mitigations")),
        ("rollback", ("rollback", "fallback", "backout")),
        ("tests", ("test", "tests", "validation", "verify")),
        ("recommendation", ("recommendation", "recommend")),
        ("confidence", ("confidence", "confidence=")),
    ]
    steps = [label for label, needles in ordered_steps if any(needle in combined for needle in needles)]
    if len(steps) < 2:
        return ""
    return " -> ".join(steps[:5])


def _contains_concept(text: str, concept: str) -> bool:
    text_l = text.lower()
    concept_l = concept.lower().strip()
    if not concept_l:
        return True
    variants = {concept_l, concept_l.replace("-", " ")}
    if concept_l.endswith("s"):
        variants.add(concept_l[:-1])
    else:
        variants.add(f"{concept_l}s")
    return any(variant and variant in text_l for variant in variants)
