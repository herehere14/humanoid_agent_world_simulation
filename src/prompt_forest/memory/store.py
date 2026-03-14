from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..config import MemoryConfig
from ..contracts import infer_output_contract
from ..types import MemoryRecord, TaskInput
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
        self.memory_path = Path(memory_path) if memory_path else None
        user_profiles_path = self.memory_path.parent / "user_profiles.json" if self.memory_path else None
        self._profiles = UserProfileStore(user_profiles_path)
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

    def retrieve_similar(self, task_type: str, limit: int | None = None, user_id: str | None = None) -> list[MemoryRecord]:
        n = limit or self.config.similarity_window
        out = [r for r in reversed(self._records) if r.task_type == task_type and (not user_id or r.user_id == user_id)]
        return list(reversed(out[:n]))

    def routing_context_key_for_task(self, task: TaskInput, parent_id: str) -> str:
        return self._build_routing_context(task.text, task.task_type, task.metadata, parent_id).key()

    def sibling_routing_scores(
        self,
        task: TaskInput,
        task_type: str,
        parent_id: str,
        child_ids: list[str],
        user_id: str | None = None,
    ) -> dict[str, float]:
        if not parent_id or not child_ids:
            return {}

        context = self._build_routing_context(task.text, task_type, task.metadata, parent_id)
        if user_id:
            user_scores = self._contextual_sibling_scores(
                context=context,
                parent_id=parent_id,
                child_ids=child_ids,
                user_id=user_id,
            )
            if user_scores:
                return user_scores
        return self._contextual_sibling_scores(
            context=context,
            parent_id=parent_id,
            child_ids=child_ids,
            user_id=None,
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

    def stats(self) -> dict[str, Any]:
        if not self._records:
            return {"records": 0, "avg_reward": 0.0, "route_contexts": 0, "user_profiles": self._profiles.count}
        avg_reward = sum(record.reward_score for record in self._records) / len(self._records)
        task_counts = Counter(record.task_type for record in self._records)
        route_contexts = {record.routing_context_key for record in self._records if record.routing_context_key}
        return {
            "records": len(self._records),
            "avg_reward": round(avg_reward, 4),
            "task_distribution": dict(task_counts),
            "route_contexts": len(route_contexts),
            "user_profiles": self._profiles.count,
        }

    def dump_records(self) -> list[dict[str, Any]]:
        return [asdict(record) for record in self._records]

    def persist_records(self) -> None:
        if self.memory_path:
            write_jsonl(self.memory_path, [record.to_dict() for record in self._records[-self.config.max_records :]])

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

    def _contextual_sibling_scores(
        self,
        *,
        context: RoutingContext,
        parent_id: str,
        child_ids: list[str],
        user_id: str | None,
    ) -> dict[str, float]:
        reward_sum: dict[str, float] = defaultdict(float)
        weight_sum: dict[str, float] = defaultdict(float)
        match_count: Counter[str] = Counter()

        recent_records = self._records[-self.config.similarity_window :]
        for age, record in enumerate(reversed(recent_records)):
            if user_id and record.user_id != user_id:
                continue

            leaf = self._selected_leaf_under_parent(record, parent_id)
            if leaf not in child_ids:
                continue

            other_context = self._build_routing_context(record.input_text, record.task_type, record.task_metadata, parent_id)
            similarity = context.similarity(other_context)
            if similarity < self.config.routing_min_similarity:
                continue

            recency_weight = self.config.recency_decay**age
            total_weight = max(1e-6, similarity * recency_weight)
            reward_sum[leaf] += record.reward_score * total_weight
            weight_sum[leaf] += total_weight
            match_count[leaf] += 1

        if not match_count:
            return {}

        ranked_support = sorted(match_count.values(), reverse=True)
        if ranked_support[0] < self.config.routing_min_support:
            return {}

        means = {
            child_id: reward_sum[child_id] / max(1e-8, weight_sum[child_id])
            for child_id in child_ids
            if weight_sum[child_id] > 0.0
        }
        if not means:
            return {}

        ordered = sorted(means.items(), key=lambda item: item[1], reverse=True)
        best_child, best_mean = ordered[0]
        second_mean = ordered[1][1] if len(ordered) > 1 else 0.5

        if len(ordered) > 1:
            if (best_mean - second_mean) < self.config.routing_margin:
                return {}
        elif best_mean < self.config.execution_hint_min_reward:
            return {}

        out: dict[str, float] = {}
        sibling_center = sum(means.values()) / len(means)
        for child_id, mean_reward in means.items():
            n = match_count[child_id]
            shrink = n / (n + self.config.routing_shrinkage_k)
            relative_term = (mean_reward - sibling_center) * 3.0
            absolute_term = (mean_reward - 0.55) * 2.6
            raw_score = (relative_term + absolute_term) * shrink
            capped = max(-1.0, min(1.0, raw_score))
            if abs(capped) < 1e-6:
                continue
            out[child_id] = capped

        if out.get(best_child, 0.0) <= 0.0:
            return {}
        return out

    def _execution_guidance(
        self,
        *,
        task: TaskInput,
        branch_name: str,
        user_id: str | None,
        limit: int,
    ) -> list[str]:
        task_meta = dict(task.metadata or {})
        task_contract = infer_output_contract(task.text, task_meta) or str(task_meta.get("output_contract", "")).strip().lower()
        task_aspect = str(task_meta.get("aspect", "")).strip().lower()
        task_terms = _lexical_terms(task.text)
        task_tags = _pattern_tags(task.text, task_contract)

        scored: list[tuple[float, MemoryRecord]] = []
        recent_records = self._records[-self.config.similarity_window :]
        for age, record in enumerate(reversed(recent_records)):
            if user_id and record.user_id != user_id:
                continue
            if record.selected_branch != branch_name or record.task_type != task.task_type:
                continue
            if record.reward_score < self.config.execution_hint_min_reward:
                continue

            record_meta = dict(record.task_metadata or {})
            record_contract = infer_output_contract(record.input_text, record_meta) or str(record_meta.get("output_contract", "")).strip().lower()
            if task_contract != record_contract:
                if task_contract or record_contract:
                    continue

            record_aspect = str(record_meta.get("aspect", "")).strip().lower()
            term_overlap = _jaccard(task_terms, _lexical_terms(record.input_text))
            tag_overlap = _jaccard(task_tags, _pattern_tags(record.input_text, record_contract))

            similarity = 0.2 + (0.45 * term_overlap) + (0.2 * tag_overlap)
            if task_aspect and task_aspect == record_aspect:
                similarity += 0.1
            similarity += max(0.0, record.reward_score - 0.7) * 0.25
            if similarity < self.config.routing_min_similarity:
                continue

            recency_weight = self.config.recency_decay**age
            scored.append((similarity * recency_weight, record))

        if not scored:
            return []

        scored.sort(key=lambda item: item[0], reverse=True)
        hints: list[str] = []
        seen: set[str] = set()
        for _, record in scored:
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

    @staticmethod
    def _compact_output_excerpt(text: str, max_chars: int = 160) -> str:
        cleaned = re.sub(r"\s+", " ", text.replace("composer-fusion:", " ").strip())
        if len(cleaned) <= max_chars:
            return cleaned
        return cleaned[:max_chars].rstrip() + "..."

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


def _jaccard(left: tuple[str, ...], right: tuple[str, ...]) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)
