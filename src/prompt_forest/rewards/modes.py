from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from ..types import TaskInput
from .verifiers import ExternalVerifierReward


def _norm(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


class RewardFunction(Protocol):
    def score(self, output: str, task: TaskInput) -> tuple[float, str]:
        ...


@dataclass
class ExactMatchReward:
    weight: float = 1.0

    def score(self, output: str, task: TaskInput) -> tuple[float, str]:
        expected = task.metadata.get("expected_answer")
        if not expected:
            return 0.5 * self.weight, "no_exact_reference"
        val = 1.0 if _norm(expected) in _norm(output) else 0.0
        reason = "exact_match" if val > 0 else "exact_mismatch"
        return val * self.weight, reason


@dataclass
class KeywordReward:
    weight: float = 1.0

    def score(self, output: str, task: TaskInput) -> tuple[float, str]:
        keywords = task.metadata.get("expected_keywords", [])
        if not keywords:
            return 0.5 * self.weight, "no_keyword_reference"
        output_n = _norm(output)
        hit = sum(1 for k in keywords if _norm(str(k)) in output_n)
        ratio = hit / max(1, len(keywords))
        return ratio * self.weight, f"keyword_coverage:{hit}/{len(keywords)}"


@dataclass
class RuleBasedReward:
    weight: float = 1.0

    def score(self, output: str, task: TaskInput) -> tuple[float, str]:
        rules = task.metadata.get("required_substrings", [])
        if not rules:
            return 0.5 * self.weight, "no_rule_reference"

        output_n = _norm(output)
        failures = [r for r in rules if _norm(str(r)) not in output_n]
        pass_ratio = 1.0 - (len(failures) / max(1, len(rules)))
        reason = "rules_passed" if not failures else f"rule_miss:{','.join(failures[:3])}"
        return pass_ratio * self.weight, reason


@dataclass
class TaskSpecificReward:
    weight: float = 1.0

    def score(self, output: str, task: TaskInput) -> tuple[float, str]:
        output_n = _norm(output)
        checks: dict[str, list[str]] = {
            "math": ["symbolic", "equation", "derive", "constraint", "solve"],
            "planning": ["plan", "timeline", "milestone", "deadline", "risk"],
            "factual": ["fact", "evidence", "grounded", "source", "reference"],
            "code": ["code", "bug", "algorithm", "test", "refactor"],
            "creative": ["creative", "novel", "idea", "diverse", "option"],
            "general": ["verification", "robust", "confidence", "check"],
        }
        expected = checks.get(task.task_type, checks["general"])
        hit = sum(1 for token in expected if token in output_n)
        ratio = hit / max(1, min(3, len(expected)))
        clipped = max(0.0, min(1.0, ratio))
        return clipped * self.weight, f"task_specific:{hit}/{len(expected)}"


@dataclass
class HybridReward:
    exact: ExactMatchReward
    keyword: KeywordReward
    rule: RuleBasedReward
    task_specific: TaskSpecificReward
    external: ExternalVerifierReward | None = None

    def score(self, output: str, task: TaskInput) -> tuple[float, str]:
        s1, r1 = self.exact.score(output, task)
        s2, r2 = self.keyword.score(output, task)
        s3, r3 = self.rule.score(output, task)
        s4, r4 = self.task_specific.score(output, task)
        s5, r5 = (0.0, "no_external_verifier")
        if self.external and self.external.weight > 0:
            s5, r5 = self.external.score(output, task)

        total_weight = self.exact.weight + self.keyword.weight + self.rule.weight + self.task_specific.weight
        if self.external:
            total_weight += self.external.weight
        score = (s1 + s2 + s3 + s4 + s5) / max(1e-8, total_weight)
        score -= self._shallow_keyword_penalty(output, task, s1=s1, s2=s2, s3=s3)
        pref_penalty, pref_reason = self._preference_penalty(output, task)
        score -= pref_penalty
        score = max(0.0, min(1.0, score))

        if score >= 0.75:
            top_reason = "high_quality"
        elif score >= 0.5:
            top_reason = "medium_quality"
        else:
            top_reason = "low_quality"

        reason = f"{top_reason}|{r1}|{r2}|{r3}|{r4}|{r5}|{pref_reason}"
        return score, reason

    def _shallow_keyword_penalty(self, output: str, task: TaskInput, s1: float, s2: float, s3: float) -> float:
        keywords = task.metadata.get("expected_keywords", [])
        output_n = _norm(output)
        keyword_hits = sum(1 for k in keywords if _norm(str(k)) in output_n) if keywords else 0
        keyword_ratio = keyword_hits / max(1, len(keywords)) if keywords else 0.0

        token_count = len(re.findall(r"\w+", output_n))
        structured_audit = any(marker in output_n for marker in ("evidence:", "verification:", "result:", "summary:"))
        if keyword_ratio >= 0.8 and (s1 <= 0.3 and s3 <= 0.6) and token_count < 180 and not structured_audit:
            return 0.12
        if keyword_ratio >= 0.7 and token_count < 24:
            return 0.08
        return 0.0

    def _preference_penalty(self, output: str, task: TaskInput) -> tuple[float, str]:
        prefs = task.metadata.get("user_preferences", {})
        if not isinstance(prefs, dict) or not prefs:
            return 0.0, "no_user_preference_profile"

        output_n = _norm(output)
        penalty = 0.0
        reasons: list[str] = []

        hard_constraints = prefs.get("hard_constraints", [])
        if isinstance(hard_constraints, list) and hard_constraints:
            missing = [c for c in hard_constraints if _norm(str(c)) not in output_n]
            if missing:
                penalty += min(0.2, 0.05 * len(missing))
                reasons.append(f"missing_constraints:{len(missing)}")

        verbosity = str(prefs.get("verbosity", "")).strip().lower()
        token_count = len(re.findall(r"\w+", output_n))
        if verbosity == "concise" and token_count > 140:
            penalty += 0.08
            reasons.append("verbosity_mismatch:too_long")
        elif verbosity in {"detailed", "high"} and token_count < 50:
            penalty += 0.06
            reasons.append("verbosity_mismatch:too_short")

        style = str(prefs.get("style", "")).strip().lower()
        if style in {"bullet", "list"} and "-" not in output and "*" not in output:
            penalty += 0.04
            reasons.append("style_mismatch:missing_bullets")

        penalty = min(0.25, penalty)
        if not reasons:
            return 0.0, "preferences_aligned"
        return penalty, ",".join(reasons)
