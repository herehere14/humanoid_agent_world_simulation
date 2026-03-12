from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from ..types import TaskInput


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
class HybridReward:
    exact: ExactMatchReward
    keyword: KeywordReward
    rule: RuleBasedReward

    def score(self, output: str, task: TaskInput) -> tuple[float, str]:
        s1, r1 = self.exact.score(output, task)
        s2, r2 = self.keyword.score(output, task)
        s3, r3 = self.rule.score(output, task)

        total_weight = self.exact.weight + self.keyword.weight + self.rule.weight
        score = (s1 + s2 + s3) / max(1e-8, total_weight)

        if score >= 0.75:
            top_reason = "high_quality"
        elif score >= 0.5:
            top_reason = "medium_quality"
        else:
            top_reason = "low_quality"

        reason = f"{top_reason}|{r1}|{r2}|{r3}"
        return score, reason
