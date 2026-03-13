from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from ..types import TaskInput


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


@dataclass
class ExternalVerifierReward:
    """Metadata-driven external verifier proxy for reward shaping."""

    weight: float = 1.0

    def score(self, output: str, task: TaskInput) -> tuple[float, str]:
        spec = task.metadata.get("verifier_spec", {})
        if not isinstance(spec, dict) or not spec:
            return self._fallback_score(output, task)

        output_n = _norm(output)
        checks: list[tuple[float, str]] = []

        must_include = [str(x).strip() for x in spec.get("must_include", []) if str(x).strip()]
        if must_include:
            hit = sum(1 for token in must_include if _norm(token) in output_n)
            checks.append((hit / len(must_include), f"must_include:{hit}/{len(must_include)}"))

        must_exclude = [str(x).strip() for x in spec.get("must_exclude", []) if str(x).strip()]
        if must_exclude:
            misses = sum(1 for token in must_exclude if _norm(token) in output_n)
            checks.append((1.0 - (misses / len(must_exclude)), f"must_exclude_ok:{len(must_exclude)-misses}/{len(must_exclude)}"))

        regex_must_match = [str(x).strip() for x in spec.get("regex_must_match", []) if str(x).strip()]
        if regex_must_match:
            matched = 0
            for pattern in regex_must_match:
                try:
                    if re.search(pattern, output, flags=re.IGNORECASE):
                        matched += 1
                except re.error:
                    continue
            checks.append((matched / len(regex_must_match), f"regex_match:{matched}/{len(regex_must_match)}"))

        min_token_count = spec.get("min_token_count")
        if isinstance(min_token_count, int) and min_token_count > 0:
            token_count = len(re.findall(r"\w+", output_n))
            checks.append((1.0 if token_count >= min_token_count else (token_count / min_token_count), f"length:{token_count}/{min_token_count}"))

        confidence_range = spec.get("confidence_range")
        if isinstance(confidence_range, list) and len(confidence_range) == 2:
            confidence = self._extract_confidence(output_n)
            lo = float(confidence_range[0])
            hi = float(confidence_range[1])
            if confidence is None:
                conf_score = 0.0
            elif lo <= confidence <= hi:
                conf_score = 1.0
            else:
                conf_score = 0.0
            checks.append((conf_score, f"confidence_range:{'ok' if conf_score > 0 else 'miss'}"))

        if not checks:
            return self._fallback_score(output, task)

        score = sum(v for v, _ in checks) / len(checks)
        reason = "verifier|" + "|".join(r for _, r in checks[:4])
        return max(0.0, min(1.0, score)) * self.weight, reason

    def _fallback_score(self, output: str, task: TaskInput) -> tuple[float, str]:
        output_n = _norm(output)
        expectations = {
            "math": ["equation", "constraint", "confidence"],
            "planning": ["plan", "risk", "confidence"],
            "factual": ["evidence", "source", "confidence"],
            "code": ["bug", "test", "confidence"],
            "creative": ["novel", "option", "confidence"],
            "general": ["verification", "confidence", "key-points"],
        }
        required = expectations.get(task.task_type, expectations["general"])
        hit = sum(1 for token in required if token in output_n)
        ratio = hit / max(1, len(required))
        return ratio * self.weight, f"fallback_verifier:{hit}/{len(required)}"

    @staticmethod
    def _extract_confidence(output_n: str) -> float | None:
        m = re.search(r"confidence=([0-9]*\.?[0-9]+)", output_n)
        if not m:
            return None
        try:
            parsed = float(m.group(1))
        except ValueError:
            return None
        return max(0.0, min(1.0, parsed))
