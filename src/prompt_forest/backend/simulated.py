from __future__ import annotations

import json
import random
import re
from typing import Any

from ..contracts import infer_output_contract
from ..types import TaskInput
from .base import LLMBackend


class DomainShiftBackend(LLMBackend):
    """Backend with deliberately shifted branch-task optima for adaptation validation."""

    def __init__(self, quality_matrix: dict[str, dict[str, float]], noise: float = 0.03, seed: int = 13) -> None:
        self.quality_matrix = quality_matrix
        self.noise = noise
        self._rng = random.Random(seed)

    def best_branch(self, task_type: str, candidates: list[str] | None = None) -> str:
        if candidates:
            scored = [(name, self._quality_for_branch(task_type, name)) for name in candidates]
            return max(scored, key=lambda x: x[1])[0]
        scores = self.quality_matrix[task_type]
        return max(scores.items(), key=lambda x: x[1])[0]

    def best_branch_for_task(self, task: TaskInput, candidates: list[str] | None = None) -> str:
        return self.best_branch(task.task_type, candidates=candidates)

    def generate(self, prompt: str, task: TaskInput, branch_name: str) -> tuple[str, dict[str, Any]]:
        task_type = task.task_type if task.task_type != "auto" else "general"
        quality = self._quality_for_branch(task_type, branch_name)
        quality += self._rng.uniform(-self.noise, self.noise)
        quality = max(0.01, min(0.99, quality))
        contract = infer_output_contract(task.text, task.metadata)
        contract_output = self._strict_contract_output(
            task=task,
            branch_name=branch_name,
            quality=quality,
            contract=contract,
        )
        if contract_output is not None:
            return contract_output, {"quality": quality, "task_type": task_type, "branch": branch_name, "contract": contract}

        expected_keywords = task.metadata.get("expected_keywords", [])
        included_count = int(round(quality * len(expected_keywords))) if expected_keywords else 0
        included = expected_keywords[:included_count]
        evidence = "; ".join(included) if included else "limited-grounding"

        answer = (
            f"[{branch_name}] shifted-sim response: {task.text} | "
            f"key-points={evidence} | confidence={quality:.2f}"
        )
        return answer, {"quality": quality, "task_type": task_type, "branch": branch_name}

    def _strict_contract_output(
        self,
        task: TaskInput,
        branch_name: str,
        quality: float,
        contract: str | None,
    ) -> str | None:
        if not contract or branch_name != contract:
            return None
        if contract == "json_lock":
            payload = {
                "answer": f"{task.task_type}_result",
                "confidence": round(quality, 2),
            }
            return json.dumps(payload, separators=(",", ":"))
        if contract == "csv_lock":
            rows = self._csv_rows_from_task(task.text)
            return "\n".join(rows)
        if contract == "code_patch_lock":
            return (
                "FIX:\n"
                "def solve(x):\n"
                "    return x\n"
                "TESTS:\n"
                "- handles baseline case\n"
                "- handles edge case"
            )
        if contract == "bullet_lock":
            count = self._bullet_count(task.text)
            return "\n".join(f"- item {i + 1}: verified step" for i in range(count))
        return None

    @staticmethod
    def _csv_rows_from_task(text: str) -> list[str]:
        rows: list[str] = []
        for line in text.splitlines():
            cleaned = line.strip()
            if "," not in cleaned:
                continue
            if cleaned.lower().startswith(("output", "holdout", "train", "task")):
                continue
            left, right, *_ = [part.strip() for part in cleaned.split(",")] + ["ok"]
            rows.append(f"{left},{right}")
        if rows:
            return rows
        return ["a,ok", "b,ok"]

    @staticmethod
    def _bullet_count(text: str) -> int:
        match = re.search(r"exactly\s+(\d+)\s+bullet", text, flags=re.IGNORECASE)
        if not match:
            return 3
        try:
            return max(1, min(8, int(match.group(1))))
        except ValueError:
            return 3

    def _quality_for_branch(self, task_type: str, branch_name: str) -> float:
        scores = self.quality_matrix.get(task_type, self.quality_matrix["general"])
        if branch_name in scores:
            return scores[branch_name]

        macro = branch_name.split("_")[0]
        base = scores.get(macro, 0.4)
        tokens = set(branch_name.split("_"))

        bonus = 0.0
        if task_type == "math":
            if {"symbolic", "solver", "constraint", "checker"} & tokens:
                bonus += 0.2
            if {"causal", "decomposer"} & tokens:
                bonus += 0.04
        elif task_type == "planning":
            if {"timeline", "optimizer", "risk", "allocator"} & tokens:
                bonus += 0.2
            if {"constraint", "innovator"} & tokens:
                bonus += 0.05
        elif task_type == "factual":
            if {"evidence", "tracer", "source", "triage"} & tokens:
                bonus += 0.2
            if {"consistency", "auditor"} & tokens:
                bonus += 0.07
        elif task_type == "code":
            if {"adversarial", "probe", "consistency", "auditor"} & tokens:
                bonus += 0.2
            if {"symbolic", "solver"} & tokens:
                bonus += 0.04
        elif task_type == "creative":
            if {"divergent", "generator", "innovator"} & tokens:
                bonus += 0.2
            if {"timeline", "optimizer"} & tokens:
                bonus += 0.03
        elif task_type == "general":
            if {"consistency", "auditor", "constraint", "checker"} & tokens:
                bonus += 0.15
            if {"evidence", "tracer"} & tokens:
                bonus += 0.08

        # Minor penalty for branches with no observed specialization markers.
        if len(tokens) == 1:
            bonus -= 0.02
        return max(0.01, min(0.99, base + bonus))


def shifted_quality_matrix() -> dict[str, dict[str, float]]:
    # Intentionally conflicts with default router affinity so adaptation is required.
    return {
        "math": {
            "analytical": 0.28,
            "planner": 0.2,
            "retrieval": 0.22,
            "critique": 0.25,
            "verification": 0.35,
            "creative": 0.92,
        },
        "planning": {
            "analytical": 0.26,
            "planner": 0.28,
            "retrieval": 0.92,
            "critique": 0.3,
            "verification": 0.32,
            "creative": 0.27,
        },
        "factual": {
            "analytical": 0.25,
            "planner": 0.91,
            "retrieval": 0.27,
            "critique": 0.24,
            "verification": 0.31,
            "creative": 0.2,
        },
        "code": {
            "analytical": 0.33,
            "planner": 0.25,
            "retrieval": 0.2,
            "critique": 0.91,
            "verification": 0.36,
            "creative": 0.22,
        },
        "creative": {
            "analytical": 0.92,
            "planner": 0.26,
            "retrieval": 0.2,
            "critique": 0.3,
            "verification": 0.22,
            "creative": 0.29,
        },
        "general": {
            "analytical": 0.34,
            "planner": 0.34,
            "retrieval": 0.34,
            "critique": 0.34,
            "verification": 0.9,
            "creative": 0.34,
        },
    }
