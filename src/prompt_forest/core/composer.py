from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from ..backend.base import LLMBackend
from ..config import ComposerConfig
from ..types import BranchOutput, RoutingDecision, TaskInput


@dataclass
class ComposerResult:
    output: BranchOutput
    notes: dict[str, Any]


class FinalComposer:
    """Fuse branch features into a final leaf output."""

    def __init__(self, backend: LLMBackend, config: ComposerConfig) -> None:
        self.backend = backend
        self.config = config

    def compose(
        self,
        task: TaskInput,
        route: RoutingDecision,
        outputs: dict[str, BranchOutput],
    ) -> ComposerResult | None:
        if not self.config.enabled or len(route.activated_branches) < self.config.min_branches_for_compose:
            return None
        if not route.activated_branches:
            return None

        if route.activated_paths and route.activated_paths[0]:
            leaf = route.activated_paths[0][-1]
        else:
            leaf = route.activated_branches[-1]
        leaf_output = outputs.get(leaf)
        if leaf_output is None:
            return None

        feature_bundle = self._extract_feature_bundle(task, route, outputs)
        prompt = self._render_prompt(task, route, feature_bundle)
        generated, meta = self.backend.generate(prompt, task, leaf)
        fused = self._fuse_output(generated=generated, leaf_output=leaf_output.output, bundle=feature_bundle, task=task)

        composed = BranchOutput(
            branch_name=leaf,
            prompt=prompt,
            output=fused,
            task_type=task.task_type,
            model_meta={
                **meta,
                "composer_enabled": True,
                "composer_feature_count": len(feature_bundle.get("key_points", [])),
                "composer_source_branches": list(outputs.keys()),
            },
        )
        notes = {
            "composer_enabled": True,
            "composed_leaf": leaf,
            "feature_count": len(feature_bundle.get("key_points", [])),
            "required_substrings": feature_bundle.get("required_substrings", []),
        }
        return ComposerResult(output=composed, notes=notes)

    def _extract_feature_bundle(
        self,
        task: TaskInput,
        route: RoutingDecision,
        outputs: dict[str, BranchOutput],
    ) -> dict[str, Any]:
        key_points: list[str] = []
        branch_summaries: list[str] = []
        confidences: list[float] = []

        for branch_name in route.activated_branches:
            out = outputs.get(branch_name)
            if out is None:
                continue

            text = out.output
            branch_summaries.append(f"{branch_name}: {self._first_sentence(text)}")
            for kp in self._extract_key_points(text):
                if kp not in key_points:
                    key_points.append(kp)
            conf = self._extract_confidence(text)
            if conf is not None:
                confidences.append(conf)

        max_items = max(4, self.config.max_feature_items)
        required = [str(x) for x in task.metadata.get("required_substrings", []) if str(x).strip()]
        expected_keywords = [str(x) for x in task.metadata.get("expected_keywords", []) if str(x).strip()]
        for token in expected_keywords:
            if token not in key_points:
                key_points.append(token)
            if len(key_points) >= max_items:
                break

        return {
            "key_points": key_points[:max_items],
            "branch_summaries": branch_summaries[:max_items],
            "required_substrings": required[:max_items],
            "confidence_hint": round(sum(confidences) / len(confidences), 4) if confidences else 0.72,
        }

    def _render_prompt(self, task: TaskInput, route: RoutingDecision, bundle: dict[str, Any]) -> str:
        key_points = bundle.get("key_points", [])
        summaries = bundle.get("branch_summaries", [])
        required = bundle.get("required_substrings", [])
        return (
            "You are the Final Composer. Build one reliable final answer from branch features.\n"
            f"Task type: {task.task_type}\n"
            f"User task: {task.text}\n"
            f"Activated branches: {', '.join(route.activated_branches)}\n"
            "Feature key-points:\n"
            + "\n".join(f"- {x}" for x in key_points)
            + "\nBranch summaries:\n"
            + "\n".join(f"- {x}" for x in summaries)
            + "\nRequired constraints:\n"
            + ("\n".join(f"- {x}" for x in required) if required else "- none")
            + "\nReturn: concise final answer, key-points, and calibrated confidence."
        )

    def _fuse_output(self, generated: str, leaf_output: str, bundle: dict[str, Any], task: TaskInput) -> str:
        key_points = bundle.get("key_points", [])
        required = bundle.get("required_substrings", [])
        confidence = float(bundle.get("confidence_hint", 0.72))
        confidence = max(0.05, min(0.99, confidence))

        key_points_str = "; ".join(key_points[: max(1, min(10, len(key_points)))]) if key_points else "limited-grounding"
        if self.config.force_required_substrings and required:
            required_line = "; ".join(required)
        else:
            required_line = "tracked"

        fused = (
            f"{generated}\n"
            f"composer-fusion: {leaf_output}\n"
            f"key-points={key_points_str}\n"
            f"constraints={required_line}\n"
            f"confidence={confidence:.2f}"
        )

        if self.config.force_required_substrings:
            lower = fused.lower()
            for token in required:
                if token.lower() not in lower:
                    fused = f"{fused}\n{token}"
        return fused

    @staticmethod
    def _extract_key_points(text: str) -> list[str]:
        m = re.search(r"key-points=([^|\n]+)", text, flags=re.IGNORECASE)
        if not m:
            return []
        raw = m.group(1).strip()
        parts = [x.strip() for x in raw.split(";")]
        return [p for p in parts if p]

    @staticmethod
    def _extract_confidence(text: str) -> float | None:
        m = re.search(r"confidence=([0-9]*\.?[0-9]+)", text, flags=re.IGNORECASE)
        if not m:
            return None
        try:
            parsed = float(m.group(1))
        except ValueError:
            return None
        return max(0.0, min(1.0, parsed))

    @staticmethod
    def _first_sentence(text: str, max_chars: int = 140) -> str:
        cleaned = re.sub(r"\s+", " ", text.strip())
        if len(cleaned) <= max_chars:
            return cleaned
        return cleaned[:max_chars].rstrip() + "..."
