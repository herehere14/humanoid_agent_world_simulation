from __future__ import annotations

from typing import Any


def _fmt_float(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.3f}"
    return "n/a"


def _short_reason(reason: str, max_chars: int = 64) -> str:
    if not reason:
        return "ok"
    if len(reason) <= max_chars:
        return reason
    return f"{reason[: max_chars - 3]}..."


def format_turn_trace(payload: dict[str, Any], visibility: str = "full", top_branches: int = 5) -> str:
    """Render a readable per-turn trace from engine payload data."""
    visibility = visibility.lower().strip()
    if visibility not in {"minimal", "eval", "opt", "full"}:
        visibility = "full"

    task = payload.get("task", {})
    routing = payload.get("routing", {})
    branch_scores = payload.get("branch_scores", {})
    signal = payload.get("evaluation_signal", {})
    optimization = payload.get("optimization", {})

    lines: list[str] = []
    task_id = str(task.get("task_id", ""))[:8]
    lines.append(f"[task] id={task_id} type={routing.get('task_type', task.get('task_type', 'auto'))}")

    path = routing.get("activated_branches", []) or []
    path_text = " -> ".join(path) if path else "(none)"
    lines.append(f"[routing] path={path_text}")

    raw_scores = routing.get("branch_scores", {}) or {}
    if raw_scores:
        ordered = sorted(raw_scores.items(), key=lambda kv: kv[1], reverse=True)[: max(1, top_branches)]
        score_text = ", ".join(f"{name}:{_fmt_float(score)}" for name, score in ordered)
        lines.append(f"[routing] top_scores={score_text}")

    if visibility in {"eval", "full"}:
        lines.append(
            "[evaluator] "
            f"selected={signal.get('selected_branch', 'none')} "
            f"reward={_fmt_float(signal.get('reward_score'))} "
            f"confidence={_fmt_float(signal.get('confidence'))} "
            f"reason={_short_reason(str(signal.get('failure_reason', '')))}"
        )
        notes = signal.get("aggregator_notes", {}) or {}
        if notes:
            lines.append(f"[evaluator] aggregator_notes={notes}")

        feedback = signal.get("branch_feedback", {}) or {}
        if feedback and path:
            lines.append("[evaluator] branch_feedback:")
            for branch_name in path:
                fb = feedback.get(branch_name)
                if not fb:
                    continue
                judge = branch_scores.get(branch_name, {}) or {}
                lines.append(
                    "  "
                    f"{branch_name}: judge={_fmt_float(judge.get('reward'))} "
                    f"branch_reward={_fmt_float(fb.get('reward'))} "
                    f"branch_conf={_fmt_float(fb.get('confidence'))} "
                    f"reason={_short_reason(str(fb.get('failure_reason', '')))} "
                    f"improve={fb.get('suggested_improvement_direction', 'n/a')}"
                )

    if visibility in {"opt", "full"}:
        lines.append(
            "[optimizer] "
            f"task_baseline={_fmt_float(optimization.get('task_baseline_before'))}"
            f"->{_fmt_float(optimization.get('task_baseline_after'))}"
        )
        update_details = optimization.get("update_details", {}) or {}
        if update_details and path:
            lines.append("[optimizer] branch_updates:")
            for branch_name in path:
                detail = update_details.get(branch_name)
                if not detail:
                    continue
                lines.append(
                    "  "
                    f"{branch_name}: "
                    f"{_fmt_float(detail.get('old_weight'))}->{_fmt_float(detail.get('new_weight'))} "
                    f"(adv={_fmt_float(detail.get('advantage'))}, "
                    f"delta={_fmt_float(detail.get('delta'))}, "
                    f"decay={_fmt_float(detail.get('decay'))}, "
                    f"rewrite={detail.get('prompt_rewritten', False)}, "
                    f"status={detail.get('status_before', 'n/a')}->{detail.get('status_after', 'n/a')})"
                )

        for key, label in (
            ("rewritten_prompts", "rewritten"),
            ("created_candidates", "created"),
            ("promoted_candidates", "promoted"),
            ("archived_candidates", "archived"),
        ):
            items = optimization.get(key, []) or []
            if items:
                lines.append(f"[optimizer] {label}={items}")

    return "\n".join(lines)
