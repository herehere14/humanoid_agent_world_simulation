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
    runtime = payload.get("runtime", {})

    lines: list[str] = []
    task_id = str(task.get("task_id", ""))[:8]
    lines.append(f"[task] id={task_id} type={routing.get('task_type', task.get('task_type', 'auto'))}")

    path = routing.get("activated_branches", []) or []
    path_text = " -> ".join(path) if path else "(none)"
    lines.append(f"[routing] path={path_text}")
    paths = routing.get("activated_paths", []) or []
    if paths:
        preview = " | ".join(" -> ".join(p) for p in paths[:3])
        if len(paths) > 3:
            preview = f"{preview} | ...(+{len(paths)-3})"
        lines.append(f"[routing] activated_paths={preview}")
    if runtime:
        lines.append(
            "[runtime] "
            f"primary_backend={runtime.get('primary_backend', 'n/a')} "
            f"primary_model={runtime.get('primary_model', 'n/a')} "
            f"evaluator_llm={runtime.get('evaluator_llm_enabled', False)} "
            f"optimizer_llm={runtime.get('optimizer_llm_enabled', False)}"
        )

    raw_scores = routing.get("branch_scores", {}) or {}
    if raw_scores:
        ordered = sorted(raw_scores.items(), key=lambda kv: kv[1], reverse=True)[: max(1, top_branches)]
        score_text = ", ".join(f"{name}:{_fmt_float(score)}" for name, score in ordered)
        lines.append(f"[routing] top_scores={score_text}")
    sibling_decisions = routing.get("sibling_decisions", {}) or {}
    if sibling_decisions:
        lines.append("[routing] sibling_decisions:")
        for parent_id, detail in sibling_decisions.items():
            lines.append(
                "  "
                f"{parent_id}: selected={detail.get('selected_child', detail.get('selected_by_score', 'n/a'))} "
                f"preferred={detail.get('preferred_child') or '-'} "
                f"override={detail.get('override_child') or '-'} "
                f"support={detail.get('support', 0)} "
                f"win_rate={_fmt_float(detail.get('win_rate'))} "
                f"margin={_fmt_float(detail.get('expected_margin'))}"
            )
            probes = detail.get("probe_candidates", []) or []
            if probes:
                lines.append(f"    probes={probes}")
    routing_probes = payload.get("routing_probes", []) or []
    if routing_probes:
        lines.append(f"[routing] probes={routing_probes}")
    routing_preferences = payload.get("routing_preferences", []) or []
    if routing_preferences:
        lines.append(f"[routing] learned_preferences={routing_preferences}")

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
        reward_components = payload.get("reward_components", {}) or {}
        if reward_components:
            component_text = ", ".join(
                f"{name}:{_fmt_float(score)}" for name, score in reward_components.items()
            )
            lines.append(f"[evaluator] reward_components={component_text}")

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
        if "advisor_used" in optimization:
            lines.append(f"[optimizer] advisor_used={optimization.get('advisor_used', False)}")
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
                    f"block={detail.get('weight_update_block_reason', 'none')}, "
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
        if optimization.get("advisor_error"):
            lines.append(f"[optimizer] advisor_error={_short_reason(str(optimization.get('advisor_error')))}")

    return "\n".join(lines)


def format_comparison_trace(comparison: dict[str, Any]) -> str:
    adaptive = comparison.get("adaptive_system", {}) or {}
    base = comparison.get("base_model", {}) or {}
    adaptive_metrics = adaptive.get("objective_metrics", {}) or {}
    base_metrics = base.get("objective_metrics", {}) or {}
    delta = comparison.get("delta", {}) or {}

    lines = [
        "[compare] "
        f"adaptive_reward={_fmt_float(adaptive_metrics.get('hybrid_verifier_reward'))} "
        f"base_reward={_fmt_float(base_metrics.get('hybrid_verifier_reward'))} "
        f"delta={_fmt_float(delta.get('hybrid_verifier_reward'))} "
        f"winner={comparison.get('winner', 'tie')}",
        "[compare] "
        f"adaptive_branch={adaptive.get('selected_branch', 'n/a')} "
        f"base_branch={base.get('selected_branch', 'base_model_direct')}",
    ]

    if adaptive_metrics or base_metrics:
        lines.append(
            "[compare] "
            f"adaptive_breakdown="
            f"keyword:{_fmt_float(adaptive_metrics.get('keyword_score'))},"
            f" rule:{_fmt_float(adaptive_metrics.get('rule_score'))},"
            f" external:{_fmt_float(adaptive_metrics.get('external_verifier_score'))}"
        )
        lines.append(
            "[compare] "
            f"base_breakdown="
            f"keyword:{_fmt_float(base_metrics.get('keyword_score'))},"
            f" rule:{_fmt_float(base_metrics.get('rule_score'))},"
            f" external:{_fmt_float(base_metrics.get('external_verifier_score'))}"
        )

    base_reason = str(base_metrics.get("hybrid_verifier_reason", "")).strip()
    adaptive_reason = str(adaptive_metrics.get("hybrid_verifier_reason", "")).strip()
    if base_reason:
        lines.append(f"[compare] base_reason={_short_reason(base_reason, max_chars=120)}")
    if adaptive_reason:
        lines.append(f"[compare] adaptive_reason={_short_reason(adaptive_reason, max_chars=120)}")

    return "\n".join(lines)
