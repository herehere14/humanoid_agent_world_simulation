from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .backend.base import LLMBackend
from .backend.mock import MockLLMBackend
from .backend.openai_chat import OpenAIChatBackend
from .config import apply_latency_profile, load_config
from .core.engine import PromptForestEngine
from .observability.trace import format_turn_trace


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="JSON bridge for OpenClaw plugin integration")
    parser.add_argument("--config", type=str, default="configs/default.json", help="Path to config JSON")
    parser.add_argument(
        "--input-file",
        type=str,
        default="-",
        help="JSON input path. Use '-' to read from stdin.",
    )

    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("run", help="Run one task and emit JSON")
    sub.add_parser("feedback", help="Apply feedback to a previous task and emit JSON")
    sub.add_parser("state", help="Return engine state as JSON")
    sub.add_parser("ingest", help="Process an OpenClaw trajectory event and emit JSON")
    return parser


def _read_payload(input_file: str) -> dict[str, Any]:
    if input_file == "-":
        raw = sys.stdin.read().strip()
        return json.loads(raw) if raw else {}
    return json.loads(Path(input_file).read_text(encoding="utf-8"))


def _print_json(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, indent=2))
    sys.stdout.write("\n")


def _normalized_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _build_backend(payload: dict[str, Any]) -> LLMBackend:
    model = str(payload.get("model", "") or "").strip()
    if not model:
        seed = payload.get("seed")
        if seed is None:
            return MockLLMBackend()
        return MockLLMBackend(seed=int(seed))
    return OpenAIChatBackend(
        model=model,
        api_key_env=str(payload.get("api_key_env", "OPENAI_API_KEY") or "OPENAI_API_KEY"),
        base_url=str(payload.get("base_url", "https://api.openai.com/v1") or "https://api.openai.com/v1"),
        temperature=float(payload.get("temperature", 0.2) or 0.2),
        max_output_tokens=int(payload.get("max_output_tokens", 700) or 700),
        api_mode=str(payload.get("api_mode", "chat_completions") or "chat_completions"),
        reasoning_effort=(str(payload.get("reasoning_effort", "") or "").strip() or None),
    )


def _build_engine(args: argparse.Namespace, payload: dict[str, Any]) -> PromptForestEngine:
    cfg = load_config(args.config)
    latency_mode = str(payload.get("latency_mode", "full") or "full")
    cfg = apply_latency_profile(cfg, latency_mode)
    artifacts_dir = str(payload.get("artifacts_dir", "") or "").strip()
    if artifacts_dir:
        cfg.artifacts_dir = artifacts_dir
    return PromptForestEngine(config=cfg, backend=_build_backend(payload))


def _merge_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(payload.get("metadata", {}) or {})
    user_id = str(payload.get("user_id", metadata.get("user_id", "global")) or "global").strip() or "global"
    metadata["user_id"] = user_id

    expected_keywords = _normalized_list(payload.get("expected_keywords"))
    if expected_keywords:
        metadata["expected_keywords"] = expected_keywords

    required_checks = _normalized_list(payload.get("required_checks"))
    if required_checks:
        metadata["required_substrings"] = required_checks

    context_seed = str(payload.get("context_seed", metadata.get("context_seed", "")) or "").strip()
    if context_seed:
        metadata["context_seed"] = context_seed

    metadata.setdefault("source", "openclaw_plugin")
    return metadata


def _summarize_run_result(
    result: dict[str, Any],
    *,
    visibility: str,
    include_trace: bool,
    include_raw: bool,
) -> dict[str, Any]:
    signal = result.get("evaluation_signal", {}) or {}
    routing = result.get("routing", {}) or {}
    summary = {
        "ok": True,
        "task_id": result.get("task", {}).get("task_id", ""),
        "task_type": routing.get("task_type", result.get("task", {}).get("task_type", "auto")),
        "answer": signal.get("selected_output", ""),
        "selected_branch": signal.get("selected_branch", ""),
        "reward": signal.get("reward_score", 0.0),
        "confidence": signal.get("confidence", 0.0),
        "failure_reason": signal.get("failure_reason", ""),
        "activated_branches": routing.get("activated_branches", []),
        "activated_paths": routing.get("activated_paths", []),
        "routing_preferences": result.get("routing_preferences", []),
        "reward_components": result.get("reward_components", {}),
        "optimization": result.get("optimization", {}),
    }
    if include_trace:
        summary["trace"] = format_turn_trace(result, visibility=visibility, top_branches=5)
    if include_raw:
        summary["raw"] = result
    return summary


def _handle_run(args: argparse.Namespace, payload: dict[str, Any]) -> dict[str, Any]:
    engine = _build_engine(args, payload)
    result = engine.run_task(
        text=str(payload.get("task", "") or ""),
        task_type=str(payload.get("task_type", "auto") or "auto"),
        metadata=_merge_metadata(payload),
    )
    visibility = str(payload.get("visibility", "minimal") or "minimal")
    include_trace = bool(payload.get("include_trace", False))
    include_raw = bool(payload.get("include_raw", False))
    return _summarize_run_result(
        result,
        visibility=visibility,
        include_trace=include_trace,
        include_raw=include_raw,
    )


def _handle_feedback(args: argparse.Namespace, payload: dict[str, Any]) -> dict[str, Any]:
    engine = _build_engine(args, payload)
    accepted = payload.get("accepted")
    if accepted is None and payload.get("rejected") is True:
        accepted = False
    return engine.apply_feedback(
        task_id=str(payload.get("task_id", "") or ""),
        score=float(payload.get("score", 0.5) or 0.5),
        accepted=accepted,
        corrected_answer=str(payload.get("corrected_answer", "") or ""),
        feedback_text=str(payload.get("feedback_text", "") or ""),
        user_id=(str(payload.get("user_id", "") or "").strip() or None),
        style=(str(payload.get("style", "") or "").strip() or None),
        verbosity=(str(payload.get("verbosity", "") or "").strip() or None),
        domain_preferences=_normalized_list(payload.get("domain_preferences")) or None,
        hard_constraints=_normalized_list(payload.get("hard_constraints")) or None,
    )


def _handle_state(args: argparse.Namespace, payload: dict[str, Any]) -> dict[str, Any]:
    engine = _build_engine(args, payload)
    memory_stats = engine.memory.stats()
    return {
        "ok": True,
        "branches": engine.branch_snapshot(),
        "memory": memory_stats,
        "routing_histogram": engine.routing_histogram(),
        "runtime": {
            "evaluator_llm_enabled": engine.config.agent_runtimes.evaluator.enabled,
            "optimizer_llm_enabled": engine.config.agent_runtimes.optimizer.enabled,
            "evaluator_provider": engine.config.agent_runtimes.evaluator.provider,
            "optimizer_provider": engine.config.agent_runtimes.optimizer.provider,
        },
    }


def _handle_ingest(args: argparse.Namespace, payload: dict[str, Any]) -> dict[str, Any]:
    engine = _build_engine(args, payload)
    result = engine.openclaw_ingest(dict(payload.get("event", payload)))
    visibility = str(payload.get("visibility", "minimal") or "minimal")
    include_trace = bool(payload.get("include_trace", False))
    include_raw = bool(payload.get("include_raw", False))
    return _summarize_run_result(
        result,
        visibility=visibility,
        include_trace=include_trace,
        include_raw=include_raw,
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    payload = _read_payload(args.input_file)
    handlers = {
        "run": _handle_run,
        "feedback": _handle_feedback,
        "state": _handle_state,
        "ingest": _handle_ingest,
    }
    try:
        result = handlers[args.command](args, payload)
    except Exception as exc:  # pragma: no cover - defensive CLI boundary
        _print_json({"ok": False, "error": str(exc), "command": args.command})
        raise SystemExit(1) from exc
    _print_json(result)


if __name__ == "__main__":
    main()
