"""Tests for the ModeOrchestrator: both operating modes."""

from __future__ import annotations

import pytest

from prompt_forest.backend.mock import MockLLMBackend
from prompt_forest.modes.orchestrator import ModeOrchestrator
from prompt_forest.modes.registry import ModeConfig, OperatingMode, get_mode_config


# ── Agent Improvement Mode ───────────────────────────────────────────────────

def test_agent_mode_construction():
    orch = ModeOrchestrator(mode="agent_improvement")
    assert orch.mode == OperatingMode.AGENT_IMPROVEMENT
    assert orch.human_state is None
    assert orch.human_router is None


def test_agent_mode_run_task():
    orch = ModeOrchestrator(mode="agent_improvement")
    result = orch.run_task("Explain how trees grow", task_type="general")
    assert result["mode"] == "agent_improvement"
    assert "task" in result or "selected_output" in result or "routing" in result


def test_agent_mode_get_state():
    orch = ModeOrchestrator(mode="agent_improvement")
    state = orch.get_state()
    assert state["mode"] == "agent_improvement"
    assert "branches" in state
    assert "human_state" not in state


def test_agent_mode_inject_event_returns_error():
    orch = ModeOrchestrator(mode="agent_improvement")
    result = orch.inject_event("threat")
    assert "error" in result


# ── Human Mode ───────────────────────────────────────────────────────────────

def test_human_mode_construction():
    orch = ModeOrchestrator(mode="human_mode")
    assert orch.mode == OperatingMode.HUMAN_MODE
    assert orch.human_state is not None
    assert orch.human_router is not None
    assert orch.human_evaluator is not None
    assert orch.human_memory is not None


def test_human_mode_run_task():
    orch = ModeOrchestrator(mode="human_mode")
    result = orch.run_task("Should I take a risk?", task_type="auto")

    assert result["mode"] == "human_mode"
    assert "routing" in result
    assert "evaluation_signal" in result
    assert "human_state" in result
    assert "before" in result["human_state"]
    assert "after" in result["human_state"]
    assert "experiential_memory" in result
    assert "timings" in result
    assert "branch_weights" in result


def test_human_mode_get_state():
    orch = ModeOrchestrator(mode="human_mode")
    state = orch.get_state()
    assert state["mode"] == "human_mode"
    assert "human_state" in state
    assert "mood_valence" in state
    assert "arousal" in state
    assert "dominant_drives" in state


def test_human_mode_inject_event():
    orch = ModeOrchestrator(mode="human_mode")
    result = orch.inject_event("threat", intensity=0.8)
    assert "variables" in result
    assert "turn_index" in result
    # The result should contain the updated state snapshot
    assert result["turn_index"] == 1


def test_human_mode_custom_initial_state():
    orch = ModeOrchestrator(
        mode="human_mode",
        initial_state={"confidence": 0.9, "fear": 0.8},
    )
    assert orch.human_state.get("confidence") == 0.9
    assert orch.human_state.get("fear") == 0.8


def test_human_mode_state_evolves_over_tasks():
    orch = ModeOrchestrator(mode="human_mode")
    state_before = orch.human_state.snapshot()

    orch.run_task("What should I do?", task_type="auto")

    state_after = orch.human_state.snapshot()
    assert state_after.turn_index > state_before.turn_index


def test_human_mode_memory_grows():
    orch = ModeOrchestrator(mode="human_mode")
    assert orch.human_memory.memory_count == 0

    orch.run_task("First task", task_type="auto")
    assert orch.human_memory.memory_count == 1

    orch.run_task("Second task", task_type="auto")
    assert orch.human_memory.memory_count == 2


def test_human_mode_event_callback():
    events = []
    orch = ModeOrchestrator(mode="human_mode")
    orch.run_task(
        "Test with callback",
        task_type="auto",
        event_callback=lambda e: events.append(e),
    )
    event_types = {e["type"] for e in events}
    assert "human_state" in event_types
    assert "routing" in event_types
    assert "evaluation" in event_types
    assert "task_complete" in event_types


# ── Same Task, Different State ───────────────────────────────────────────────

def test_different_states_produce_different_routing():
    """Core requirement: same task under different internal states → different behavior."""
    task_text = "Should I invest in this opportunity?"

    # Confident, curious agent
    orch_confident = ModeOrchestrator(
        mode="human_mode",
        initial_state={"confidence": 0.9, "curiosity": 0.8, "fear": 0.1, "stress": 0.1},
    )
    result_confident = orch_confident.run_task(task_text)

    # Fearful, stressed agent
    orch_fearful = ModeOrchestrator(
        mode="human_mode",
        initial_state={"confidence": 0.2, "curiosity": 0.2, "fear": 0.9, "stress": 0.8},
    )
    result_fearful = orch_fearful.run_task(task_text)

    # Routing should differ (different branch scores)
    scores_confident = result_confident["routing"]["branch_scores"]
    scores_fearful = result_fearful["routing"]["branch_scores"]

    # fear_risk should score higher when fear is high
    assert scores_fearful.get("fear_risk", 0) > scores_confident.get("fear_risk", 0)


# ── Mode Config ──────────────────────────────────────────────────────────────

def test_mode_config_with_overrides():
    cfg = get_mode_config("human_mode", overrides={"noise_level": 0.2})
    assert cfg.human_mode.noise_level == 0.2


def test_mode_config_defaults():
    cfg = get_mode_config("agent_improvement")
    assert cfg.is_agent_mode
    assert not cfg.is_human_mode


def test_human_mode_config_defaults():
    cfg = get_mode_config("human_mode")
    assert cfg.is_human_mode
    assert cfg.human_mode.coherence_weight == 0.3
    assert cfg.human_mode.conflict_resolution_strategy == "weighted_compromise"
