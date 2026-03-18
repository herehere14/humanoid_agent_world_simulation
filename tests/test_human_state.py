"""Tests for the Human Mode state system."""

from __future__ import annotations

import pytest

from prompt_forest.state.human_state import (
    DEFAULT_BASELINES,
    DRIVE_OPPOSITIONS,
    DriveConflict,
    HumanState,
    StateSnapshot,
)


# ── Construction ──────────────────────────────────────────────────────────────

def test_default_construction():
    state = HumanState()
    assert state.turn_index == 0
    for var, baseline in DEFAULT_BASELINES.items():
        assert state.get(var) == baseline


def test_custom_initial_values():
    state = HumanState(initial_values={"confidence": 0.9, "stress": 0.8})
    assert state.get("confidence") == 0.9
    assert state.get("stress") == 0.8
    # Others should still be baseline
    assert state.get("trust") == DEFAULT_BASELINES["trust"]


def test_values_clamped():
    state = HumanState(initial_values={"confidence": 1.5, "stress": -0.3})
    assert state.get("confidence") == 1.0
    assert state.get("stress") == 0.0


# ── Mood & Drives ────────────────────────────────────────────────────────────

def test_mood_valence_range():
    state = HumanState()
    mood = state.mood_valence()
    assert -1.0 <= mood <= 1.0


def test_positive_mood():
    state = HumanState(initial_values={
        "confidence": 0.9, "motivation": 0.9, "curiosity": 0.9,
        "trust": 0.9, "ambition": 0.9,
        "stress": 0.1, "frustration": 0.1, "fear": 0.1,
        "fatigue": 0.1, "self_protection": 0.1,
    })
    assert state.mood_valence() > 0.3


def test_negative_mood():
    state = HumanState(initial_values={
        "confidence": 0.1, "motivation": 0.1, "curiosity": 0.1,
        "trust": 0.1, "ambition": 0.1,
        "stress": 0.9, "frustration": 0.9, "fear": 0.9,
        "fatigue": 0.9, "self_protection": 0.9,
    })
    assert state.mood_valence() < -0.3


def test_dominant_drives():
    state = HumanState(initial_values={"curiosity": 0.95, "fear": 0.90, "ambition": 0.85})
    top = state.dominant_drives(top_k=3)
    assert len(top) == 3
    assert top[0] == "curiosity"


def test_arousal_level():
    state = HumanState()
    arousal = state.arousal_level()
    assert 0.0 <= arousal <= 1.0


def test_drive_strength_penalised_by_fatigue():
    state = HumanState(initial_values={"curiosity": 0.8, "fatigue": 0.0})
    strong = state.drive_strength("curiosity")
    state.set("fatigue", 0.9)
    weak = state.drive_strength("curiosity")
    assert weak < strong


# ── State Updates ────────────────────────────────────────────────────────────

def test_update_increments_turn():
    state = HumanState()
    state.update({"confidence": 0.1})
    assert state.turn_index == 1
    state.update({"confidence": 0.1})
    assert state.turn_index == 2


def test_update_returns_snapshot():
    state = HumanState()
    snap = state.update({"stress": 0.2})
    assert isinstance(snap, StateSnapshot)
    assert snap.turn_index == 1


def test_momentum_dampens_changes():
    """With momentum 0.7, changes should be gradual not instantaneous."""
    state = HumanState(initial_values={"confidence": 0.5}, momentum=0.7, noise_level=0.0)
    state.update({"confidence": 0.5})  # Try to jump to 1.0
    # Should NOT be 1.0 due to momentum and decay
    assert state.get("confidence") < 0.95


def test_decay_toward_baseline():
    """Variables should decay toward baseline over many turns."""
    state = HumanState(initial_values={"stress": 0.9}, noise_level=0.0, decay_rate=0.1)
    baseline = DEFAULT_BASELINES["stress"]
    for _ in range(20):
        state.update({})
    # Should have moved toward baseline
    assert abs(state.get("stress") - baseline) < abs(0.9 - baseline)


# ── Conflict Detection ──────────────────────────────────────────────────────

def test_conflict_detection_opposing_drives():
    """When two opposing drives are both high and close, a conflict emerges."""
    state = HumanState(
        initial_values={"curiosity": 0.7, "fear": 0.65},
        noise_level=0.0,
    )
    state.update({})  # Trigger detection
    conflicts = state.active_conflicts
    # Should detect curiosity vs fear
    conflict_pairs = {(c.drive_a, c.drive_b) for c in conflicts}
    assert any(
        {"curiosity", "fear"} == {a, b}
        for a, b in conflict_pairs
    ), f"Expected curiosity/fear conflict, got {conflict_pairs}"


def test_no_conflict_when_drives_low():
    state = HumanState(
        initial_values={"curiosity": 0.2, "fear": 0.2},
        noise_level=0.0,
    )
    state.update({})
    conflicts = state.active_conflicts
    # No curiosity/fear conflict expected
    for c in conflicts:
        pair = {c.drive_a, c.drive_b}
        assert pair != {"curiosity", "fear"}


def test_conflict_resolution_dominant():
    conflict = DriveConflict(drive_a="curiosity", drive_b="fear", intensity=0.5)
    state = HumanState(initial_values={"curiosity": 0.8, "fear": 0.5})
    resolved = state.resolve_conflict(conflict, strategy="dominant")
    assert resolved.resolution == "curiosity"
    assert resolved.resolution_weight < 0.5  # drive_a wins → weight closer to 0


def test_conflict_resolution_compromise():
    conflict = DriveConflict(drive_a="curiosity", drive_b="fear", intensity=0.5)
    state = HumanState(initial_values={"curiosity": 0.6, "fear": 0.6})
    resolved = state.resolve_conflict(conflict, strategy="weighted_compromise")
    assert resolved.resolution == "compromise"
    assert 0.3 < resolved.resolution_weight < 0.7  # roughly balanced


# ── Outcome Application ─────────────────────────────────────────────────────

def test_positive_outcome_boosts_confidence():
    state = HumanState(initial_values={"confidence": 0.5}, noise_level=0.0)
    state.apply_outcome(reward=0.9, task_type="general")
    assert state.get("confidence") > 0.5


def test_negative_outcome_increases_stress():
    state = HumanState(initial_values={"stress": 0.2}, noise_level=0.0)
    state.apply_outcome(reward=0.1, task_type="general")
    assert state.get("stress") > 0.2


def test_repeated_failure_compounds():
    """Repeated failures should compound frustration."""
    state = HumanState(initial_values={"frustration": 0.1}, noise_level=0.0)
    for _ in range(5):
        state.apply_outcome(reward=0.1, task_type="general")
    assert state.get("frustration") > 0.25


# ── Event Injection ──────────────────────────────────────────────────────────

def test_threat_event_increases_fear():
    state = HumanState(initial_values={"fear": 0.2}, noise_level=0.0)
    state.inject_event("threat", intensity=0.8)
    assert state.get("fear") > 0.2


def test_rest_event_reduces_fatigue():
    state = HumanState(initial_values={"fatigue": 0.7}, noise_level=0.0)
    state.inject_event("rest", intensity=0.8)
    assert state.get("fatigue") < 0.7


def test_unknown_event_type_no_crash():
    state = HumanState()
    snap = state.inject_event("unknown_event", intensity=0.5)
    assert isinstance(snap, StateSnapshot)


# ── Cross Effects ────────────────────────────────────────────────────────────

def test_high_stress_reduces_reflection():
    state = HumanState(
        initial_values={"stress": 0.85, "reflection": 0.6},
        noise_level=0.0,
    )
    state.update({})
    assert state.get("reflection") < 0.6


def test_high_fatigue_reduces_motivation():
    state = HumanState(
        initial_values={"fatigue": 0.9, "motivation": 0.6},
        noise_level=0.0,
    )
    state.update({})
    assert state.get("motivation") < 0.6


# ── Serialisation ────────────────────────────────────────────────────────────

def test_to_dict_and_from_dict():
    state = HumanState(initial_values={"confidence": 0.8, "stress": 0.3})
    state.update({"fear": 0.1})
    data = state.to_dict()

    restored = HumanState.from_dict(data)
    assert restored.get("confidence") == pytest.approx(state.get("confidence"), abs=0.01)
    assert restored.get("stress") == pytest.approx(state.get("stress"), abs=0.01)


def test_snapshot_to_dict():
    state = HumanState()
    snap = state.snapshot()
    d = snap.to_dict()
    assert "variables" in d
    assert "mood_valence" in d
    assert "dominant_drives" in d


# ── Routing Bias ─────────────────────────────────────────────────────────────

def test_routing_bias_structure():
    state = HumanState()
    biases = state.routing_bias()
    assert isinstance(biases, dict)
    assert "reflective_reasoning" in biases
    assert "impulse_response" in biases
    for v in biases.values():
        assert 0.0 <= v <= 1.0
