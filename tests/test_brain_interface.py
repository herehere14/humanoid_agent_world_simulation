from __future__ import annotations

from prompt_forest.adapters.openclaw_adapter import OpenClawAdapter, OpenClawTrajectory
from prompt_forest.behavioral.predictor import BehavioralPredictor
from prompt_forest.brain import BrainController
from prompt_forest.modes.orchestrator import ModeOrchestrator
from prompt_forest.state.human_state import DriveConflict, HumanState
from prompt_forest.types import RoutingDecision


def test_brain_controller_builds_portable_output():
    controller = BrainController()
    state = HumanState(
        initial_values={
            "fear": 0.75,
            "caution": 0.72,
            "ambition": 0.48,
            "curiosity": 0.33,
            "self_protection": 0.62,
        },
        noise_level=0.0,
    )
    route = RoutingDecision(
        task_type="threat_response",
        activated_branches=["fear_risk", "loss_aversion", "self_protection"],
        branch_scores={"fear_risk": 0.82, "loss_aversion": 0.76, "self_protection": 0.63},
        activated_paths=[["affect", "fear_risk", "loss_aversion"]],
    )
    conflicts = [
        DriveConflict(
            drive_a="fear",
            drive_b="curiosity",
            intensity=0.44,
            resolution="compromise",
            resolution_weight=0.32,
        )
    ]

    output = controller.build_output(
        state=state,
        route=route,
        conflicts=conflicts,
        human_memory=None,
        branch_weights={"fear_risk": 1.2, "loss_aversion": 1.1, "self_protection": 0.9},
    )

    assert output.regime == "guarded_avoidant"
    assert output.control_signals.avoidance_drive > output.control_signals.approach_drive
    assert output.action_tendencies.inhibit > output.action_tendencies.act
    assert output.conflicts[0].name == "fear_vs_curiosity"


def test_human_mode_orchestrator_emits_brain_output():
    orch = ModeOrchestrator(
        mode="human_mode",
        initial_state={"confidence": 0.8, "curiosity": 0.75, "fear": 0.2},
    )
    result = orch.run_task("Should I explore this option?", task_type="auto")

    assert "brain_output" in result
    assert result["brain_output"]["regime"]
    assert "control_signals" in result["brain_output"]
    assert "action_tendencies" in result["brain_output"]

    state = orch.get_state()
    assert "brain_state" in state
    assert orch.get_last_brain_output() is not None


def test_predictor_flattens_brain_output():
    brain_output = {
        "regime": "exploratory_open",
        "state": {"curiosity": 0.8, "fear": 0.2},
        "dominant_drives": ["curiosity", "confidence"],
        "branch_activations": {"curiosity_exploration": 0.91},
        "control_signals": {"exploration_drive": 0.82, "avoidance_drive": 0.18},
        "action_tendencies": {"explore": 0.88, "act": 0.72},
        "memory_biases": {"curiosity_exploration": 0.12},
        "state_summary": {"mood_valence": 0.24},
        "conflicts": [{"name": "curiosity_vs_fear", "intensity": 0.21}],
    }

    features = BehavioralPredictor.brain_output_features(brain_output)

    assert features["brain_state::curiosity"] == 0.8
    assert features["brain_drive::curiosity"] == 1.0
    assert features["brain_control::exploration_drive"] == 0.82
    assert features["brain_regime::exploratory_open"] == 1.0
    assert features["brain_conflict_count"] == 1.0


def test_openclaw_adapter_surfaces_brain_output_for_orchestrator():
    orch = ModeOrchestrator(mode="human_mode")
    adapter = OpenClawAdapter(orch)
    trajectory = OpenClawTrajectory(
        episode_id="ep-1",
        task="Decide whether to take a risky shortcut.",
        task_type="general",
    )

    payload = adapter.process_trajectory(trajectory)

    assert payload["episode_id"] == "ep-1"
    assert "brain_output" in payload
    assert "brain_state" in payload
