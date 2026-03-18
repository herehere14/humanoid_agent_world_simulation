from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"


def _run_bridge(command: str, payload: dict[str, object]) -> dict[str, object]:
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{SRC_DIR}{os.pathsep}{env.get('PYTHONPATH', '')}".rstrip(os.pathsep)
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "prompt_forest.openclaw_bridge",
            "--config",
            str(PROJECT_ROOT / "configs" / "default.json"),
            command,
        ],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        env=env,
        cwd=PROJECT_ROOT,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    return json.loads(proc.stdout)


def test_openclaw_bridge_run_feedback_and_state(tmp_path):
    artifacts_dir = tmp_path / "artifacts"
    run_result = _run_bridge(
        "run",
        {
            "task": "Plan a migration rollout with owners, risks, and confidence.",
            "task_type": "planning",
            "user_id": "openclaw-user",
            "expected_keywords": ["owners", "risks", "confidence"],
            "required_checks": ["confidence"],
            "visibility": "full",
            "include_trace": True,
            "seed": 17,
            "artifacts_dir": str(artifacts_dir),
        },
    )

    assert run_result["ok"] is True
    assert run_result["answer"]
    assert run_result["selected_branch"]
    assert run_result["task_type"] == "planning"
    assert "trace" in run_result
    assert "[task]" in str(run_result["trace"])

    feedback_result = _run_bridge(
        "feedback",
        {
            "task_id": run_result["task_id"],
            "score": 5,
            "accepted": True,
            "feedback_text": "This routing was useful.",
            "user_id": "openclaw-user",
            "seed": 17,
            "artifacts_dir": str(artifacts_dir),
        },
    )

    assert feedback_result["ok"] is True
    assert feedback_result["task_id"] == run_result["task_id"]
    assert float(feedback_result["new_reward"]) >= 0.0

    state_result = _run_bridge(
        "state",
        {
            "seed": 17,
            "artifacts_dir": str(artifacts_dir),
        },
    )

    assert state_result["ok"] is True
    assert int(state_result["memory"]["records"]) >= 1


def test_openclaw_bridge_ingest_supports_trajectory_events(tmp_path):
    result = _run_bridge(
        "ingest",
        {
            "event": {
                "episode_id": "ep-001",
                "task": "Verify a draft answer against strict constraints and include confidence.",
                "task_type": "general",
                "user_id": "trajectory-user",
                "expected_keywords": ["verify", "constraints", "confidence"],
                "required_checks": ["confidence"],
            },
            "include_trace": True,
            "seed": 13,
            "artifacts_dir": str(tmp_path / "artifacts"),
        },
    )

    assert result["ok"] is True
    assert result["answer"]
    assert result["selected_branch"]
    assert "trace" in result
