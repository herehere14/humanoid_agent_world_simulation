from __future__ import annotations

import json
from pathlib import Path


def test_openclaw_plugin_assets_are_consistent():
    project_root = Path(__file__).resolve().parents[1]
    plugin_root = project_root / "extensions" / "prompt-forest"

    package_json = json.loads((plugin_root / "package.json").read_text(encoding="utf-8"))
    manifest = json.loads((plugin_root / "openclaw.plugin.json").read_text(encoding="utf-8"))
    legacy_manifest = json.loads((plugin_root / "moltbot.plugin.json").read_text(encoding="utf-8"))
    skill_path = plugin_root / "skills" / "prompt-forest" / "SKILL.md"
    index_path = plugin_root / "index.js"

    assert package_json["name"] == "openclaw-prompt-forest"
    assert package_json["openclaw"]["extensions"] == ["./index.js"]
    assert manifest["id"] == "openclaw-prompt-forest"
    assert legacy_manifest["id"] == manifest["id"]
    assert manifest["skills"] == ["skills/prompt-forest"]
    assert skill_path.exists()
    assert index_path.exists()

    skill_text = skill_path.read_text(encoding="utf-8")
    assert skill_text.startswith("---\n")
    assert "name: prompt-forest" in skill_text
    assert "description:" in skill_text
    assert "prompt_forest_assist" in skill_text
    assert "prompt_forest_feedback" in skill_text
