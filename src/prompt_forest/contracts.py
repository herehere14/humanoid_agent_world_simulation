from __future__ import annotations

import json
import re
from typing import Any


_KNOWN_CONTRACTS = {"json_lock", "csv_lock", "code_patch_lock", "bullet_lock"}


def infer_output_contract(task_text: str, metadata: dict[str, Any] | None = None) -> str | None:
    meta = metadata or {}
    explicit = str(meta.get("output_contract", "")).strip().lower()
    if explicit in _KNOWN_CONTRACTS:
        return explicit

    text = task_text.lower()
    if "json" in text and ("respond only" in text or "output only" in text or "json with keys" in text):
        return "json_lock"
    if "csv" in text and ("output only" in text or "csv lines" in text or "no header" in text):
        return "csv_lock"
    if "fix:" in text and "tests:" in text:
        return "code_patch_lock"
    if "bullet" in text and ("output bullets only" in text or "each bullet must start" in text):
        return "bullet_lock"
    return None


def evaluate_output_contract(output: str, contract: str, task_text: str = "") -> tuple[bool, str]:
    if contract == "json_lock":
        return _check_json_lock(output)
    if contract == "csv_lock":
        return _check_csv_lock(output, task_text)
    if contract == "code_patch_lock":
        return _check_code_patch_lock(output)
    if contract == "bullet_lock":
        return _check_bullet_lock(output, task_text)
    return True, "no_contract"


def _strip_fences(text: str) -> str:
    t = text.strip()
    if not t.startswith("```"):
        return t
    lines = [ln for ln in t.splitlines() if not ln.strip().startswith("```")]
    return "\n".join(lines).strip()


def _check_json_lock(output: str) -> tuple[bool, str]:
    cleaned = _strip_fences(output).strip()
    if not (cleaned.startswith("{") and cleaned.endswith("}")):
        return False, "not_pure_json_object"
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return False, "json_parse_error"
    if not isinstance(parsed, dict):
        return False, "json_not_object"
    return True, "json_valid"


def _check_csv_lock(output: str, task_text: str) -> tuple[bool, str]:
    cleaned = _strip_fences(output).strip()
    lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
    if not lines:
        return False, "empty_csv"
    if any(ln.count(",") != 1 for ln in lines):
        return False, "malformed_csv_rows"
    if any(re.search(r"\b(pass|because|verification)\b", ln, flags=re.IGNORECASE) for ln in lines):
        return False, "extra_explanatory_tokens"

    expected_rows = _extract_expected_csv_rows(task_text)
    if expected_rows and len(lines) != expected_rows:
        return False, "wrong_row_count"
    return True, "csv_valid"


def _extract_expected_csv_rows(task_text: str) -> int:
    lines = [ln.strip() for ln in task_text.splitlines() if ln.strip()]
    data_lines = [ln for ln in lines if "," in ln and not ln.lower().startswith("output")]
    return len(data_lines)


def _check_code_patch_lock(output: str) -> tuple[bool, str]:
    cleaned = output.strip()
    if "FIX:" not in cleaned or "TESTS:" not in cleaned:
        return False, "missing_fix_or_tests_header"
    if "```" in cleaned:
        return False, "contains_code_fences"

    try:
        after_fix = cleaned.split("FIX:", 1)[1]
        fix_section, tests_section = after_fix.split("TESTS:", 1)
    except ValueError:
        return False, "malformed_fix_tests_layout"

    if "def " not in fix_section:
        return False, "missing_function_def"
    test_lines = [ln.strip() for ln in tests_section.splitlines() if ln.strip().startswith("-")]
    if len(test_lines) < 2:
        return False, "insufficient_tests"
    return True, "code_patch_valid"


def _check_bullet_lock(output: str, task_text: str) -> tuple[bool, str]:
    cleaned = _strip_fences(output).strip()
    lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
    if not lines:
        return False, "empty_bullet_output"
    if any(not ln.startswith("- ") for ln in lines):
        return False, "non_bullet_line_detected"

    expected = _extract_expected_bullet_count(task_text)
    if expected is not None and len(lines) != expected:
        return False, "wrong_bullet_count"
    return True, "bullet_valid"


def _extract_expected_bullet_count(task_text: str) -> int | None:
    m = re.search(r"exactly\s+(\d+)\s+bullet", task_text, flags=re.IGNORECASE)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None
