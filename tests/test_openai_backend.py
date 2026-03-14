from __future__ import annotations

import json
import os
from urllib.error import URLError

from prompt_forest.backend.openai_chat import OpenAIChatBackend
from prompt_forest.types import TaskInput


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


def test_openai_backend_parses_text_and_usage(monkeypatch):
    payload = {
        "id": "resp_123",
        "choices": [{"message": {"content": "hello world"}}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
    }

    def fake_urlopen(req, timeout):  # noqa: ANN001
        assert timeout == 12
        assert req.full_url.endswith("/chat/completions")
        return _FakeResponse(payload)

    monkeypatch.setattr("prompt_forest.backend.openai_chat.urlopen", fake_urlopen)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    backend = OpenAIChatBackend(model="gpt-4.1-mini", timeout_seconds=12)
    task = TaskInput(task_id="t1", text="hi", task_type="general")
    text, meta = backend.generate("say hi", task, "branch_x")

    assert text == "hello world"
    assert meta["response_id"] == "resp_123"
    assert meta["prompt_tokens"] == 11
    assert meta["completion_tokens"] == 7
    assert meta["total_tokens"] == 18
    assert backend.usage_summary()["total_tokens"] == 18

    os.environ.pop("OPENAI_API_KEY", None)


def test_openai_backend_retries_transient_network_errors(monkeypatch):
    payload = {
        "choices": [{"message": {"content": "hello again"}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 4, "total_tokens": 9},
    }
    attempts = {"count": 0}

    def fake_urlopen(req, timeout):  # noqa: ANN001
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise URLError("temporary failure")
        return _FakeResponse(payload)

    monkeypatch.setattr("prompt_forest.backend.openai_chat.urlopen", fake_urlopen)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    backend = OpenAIChatBackend(model="gpt-4.1-mini", timeout_seconds=12, max_retries=2, retry_backoff_seconds=0.0)
    task = TaskInput(task_id="t2", text="hi", task_type="general")
    text, meta = backend.generate("say hi", task, "branch_y")

    assert text == "hello again"
    assert meta["total_tokens"] == 9
    assert attempts["count"] == 3


def test_openai_backend_supports_responses_api(monkeypatch):
    payload = {
        "id": "resp_456",
        "output": [
            {
                "type": "message",
                "content": [
                    {"type": "output_text", "text": "ok from responses"},
                ],
            }
        ],
        "usage": {"input_tokens": 13, "output_tokens": 8, "total_tokens": 21},
    }

    def fake_urlopen(req, timeout):  # noqa: ANN001
        assert timeout == 12
        assert req.full_url.endswith("/responses")
        body = json.loads(req.data.decode("utf-8"))
        assert body["model"] == "gpt-5.3-codex"
        assert body["input"] == "say hi"
        assert body["instructions"]
        return _FakeResponse(payload)

    monkeypatch.setattr("prompt_forest.backend.openai_chat.urlopen", fake_urlopen)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    backend = OpenAIChatBackend(model="gpt-5.3-codex", timeout_seconds=12, api_mode="responses")
    task = TaskInput(task_id="t3", text="hi", task_type="general")
    text, meta = backend.generate("say hi", task, "branch_z")

    assert text == "ok from responses"
    assert meta["response_id"] == "resp_456"
    assert meta["prompt_tokens"] == 13
    assert meta["completion_tokens"] == 8
    assert meta["total_tokens"] == 21
