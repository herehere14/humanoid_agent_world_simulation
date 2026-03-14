from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ..types import TaskInput
from .base import LLMBackend


class OpenAIChatBackend(LLMBackend):
    """Real text backend for OpenAI-compatible chat generation."""

    def __init__(
        self,
        *,
        model: str,
        api_key_env: str = "OPENAI_API_KEY",
        base_url: str = "https://api.openai.com/v1",
        temperature: float = 0.2,
        max_output_tokens: int = 700,
        timeout_seconds: int = 90,
        max_retries: int = 2,
        retry_backoff_seconds: float = 1.5,
        api_mode: str = "chat_completions",
        reasoning_effort: str | None = None,
        seed: int | None = 42,
        system_prompt: str = "You are a careful assistant. Follow the user's instructions exactly.",
    ) -> None:
        self.model = model
        self.api_key_env = api_key_env
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.timeout_seconds = timeout_seconds
        self.max_retries = max(0, int(max_retries))
        self.retry_backoff_seconds = max(0.0, float(retry_backoff_seconds))
        self.api_mode = api_mode
        self.reasoning_effort = reasoning_effort
        self.seed = seed
        self.system_prompt = system_prompt
        self._calls: list[dict[str, Any]] = []

    def generate(self, prompt: str, task: TaskInput, branch_name: str) -> tuple[str, dict[str, Any]]:
        _ = task
        key = os.getenv(self.api_key_env, "").strip()
        if not key:
            raise RuntimeError(f"Missing API key env var: {self.api_key_env}")

        url, body = self._request_payload(prompt)

        payload = json.dumps(body, ensure_ascii=True).encode("utf-8")
        req = Request(
            url,
            data=payload,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        raw = ""
        last_error = ""
        for attempt in range(self.max_retries + 1):
            started = time.perf_counter()
            try:
                with urlopen(req, timeout=self.timeout_seconds) as resp:
                    raw = resp.read().decode("utf-8")
                break
            except HTTPError as exc:
                body_text = exc.read().decode("utf-8", errors="replace")
                last_error = f"HTTPError {exc.code}: {body_text}"
                retryable = exc.code in {408, 409, 429, 500, 502, 503, 504}
                if retryable and attempt < self.max_retries:
                    time.sleep(self.retry_backoff_seconds * (attempt + 1))
                    continue
                self._record_call(
                    endpoint=url,
                    branch_name=branch_name,
                    latency_ms=(time.perf_counter() - started) * 1000.0,
                    usage={},
                    ok=False,
                    error=last_error,
                )
                raise RuntimeError(last_error) from exc
            except (URLError, TimeoutError) as exc:
                last_error = f"Network error: {exc}"
                if attempt < self.max_retries:
                    time.sleep(self.retry_backoff_seconds * (attempt + 1))
                    continue
                self._record_call(
                    endpoint=url,
                    branch_name=branch_name,
                    latency_ms=(time.perf_counter() - started) * 1000.0,
                    usage={},
                    ok=False,
                    error=last_error,
                )
                raise RuntimeError(last_error) from exc

        latency_ms = (time.perf_counter() - started) * 1000.0
        data = json.loads(raw)
        text = self._extract_text(data)
        usage = self._normalize_usage(data.get("usage", {}))
        if not text:
            self._record_call(
                endpoint=url,
                branch_name=branch_name,
                latency_ms=latency_ms,
                usage=usage,
                ok=False,
                error="No text returned from OpenAI-compatible backend.",
            )
            raise RuntimeError("No text returned from OpenAI-compatible backend.")

        self._record_call(
            endpoint=url,
            branch_name=branch_name,
            latency_ms=latency_ms,
            usage=usage,
            ok=True,
        )

        meta = {
            "provider": "openai_compatible",
            "model": self.model,
            "branch": branch_name,
            "latency_ms": round(latency_ms, 3),
            **usage,
        }
        if "id" in data:
            meta["response_id"] = data["id"]
        return text, meta

    def generate_stream(
        self,
        prompt: str,
        task: TaskInput,
        branch_name: str,
        *,
        on_delta: Callable[[str], None] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        if on_delta is None or self.api_mode != "chat_completions":
            return super().generate_stream(prompt, task, branch_name, on_delta=on_delta)

        _ = task
        key = os.getenv(self.api_key_env, "").strip()
        if not key:
            raise RuntimeError(f"Missing API key env var: {self.api_key_env}")

        url = f"{self.base_url}/chat/completions"
        body = self._chat_completions_body(prompt)
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
        payload = json.dumps(body, ensure_ascii=True).encode("utf-8")
        req = Request(
            url,
            data=payload,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        chunks: list[str] = []
        usage: dict[str, int] = {}
        response_id = ""
        last_error = ""
        started = time.perf_counter()

        for attempt in range(self.max_retries + 1):
            chunks.clear()
            usage = {}
            response_id = ""
            started = time.perf_counter()
            try:
                with urlopen(req, timeout=self.timeout_seconds) as resp:
                    for raw_line in resp:
                        line = raw_line.decode("utf-8", errors="replace").strip()
                        if not line or not line.startswith("data:"):
                            continue
                        data_str = line[5:].strip()
                        if not data_str:
                            continue
                        if data_str == "[DONE]":
                            break
                        event = json.loads(data_str)
                        if not response_id:
                            response_id = str(event.get("id", "") or "")
                        if "usage" in event:
                            usage = self._normalize_usage(event.get("usage", {}))
                        for choice in event.get("choices", []) or []:
                            delta = choice.get("delta", {}) or {}
                            piece = self._delta_text(delta)
                            if piece:
                                chunks.append(piece)
                                on_delta(piece)
                break
            except HTTPError as exc:
                body_text = exc.read().decode("utf-8", errors="replace")
                last_error = f"HTTPError {exc.code}: {body_text}"
                retryable = exc.code in {408, 409, 429, 500, 502, 503, 504}
                if retryable and attempt < self.max_retries:
                    time.sleep(self.retry_backoff_seconds * (attempt + 1))
                    continue
                self._record_call(
                    endpoint=url,
                    branch_name=branch_name,
                    latency_ms=(time.perf_counter() - started) * 1000.0,
                    usage={},
                    ok=False,
                    error=last_error,
                )
                raise RuntimeError(last_error) from exc
            except (URLError, TimeoutError) as exc:
                last_error = f"Network error: {exc}"
                if attempt < self.max_retries:
                    time.sleep(self.retry_backoff_seconds * (attempt + 1))
                    continue
                self._record_call(
                    endpoint=url,
                    branch_name=branch_name,
                    latency_ms=(time.perf_counter() - started) * 1000.0,
                    usage={},
                    ok=False,
                    error=last_error,
                )
                raise RuntimeError(last_error) from exc

        latency_ms = (time.perf_counter() - started) * 1000.0
        text = "".join(chunks).strip()
        if not text:
            self._record_call(
                endpoint=url,
                branch_name=branch_name,
                latency_ms=latency_ms,
                usage=usage,
                ok=False,
                error="No text returned from OpenAI-compatible backend stream.",
            )
            raise RuntimeError("No text returned from OpenAI-compatible backend stream.")

        self._record_call(
            endpoint=url,
            branch_name=branch_name,
            latency_ms=latency_ms,
            usage=usage,
            ok=True,
        )
        meta = {
            "provider": "openai_compatible",
            "model": self.model,
            "branch": branch_name,
            "latency_ms": round(latency_ms, 3),
            **usage,
            "streamed": True,
        }
        if response_id:
            meta["response_id"] = response_id
        return text, meta

    def _request_payload(self, prompt: str) -> tuple[str, dict[str, Any]]:
        if self.api_mode == "responses":
            url = f"{self.base_url}/responses"
            body: dict[str, Any] = {
                "model": self.model,
                "input": prompt,
                "instructions": self.system_prompt,
                "max_output_tokens": self.max_output_tokens,
            }
            if self.reasoning_effort:
                body["reasoning"] = {"effort": self.reasoning_effort}
            return url, body

        url = f"{self.base_url}/chat/completions"
        return url, self._chat_completions_body(prompt)

    def _extract_text(self, data: dict[str, Any]) -> str:
        if self.api_mode == "responses":
            output = data.get("output", [])
            parts: list[str] = []
            for item in output:
                if not isinstance(item, dict):
                    continue
                if item.get("type") != "message":
                    continue
                for content in item.get("content", []):
                    if not isinstance(content, dict):
                        continue
                    if content.get("type") == "output_text":
                        parts.append(str(content.get("text", "")))
            return "\n".join(part for part in parts if part).strip()

        choices = data.get("choices", [])
        if not choices:
            return ""
        content = choices[0].get("message", {}).get("content", "")
        if isinstance(content, list):
            parts = [part.get("text", "") for part in content if isinstance(part, dict)]
            return "\n".join(parts).strip()
        return str(content).strip()

    def _chat_completions_body(self, prompt: str) -> dict[str, Any]:
        body = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_output_tokens,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
        }
        if self.seed is not None:
            body["seed"] = int(self.seed)
        return body

    @staticmethod
    def _delta_text(delta: dict[str, Any]) -> str:
        content = delta.get("content", "")
        if isinstance(content, list):
            parts = [str(part.get("text", "")) for part in content if isinstance(part, dict)]
            return "".join(parts)
        return str(content or "")

    def call_log(self) -> list[dict[str, Any]]:
        return [dict(item) for item in self._calls]

    def usage_summary(self) -> dict[str, Any]:
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        total_latency_ms = 0.0
        ok_calls = 0
        error_calls = 0
        for item in self._calls:
            usage = item.get("usage", {}) or {}
            prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
            completion_tokens += int(usage.get("completion_tokens", 0) or 0)
            total_tokens += int(usage.get("total_tokens", 0) or 0)
            total_latency_ms += float(item.get("latency_ms", 0.0) or 0.0)
            if item.get("ok", False):
                ok_calls += 1
            else:
                error_calls += 1
        return {
            "call_count": len(self._calls),
            "ok_calls": ok_calls,
            "error_calls": error_calls,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "total_latency_ms": round(total_latency_ms, 3),
            "mean_latency_ms": round(total_latency_ms / max(1, len(self._calls)), 3),
        }

    def reset_usage(self) -> None:
        self._calls.clear()

    @staticmethod
    def _normalize_usage(usage: dict[str, Any] | None) -> dict[str, int]:
        payload = usage or {}
        prompt_tokens = int(payload.get("prompt_tokens", payload.get("input_tokens", 0)) or 0)
        completion_tokens = int(payload.get("completion_tokens", payload.get("output_tokens", 0)) or 0)
        total_tokens = int(payload.get("total_tokens", prompt_tokens + completion_tokens) or 0)
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

    def _record_call(
        self,
        *,
        endpoint: str,
        branch_name: str,
        latency_ms: float,
        usage: dict[str, Any],
        ok: bool,
        error: str = "",
    ) -> None:
        self._calls.append(
            {
                "endpoint": endpoint,
                "model": self.model,
                "branch_name": branch_name,
                "latency_ms": round(latency_ms, 3),
                "usage": dict(usage),
                "ok": ok,
                "error": error,
            }
        )
