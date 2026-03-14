from __future__ import annotations

import json
import os
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ..config import AgentRuntimeConfig


class AgentRuntimeClient:
    """Thin JSON-generation client for evaluator/optimizer LLM runtime calls."""

    def __init__(self, config: AgentRuntimeConfig) -> None:
        self.config = config
        self._calls: list[dict[str, Any]] = []

    def is_enabled(self) -> bool:
        return bool(self.config.enabled)

    def generate_json(self, system_prompt: str, user_payload: dict[str, Any]) -> dict[str, Any]:
        if not self.is_enabled():
            raise RuntimeError("Agent runtime is disabled.")

        provider = self.config.provider.strip().lower()
        if provider in {"openai", "openai_compatible", "openai-compatible"}:
            text = self._call_openai_compatible(system_prompt, user_payload)
        elif provider in {"gemini", "google_gemini"}:
            text = self._call_gemini(system_prompt, user_payload)
        else:
            raise RuntimeError(f"Unsupported agent runtime provider: {self.config.provider}")

        return self._parse_json_text(text)

    def call_log(self) -> list[dict[str, Any]]:
        return [dict(item) for item in self._calls]

    def usage_summary(self) -> dict[str, Any]:
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        ok_calls = 0
        error_calls = 0
        latency_ms = 0.0
        for item in self._calls:
            usage = item.get("usage", {}) or {}
            prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
            completion_tokens += int(usage.get("completion_tokens", 0) or 0)
            total_tokens += int(usage.get("total_tokens", 0) or 0)
            latency_ms += float(item.get("latency_ms", 0.0) or 0.0)
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
            "total_latency_ms": round(latency_ms, 3),
            "mean_latency_ms": round(latency_ms / max(1, len(self._calls)), 3),
        }

    def reset_usage(self) -> None:
        self._calls.clear()

    def _call_openai_compatible(self, system_prompt: str, user_payload: dict[str, Any]) -> str:
        key = os.getenv(self.config.api_key_env, "").strip()
        if not key:
            raise RuntimeError(f"Missing API key env var: {self.config.api_key_env}")

        url = f"{self.config.base_url.rstrip('/')}/chat/completions"
        body = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_output_tokens,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=True)},
            ],
        }
        if self.config.seed is not None:
            body["seed"] = int(self.config.seed)
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
        started = time.perf_counter()
        try:
            text = self._post(req)
        except Exception as exc:
            self._record_call(
                provider="openai_compatible",
                endpoint=url,
                model=self.config.model,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                usage={},
                ok=False,
                error=str(exc),
            )
            raise

        latency_ms = (time.perf_counter() - started) * 1000.0
        data = json.loads(text)
        usage = self._normalize_openai_usage(data.get("usage", {}))
        self._record_call(
            provider="openai_compatible",
            endpoint=url,
            model=self.config.model,
            latency_ms=latency_ms,
            usage=usage,
            ok=True,
        )
        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError("No choices returned from openai-compatible runtime.")
        content = choices[0].get("message", {}).get("content", "")
        if isinstance(content, list):
            parts = [p.get("text", "") for p in content if isinstance(p, dict)]
            return "\n".join(parts).strip()
        return str(content)

    def _call_gemini(self, system_prompt: str, user_payload: dict[str, Any]) -> str:
        key = os.getenv(self.config.api_key_env, "").strip()
        if not key:
            raise RuntimeError(f"Missing API key env var: {self.config.api_key_env}")

        model = self.config.model.strip()
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
        body = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"parts": [{"text": json.dumps(user_payload, ensure_ascii=True)}]}],
            "generationConfig": {
                "temperature": self.config.temperature,
                "maxOutputTokens": self.config.max_output_tokens,
                "responseMimeType": "application/json",
            },
        }
        payload = json.dumps(body, ensure_ascii=True).encode("utf-8")
        req = Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = time.perf_counter()
        try:
            text = self._post(req)
        except Exception as exc:
            self._record_call(
                provider="gemini",
                endpoint=url,
                model=self.config.model,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                usage={},
                ok=False,
                error=str(exc),
            )
            raise

        latency_ms = (time.perf_counter() - started) * 1000.0
        data = json.loads(text)
        self._record_call(
            provider="gemini",
            endpoint=url,
            model=self.config.model,
            latency_ms=latency_ms,
            usage={},
            ok=True,
        )
        candidates = data.get("candidates", [])
        if not candidates:
            raise RuntimeError("No candidates returned from Gemini runtime.")
        parts = candidates[0].get("content", {}).get("parts", [])
        joined = "\n".join(str(part.get("text", "")) for part in parts if isinstance(part, dict)).strip()
        if not joined:
            raise RuntimeError("Gemini runtime response missing text content.")
        return joined

    def _post(self, req: Request) -> str:
        try:
            with urlopen(req, timeout=self.config.timeout_seconds) as resp:
                return resp.read().decode("utf-8")
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTPError {exc.code}: {body}") from exc
        except URLError as exc:
            raise RuntimeError(f"Network error: {exc}") from exc

    @staticmethod
    def _parse_json_text(raw: str) -> dict[str, Any]:
        text = raw.strip()
        if not text:
            raise RuntimeError("LLM runtime returned empty response.")
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            candidate = text[start : end + 1]
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Failed to parse JSON response: {exc}") from exc
            if isinstance(parsed, dict):
                return parsed
        raise RuntimeError("LLM runtime response was not valid JSON object.")

    @staticmethod
    def _normalize_openai_usage(usage: dict[str, Any] | None) -> dict[str, int]:
        payload = usage or {}
        prompt_tokens = int(payload.get("prompt_tokens", 0) or 0)
        completion_tokens = int(payload.get("completion_tokens", 0) or 0)
        total_tokens = int(payload.get("total_tokens", prompt_tokens + completion_tokens) or 0)
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

    def _record_call(
        self,
        *,
        provider: str,
        endpoint: str,
        model: str,
        latency_ms: float,
        usage: dict[str, Any],
        ok: bool,
        error: str = "",
    ) -> None:
        self._calls.append(
            {
                "provider": provider,
                "endpoint": endpoint,
                "model": model,
                "latency_ms": round(latency_ms, 3),
                "usage": dict(usage),
                "ok": ok,
                "error": error,
            }
        )
