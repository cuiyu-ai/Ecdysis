"""Chat-completions client for Ecdysis (DashScope / OpenRouter / OpenAI-compatible).

Auto-detects the right "disable thinking" body key from ``base_url``:

- DashScope (``dashscope.aliyuncs.com``) ->``{"enable_thinking": False}``
- OpenRouter (``openrouter.ai``) ->``{"reasoning": {"enabled": False}}``
- Other OpenAI-compatible ->no special reasoning field

Kept a ``OpenRouterClient`` name alias for back-compat with older imports.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any

try:
    from openai import APIConnectionError, APIError, OpenAI, RateLimitError
except ModuleNotFoundError:
    OpenAI = None

    class APIConnectionError(Exception):
        pass

    class APIError(Exception):
        pass

    class RateLimitError(Exception):
        pass

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_API_KEY_ENV = "DASHSCOPE_API_KEY"
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BACKOFF = 2.0

_LITELLM_PREFIX_RE = re.compile(r"^openai/")


def _normalize_model(model: str) -> str:
    if not model:
        raise ValueError("model name must be a non-empty string")
    return _LITELLM_PREFIX_RE.sub("", model, count=1)


def _read_default_api_key() -> str | None:
    for env in (
        DEFAULT_API_KEY_ENV,
        "OPENROUTER_API_KEY",
        "OPENAI_API_KEY",
        "DASHSCOPE_API_KEY",
    ):
        if os.getenv(env):
            return os.getenv(env)
    return None


def _detect_disable_format(base_url: str) -> str:
    """Return the right body fragment key for disabling thinking/reasoning."""
    url = (base_url or "").lower()
    if "dashscope" in url:
        return "dashscope"
    if "openrouter" in url:
        return "openrouter"
    return "none"


def _build_disable_kwargs(disable_reasoning: bool, base_url: str) -> dict[str, Any]:
    """Return the extra_body fragment that turns off thinking/reasoning."""
    if not disable_reasoning:
        return {}
    fmt = _detect_disable_format(base_url)
    if fmt == "dashscope":
        return {"extra_body": {"enable_thinking": False}}
    if fmt == "openrouter":
        return {"extra_body": {"reasoning": {"enabled": False}}}
    return {}


def _extract_json_object(text: str) -> dict | None:
    if not text:
        return None
    text = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fence_match.group(1) if fence_match else text
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    snippet = candidate[start : end + 1]
    try:
        return json.loads(snippet)
    except json.JSONDecodeError:
        cleaned = re.sub(r",\s*([}\]])", r"\1", snippet)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return None


class LLMClient:
    """Synchronous OpenAI-compatible chat-completions client.

    Auto-detects the right way to disable thinking/reasoning based on
    ``base_url``. Supports DashScope, OpenRouter, and any other
    OpenAI-compatible endpoint.
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        temperature: float = 0.0,
        disable_reasoning: bool = True,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_backoff: float = DEFAULT_RETRY_BACKOFF,
        timeout: float = 120.0,
    ) -> None:
        if not model:
            raise ValueError("model is required")
        if OpenAI is None:
            raise RuntimeError(
                "The 'openai' package is required for real LLM calls. "
                "Install project dependencies before running without --dry-run."
            )
        self.model = _normalize_model(model)
        self.base_url = base_url
        resolved_key = api_key or _read_default_api_key()
        if not resolved_key:
            raise ValueError(
                f"API key not found; set {DEFAULT_API_KEY_ENV} "
                "or pass api_key explicitly"
            )
        self.client = OpenAI(
            api_key=resolved_key,
            base_url=base_url,
            timeout=timeout,
        )
        self.temperature = temperature
        self.disable_reasoning = disable_reasoning
        self.disable_format = _detect_disable_format(base_url)
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff

    def _build_kwargs(self, messages: list[dict[str, str]], **overrides: Any) -> dict:
        temp = overrides.pop("temperature", self.temperature)
        if temp is None:
            temp = self.temperature
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temp,
        }
        disable_kwargs = _build_disable_kwargs(self.disable_reasoning, self.base_url)
        for key, value in disable_kwargs.items():
            kwargs.setdefault(key, value)
        for key, value in overrides.items():
            if value is not None:
                kwargs[key] = value
        return kwargs

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        **overrides: Any,
    ) -> str:
        kwargs = self._build_kwargs(messages, temperature=temperature, **overrides)
        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(**kwargs)
                return response.choices[0].message.content or ""
            except (RateLimitError, APIConnectionError) as exc:
                last_exc = exc
                if attempt >= self.max_retries:
                    raise
                sleep_for = self.retry_backoff ** attempt
                logger.warning(
                    "LLM transient error (attempt %d/%d): %s; retrying in %.1fs",
                    attempt,
                    self.max_retries,
                    exc,
                    sleep_for,
                )
                time.sleep(sleep_for)
            except APIError as exc:
                raise RuntimeError(f"LLM API error: {exc}") from exc
        raise RuntimeError(f"LLM call failed after retries: {last_exc}")

    def chat_json(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        **overrides: Any,
    ) -> dict | None:
        """Call the model and return the parsed JSON object, or None on failure."""
        overrides.setdefault("response_format", {"type": "json_object"})
        text = self.chat(messages, temperature=temperature, **overrides)
        return _extract_json_object(text)


# Back-compat alias for older imports.
OpenRouterClient = LLMClient


def build_client_from_exp_config(
    exp_config: dict[str, Any],
    role: str = "evolution",
) -> LLMClient:
    """Build an LLM client from experiment YAML model fields."""
    import os

    role_fields = {
        "evolution": (
            "evolution_llm",
            "evolution_api_base",
            "evolution_api_key_env",
            "agent_llm",
            "agent_api_base",
            "agent_api_key_env",
        ),
        "judge": (
            "judge_llm",
            "judge_api_base",
            "judge_api_key_env",
            "evolution_llm",
            "evolution_api_base",
            "evolution_api_key_env",
        ),
    }
    if role not in role_fields:
        raise ValueError(f"Unknown LLM role: {role}")
    model_key, base_key, key_env_key, fb_model, fb_base, fb_key_env = role_fields[
        role
    ]
    model = exp_config.get(model_key) or exp_config.get(fb_model)
    if not model:
        raise ValueError(f"No model configured for role '{role}'")
    base_url = exp_config.get(base_key) or exp_config.get(fb_base) or DEFAULT_BASE_URL
    key_env = (
        exp_config.get(key_env_key)
        or exp_config.get(fb_key_env)
        or DEFAULT_API_KEY_ENV
    )
    api_key = os.getenv(str(key_env)) if key_env else None
    return LLMClient(model=model, api_key=api_key, base_url=base_url)
