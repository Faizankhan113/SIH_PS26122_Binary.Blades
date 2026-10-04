from __future__ import annotations

import os
import random
import threading
import time
from typing import Callable, TypeVar

from dotenv import load_dotenv
from pydantic import BaseModel

from .llm_provider import (
    LLMProvider,
    ProviderError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)

load_dotenv()

T = TypeVar("T", bound=BaseModel)

# --------------------------------------------------------------------------
# Call limits (all overridable from the environment)
# --------------------------------------------------------------------------
# Hard limit for ONE provider call. Enforced twice: by the SDK's own HTTP
# timeout and by ResilientProvider, so a hung call can never block a request.
DEFAULT_TIMEOUT_SECONDS = 30.0
# Retries AFTER the first attempt, for rate-limit (429) and transient errors only.
DEFAULT_MAX_RETRIES = 2
# Wait before retry n is BASE * 2**n seconds (+ a little jitter), or the
# provider's own Retry-After hint if it sent one. Never longer than MAX_BACKOFF.
DEFAULT_BACKOFF_BASE_SECONDS = 1.0
MAX_BACKOFF_SECONDS = 20.0

TIMEOUT_ENV = "LLM_TIMEOUT_SECONDS"
MAX_RETRIES_ENV = "LLM_MAX_RETRIES"
BACKOFF_ENV = "LLM_RETRY_BASE_SECONDS"

_RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "").strip() or default)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "").strip() or default)
    except ValueError:
        return default


def request_timeout_seconds() -> float:
    return max(0.1, _env_float(TIMEOUT_ENV, DEFAULT_TIMEOUT_SECONDS))


# --------------------------------------------------------------------------
# Error classification
# --------------------------------------------------------------------------

def _status_code(exc: BaseException) -> int | None:
    """HTTP status carried by an SDK exception, whatever the SDK calls it."""
    for attr in ("status_code", "code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    response = getattr(exc, "response", None)
    value = getattr(response, "status_code", None)
    return value if isinstance(value, int) else None


def _is_timeout(exc: BaseException) -> bool:
    return isinstance(exc, (TimeoutError, ProviderTimeoutError)) or "timeout" in type(exc).__name__.lower()


def is_retryable(exc: BaseException) -> bool:
    """True only for rate-limit and transient failures.

    Never retried: timeouts (a hung call is not going to recover, and retrying
    would multiply the wait), schema/validation errors (the same prompt gives
    the same bad output), authentication and other 4xx errors, and our own
    "empty response" errors.
    """
    if _is_timeout(exc):
        return False
    if isinstance(exc, ProviderError):
        return False
    status = _status_code(exc)
    if status is not None:
        return status in _RETRYABLE_STATUS
    # No HTTP status: only plain connection-level failures count as transient.
    return isinstance(exc, ConnectionError) or "connectionerror" in type(exc).__name__.lower()


def _retry_after_seconds(exc: BaseException) -> float | None:
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if not headers:
        return None
    try:
        value = headers.get("retry-after") or headers.get("Retry-After")
        return float(value) if value is not None else None
    except (TypeError, ValueError, AttributeError):
        return None


class ResilientProvider(LLMProvider):
    """Wraps a provider with a hard timeout and a limited retry with backoff."""

    def __init__(
        self,
        inner: LLMProvider,
        *,
        timeout_seconds: float | None = None,
        max_retries: int | None = None,
        backoff_base_seconds: float | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.inner = inner
        self.timeout_seconds = request_timeout_seconds() if timeout_seconds is None else timeout_seconds
        self.max_retries = max(0, _env_int(MAX_RETRIES_ENV, DEFAULT_MAX_RETRIES) if max_retries is None else max_retries)
        self.backoff_base_seconds = (
            _env_float(BACKOFF_ENV, DEFAULT_BACKOFF_BASE_SECONDS) if backoff_base_seconds is None else backoff_base_seconds
        )
        self._sleep = sleep

    def __getattr__(self, name: str):  # expose model/client of the wrapped provider
        if name == "inner":  # not set yet (e.g. while copying): avoid endless recursion
            raise AttributeError(name)
        return getattr(self.inner, name)

    def _call_once(self, prompt: str, response_model: type[T]) -> T:
        """Run one call in a daemon thread and give up after the time limit.

        A truly hung SDK call cannot be killed, but it no longer holds the
        caller: the thread is a daemon and ends when the SDK's own timeout
        fires (or when the process exits).
        """
        box: dict[str, object] = {}
        done = threading.Event()

        def target() -> None:
            try:
                box["value"] = self.inner.generate_structured(prompt, response_model)
            except BaseException as exc:  # noqa: BLE001 - re-raised in the caller
                box["error"] = exc
            finally:
                done.set()

        threading.Thread(target=target, name="llm-call", daemon=True).start()
        if not done.wait(self.timeout_seconds):
            raise ProviderTimeoutError(
                f"{type(self.inner).__name__} did not answer within {self.timeout_seconds:g} seconds."
            )
        if "error" in box:
            raise box["error"]  # type: ignore[misc]
        return box["value"]  # type: ignore[return-value]

    def generate_structured(self, prompt: str, response_model: type[T]) -> T:
        attempt = 0
        while True:
            try:
                return self._call_once(prompt, response_model)
            except ProviderTimeoutError:
                raise
            except Exception as exc:
                if _is_timeout(exc):  # the SDK's own timeout fired first
                    raise ProviderTimeoutError(
                        f"{type(self.inner).__name__} timed out after {self.timeout_seconds:g} seconds."
                    ) from exc
                if not is_retryable(exc):
                    raise  # schema errors, bad requests, auth problems: fail as they are
                if attempt >= self.max_retries:
                    raise ProviderUnavailableError(
                        f"{type(self.inner).__name__} still failing after {attempt + 1} attempt(s): "
                        f"{type(exc).__name__}: {exc}"
                    ) from exc
                delay = _retry_after_seconds(exc)
                if delay is None:
                    delay = self.backoff_base_seconds * (2 ** attempt) * (1 + random.random() * 0.25)
                self._sleep(min(max(delay, 0.0), MAX_BACKOFF_SECONDS))
                attempt += 1


class GeminiProvider(LLMProvider):
    """Gemini implementation using Google's current GenAI Python SDK."""

    def __init__(self, model: str | None = None) -> None:
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError(
                "Gemini SDK missing. Install dependencies from requirements.txt."
            ) from exc

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set in the environment.")

        self.model = model or os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
        # The SDK takes its timeout in milliseconds.
        self.client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=int(request_timeout_seconds() * 1000)),
        )

    def generate_structured(self, prompt: str, response_model: type[T]) -> T:
        from google.genai import types

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_json_schema=response_model.model_json_schema(),
                temperature=0,
            ),
        )

        if not response.text:
            raise RuntimeError("Gemini returned an empty response.")

        return response_model.model_validate_json(response.text)


class GroqProvider(LLMProvider):
    """Groq implementation using an OpenAI-compatible API."""

    def __init__(self, model: str | None = None) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "OpenAI Python SDK missing. Install dependencies from requirements.txt."
            ) from exc

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set in the environment.")

        self.model = model or os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
            timeout=request_timeout_seconds(),
            max_retries=0,  # retries are handled once, by ResilientProvider
        )

    def generate_structured(self, prompt: str, response_model: type[T]) -> T:
        schema = response_model.model_json_schema()

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": response_model.__name__,
                    "strict": True,
                    "schema": schema,
                },
            },
        )

        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("Groq returned an empty response.")

        return response_model.model_validate_json(content)


def _provider_name(name: str | None) -> str:
    return (name or os.getenv("LLM_PROVIDER", "gemini")).strip().lower()


def build_provider(name: str | None = None) -> LLMProvider:
    """Create a NEW raw provider selected by LLM_PROVIDER (no timeout wrapper, no cache).

    Application code should call `get_provider()` instead.
    """

    provider_name = _provider_name(name)

    if provider_name == "gemini":
        return GeminiProvider()
    if provider_name == "groq":
        return GroqProvider()

    raise ValueError(
        f"Unsupported LLM_PROVIDER={provider_name!r}. Use 'gemini' or 'groq'."
    )


_PROVIDERS: dict[str, LLMProvider] = {}
_PROVIDERS_LOCK = threading.Lock()


def get_provider(name: str | None = None) -> LLMProvider:
    """Return the process-wide provider for `name` (or LLM_PROVIDER), building it once.

    Providers come wrapped in `ResilientProvider` (timeout + retry). A provider
    that fails to build (for example a missing API key) is not cached, so
    fixing the environment and retrying works without a restart.
    """
    key = _provider_name(name)
    provider = _PROVIDERS.get(key)
    if provider is not None:
        return provider
    with _PROVIDERS_LOCK:
        provider = _PROVIDERS.get(key)
        if provider is None:
            raw = build_provider(key)
            provider = ResilientProvider(raw)
            _PROVIDERS[key] = provider
        return provider


def provider_label(provider: LLMProvider) -> str:
    """Class name of the real provider (``GeminiProvider``), looking through the wrapper.

    Used for the audit trail, so a match is recorded as made by Gemini rather
    than by ``ResilientProvider``.
    """
    return type(provider.inner if isinstance(provider, ResilientProvider) else provider).__name__


def clear_provider_cache() -> None:
    """Forget cached providers (after changing keys in the environment)."""
    with _PROVIDERS_LOCK:
        _PROVIDERS.clear()
