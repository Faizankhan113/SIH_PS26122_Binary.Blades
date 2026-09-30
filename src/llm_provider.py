from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class ProviderError(RuntimeError):
    """The LLM provider could not be reached or did not answer in time.

    These are the failures where falling back to a degraded mode is reasonable.
    They are deliberately different from schema / validation errors
    (``pydantic.ValidationError``, ``ValueError``), which are never retried and
    never wrapped in this class.
    """


class ProviderTimeoutError(ProviderError):
    """One provider call ran longer than the configured limit."""


class ProviderUnavailableError(ProviderError):
    """A rate-limit or transient error persisted after every allowed retry."""


class LLMProvider(ABC):
    """Provider-neutral interface for the MVP.

    The rest of the application should depend on this interface, not on
    Gemini/Groq SDKs. This lets us benchmark providers without rewriting
    extraction or matching logic.
    """

    @abstractmethod
    def generate_structured(self, prompt: str, response_model: type[T]) -> T:
        raise NotImplementedError
