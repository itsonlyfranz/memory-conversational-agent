"""The LLM boundary: a tiny client protocol plus the Claude implementation (langchain-anthropic)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from .models import ChatMessage


class LLMError(RuntimeError):
    """Raised when the underlying model call fails."""


class LLMClient(Protocol):
    def complete(
        self, *, system: str, messages: Sequence[ChatMessage], max_tokens: int = 1024
    ) -> str: ...


class AnthropicClient:
    """LLMClient backed by Claude through ``langchain_anthropic.ChatAnthropic``."""

    def __init__(self, api_key: str | None, model: str) -> None:
        from langchain_anthropic import ChatAnthropic

        if not api_key:
            raise LLMError("ANTHROPIC_API_KEY is not set (put it in the environment or .env)")
        self.model = model
        self._chat = ChatAnthropic(model=model, api_key=api_key, max_tokens=1024)

    def complete(
        self, *, system: str, messages: Sequence[ChatMessage], max_tokens: int = 1024
    ) -> str:
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        lc = [SystemMessage(content=system)] + [
            HumanMessage(content=m.content) if m.role == "user" else AIMessage(content=m.content)
            for m in messages
        ]
        try:
            response = self._chat.bind(max_tokens=max_tokens).invoke(lc)
        except Exception as exc:  # provider/network errors surface as many types
            raise LLMError(f"Claude call failed: {exc}") from exc
        return (
            response.text
            if hasattr(response, "text") and isinstance(response.text, str)
            else str(response.content)
        )
