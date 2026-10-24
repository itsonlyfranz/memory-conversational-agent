"""Context compression: fold old messages into a bounded rolling summary."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence

from .llm import LLMClient, LLMError
from .models import ChatMessage
from .prompts import COMPRESS_SYSTEM

log = logging.getLogger(__name__)

TokenCounter = Callable[[str], int]


def estimate_tokens(text: str) -> int:
    """Cheap, dependency-free estimate (~4 characters per token)."""
    return max(1, (len(text) + 3) // 4) if text else 0


def render_transcript(messages: Sequence[ChatMessage]) -> str:
    return "\n".join(
        f"{'User' if m.role == 'user' else 'Assistant'}: {m.content}" for m in messages
    )


def extractive_summary(previous: str, messages: Sequence[ChatMessage], max_tokens: int) -> str:
    """Offline fallback: keep the user's own statements, truncated to fit the budget."""
    parts = [previous] if previous else []
    parts += [m.content.strip() for m in messages if m.role == "user"]
    text = " | ".join(p for p in parts if p)
    budget = max_tokens * 4
    return text if len(text) <= budget else "…" + text[-(budget - 1) :]


class ContextCompressor:
    def __init__(self, llm: LLMClient, *, count_tokens: TokenCounter = estimate_tokens) -> None:
        self.llm = llm
        self.count_tokens = count_tokens

    def compress(
        self, previous_summary: str, messages: Sequence[ChatMessage], max_tokens: int = 300
    ) -> str:
        """Return an updated summary of at most ~``max_tokens``; never raises."""
        content = (
            f"Existing summary:\n{previous_summary or '(none)'}\n\n"
            f"New messages:\n{render_transcript(messages)}"
        )
        try:
            summary = self.llm.complete(
                system=COMPRESS_SYSTEM.format(max_tokens=max_tokens),
                messages=[ChatMessage(role="user", content=content)],
                max_tokens=max_tokens * 2,
            ).strip()
        except LLMError as exc:
            log.warning("LLM compression failed, using extractive fallback: %s", exc)
            summary = ""
        if not summary:
            return extractive_summary(previous_summary, messages, max_tokens)
        if self.count_tokens(summary) > max_tokens * 1.5:
            # The model overshot its budget; hard-trim rather than let the summary grow unbounded.
            return extractive_summary("", [ChatMessage(role="user", content=summary)], max_tokens)
        return summary
