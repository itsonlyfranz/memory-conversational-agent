"""Short-term buffer: recent turns verbatim, older turns compressed into a rolling summary."""

from __future__ import annotations

from pydantic import BaseModel, Field

from .compression import ContextCompressor, TokenCounter, estimate_tokens
from .models import ChatMessage


class BufferState(BaseModel):
    summary: str = ""
    messages: list[ChatMessage] = Field(default_factory=list)


class ShortTermBuffer:
    """Holds the live conversation within a token budget.

    When ``summary + messages`` exceed ``max_tokens``, everything except the last
    ``keep_recent`` messages is folded into the summary via the compressor. The kept tail
    always starts on a user turn so the transcript stays valid for the chat API.
    """

    def __init__(
        self,
        compressor: ContextCompressor,
        *,
        max_tokens: int = 1200,
        keep_recent: int = 4,
        summary_max_tokens: int = 300,
        count_tokens: TokenCounter = estimate_tokens,
    ) -> None:
        self.compressor = compressor
        self.max_tokens = max_tokens
        self.keep_recent = keep_recent
        self.summary_max_tokens = summary_max_tokens
        self.count_tokens = count_tokens
        self.summary = ""
        self.messages: list[ChatMessage] = []
        self.compressions = 0

    @property
    def tokens(self) -> int:
        return self.count_tokens(self.summary) + sum(
            self.count_tokens(m.content) for m in self.messages
        )

    def add(self, *messages: ChatMessage) -> bool:
        """Append messages; compress if over budget. Returns True if compression ran."""
        self.messages.extend(messages)
        return self.compress_if_needed()

    def compress_if_needed(self) -> bool:
        if self.tokens <= self.max_tokens:
            return False
        cut = len(self.messages) - self.keep_recent
        while cut < len(self.messages) and self.messages[cut].role != "user":
            cut += 1
        if cut <= 0 or cut >= len(self.messages):
            return False  # nothing safely compressible; recent turns alone exceed the budget
        self.summary = self.compressor.compress(
            self.summary, self.messages[:cut], self.summary_max_tokens
        )
        self.messages = self.messages[cut:]
        self.compressions += 1
        return True

    def state(self) -> BufferState:
        return BufferState(summary=self.summary, messages=list(self.messages))

    def load(self, state: BufferState) -> None:
        self.summary = state.summary
        self.messages = list(state.messages)
