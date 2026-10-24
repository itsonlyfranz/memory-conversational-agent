"""The conversational agent: a thin session API over the LangGraph turn graph."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver

from .compression import ContextCompressor, estimate_tokens
from .graph import GraphConfig, build_graph
from .ids import safe_id
from .llm import LLMClient
from .memory import LongTermMemory
from .models import ChatMessage, MemoryRecord, ScoredMemory

RECURSION_LIMIT = 12  # the turn graph is acyclic (~6 supersteps); this is a safety net


@dataclass
class TurnResult:
    reply: str
    recalled: list[ScoredMemory] = field(default_factory=list)
    learned: list[MemoryRecord] = field(default_factory=list)
    compressed: bool = False


class MemoryAgent:
    def __init__(
        self,
        llm: LLMClient,
        memory: LongTermMemory,
        checkpointer: BaseCheckpointSaver | None = None,
        *,
        max_context_tokens: int = 1200,
        keep_recent: int = 4,
        summary_max_tokens: int = 300,
        top_k: int = 5,
        min_score: float = 0.25,
        extract: bool = True,
        reinforce: bool = True,
    ) -> None:
        self.llm = llm
        self.memory = memory
        self.compressor = ContextCompressor(llm)
        self.summary_max_tokens = summary_max_tokens
        self.checkpointer = checkpointer or InMemorySaver()
        self.graph = build_graph(
            llm,
            memory,
            GraphConfig(
                max_context_tokens=max_context_tokens,
                keep_recent=keep_recent,
                summary_max_tokens=summary_max_tokens,
                top_k=top_k,
                min_score=min_score,
                extract=extract,
                reinforce=reinforce,
            ),
            self.checkpointer,
        )

    def open_session(self, user_id: str, session_id: str | None = None) -> ChatSession:
        """Start a session. Re-opening an existing ``session_id`` resumes its saved buffer."""
        safe_id(user_id, "user id")
        session_id = safe_id(session_id or uuid.uuid4().hex[:12], "session id")
        return ChatSession(self, user_id, session_id)


class ChatSession:
    def __init__(self, agent: MemoryAgent, user_id: str, session_id: str) -> None:
        self.agent = agent
        self.user_id = user_id
        self.session_id = session_id
        self.closed = False
        self._config = {
            "configurable": {"thread_id": f"{user_id}:{session_id}"},
            "recursion_limit": RECURSION_LIMIT,
        }

    # -- short-term buffer view (state lives in the graph checkpoint) ------------------------
    def _values(self) -> dict:
        return self.agent.graph.get_state(self._config).values or {}

    @property
    def summary(self) -> str:
        return self._values().get("summary", "")

    @property
    def messages(self) -> list[ChatMessage]:
        return list(self._values().get("messages", []))

    @property
    def compressions(self) -> int:
        return self._values().get("compressions", 0)

    @property
    def buffer_tokens(self) -> int:
        return estimate_tokens(self.summary) + sum(
            estimate_tokens(m.content) for m in self.messages
        )

    # -- turns ------------------------------------------------------------------------------
    def send(self, text: str) -> TurnResult:
        inputs = {"user_id": self.user_id, "session_id": self.session_id, "user_text": text}
        out = self.agent.graph.invoke(inputs, self._config)
        return TurnResult(
            reply=out.get("reply", ""),
            recalled=out.get("recalled", []),
            learned=out.get("learned", []),
            compressed=out.get("compressed", False),
        )

    def close(self) -> MemoryRecord | None:
        """End the session by storing a one-paragraph episodic memory of it."""
        if self.closed:
            return None
        self.closed = True
        messages = self.messages
        if not messages:
            return None
        summary = self.agent.compressor.compress(
            self.summary, messages, self.agent.summary_max_tokens
        )
        return self.agent.memory.remember(
            self.user_id,
            f"Earlier conversation: {summary}",
            kind="episode",
            importance=0.4,
            session_id=self.session_id,
        )
