"""The agent's turn as a LangGraph ``StateGraph``.

    START → recall → generate ─┬→ update_buffer ─┐        (parallel branches)
                               └→ extract ───────┴→ review ─┬→ confirm (interrupt) ─┐
                                                            └────────────────────────┴→ store → END

* ``recall``        long-term retrieval, ranked by the relevance scorer
* ``generate``      one chat completion over summary + recalled memories + recent buffer
* ``update_buffer`` append the turn to the short-term buffer, compress if over budget
* ``extract``       LLM pulls durable facts out of the exchange (skipped when disabled)
* ``review``        join point; a conditional edge decides whether a human must confirm
* ``confirm``       optional ``interrupt`` so the user approves sensitive memories first
* ``store``         write the (approved) memories to long-term memory

The checkpointer persists ``summary``/``messages`` per thread, which is what makes a session
resumable; the run's ``recursion_limit`` bounds the graph defensively.
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from .buffer import BufferState, ShortTermBuffer
from .compression import ContextCompressor
from .extraction import MemoryExtractor
from .llm import LLMClient
from .memory import LongTermMemory
from .models import ChatMessage, ExtractedMemory, MemoryRecord, ScoredMemory
from .prompts import CHAT_SYSTEM


class TurnState(TypedDict, total=False):
    # inputs for this turn
    user_id: str
    session_id: str
    user_text: str
    # persisted across turns by the checkpointer (the short-term buffer)
    summary: str
    messages: list[ChatMessage]
    compressions: int
    # per-turn scratch / outputs
    recalled: list[ScoredMemory]
    reply: str
    compressed: bool
    candidates: list[ExtractedMemory]
    learned: list[MemoryRecord]
    declined: list[str]


def format_memories(recalled: list[ScoredMemory]) -> str:
    if not recalled:
        return "(none)"
    return "\n".join(f"- {s.record.text} (relevance {s.score:.2f})" for s in recalled)


class GraphConfig(TypedDict):
    max_context_tokens: int
    keep_recent: int
    summary_max_tokens: int
    top_k: int
    min_score: float
    extract: bool
    confirm_threshold: float | None


def build_graph(
    llm: LLMClient,
    memory: LongTermMemory,
    cfg: GraphConfig,
    checkpointer: BaseCheckpointSaver | None = None,
):
    compressor = ContextCompressor(llm)
    extractor = MemoryExtractor(llm)

    def recall(state: TurnState) -> dict:
        recalled = memory.recall(
            state["user_id"], state["user_text"], top_k=cfg["top_k"], min_score=cfg["min_score"]
        )
        # Reset per-turn scratch so nothing leaks from the previous checkpointed turn.
        return {
            "recalled": recalled,
            "candidates": [],
            "learned": [],
            "declined": [],
            "compressed": False,
        }

    def generate(state: TurnState) -> dict:
        system = CHAT_SYSTEM.format(
            summary=state.get("summary") or "(none)",
            memories=format_memories(state.get("recalled", [])),
        )
        user_msg = ChatMessage(role="user", content=state["user_text"])
        reply = llm.complete(system=system, messages=[*state.get("messages", []), user_msg])
        return {"reply": reply}

    def update_buffer(state: TurnState) -> dict:
        buffer = ShortTermBuffer(
            compressor,
            max_tokens=cfg["max_context_tokens"],
            keep_recent=cfg["keep_recent"],
            summary_max_tokens=cfg["summary_max_tokens"],
        )
        buffer.load(
            BufferState(summary=state.get("summary", ""), messages=state.get("messages", []))
        )
        compressed = buffer.add(
            ChatMessage(role="user", content=state["user_text"]),
            ChatMessage(role="assistant", content=state["reply"]),
        )
        return {
            "summary": buffer.summary,
            "messages": buffer.messages,
            "compressed": compressed,
            "compressions": state.get("compressions", 0) + int(compressed),
        }

    def extract(state: TurnState) -> dict:
        return {"candidates": extractor.extract(state["user_text"], state["reply"])}

    def review(state: TurnState) -> dict:
        return {}

    def needs_confirmation(state: TurnState) -> str:
        threshold = cfg["confirm_threshold"]
        if threshold is not None and any(c.importance >= threshold for c in state["candidates"]):
            return "confirm"
        return "store"

    def confirm(state: TurnState) -> dict:
        threshold = cfg["confirm_threshold"] or 0.0
        sensitive = [c.text for c in state["candidates"] if c.importance >= threshold]
        approved = interrupt({"question": "Save these memories?", "memories": sensitive})
        approved_set = set(sensitive) if approved is True else set(approved or [])
        kept = [c for c in state["candidates"] if c.text not in sensitive or c.text in approved_set]
        declined = [t for t in sensitive if t not in approved_set]
        return {"candidates": kept, "declined": declined}

    def store(state: TurnState) -> dict:
        learned = [
            memory.remember(
                state["user_id"],
                c.text,
                kind=c.kind,
                importance=c.importance,
                slot=c.slot,
                session_id=state["session_id"],
            )
            for c in state["candidates"]
        ]
        return {"learned": learned}

    def fan_out(state: TurnState) -> list[str]:
        return ["update_buffer", "extract"] if cfg["extract"] else ["update_buffer"]

    g = StateGraph(TurnState)
    for name, fn in [
        ("recall", recall),
        ("generate", generate),
        ("update_buffer", update_buffer),
        ("extract", extract),
        ("review", review),
        ("confirm", confirm),
        ("store", store),
    ]:
        g.add_node(name, fn)
    g.add_edge(START, "recall")
    g.add_edge("recall", "generate")
    g.add_conditional_edges("generate", fan_out, ["update_buffer", "extract"])
    g.add_edge("update_buffer", "review")
    g.add_edge("extract", "review")
    g.add_conditional_edges("review", needs_confirmation, ["confirm", "store"])
    g.add_edge("confirm", "store")
    g.add_edge("store", END)
    return g.compile(checkpointer=checkpointer)
