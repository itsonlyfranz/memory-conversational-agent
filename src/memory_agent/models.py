"""Pydantic data models shared across the package."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field

Clock = Callable[[], datetime]


def utcnow() -> datetime:
    return datetime.now(UTC)


Role = Literal["user", "assistant"]
MemoryKind = Literal["fact", "preference", "episode"]


class ChatMessage(BaseModel):
    role: Role
    content: str
    timestamp: datetime = Field(default_factory=utcnow)


class MemoryRecord(BaseModel):
    """One durable memory about a user, stored with its embedding."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    user_id: str
    text: str
    kind: MemoryKind = "fact"
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    # A slot names a single-valued attribute ("residence", "job"); a newer memory in the
    # same slot supersedes the older one instead of coexisting with it.
    slot: str | None = None
    embedding: list[float] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utcnow)
    last_accessed: datetime = Field(default_factory=utcnow)
    access_count: int = 0
    source_session: str | None = None


class ScoredMemory(BaseModel):
    """A recalled memory with the components that produced its relevance score."""

    record: MemoryRecord
    similarity: float
    recency: float
    importance: float
    frequency: float
    score: float


class ExtractedMemory(BaseModel):
    text: str
    kind: MemoryKind = "fact"
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    slot: str | None = None
