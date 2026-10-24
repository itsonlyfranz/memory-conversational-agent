from datetime import UTC, datetime, timedelta

import pytest

from memory_agent import (
    HashingEmbedder,
    InMemoryStore,
    LongTermMemory,
    MemoryAgent,
    RelevanceScorer,
)
from memory_agent.fakes import HeuristicLLM


class FakeClock:
    """A controllable clock so recency behaviour is testable without sleeping."""

    def __init__(self) -> None:
        self.now = datetime(2026, 1, 1, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **kwargs: float) -> None:
        self.now += timedelta(**kwargs)


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder()


@pytest.fixture
def memory(clock: FakeClock) -> LongTermMemory:
    return LongTermMemory(
        InMemoryStore(), HashingEmbedder(), RelevanceScorer(clock=clock), clock=clock
    )


@pytest.fixture
def agent(memory: LongTermMemory) -> MemoryAgent:
    return MemoryAgent(HeuristicLLM(), memory, max_context_tokens=1000)
