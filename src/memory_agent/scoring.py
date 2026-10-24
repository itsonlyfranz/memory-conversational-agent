"""Relevance scoring: blends semantic similarity with recency, importance and use frequency."""

from __future__ import annotations

import math

from pydantic import BaseModel

from .embeddings import cosine
from .models import Clock, MemoryRecord, ScoredMemory, utcnow


class ScoringWeights(BaseModel):
    similarity: float = 0.6
    recency: float = 0.15
    importance: float = 0.2
    frequency: float = 0.05

    def total(self) -> float:
        return self.similarity + self.recency + self.importance + self.frequency


class RelevanceScorer:
    """score = w_sim*similarity + w_rec*recency + w_imp*importance + w_freq*frequency.

    * recency decays exponentially with the time since the memory was last used; the half-life
      stretches with importance, so important memories fade more slowly.
    * frequency saturates (log-scaled) so a memory cannot dominate just by being recalled a lot.
    * a similarity gate (``min_similarity``) keeps important-but-unrelated memories out, so the
      non-semantic signals only re-rank memories that are already topically relevant.
    """

    def __init__(
        self,
        weights: ScoringWeights | None = None,
        *,
        half_life_days: float = 30.0,
        min_similarity: float = 0.2,
        saturation: int = 10,
        clock: Clock = utcnow,
    ) -> None:
        self.weights = weights or ScoringWeights()
        self.half_life_days = half_life_days
        self.min_similarity = min_similarity
        self.saturation = saturation
        self.clock = clock

    def recency(self, record: MemoryRecord) -> float:
        age_days = max(0.0, (self.clock() - record.last_accessed).total_seconds() / 86400)
        half_life = self.half_life_days * (1 + 2 * record.importance)
        return 0.5 ** (age_days / half_life)

    def frequency(self, record: MemoryRecord) -> float:
        return min(1.0, math.log1p(record.access_count) / math.log1p(self.saturation))

    def score(self, query_vec: list[float], record: MemoryRecord) -> ScoredMemory:
        w = self.weights
        sim = max(0.0, cosine(query_vec, record.embedding))
        rec = self.recency(record)
        freq = self.frequency(record)
        total = (
            w.similarity * sim
            + w.recency * rec
            + w.importance * record.importance
            + w.frequency * freq
        ) / (w.total() or 1.0)
        return ScoredMemory(
            record=record,
            similarity=sim,
            recency=rec,
            importance=record.importance,
            frequency=freq,
            score=total,
        )

    def rank(
        self,
        query_vec: list[float],
        records: list[MemoryRecord],
        *,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> list[ScoredMemory]:
        scored = [self.score(query_vec, r) for r in records]
        kept = [s for s in scored if s.similarity >= self.min_similarity and s.score >= min_score]
        kept.sort(key=lambda s: s.score, reverse=True)
        return kept[:top_k]
