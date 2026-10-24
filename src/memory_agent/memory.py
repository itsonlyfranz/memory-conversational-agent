"""Long-term memory: embed, deduplicate, supersede, recall by relevance score, forget."""

from __future__ import annotations

import contextlib

from .embeddings import Embedder, HashingEmbedder, cosine
from .models import Clock, MemoryKind, MemoryRecord, ScoredMemory, utcnow
from .scoring import RelevanceScorer
from .store import MemoryStore


class LongTermMemory:
    def __init__(
        self,
        store: MemoryStore,
        embedder: Embedder | None = None,
        scorer: RelevanceScorer | None = None,
        *,
        dedupe_threshold: float = 0.85,
        max_per_user: int = 500,
        clock: Clock = utcnow,
    ) -> None:
        self.store = store
        self.embedder = embedder or HashingEmbedder()
        self.clock = clock
        self.scorer = scorer or RelevanceScorer(clock=clock)
        self.dedupe_threshold = dedupe_threshold
        self.max_per_user = max_per_user

    def remember(
        self,
        user_id: str,
        text: str,
        *,
        kind: MemoryKind = "fact",
        importance: float = 0.5,
        slot: str | None = None,
        session_id: str | None = None,
    ) -> MemoryRecord:
        """Store a memory. A same-slot memory is replaced; a near-duplicate is reinforced."""
        vec = self.embedder.embed(text)
        with self.store.transaction():
            now = self.clock()
            existing = self.store.list(user_id)

            if slot:
                for old in existing:
                    if old.slot == slot:
                        updated = old.model_copy(
                            update={
                                "text": text,
                                "embedding": vec,
                                "kind": kind,
                                "importance": max(importance, old.importance),
                                "last_accessed": now,
                                "source_session": session_id or old.source_session,
                            }
                        )
                        self.store.upsert(updated)
                        return updated

            best = max(existing, key=lambda r: cosine(vec, r.embedding), default=None)
            if best and cosine(vec, best.embedding) >= self.dedupe_threshold:
                reinforced = best.model_copy(
                    update={
                        "importance": max(importance, best.importance),
                        "access_count": best.access_count + 1,
                        "slot": best.slot or slot,
                        "last_accessed": now,
                    }
                )
                self.store.upsert(reinforced)
                return reinforced

            record = MemoryRecord(
                user_id=user_id,
                text=text,
                kind=kind,
                importance=importance,
                slot=slot,
                embedding=vec,
                created_at=now,
                last_accessed=now,
                source_session=session_id,
            )
            self.store.upsert(record)
            self._enforce_capacity(user_id)
            return record

    def recall(
        self,
        user_id: str,
        query: str,
        *,
        top_k: int = 5,
        min_score: float = 0.0,
        reinforce: bool = True,
    ) -> list[ScoredMemory]:
        """Return the memories most relevant to ``query``, best first."""
        vec = self.embedder.embed(query)
        with self.store.transaction() if reinforce else contextlib.nullcontext():
            ranked = self.scorer.rank(
                vec,
                self.store.list(user_id),
                top_k=top_k,
                min_score=min_score,
            )
            if reinforce:
                now = self.clock()
                for s in ranked:
                    self.store.upsert(
                        s.record.model_copy(
                            update={
                                "access_count": s.record.access_count + 1,
                                "last_accessed": now,
                            }
                        )
                    )
        return ranked

    def all(self, user_id: str) -> list[MemoryRecord]:
        records = self.store.list(user_id)
        return sorted(records, key=lambda r: r.created_at)

    def forget(self, user_id: str, memory_id: str) -> bool:
        with self.store.transaction():
            rec = self.store.get(memory_id)
            if rec is None or rec.user_id != user_id:
                return False
            return self.store.delete(memory_id)

    def forget_all(self, user_id: str) -> int:
        with self.store.transaction():
            records = self.store.list(user_id)
            for r in records:
                self.store.delete(r.id)
        return len(records)

    def _enforce_capacity(self, user_id: str) -> None:
        records = self.store.list(user_id)
        overflow = len(records) - self.max_per_user
        if overflow <= 0:
            return
        # Evict the least valuable: low importance and long unused.
        records.sort(key=lambda r: 0.7 * r.importance + 0.3 * self.scorer.recency(r))
        for r in records[:overflow]:
            self.store.delete(r.id)
