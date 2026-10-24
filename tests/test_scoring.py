import pytest

from memory_agent.embeddings import HashingEmbedder
from memory_agent.models import MemoryRecord
from memory_agent.scoring import RelevanceScorer, ScoringWeights

E = HashingEmbedder()


def rec(text, clock, **kw):
    return MemoryRecord(
        user_id="u",
        text=text,
        embedding=E.embed(text),
        created_at=clock(),
        last_accessed=clock(),
        **kw,
    )


def test_relevant_memory_outranks_unrelated(clock):
    scorer = RelevanceScorer(clock=clock)
    records = [rec("User is allergic to peanuts", clock), rec("User lives in Berlin", clock)]
    ranked = scorer.rank(E.embed("Where do I live?"), records)
    assert [r.record.text for r in ranked] == ["User lives in Berlin"]
    assert ranked[0].score > 0.5


def test_similarity_gate_blocks_important_but_unrelated(clock):
    scorer = RelevanceScorer(clock=clock)
    vital = rec("User is allergic to peanuts", clock, importance=1.0)
    assert scorer.rank(E.embed("favourite hiking trail"), [vital]) == []


def test_recency_decays_with_half_life(clock):
    scorer = RelevanceScorer(half_life_days=10, clock=clock)
    r = rec("User likes tea", clock, importance=0.0)
    assert scorer.recency(r) == pytest.approx(1.0)
    clock.advance(days=10)
    assert scorer.recency(r) == pytest.approx(0.5)
    clock.advance(days=10)
    assert scorer.recency(r) == pytest.approx(0.25)


def test_important_memories_decay_more_slowly(clock):
    scorer = RelevanceScorer(half_life_days=10, clock=clock)
    trivial = rec("User likes tea", clock, importance=0.0)
    vital = rec("User likes tea", clock, importance=1.0)
    clock.advance(days=30)
    assert scorer.recency(vital) > scorer.recency(trivial) * 2


def test_older_memory_loses_to_fresher_one_with_equal_similarity(clock):
    scorer = RelevanceScorer(half_life_days=5, clock=clock)
    old = rec("User likes jazz music", clock, importance=0.5)
    clock.advance(days=60)
    fresh = rec("User likes rock music", clock, importance=0.5)
    ranked = scorer.rank(E.embed("what music do I like"), [old, fresh])
    assert ranked[0].record.text == "User likes rock music"
    assert ranked[0].recency > ranked[1].recency


def test_frequency_saturates(clock):
    scorer = RelevanceScorer(saturation=10, clock=clock)
    assert scorer.frequency(rec("x y z", clock, access_count=0)) == 0.0
    assert scorer.frequency(rec("x y z", clock, access_count=10)) == pytest.approx(1.0)
    assert scorer.frequency(rec("x y z", clock, access_count=500)) == 1.0


def test_weights_change_ranking(clock):
    q = E.embed("tell me about coffee")
    sim_heavy = rec("User drinks coffee daily", clock, importance=0.1)
    imp_heavy = rec("User is allergic to coffee beans", clock, importance=1.0)
    by_sim = RelevanceScorer(
        ScoringWeights(similarity=1, recency=0, importance=0, frequency=0),
        clock=clock,
        min_similarity=0.1,
    )
    by_imp = RelevanceScorer(
        ScoringWeights(similarity=0.1, recency=0, importance=1, frequency=0),
        clock=clock,
        min_similarity=0.1,
    )
    assert by_sim.rank(q, [sim_heavy, imp_heavy])[0].record is sim_heavy
    assert by_imp.rank(q, [sim_heavy, imp_heavy])[0].record is imp_heavy


def test_top_k_and_min_score(clock):
    scorer = RelevanceScorer(clock=clock, min_similarity=0.1)
    records = [rec(f"User likes fruit {i}", clock) for i in range(6)]
    q = E.embed("what fruit do I like")
    assert len(scorer.rank(q, records, top_k=3)) == 3
    assert scorer.rank(q, records, min_score=0.99) == []


def test_score_components_are_exposed(clock):
    s = RelevanceScorer(clock=clock).score(E.embed("Berlin"), rec("User lives in Berlin", clock))
    assert {"similarity", "recency", "importance", "frequency", "score"} <= set(s.model_dump())
