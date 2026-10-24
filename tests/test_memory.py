from memory_agent import HashingEmbedder, InMemoryStore, LongTermMemory, RelevanceScorer


def test_remember_then_recall_most_relevant(memory):
    memory.remember("alice", "User lives in Berlin", importance=0.8)
    memory.remember("alice", "User is allergic to peanuts", importance=1.0)
    memory.remember("alice", "User likes hiking", kind="preference")
    top = memory.recall("alice", "Where do I live?")
    assert top[0].record.text == "User lives in Berlin"
    assert memory.recall("alice", "what am I allergic to")[0].record.text.endswith("peanuts")


def test_users_are_isolated(memory):
    memory.remember("alice", "User lives in Berlin")
    assert memory.recall("bob", "Where do I live?") == []
    assert memory.all("bob") == []
    assert memory.forget("bob", memory.all("alice")[0].id) is False
    assert len(memory.all("alice")) == 1


def test_slot_supersedes_old_value(memory):
    first = memory.remember("alice", "User lives in Berlin", slot="residence")
    second = memory.remember("alice", "User lives in Lisbon", slot="residence")
    assert second.id == first.id
    texts = [r.text for r in memory.all("alice")]
    assert texts == ["User lives in Lisbon"]
    assert memory.recall("alice", "Where do I live?")[0].record.text == "User lives in Lisbon"


def test_slot_keeps_higher_importance(memory):
    memory.remember("a", "User works as a nurse", slot="job", importance=0.9)
    updated = memory.remember("a", "User works as a chef", slot="job", importance=0.3)
    assert updated.importance == 0.9


def test_near_duplicates_are_merged_and_reinforced(memory):
    a = memory.remember("alice", "User likes hiking", importance=0.4)
    b = memory.remember("alice", "User likes hiking", importance=0.7)
    assert a.id == b.id and len(memory.all("alice")) == 1
    assert b.importance == 0.7 and b.access_count == 1


def test_distinct_facts_without_slot_coexist(memory):
    memory.remember("a", "User is allergic to peanuts")
    memory.remember("a", "User is allergic to shellfish")
    assert len(memory.all("a")) == 2


def test_recall_reinforces_returned_memories(memory, clock):
    r = memory.remember("a", "User lives in Berlin")
    clock.advance(days=3)
    memory.recall("a", "Where do I live?")
    stored = memory.store.get(r.id)
    assert stored.access_count == 1 and stored.last_accessed == clock()


def test_recall_without_reinforce_leaves_store_untouched(memory):
    r = memory.remember("a", "User lives in Berlin")
    memory.recall("a", "Where do I live?", reinforce=False)
    assert memory.store.get(r.id).access_count == 0


def test_recently_used_memory_stays_fresh(memory, clock):
    used = memory.remember("a", "User likes jazz music", importance=0.0)
    unused = memory.remember("a", "User likes rock music", importance=0.0)
    for _ in range(4):
        clock.advance(days=20)
        memory.store.upsert(memory.store.get(used.id).model_copy(update={"last_accessed": clock()}))
    ranked = memory.recall("a", "what music do I like", reinforce=False)
    assert ranked[0].record.id == used.id
    assert ranked[1].record.id == unused.id


def test_forget_and_forget_all(memory):
    r = memory.remember("a", "User lives in Berlin")
    memory.remember("a", "User is allergic to peanuts")
    assert memory.forget("a", r.id)
    assert memory.recall("a", "Where do I live?") == []
    assert memory.forget_all("a") == 1
    assert memory.all("a") == []


def test_capacity_evicts_least_valuable(clock):
    mem = LongTermMemory(
        InMemoryStore(),
        HashingEmbedder(),
        RelevanceScorer(clock=clock),
        max_per_user=3,
        clock=clock,
    )
    mem.remember("a", "User is allergic to peanuts", importance=1.0)
    mem.remember("a", "User enjoys quiet mornings", importance=0.1)
    mem.remember("a", "User works at Initech", importance=0.8)
    mem.remember("a", "User owns a red bicycle", importance=0.7)
    texts = {r.text for r in mem.all("a")}
    assert len(texts) == 3 and "User enjoys quiet mornings" not in texts
