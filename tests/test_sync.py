"""Cross-session sync: independent agents sharing one on-disk store see each other live."""

from memory_agent import JsonlMemoryStore, LongTermMemory, MemoryAgent
from memory_agent.fakes import HeuristicLLM


def make_agent(tmp_path):
    memory = LongTermMemory(JsonlMemoryStore(tmp_path / "memories.jsonl"))
    return MemoryAgent(HeuristicLLM(), memory)


def test_memory_learned_in_one_live_session_is_recalled_by_another(tmp_path):
    laptop = make_agent(tmp_path).open_session("alice", "laptop")
    phone = make_agent(tmp_path).open_session("alice", "phone")
    assert phone.send("Where do I live?").recalled == []
    laptop.send("I live in Berlin.")
    assert "Berlin" in phone.send("Where do I live?").reply


def test_updates_and_forgetting_propagate(tmp_path):
    a, b = make_agent(tmp_path), make_agent(tmp_path)
    sa, sb = a.open_session("alice", "a"), b.open_session("alice", "b")
    sa.send("I live in Berlin.")
    assert "Berlin" in sb.send("Where do I live?").reply
    sa.send("I moved to Lisbon.")
    reply = sb.send("Where do I live?").reply
    assert "Lisbon" in reply and "Berlin" not in reply
    a.memory.forget_all("alice")
    assert sb.send("Where do I live?").recalled == []


def test_state_survives_process_restart(tmp_path):
    first = make_agent(tmp_path).open_session("alice", "one")
    first.send("My dog is named Biscuit.")
    first.close()
    restarted = make_agent(tmp_path)  # brand-new objects, same files
    assert "Biscuit" in restarted.open_session("alice", "two").send("What is my dog called?").reply


def test_sync_survives_log_compaction(tmp_path):
    a, b = make_agent(tmp_path), make_agent(tmp_path)
    sa, sb = a.open_session("alice", "a"), b.open_session("alice", "b")
    sa.send("I live in Berlin.")
    sb.send("Where do I live?")
    a.memory.store.compact()
    sa.send("I like espresso.")
    assert "espresso" in sb.send("What do I like?").reply
    assert "Berlin" in sb.send("Where do I live?").reply
