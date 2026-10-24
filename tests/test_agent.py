import pytest
from langgraph.checkpoint.sqlite import SqliteSaver

from memory_agent import LongTermMemory, MemoryAgent
from memory_agent.fakes import HeuristicLLM, ScriptedLLM
from memory_agent.llm import LLMError
from memory_agent.store import InMemoryStore


def test_agent_learns_from_conversation_and_recalls_in_a_new_session(agent):
    s1 = agent.open_session("alice", "s1")
    s1.send("My name is Alice and I live in Berlin.")
    s1.close()
    s2 = agent.open_session("alice", "s2")
    result = s2.send("Where do I live?")
    assert "Berlin" in result.reply
    assert result.recalled[0].record.text == "User lives in Berlin"
    assert s2.messages[0].content == "Where do I live?"  # fresh short-term buffer


def test_memories_are_injected_into_the_system_prompt(memory):
    memory.remember("alice", "User is allergic to peanuts", importance=1.0)
    llm = ScriptedLLM(["Noted.", "[]"])
    agent = MemoryAgent(llm, memory)
    agent.open_session("alice", "s").send("Can I eat a peanut cookie? I'm allergic, right?")
    chat_call = llm.calls[0]
    assert "[task:chat]" in chat_call["system"]
    assert "User is allergic to peanuts" in chat_call["system"]


def test_unrelated_memories_are_not_injected(memory):
    memory.remember("alice", "User is allergic to peanuts", importance=1.0)
    llm = ScriptedLLM(["ok", "[]"])
    MemoryAgent(llm, memory).open_session("alice", "s").send("What's the capital of France?")
    assert "peanuts" not in llm.calls[0]["system"]


def test_compression_summary_replaces_old_turns_in_prompt(memory):
    llm = ScriptedLLM(
        lambda system, messages, max_tokens: (
            "[]"
            if "[task:extract_memories]" in system
            else "SUMMARY-OF-OLD-TURNS"
            if "[task:compress_context]" in system
            else "reply " + "z" * 60
        )
    )
    agent = MemoryAgent(llm, memory, max_context_tokens=120, keep_recent=2)
    session = agent.open_session("alice", "s")
    results = [session.send(f"message number {i} " + "q" * 60) for i in range(8)]
    assert any(r.compressed for r in results)
    last_chat = [c for c in llm.calls if "[task:chat]" in c["system"]][-1]
    assert "SUMMARY-OF-OLD-TURNS" in last_chat["system"]
    assert len(last_chat["messages"]) <= 3
    assert "message number 0" not in "".join(m.content for m in last_chat["messages"])


def test_prompt_messages_always_start_with_user_and_alternate(memory):
    llm = ScriptedLLM(
        lambda system, messages, max_tokens: "[]" if "extract" in system else "r" * 50
    )
    agent = MemoryAgent(llm, memory, max_context_tokens=100, keep_recent=3)
    session = agent.open_session("alice", "s")
    for i in range(10):
        session.send(f"m{i} " + "w" * 40)
    for call in (c for c in llm.calls if "[task:chat]" in c["system"]):
        roles = [m.role for m in call["messages"]]
        assert roles[0] == "user" and roles[-1] == "user"
        assert all(a != b for a, b in zip(roles, roles[1:], strict=False))


def test_failed_llm_call_leaves_buffer_unchanged(memory):
    def boom(**_):
        raise LLMError("API down")

    session = MemoryAgent(ScriptedLLM(boom), memory).open_session("alice", "s")
    with pytest.raises(LLMError):
        session.send("hello")
    assert session.messages == []


def test_extraction_can_be_disabled(memory):
    agent = MemoryAgent(HeuristicLLM(), memory, extract=False)
    agent.open_session("alice", "s").send("My name is Alice.")
    assert memory.all("alice") == []


def test_fact_updates_replace_old_values(agent):
    s = agent.open_session("alice", "s")
    s.send("I live in Berlin.")
    s.send("I moved to Lisbon.")
    assert [r.text for r in agent.memory.all("alice")] == ["User lives in Lisbon"]
    assert "Lisbon" in s.send("Where do I live?").reply


def test_close_stores_episode_memory_and_is_idempotent(agent):
    s = agent.open_session("alice", "s")
    s.send("I'm planning a trip to Japan next spring.")
    episode = s.close()
    assert episode.kind == "episode" and "Japan" in episode.text
    assert s.close() is None
    later = agent.open_session("alice", "s2").send("What about that Japan trip?")
    assert any(m.record.kind == "episode" for m in later.recalled)


def test_empty_session_stores_no_episode(agent):
    assert agent.open_session("alice", "s").close() is None


def test_session_resume_restores_buffer_from_checkpoint(tmp_path):
    import sqlite3

    def build():
        saver = SqliteSaver(sqlite3.connect(tmp_path / "ck.sqlite", check_same_thread=False))
        return MemoryAgent(
            HeuristicLLM(),
            LongTermMemory(InMemoryStore()),
            saver,
            max_context_tokens=80,
            keep_recent=2,
        )

    s = build().open_session("alice", "work")
    for i in range(6):
        s.send(f"I like topic number {i} a lot " + "p" * 30)
    assert s.summary and s.compressions >= 1
    resumed = build().open_session("alice", "work")  # new agent, new connection, same file
    assert resumed.summary == s.summary
    assert resumed.messages == s.messages
    assert build().open_session("alice", "other").messages == []


def test_invalid_ids_rejected(agent):
    with pytest.raises(ValueError):
        agent.open_session("../x")
    with pytest.raises(ValueError):
        agent.open_session("alice", "a/b")


def test_users_do_not_leak_into_each_others_prompts(agent):
    agent.open_session("alice", "a").send("I'm allergic to peanuts.")
    reply = agent.open_session("bob", "b").send("What am I allergic to?")
    assert "peanuts" not in reply.reply and reply.recalled == []
