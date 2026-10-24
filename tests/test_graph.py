"""The turn is a LangGraph StateGraph: topology, parallel branches, interrupts, checkpoints."""

import pytest
from langgraph.errors import GraphRecursionError

from memory_agent import MemoryAgent
from memory_agent.fakes import HeuristicLLM, ScriptedLLM


def test_graph_has_the_documented_nodes_and_edges(agent):
    drawn = agent.graph.get_graph()
    assert {"recall", "generate", "update_buffer", "extract", "review", "confirm", "store"} <= set(
        drawn.nodes
    )
    edges = {(e.source, e.target) for e in drawn.edges}
    assert ("recall", "generate") in edges
    assert {("generate", "update_buffer"), ("generate", "extract")} <= edges  # parallel fan-out
    assert {("update_buffer", "review"), ("extract", "review")} <= edges  # fan-in
    assert ("review", "confirm") in edges and ("review", "store") in edges  # conditional


def test_parallel_branches_both_run_in_one_turn(agent):
    s = agent.open_session("alice", "s")
    result = s.send("I live in Berlin.")
    assert result.learned and len(s.messages) == 2  # extract branch AND buffer branch


def test_extract_branch_is_skipped_when_disabled(memory):
    llm = ScriptedLLM(["hello"])
    s = MemoryAgent(llm, memory, extract=False).open_session("alice", "s")
    s.send("hi")
    assert len(llm.calls) == 1 and "[task:chat]" in llm.calls[0]["system"]
    assert len(s.messages) == 2


def test_sensitive_memory_pauses_for_approval_then_saves(memory):
    agent = MemoryAgent(HeuristicLLM(), memory, confirm_threshold=0.95)
    s = agent.open_session("alice", "s")
    paused = s.send("I'm allergic to peanuts and I live in Berlin.")
    assert paused.pending_confirmation == ["User is allergic to peanuts"]
    assert memory.all("alice") == []  # nothing stored yet, not even the non-sensitive one
    done = s.confirm(True)
    assert {r.text for r in done.learned} == {
        "User is allergic to peanuts",
        "User lives in Berlin",
    }
    assert len(s.messages) == 2  # the turn's buffer update is not duplicated by the resume


def test_declined_sensitive_memory_is_dropped_but_others_are_kept(memory):
    agent = MemoryAgent(HeuristicLLM(), memory, confirm_threshold=0.95)
    s = agent.open_session("alice", "s")
    s.send("I'm allergic to peanuts and I live in Berlin.")
    done = s.confirm(False)
    assert done.declined == ["User is allergic to peanuts"]
    assert [r.text for r in memory.all("alice")] == ["User lives in Berlin"]


def test_no_interrupt_when_nothing_is_sensitive(memory):
    agent = MemoryAgent(HeuristicLLM(), memory, confirm_threshold=0.95)
    result = agent.open_session("alice", "s").send("I live in Berlin.")
    assert result.pending_confirmation == [] and result.learned


def test_per_turn_scratch_does_not_leak_between_turns(agent):
    s = agent.open_session("alice", "s")
    assert s.send("I live in Berlin.").learned
    assert s.send("Thanks!").learned == []


def test_recursion_limit_bounds_the_graph(memory, monkeypatch):
    monkeypatch.setattr("memory_agent.agent.RECURSION_LIMIT", 2)
    s = MemoryAgent(HeuristicLLM(), memory).open_session("alice", "s")
    with pytest.raises(GraphRecursionError):
        s.send("hello")
