"""The turn is a LangGraph StateGraph: topology, parallel branches, checkpoints."""

import pytest
from langgraph.errors import GraphRecursionError

from memory_agent import MemoryAgent
from memory_agent.fakes import HeuristicLLM, ScriptedLLM


def test_graph_has_the_documented_nodes_and_edges(agent):
    drawn = agent.graph.get_graph()
    assert {"recall", "generate", "update_buffer", "extract", "store"} <= set(drawn.nodes)
    edges = {(e.source, e.target) for e in drawn.edges}
    assert ("recall", "generate") in edges
    assert {("generate", "update_buffer"), ("generate", "extract")} <= edges  # parallel fan-out
    assert {("update_buffer", "store"), ("extract", "store")} <= edges  # fan-in


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


def test_reinforce_can_be_disabled_on_the_agent(memory):
    r = memory.remember("alice", "User lives in Berlin")
    agent = MemoryAgent(HeuristicLLM(), memory, extract=False, reinforce=False)
    s = agent.open_session("alice", "s")
    assert s.send("Where do I live?").recalled
    assert memory.store.get(r.id).access_count == 0


def test_per_turn_scratch_does_not_leak_between_turns(agent):
    s = agent.open_session("alice", "s")
    assert s.send("I live in Berlin.").learned
    assert s.send("Thanks!").learned == []


def test_recursion_limit_bounds_the_graph(memory, monkeypatch):
    monkeypatch.setattr("memory_agent.agent.RECURSION_LIMIT", 2)
    s = MemoryAgent(HeuristicLLM(), memory).open_session("alice", "s")
    with pytest.raises(GraphRecursionError):
        s.send("hello")
