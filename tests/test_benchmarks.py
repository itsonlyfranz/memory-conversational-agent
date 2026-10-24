"""Small measured checks whose numbers are quoted in the README."""

from memory_agent import MemoryAgent
from memory_agent.compression import estimate_tokens
from memory_agent.fakes import HeuristicLLM

FACTS = [
    "My name is Alice.",
    "I live in Berlin.",
    "I work as a data engineer.",
    "I work at Initech.",
    "I'm allergic to peanuts.",
    "I love hiking in the Alps.",
    "I like espresso.",
    "I hate cilantro.",
    "My dog is named Biscuit.",
    "My cat is named Miso.",
]
# (question, substring expected in the top-ranked memory)
QUERIES = [
    ("What is my name?", "Alice"),
    ("Where do I live?", "Berlin"),
    ("What is my job?", "data engineer"),
    ("Which company employs me?", "Initech"),
    ("What am I allergic to?", "peanuts"),
    ("What outdoor activity do I enjoy?", "hiking"),
    ("What coffee drink do I like?", "espresso"),
    ("What food do I dislike?", "cilantro"),
    ("What is my dog called?", "Biscuit"),
    ("What is my cat called?", "Miso"),
]


def test_recall_at_1_on_labelled_queries(agent):
    session = agent.open_session("alice", "seed")
    for fact in FACTS:
        session.send(fact)
    hits = 0
    for question, expected in QUERIES:
        top = agent.memory.recall("alice", question, top_k=1, reinforce=False)
        hits += bool(top) and expected.lower() in top[0].record.text.lower()
    print(f"recall@1 = {hits}/{len(QUERIES)}")
    assert hits >= 7  # lexical embedder: purely semantic paraphrases are the misses


def test_buffer_stays_bounded_over_a_long_conversation(memory):
    agent = MemoryAgent(HeuristicLLM(), memory, max_context_tokens=400, keep_recent=4)
    session = agent.open_session("alice", "long")
    raw = 0
    peak = 0
    for i in range(100):
        text = f"Turn {i}: I want to talk about topic {i} in some detail, please."
        result = session.send(text)
        raw += estimate_tokens(text) + estimate_tokens(result.reply)
        peak = max(peak, session.buffer_tokens)
    print(f"uncompressed={raw} tokens, buffer peak={peak}, final={session.buffer_tokens}")
    assert peak <= 400 + 100
    assert session.buffer_tokens < raw / 4
