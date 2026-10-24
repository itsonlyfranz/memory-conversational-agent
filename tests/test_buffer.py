from memory_agent.buffer import BufferState, ShortTermBuffer
from memory_agent.compression import ContextCompressor, estimate_tokens
from memory_agent.fakes import HeuristicLLM, ScriptedLLM
from memory_agent.models import ChatMessage


def turn(i, size=40):
    return [
        ChatMessage(role="user", content=f"user message {i} " + "x" * size),
        ChatMessage(role="assistant", content=f"assistant reply {i} " + "y" * size),
    ]


def make(max_tokens=200, keep_recent=4, llm=None):
    return ShortTermBuffer(
        ContextCompressor(llm or HeuristicLLM()),
        max_tokens=max_tokens,
        keep_recent=keep_recent,
        summary_max_tokens=60,
    )


def test_under_budget_nothing_is_compressed():
    buf = make(max_tokens=1000)
    for i in range(3):
        assert buf.add(*turn(i)) is False
    assert buf.summary == "" and len(buf.messages) == 6


def test_over_budget_compresses_old_turns_and_keeps_recent_verbatim():
    buf = make(max_tokens=200, keep_recent=4)
    compressed = [buf.add(*turn(i)) for i in range(10)]
    assert any(compressed) and buf.compressions >= 1
    assert buf.tokens <= 200
    assert buf.messages[-1].content.startswith("assistant reply 9")
    assert buf.messages[-4].content.startswith("user message 8") or len(buf.messages) >= 4
    assert "user message" in buf.summary and buf.summary.count("user message") >= 2


def test_kept_tail_always_starts_with_user_turn():
    buf = make(max_tokens=120, keep_recent=3)  # odd keep_recent would split a pair
    for i in range(8):
        buf.add(*turn(i))
        assert not buf.messages or buf.messages[0].role == "user"


def test_summary_stays_bounded_across_many_compressions():
    buf = make(max_tokens=150, keep_recent=2)
    for i in range(60):
        buf.add(*turn(i))
    assert estimate_tokens(buf.summary) <= 60 * 1.5 + 1
    assert buf.tokens <= 150 + 60


def test_summary_is_fed_back_into_next_compression():
    llm = ScriptedLLM(["S1" + "." * 150, "S2" + "." * 150])
    buf = make(max_tokens=100, keep_recent=2, llm=llm)
    for i in range(6):
        buf.add(*turn(i))
    assert buf.summary.startswith(("S1", "S2"))
    assert "Existing summary:\nS1" in llm.calls[1]["messages"][0].content


def test_cannot_compress_when_only_recent_messages_exceed_budget():
    buf = make(max_tokens=10, keep_recent=4)
    assert buf.add(*turn(0, size=500)) is False
    assert len(buf.messages) == 2


def test_state_roundtrip():
    buf = make(max_tokens=100, keep_recent=2)
    for i in range(5):
        buf.add(*turn(i))
    restored = make()
    restored.load(BufferState.model_validate_json(buf.state().model_dump_json()))
    assert restored.summary == buf.summary and restored.messages == buf.messages
