from memory_agent.compression import ContextCompressor, estimate_tokens
from memory_agent.fakes import ScriptedLLM
from memory_agent.llm import LLMError
from memory_agent.models import ChatMessage

MSGS = [
    ChatMessage(role="user", content="I'm planning a trip to Japan in April."),
    ChatMessage(role="assistant", content="Lovely! Cherry blossom season."),
]


def test_uses_llm_summary_and_passes_previous_summary_and_messages():
    llm = ScriptedLLM(["User is planning a Japan trip in April."])
    out = ContextCompressor(llm).compress("User is vegetarian.", MSGS, 100)
    assert out == "User is planning a Japan trip in April."
    sent = llm.calls[0]["messages"][0].content
    assert "User is vegetarian." in sent and "trip to Japan" in sent
    assert "[task:compress_context]" in llm.calls[0]["system"]


def test_falls_back_to_extractive_when_llm_fails():
    def boom(**_):
        raise LLMError("down")

    out = ContextCompressor(ScriptedLLM(boom)).compress("", MSGS, 100)
    assert "trip to Japan" in out
    assert "Cherry blossom" not in out  # only the user's own statements are kept


def test_falls_back_when_llm_returns_empty():
    out = ContextCompressor(ScriptedLLM(["   "])).compress("prev", MSGS, 100)
    assert "prev" in out and "Japan" in out


def test_overlong_model_output_is_hard_trimmed():
    out = ContextCompressor(ScriptedLLM(["word " * 2000])).compress("", MSGS, 50)
    assert estimate_tokens(out) <= 55
