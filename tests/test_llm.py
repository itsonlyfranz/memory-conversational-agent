import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from memory_agent.llm import AnthropicClient, LLMError
from memory_agent.models import ChatMessage


class StubChat:
    def __init__(self, reply=None, error=None):
        self.reply, self.error, self.bound, self.sent = reply, error, None, None

    def bind(self, **kwargs):
        self.bound = kwargs
        return self

    def invoke(self, messages):
        self.sent = messages
        if self.error:
            raise self.error
        return self.reply


def make_client(stub):
    client = AnthropicClient("test-key", "claude-sonnet-5-5")
    client._chat = stub
    return client


def test_converts_messages_and_returns_text():
    stub = StubChat(reply=AIMessage(content="Hello there"))
    out = make_client(stub).complete(
        system="sys",
        messages=[
            ChatMessage(role="user", content="hi"),
            ChatMessage(role="assistant", content="yo"),
        ],
        max_tokens=50,
    )
    assert out == "Hello there"
    assert stub.bound == {"max_tokens": 50}
    assert [type(m) for m in stub.sent] == [SystemMessage, HumanMessage, AIMessage]
    assert stub.sent[0].content == "sys"


def test_model_is_configurable():
    assert make_client(StubChat()).model == "claude-sonnet-5-5"
    assert AnthropicClient("k", "claude-haiku-5-5")._chat.model == "claude-haiku-5-5"


def test_provider_errors_become_llm_errors():
    stub = StubChat(error=RuntimeError("boom"))
    with pytest.raises(LLMError, match="boom"):
        make_client(stub).complete(system="s", messages=[ChatMessage(role="user", content="x")])


def test_missing_api_key_is_a_clear_error():
    with pytest.raises(LLMError, match="ANTHROPIC_API_KEY"):
        AnthropicClient(None, "claude-sonnet-5-5")
