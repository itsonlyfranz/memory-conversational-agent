from memory_agent.extraction import MemoryExtractor, parse_extraction
from memory_agent.fakes import ScriptedLLM
from memory_agent.llm import LLMError


def test_parses_plain_json_array():
    items = parse_extraction(
        '[{"text": "User lives in Rome", "importance": 0.8, "slot": "residence"}]'
    )
    assert items[0].text == "User lives in Rome" and items[0].slot == "residence"
    assert items[0].kind == "fact"


def test_tolerates_code_fences_and_prose():
    raw = 'Sure!\n```json\n[{"text": "User likes tea", "kind": "preference"}]\n```'
    assert parse_extraction(raw)[0].kind == "preference"


def test_drops_invalid_items_and_garbage():
    raw = '[{"text": ""}, {"nope": 1}, {"text": "ok", "importance": 5}, {"text": "fine"}]'
    assert [m.text for m in parse_extraction(raw)] == ["fine"]
    assert parse_extraction("no json here") == []
    assert parse_extraction("[not, json]") == []
    assert parse_extraction('{"text": "x"}') == []


def test_extractor_sends_exchange_and_handles_llm_failure():
    llm = ScriptedLLM(['[{"text": "User has a cat"}]'])
    out = MemoryExtractor(llm).extract("I have a cat", "Nice!")
    assert out[0].text == "User has a cat"
    sent = llm.calls[0]["messages"][0].content
    assert "I have a cat" in sent and "Nice!" in sent

    def boom(**_):
        raise LLMError("down")

    assert MemoryExtractor(ScriptedLLM(boom)).extract("x", "y") == []
