import pytest

from memory_agent.ids import safe_id


@pytest.mark.parametrize("ok", ["alice", "a-b_c.1", "x" * 64])
def test_valid_ids(ok):
    assert safe_id(ok) == ok


@pytest.mark.parametrize("bad", ["../evil", "a/b", "", "..", ".", "x" * 65, "sp ace"])
def test_unsafe_ids_are_rejected(bad):
    with pytest.raises(ValueError):
        safe_id(bad)
