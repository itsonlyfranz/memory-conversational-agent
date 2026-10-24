import multiprocessing as mp
import threading

from memory_agent.models import MemoryRecord
from memory_agent.store import InMemoryStore, JsonlMemoryStore


def make(text="User lives in Berlin", user="u", **kw):
    return MemoryRecord(user_id=user, text=text, embedding=[0.1, 0.2], **kw)


def test_in_memory_store_roundtrip_and_isolation_of_copies():
    s = InMemoryStore()
    r = make()
    s.upsert(r)
    got = s.get(r.id)
    got.text = "mutated"
    assert s.get(r.id).text == "User lives in Berlin"
    assert [x.id for x in s.list("u")] == [r.id]
    assert s.list("other") == []
    assert s.delete(r.id) and not s.delete(r.id)


def test_jsonl_store_persists_across_instances(tmp_path):
    path = tmp_path / "m.jsonl"
    r = make()
    JsonlMemoryStore(path).upsert(r)
    reopened = JsonlMemoryStore(path)
    assert reopened.get(r.id).text == "User lives in Berlin"
    assert reopened.list("u")[0].embedding == [0.1, 0.2]


def test_jsonl_upsert_overrides_and_delete_persists(tmp_path):
    path = tmp_path / "m.jsonl"
    s = JsonlMemoryStore(path)
    r = make()
    s.upsert(r)
    s.upsert(r.model_copy(update={"text": "User lives in Lisbon"}))
    assert JsonlMemoryStore(path).get(r.id).text == "User lives in Lisbon"
    assert s.delete(r.id)
    assert JsonlMemoryStore(path).get(r.id) is None
    assert not s.delete(r.id)


def test_other_instance_sees_new_writes_without_reopening(tmp_path):
    path = tmp_path / "m.jsonl"
    a, b = JsonlMemoryStore(path), JsonlMemoryStore(path)
    assert b.list("u") == []
    a.upsert(make())
    assert len(b.list("u")) == 1


def test_corrupt_lines_are_skipped(tmp_path):
    path = tmp_path / "m.jsonl"
    s = JsonlMemoryStore(path)
    s.upsert(make("first"))
    with open(path, "ab") as fh:
        fh.write(b'{not json}\n{"op":"upsert","record":{"nope":1}}\n')
    s.upsert(make("second"))
    assert sorted(r.text for r in JsonlMemoryStore(path).list("u")) == ["first", "second"]


def test_partial_trailing_line_is_ignored_until_complete(tmp_path):
    path = tmp_path / "m.jsonl"
    s = JsonlMemoryStore(path)
    r = make()
    line = ('{"op":"upsert","record":' + r.model_dump_json() + "}\n").encode()
    with open(path, "ab") as fh:
        fh.write(line[:20])
    assert s.list("u") == []
    with open(path, "ab") as fh:
        fh.write(line[20:])
    assert len(s.list("u")) == 1


def test_compaction_shrinks_log_and_other_readers_follow(tmp_path):
    path = tmp_path / "m.jsonl"
    writer, reader = JsonlMemoryStore(path), JsonlMemoryStore(path)
    r = make()
    for i in range(10):
        writer.upsert(r.model_copy(update={"text": f"v{i}"}))
    assert reader.get(r.id).text == "v9"
    before = path.stat().st_size
    writer.compact()
    assert path.stat().st_size < before
    writer.upsert(make("another"))
    assert {x.text for x in reader.list("u")} == {"v9", "another"}


def test_reader_rebuilds_when_a_compacted_log_reuses_the_same_inode(tmp_path):
    path = tmp_path / "m.jsonl"
    reader = JsonlMemoryStore(path)
    forgotten = make("forgotten")
    reader.upsert(forgotten)
    assert reader.get(forgotten.id)
    other = tmp_path / "other.jsonl"
    writer = JsonlMemoryStore(other)
    for i in range(5):
        writer.upsert(make(f"kept-{i}"))
    writer.compact()
    with open(path, "r+b") as fh:  # same inode, new generation, larger than the reader's offset
        fh.write(other.read_bytes())
        fh.truncate()
    assert sorted(r.text for r in reader.list("u")) == [f"kept-{i}" for i in range(5)]


def test_log_compacts_itself_once_mostly_stale(tmp_path):
    path = tmp_path / "m.jsonl"
    s = JsonlMemoryStore(path, compact_slack=5)
    r = make()
    for i in range(50):
        s.upsert(r.model_copy(update={"text": f"v{i}"}))
    assert len(path.read_bytes().splitlines()) < 10
    assert JsonlMemoryStore(path).get(r.id).text == "v49"


def test_writes_from_another_instance_wait_for_an_open_transaction(tmp_path):
    path = tmp_path / "m.jsonl"
    a, b = JsonlMemoryStore(path), JsonlMemoryStore(path)
    r = make()
    a.upsert(r)
    other = threading.Thread(target=b.upsert, args=(r.model_copy(update={"text": "Lisbon"}),))
    with a.transaction():
        stale = a.get(r.id)
        other.start()
        other.join(0.3)
        assert other.is_alive()
        a.upsert(stale.model_copy(update={"access_count": stale.access_count + 1}))
    other.join(5)
    assert JsonlMemoryStore(path).get(r.id).text == "Lisbon"


def _worker(path, n, tag):
    s = JsonlMemoryStore(path)
    for i in range(n):
        s.upsert(make(f"{tag}-{i}"))


def test_concurrent_processes_do_not_lose_writes(tmp_path):
    path = tmp_path / "m.jsonl"
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=_worker, args=(path, 15, f"p{i}")) for i in range(3)]
    [p.start() for p in procs]
    [p.join(60) for p in procs]
    assert all(p.exitcode == 0 for p in procs)
    assert len(JsonlMemoryStore(path).list("u")) == 45
