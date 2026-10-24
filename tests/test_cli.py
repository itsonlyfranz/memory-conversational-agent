import builtins

from memory_agent.cli import main


def test_demo_runs_offline_and_shows_headline_behaviours(capsys, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert main(["demo"]) == 0
    out = capsys.readouterr().out
    assert "context compressed into summary" in out
    assert "From what I remember: User lives in Berlin." in out
    assert "From what I remember: User lives in Lisbon." in out
    assert "From what I remember: User likes espresso" in out
    assert "=== 5. Isolation" in out and "I don't have anything about that yet." in out
    assert "paused for approval: ['User is allergic to penicillin']" in out


def test_live_demo_without_key_fails_cleanly(capsys, monkeypatch, tmp_path):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)  # no .env here
    assert main(["demo", "--live"]) == 2
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err


def test_chat_persists_memories_and_memories_command_lists_them(capsys, monkeypatch, tmp_path):
    lines = iter(["My name is Dana.", "/memories", "/quit"])
    monkeypatch.setattr(builtins, "input", lambda _="": next(lines))
    data = str(tmp_path / "data")
    assert main(["chat", "--offline", "--user", "dana", "--data-dir", data]) == 0
    assert main(["memories", "--user", "dana", "--data-dir", data]) == 0
    out = capsys.readouterr().out
    assert "User's name is Dana" in out
    assert out.count("User's name is Dana") >= 3  # turn log, /memories, memories command


def test_chat_forget_command(capsys, monkeypatch, tmp_path):
    data = str(tmp_path / "d")
    first = iter(["I live in Oslo.", "/quit"])
    monkeypatch.setattr(builtins, "input", lambda _="": next(first))
    main(["chat", "--offline", "--user", "eve", "--data-dir", data])
    from memory_agent import JsonlMemoryStore, LongTermMemory

    mem = LongTermMemory(JsonlMemoryStore(tmp_path / "d" / "memories.jsonl"))
    prefix = next(r.id for r in mem.all("eve") if r.kind == "fact")[:8]
    second = iter([f"/forget {prefix}", "/quit"])
    monkeypatch.setattr(builtins, "input", lambda _="": next(second))
    main(["chat", "--offline", "--user", "eve", "--data-dir", data])
    assert "forgotten" in capsys.readouterr().out
    assert not [r for r in mem.all("eve") if r.kind == "fact"]


def test_chat_confirm_sensitive_prompts_and_respects_no(capsys, monkeypatch, tmp_path):
    lines = iter(["I'm allergic to peanuts.", "n", "/memories", "/quit"])
    monkeypatch.setattr(builtins, "input", lambda _="": next(lines))
    data = str(tmp_path / "d")
    assert (
        main(["chat", "--offline", "--confirm-sensitive", "--user", "fay", "--data-dir", data]) == 0
    )
    out = capsys.readouterr().out
    assert "save these sensitive memories?" in out
    assert "not saved (declined): User is allergic to peanuts" in out
