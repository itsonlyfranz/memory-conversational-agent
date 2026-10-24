from memory_agent.config import DEFAULT_MODEL, Settings, load_dotenv


def test_defaults(monkeypatch):
    for k in (
        "ANTHROPIC_API_KEY",
        "MEMORY_AGENT_MODEL",
        "MEMORY_AGENT_DATA_DIR",
        "MEMORY_AGENT_MAX_CONTEXT_TOKENS",
    ):
        monkeypatch.delenv(k, raising=False)
    s = Settings.from_env(dotenv=None)
    assert s.model == DEFAULT_MODEL == "claude-sonnet-5-5"
    assert s.api_key is None and s.max_context_tokens == 1200


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("MEMORY_AGENT_MODEL", "claude-haiku-5-5")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setenv("MEMORY_AGENT_MAX_CONTEXT_TOKENS", "500")
    s = Settings.from_env(dotenv=None)
    assert (s.model, s.api_key, s.max_context_tokens) == ("claude-haiku-5-5", "k", 500)


def test_dotenv_loads_but_never_overrides_real_env(tmp_path, monkeypatch):
    monkeypatch.delenv("FOO_A", raising=False)
    monkeypatch.setenv("FOO_B", "real")
    env = tmp_path / ".env"
    env.write_text("# comment\nFOO_A='from-file'\nexport FOO_C=3\nFOO_B=file\nbadline\n")
    monkeypatch.delenv("FOO_C", raising=False)
    load_dotenv(env)
    import os

    assert os.environ["FOO_A"] == "from-file" and os.environ["FOO_C"] == "3"
    assert os.environ["FOO_B"] == "real"
    monkeypatch.delenv("FOO_A")
    monkeypatch.delenv("FOO_C")
    load_dotenv(tmp_path / "missing.env")  # no error
