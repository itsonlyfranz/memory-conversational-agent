"""Command line: interactive chat, memory inspection, and a scripted offline demo."""

from __future__ import annotations

import argparse
import sqlite3
import sys
import tempfile
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

from .agent import ChatSession, MemoryAgent, TurnResult
from .config import Settings
from .fakes import HeuristicLLM
from .llm import AnthropicClient, LLMClient, LLMError
from .memory import LongTermMemory
from .store import JsonlMemoryStore


def build_agent(
    data_dir: Path,
    llm: LLMClient,
    *,
    max_context_tokens: int = 1200,
    keep_recent: int = 4,
) -> MemoryAgent:
    """Wire an agent whose memories and session checkpoints live under ``data_dir``."""
    data_dir.mkdir(parents=True, exist_ok=True)
    memory = LongTermMemory(JsonlMemoryStore(data_dir / "memories.jsonl"))
    checkpointer = SqliteSaver(
        sqlite3.connect(data_dir / "checkpoints.sqlite", check_same_thread=False)
    )
    return MemoryAgent(
        llm,
        memory,
        checkpointer,
        max_context_tokens=max_context_tokens,
        keep_recent=keep_recent,
    )


def _llm_from_settings(settings: Settings, offline: bool) -> LLMClient:
    if offline:
        return HeuristicLLM()
    return AnthropicClient(settings.api_key, settings.model)


def _show_turn(result: TurnResult) -> None:
    for s in result.recalled:
        print(
            f"   · recalled [{s.score:.2f}] {s.record.text}  "
            f"(sim {s.similarity:.2f}, rec {s.recency:.2f}, imp {s.importance:.2f})"
        )
    for r in result.learned:
        print(f"   · remembered: {r.text}")
    if result.compressed:
        print("   · context compressed into summary")


def cmd_chat(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    data_dir = Path(args.data_dir or settings.data_dir)
    try:
        llm = _llm_from_settings(settings, args.offline)
    except LLMError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    agent = build_agent(data_dir, llm, max_context_tokens=settings.max_context_tokens)
    session = agent.open_session(args.user, args.session)
    print(f"session {session.session_id} for {args.user} — /memories, /forget <id>, /quit")
    try:
        while True:
            try:
                line = input("you> ").strip()
            except EOFError:
                break
            if not line:
                continue
            if line == "/quit":
                break
            if line == "/memories":
                for r in agent.memory.all(args.user):
                    print(f"  {r.id[:8]}  [{r.kind}] {r.text}")
                continue
            if line.startswith("/forget "):
                target = line.split(None, 1)[1].strip()
                match = [r for r in agent.memory.all(args.user) if r.id.startswith(target)]
                ok = len(match) == 1 and agent.memory.forget(args.user, match[0].id)
                print("  forgotten" if ok else "  no unique memory with that id")
                continue
            try:
                result = session.send(line)
            except LLMError as exc:
                print(f"  error: {exc}")
                continue
            print(f"agent> {result.reply}")
            _show_turn(result)
    finally:
        session.close()
    return 0


def cmd_memories(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    store = JsonlMemoryStore(Path(args.data_dir or settings.data_dir) / "memories.jsonl")
    for r in LongTermMemory(store).all(args.user):
        print(f"{r.id[:8]}  [{r.kind}] imp={r.importance:.1f} used={r.access_count}  {r.text}")
    return 0


def _banner(title: str) -> None:
    print(f"\n=== {title} ===")


def _say(session: ChatSession, text: str) -> TurnResult:
    print(f"you> {text}")
    result = session.send(text)
    print(f"agent> {result.reply}")
    _show_turn(result)
    return result


def cmd_demo(args: argparse.Namespace) -> int:
    """Run the headline behaviours end to end (offline by default)."""
    settings = Settings.from_env()
    try:
        llm = _llm_from_settings(settings, offline=not args.live)
    except LLMError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    with tempfile.TemporaryDirectory(prefix="memory-demo-") as tmp:
        data_dir = Path(tmp)
        # A deliberately small buffer budget so compression is visible in a short demo.
        agent = build_agent(data_dir, llm, max_context_tokens=80, keep_recent=2)

        _banner("1. Session one: the agent learns about Alice (and compresses old turns)")
        s1 = agent.open_session("alice", "mon")
        for text in (
            "Hi! My name is Alice.",
            "I live in Berlin and I work as a data engineer.",
            "I'm allergic to peanuts.",
            "I love hiking in the Alps on weekends.",
            "My dog is named Biscuit.",
            "Work has been busy lately with a long migration project and many meetings.",
        ):
            _say(s1, text)
        print(f"   buffer: {s1.buffer_tokens} tokens, {s1.compressions} compression(s)")
        print(f"   summary: {s1.summary}")
        s1.close()

        _banner("2. A brand-new process and session: long-term recall across sessions")
        agent2 = build_agent(data_dir, llm, max_context_tokens=80)
        s2 = agent2.open_session("alice", "tue")
        _say(s2, "Where do I live?")
        _say(s2, "What am I allergic to?")
        _say(s2, "What is my dog called?")

        _banner("3. Facts change: the newer value supersedes the old one")
        _say(s2, "I moved to Lisbon.")
        _say(s2, "Where do I live?")

        _banner("4. Cross-session sync: two live sessions share memory instantly")
        agent_a = build_agent(data_dir, llm)
        agent_b = build_agent(data_dir, llm)
        a = agent_a.open_session("alice", "laptop")
        b = agent_b.open_session("alice", "phone")
        _say(a, "I like espresso.")
        _say(b, "What do I like?")

        _banner("5. Isolation: another user sees none of it")
        other = agent.open_session("bob", "bob1")
        _say(other, "Where do I live?")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="memory-agent", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    chat = sub.add_parser("chat", help="interactive chat with long-term memory")
    chat.add_argument("--user", default="default")
    chat.add_argument(
        "--session", help="session id; an existing id resumes its saved buffer (default: new)"
    )
    chat.add_argument("--offline", action="store_true", help="use the rule-based fake model")
    chat.add_argument("--data-dir")
    chat.set_defaults(func=cmd_chat)

    mem = sub.add_parser("memories", help="list stored memories for a user")
    mem.add_argument("--user", default="default")
    mem.add_argument("--data-dir")
    mem.set_defaults(func=cmd_memories)

    demo = sub.add_parser("demo", help="scripted walkthrough of the headline behaviours")
    demo.add_argument("--live", action="store_true", help="use Claude instead of the offline fake")
    demo.set_defaults(func=cmd_demo)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
