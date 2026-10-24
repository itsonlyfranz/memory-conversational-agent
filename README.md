# Memory-Enabled Conversational Agent

A LangGraph conversational agent that remembers its users: a token-budgeted short-term buffer, long-term vector recall, LLM-driven context compression, a multi-signal relevance scorer, and cross-session sync, all running on local files with Claude as the model.

**Technologies:** Python 3.11+, LangGraph, langchain-anthropic (Claude, default `claude-sonnet-5-5`), Pydantic v2, SQLite checkpoints, pytest, uv-compatible packaging. **Category:** AI agents / memory. No external services: stores are local files, and the whole test suite runs offline.

## Highlights

- **Each turn is a LangGraph `StateGraph`** with typed state, parallel branches (buffer update and memory extraction run concurrently), a conditional edge, a SQLite checkpointer for resumable sessions, and a recursion limit.
- **Two-tier memory.** Recent turns stay verbatim in a short-term buffer; durable facts are extracted by the LLM after each turn, embedded, and stored for recall in later sessions.
- **Bounded context.** In a measured 100-turn conversation (`tests/test_benchmarks.py`), the raw transcript is ~2,290 estimated tokens while the buffer never exceeds ~393, with older turns folded into a rolling summary.
- **Relevance, not just similarity.** Recall ranks by `similarity + recency decay + importance + use frequency`, with a similarity gate so an "important" memory never surfaces for an unrelated question.
- **Memories that update.** Single-valued facts use slots ("residence", "job"), so "I moved to Lisbon" replaces "I live in Berlin" instead of contradicting it; near-duplicates are merged and reinforced.
- **Cross-session sync with no server.** An append-only JSONL log with file locking lets several live agents (or processes) share one memory; concurrent-writer and log-compaction behavior is tested (3 processes x 15 writes, none lost).
- **Offline by design.** The LLM sits behind a two-method interface, so all tests and the demo run with a deterministic fake, no network or API key.

## Architecture

| Component | Module | Responsibility |
|---|---|---|
| Turn graph (`build_graph`) | `graph.py` | LangGraph `StateGraph` over a typed `TurnState`; nodes `recall → generate → (update_buffer ∥ extract) → store`. |
| `MemoryAgent` / `ChatSession` | `agent.py` | Session API over the graph: thread id per user+session, checkpointed buffer view, `close()` writes an episodic summary. |
| `ShortTermBuffer` | `buffer.py` | Budget logic used by the `update_buffer` node (its state is checkpointed by LangGraph); folds old turns into the summary when over budget; when over, folds all but the last `keep_recent` messages into the summary. The kept tail always starts on a user turn. |
| `ContextCompressor` | `compression.py` | LLM summarizer that merges the old summary with new messages under a token cap; falls back to an extractive summary if the model fails or overshoots. |
| `MemoryExtractor` | `extraction.py` | Asks the LLM for a JSON list of durable facts/preferences (with importance and slot); tolerant parsing of fences and malformed items. |
| `LongTermMemory` | `memory.py` | Embeds, deduplicates, supersedes by slot, recalls, reinforces, forgets, and evicts the least valuable memories at capacity. Strictly per-user. |
| `RelevanceScorer` | `scoring.py` | Weighted blend of similarity, recency (half-life that grows with importance), importance and saturating frequency; exposes every component. |
| `HashingEmbedder` | `embeddings.py` | Local, deterministic embeddings (stemmed unigrams, bigrams, char trigrams hashed into 512 dims). Behind an `Embedder` protocol. |
| `JsonlMemoryStore` / `InMemoryStore` | `store.py` | Persistence behind a `MemoryStore` protocol. The JSONL store tails other writers' appends under `flock`, runs read-modify-writes in an exclusive `transaction()`, and compacts itself once the log is mostly stale. |
| Checkpointer | `cli.py` / LangGraph | `SqliteSaver` (local file) persists each session's buffer; re-opening a session id resumes it. `ids.py` validates ids against path traversal. |
| `LLMClient` / `AnthropicClient` | `llm.py` | Two-method protocol; `AnthropicClient` wraps `langchain_anthropic.ChatAnthropic`. The only place the network is touched. `fakes.py` provides `HeuristicLLM` and `ScriptedLLM` for offline runs. |

```
START → recall → generate ─┬→ update_buffer ─┐   (parallel; compresses when over budget)
                           └→ extract ───────┴→ store → END
                                       JSONL memory store ◄── shared by other live sessions
```

## Quickstart

```bash
uv sync --extra dev          # or: python -m venv .venv && .venv/bin/pip install -e '.[dev]'
uv run pytest                # fully offline
uv run memory-agent demo     # offline scripted walkthrough
```

For live use with Claude, copy `.env.example` to `.env` (gitignored) and set `ANTHROPIC_API_KEY`; `MEMORY_AGENT_MODEL` overrides the default `claude-sonnet-5-5`.

## Demo usage

```bash
memory-agent demo                # offline; add --live to run the same script against Claude
memory-agent chat --user alice   # interactive; --offline for the fake model; --session ID resumes a saved buffer
memory-agent memories --user alice
```

In chat: `/memories`, `/forget <id-prefix>`, `/quit`. Each turn prints what was recalled (with score components), what was remembered, and when compression ran. Data lives in `.memory_data/` (`memories.jsonl`, `checkpoints.sqlite`) (override with `--data-dir` or `MEMORY_AGENT_DATA_DIR`).

The demo walks through five scenes: (1) learning facts while old turns are compressed, (2) a brand-new process recalling them, (3) "I moved to Lisbon" superseding Berlin, (4) two live sessions sharing memory instantly, and (5) a second user seeing nothing.

## Design decisions and trade-offs

- **Local hashing embeddings instead of a model or hosted vector DB.** Zero setup and deterministic tests. The cost is that it matches words, not meaning: on my 10-question benchmark recall@1 is 7/10, and the misses are pure paraphrases ("what outdoor activity do I enjoy?" vs "hiking"). Swap in a real embedder through the `Embedder` protocol; scoring, storage and tests are unchanged.
- **Brute-force cosine search.** O(n) per user, fine for the hundreds of memories a per-user cap allows (`max_per_user`, default 500); an ANN index would be the next step at scale.
- **Append-only JSONL + advisory file locks for sync.** Simple, inspectable, crash-tolerant (partial and corrupt lines are skipped), and needs no server. It is POSIX-oriented, serializes each remember/recall/forget under an exclusive lock, and meant for one machine, not a distributed deployment.
- **Slots for contradiction handling.** The extractor labels single-valued attributes, so updates are deterministic rather than relying on the model to notice a conflict later. Facts with no slot rely on near-duplicate merging.
- **LangGraph where it fits, not everywhere.** Parallel branches (extraction does not wait on buffer compression, and a conditional edge skips it when disabled) and a checkpointer (session resume) each replace hand-rolled code. The graph is acyclic; no agent tool loop is faked in. The LLM stays behind a two-method protocol rather than being passed as a LangChain model, so tests need no LangChain fakes.
- **Similarity gate before blending.** Importance, recency and frequency only re-rank topically relevant memories.
- **Token counts are estimates** (~4 chars/token), pluggable via `count_tokens`. If the last few messages alone exceed the budget, nothing is dropped; the buffer simply stays over budget.
- **Extraction costs one extra LLM call per turn**, and recall reinforcement writes to the store on reads. Both are simple and visible; `extract=False` and `reinforce=False` turn them off.
- **Failure behavior:** a failed chat call leaves the buffer untouched; failed extraction or compression degrades (no new memories / extractive summary) rather than crashing.

## How the tests prove each capability

Run `pytest` (98 tests, ~3 seconds, no network).

| Capability | Evidence |
|---|---|
| LangGraph flow | `test_graph.py`: node/edge topology, parallel fan-out/fan-in, skipped branch, no leakage between checkpointed turns, recursion limit; `test_agent.py::test_session_resume_restores_buffer_from_checkpoint` (SQLite) |
| Short-term buffer | `test_buffer.py`: budget respected, recent turns verbatim, tail starts on a user turn, state round-trips; `test_agent.py::test_prompt_messages_always_start_with_user_and_alternate` |
| Long-term vector recall | `test_memory.py`, `test_embeddings.py`; `test_agent.py::test_agent_learns_from_conversation_and_recalls_in_a_new_session`; `test_benchmarks.py` (recall@1 = 7/10) |
| Context compression | `test_compression.py` (LLM path, failure and overshoot fallbacks); `test_buffer.py` (summary bounded, fed back into the next compression); `test_benchmarks.py` (buffer peak ~393 vs ~2,290 raw tokens) |
| Relevance scoring | `test_scoring.py`: half-life decay, importance slows decay, frequency saturates, similarity gate, weights change ranking, top-k/min-score |
| Cross-session sync | `test_sync.py` (live sessions share, updates and forgets propagate, survives restart and compaction); `test_store.py` (concurrent processes lose no writes, corrupt and partial lines tolerated) |
| Memory hygiene | `test_memory.py`: slot supersession, dedupe/reinforce, per-user isolation, forgetting, capacity eviction |
| Offline LLM boundary | `test_llm.py` (request forwarding, error wrapping), `test_config.py`; every other test runs on fakes |

Layout: `src/memory_agent/` (package), `tests/`, `pyproject.toml`, `.env.example`.
