# Decisions

Short records of choices that shape the code. Newest last.

## 1. Rebuild instead of refactoring the legacy pipeline (2026-10-15)

The legacy scripts rejected every note submitted after 2026-03-26 on formatting, matched
duplicates by title words only, and had no tests or undo. None of the code was kept; the notes
and design docs were kept as reference and test data.

## 2. Markdown is the source of truth; the index is a cache (2026-10-15)

Every table in SQLite is derived from the vault and can be deleted. This keeps Obsidian, Git
and the AI tools agreeing on one copy of the knowledge, and makes schema changes cheap: bump
`SCHEMA_VERSION` and the index rebuilds itself.

## 3. SQLite rather than JSON files or a vector database (2026-10-15)

SQLite is in the standard library, stores everything in one file, supports joins for the link
graph, and has FTS5 for keyword search in M2. Vectors (M4) fit in the same file at this scale.
A separate vector database (Chroma, Qdrant) would be a second store to keep in sync.

## 4. Content hashes, not modification times, for change detection (2026-10-15)

A SHA-256 of the file bytes changes exactly when the content does. Timestamps change on copy
or sync without content changes and are unreliable across machines. The cost is reading every
file on each build, which is negligible for thousands of small notes.

## 5. Stable `id` plus Obsidian-style link resolution (2026-10-15)

Links resolve the way Obsidian does: by file name, case-insensitively, or by vault path when
the link contains `/`. As a fallback a link may name a note's frontmatter `id`, which survives
renames. Two notes with the same file name make a link ambiguous; it is reported, never guessed.

## 6. Real YAML parsing that tolerates the legacy dialect (2026-10-15)

PyYAML `safe_load` replaces the regex parser, so quoted values are valid. Fields the new schema
dropped (`domain`, `status`, `confidence`) are kept in the stored frontmatter but ignored.

## 7. Keyword search with SQLite FTS5 and BM25 (2026-10-15)

FTS5 lives in the same index file, ranks with BM25, and supports per-column weights and Porter
stemming. A hand-written inverted index would teach the same ideas but duplicate a correct, fast
implementation. Queries quote every word and join them with OR: quoting neutralises FTS5 syntax in
user input, and OR keeps natural-language questions from returning nothing.

Column weights are title 10, aliases 8, tags 5, summary 3, body 1. On the 30-query legacy eval set
they improved MRR only from 0.80 (equal weights) to 0.82, which is one query's difference; they stay
because they never hurt, and the eval set will judge them again as it grows.

## 8. Evaluation from the start, with private query sets kept out of the repo (2026-10-15)

`zk eval` measures Recall@k, MRR and latency on a labelled query set so each later change (embeddings,
hybrid ranking, link expansion) is judged against a recorded baseline instead of impressions. The
repository holds only the fixture query set; sets that name private notes live with the vault.

## 9. A read-only MCP server over stdio (2026-10-16)

`zk serve` exposes `search_notes`, `get_note` and `get_related` as MCP tools, so Claude Code and
Codex use the same library as the CLI. It is read-only: no write tools exist until M6, and then only
behind `--allow-writes`. Stdio rather than HTTP because there is one user on one machine; it needs no
open port and no auth. Tools rather than MCP resources because agents call tools; clients mostly
show resources to humans.

Context efficiency shapes the tools. Search returns summaries and snippets, never bodies; bodies
come one note at a time from `get_note` and are capped at 20,000 characters; `get_related` returns
summaries so an agent can walk the link graph cheaply. Results are compact JSON text (no
indentation, no duplicate structured copy), since every returned character costs context.

Each call refreshes the index before answering, so edits in Obsidian are visible immediately.
To make that cheap, `build_index` now compares a file's hash before parsing it and re-resolves links
only when something changed: a no-change refresh of the 96-note legacy copy fell from 88 ms to 7 ms.

## 10. The official `mcp` SDK as a dependency (2026-10-16)

The Model Context Protocol is JSON-RPC with a handshake, capability negotiation and schemas for tool
inputs; the official Python SDK (`mcp` 2.x) implements it and is what clients are tested against.
It brings pydantic, anyio, starlette and httpx with it, which is the heaviest dependency so far, but
writing the protocol by hand would be more code to get wrong than everything else in this repo. It
is imported only by `zk/server.py` (and lazily by `zk serve`), so the core stays usable without it.
