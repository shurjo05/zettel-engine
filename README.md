# zettel-engine

Index, search and maintain an Obsidian Zettelkasten so AI tools (Claude Code, Codex) can
retrieve the knowledge they need without loading the whole vault.

The Markdown vault is the only source of truth. Everything this tool builds is a derived,
rebuildable index stored outside the vault.

## Status

| Milestone | What | State |
| --- | --- | --- |
| M0 | Repo skeleton, tests, CI | done |
| M1 | Note catalog and `zk check` health report | done |
| M2 | Keyword search (SQLite FTS5, BM25) and evaluation set | next |
| M3 | Read-only MCP server | planned |
| M4-M9 | Embeddings, hybrid search, ingestion, maintenance, evaluation | planned |

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest
```

## Usage

```bash
uv run zk check path/to/vault          # index the vault, then print a health report
uv run zk check path/to/vault --json   # the same report as JSON
uv run zk index path/to/vault          # just build or refresh the index
```

`zk` only reads the vault. It skips hidden folders (`.obsidian`, `.git`) plus `inbox/**` and
`templates/**`; add more with `--exclude GLOB` (repeatable, vault-relative, `*` also matches `/`).

The index is a single SQLite file outside the vault: `%LOCALAPPDATA%\zk\` on Windows,
`~/.local/share/zk/` elsewhere, or `$ZK_DATA_DIR`. Pass `--db` to choose a file. Deleting it is
always safe; the next run rebuilds it.

`zk check` exits with status 1 when any note has an error (invalid YAML, missing id or type,
unknown type, duplicate id), so it can run in CI or a pre-commit hook.

## Note format

```markdown
---
id: 20261015143200
title: Reciprocal Rank Fusion for Hybrid Search
type: pattern            # concept | pattern | problem | decision
tags: [retrieval, hybrid-search]
aliases: [RRF]
sources:
  - project:ruflo-claude-flow:2026-03-23
created: 2026-10-15
updated: 2026-10-15
---
# Reciprocal Rank Fusion for Hybrid Search

One to three sentences that state the idea.

## Details
## Related
- [[bm25-first-escalate-to-semantic-search]]: RRF combines its two result lists
## Sources
```

The parser also reads the legacy dialect (quoted scalars, single `source`, `## Summary`
sections), so an old vault can be checked without rewriting it.

## Layout

```
src/zk/notes.py   parse and validate one note (pure, no file access)
src/zk/index.py   SQLite catalog: notes, links, issues; incremental builds; health report
src/zk/cli.py     the `zk` command
tests/            pytest suite and a small synthetic fixture vault
docs/             decisions and data flow
```

See [docs/decisions.md](docs/decisions.md) for why things are built this way and
[docs/data-flow.md](docs/data-flow.md) for where vault content goes.
