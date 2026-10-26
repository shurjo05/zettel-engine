# Data flow

Vault content is private. This file lists every place it goes. Update it whenever a feature
sends vault content somewhere new.

| Feature | Reads | Writes | Leaves this machine? |
| --- | --- | --- | --- |
| `zk index`, `zk check`, `zk search`, `zk show`, `zk eval` | Markdown files in the vault | SQLite index outside the vault | No |
| Semantic search, `zk index --embed` (M4) | Note text from the index | Vectors in the SQLite index | No: the model runs locally. Only the model weights are downloaded, from huggingface.co, into the zk data folder |
| `zk capture` (M6a) | A Claude Code transcript in `~/.claude/projects`, or text you pass | A new file in the vault's `inbox/`, with common secrets redacted | No |
| `zk serve --allow-writes`: ingest (M6) | Inbox captures and the index | New files in `notes/`, `inbox/held/`, `inbox/processed/`, `inbox/rejected/`, `inbox/warnings/` and `eval/questions/` (test questions, which quote your topics); one git commit per run in the vault repo; score history as JSON next to the SQLite index | Yes: captures the agent reads are sent to the client's model provider, like any tool result. A Claude Code capture read from Codex goes to OpenAI. Notes are written by that model from what it read |
| `zk capture --then COMMAND` (M6, opt-in) | The capture it just saved | A log in the zk data folder's `logs/` | Yes, through COMMAND: a headless agent such as `claude -p` reads the inbox through the zk tools and sends it to its model provider, as above |
| `zk import` (M7) | The legacy vault folder, read only | New notes in `notes/` and `inbox/held/`, an import report in `inbox/reports/`, and `eval/legacy-queries.yaml` if `--queries` is given; one git commit. `--dry-run` writes only to a temporary folder it deletes | No: conversion, dedup (local embeddings) and link repair all run on this machine |
| `zk serve` (MCP, M3) | The vault (to refresh the index) and the index | SQLite index outside the vault | Yes, the parts an agent asks for: see below |

### MCP server

The server makes no network calls. It answers the MCP client that started it (Claude Code,
Codex) over stdin/stdout. The client puts each tool result into the conversation, and the
conversation is sent to that client's model provider (Anthropic, OpenAI) under the client's own
data terms. What leaves the machine is therefore what tools return:

- `search_notes`: up to 25 hits, each with path, id, title, type, summary and a short snippet.
- `get_note`: one note's metadata and body (capped at 20,000 characters) and the titles of linked notes.
- `get_related`: up to 50 neighbouring notes' paths, titles and summaries.

The whole vault is never sent at once. Notes you would not want sent to a model provider should be
kept out of the served vault, for example with `--exclude`.

Planned features and their expected flow (to be confirmed when built):

- Optional API features (`zk ask`, API embeddings): opt-in only, documented here before use.

The repository itself contains no vault content; tests use the synthetic vault in
`tests/fixtures/vault`.
