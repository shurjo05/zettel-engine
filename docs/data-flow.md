# Data flow

Vault content is private. This file lists every place it goes. Update it whenever a feature
sends vault content somewhere new.

| Feature | Reads | Writes | Leaves this machine? |
| --- | --- | --- | --- |
| `zk index`, `zk check`, `zk search`, `zk show`, `zk eval`, `zk suggest` | Markdown files in the vault | SQLite index outside the vault | No |
| Semantic search, `zk index --embed` (M4) | Note text from the index | Vectors in the SQLite index | No: the model runs locally. Only the model weights are downloaded, from huggingface.co, into the zk data folder |
| `zk capture` (M6a) | A Claude Code transcript, or text you pass | A new file in the vault's `inbox/`, with common secrets redacted | No |
| `zk serve --allow-writes`: ingest (M6) | Inbox captures and the index | New files in `notes/`, `inbox/held/`, `inbox/processed/`, `inbox/rejected/`, `inbox/warnings/` and `eval/questions/` (test questions, which quote your topics); one git commit per run in the vault repo; score history as JSON next to the SQLite index | Yes: captures the agent reads are sent to the client's model provider, like any tool result. A Claude Code capture read from Codex goes to OpenAI. Notes are written by that model from what it read |
| `zk capture --then COMMAND` (M6, opt-in) | The capture it just saved | A log in the zk data folder's `logs/` | Yes, through COMMAND: a headless agent such as `claude -p` reads the inbox through the zk tools and sends it to its model provider, as above |
| Session-end hook, `zk capture --catch-up` (M10) | Claude Code transcripts in `~/.claude/projects`, from every project on this machine | New captures in `inbox/`, secrets redacted; `sessions.json` (session ids and capture names) in the zk data folder | No |
| `zk process`, after each session and in the daily `zk catch-up` task (M10) | Inbox captures and held notes, through the zk tools | What ingest and resolve write (rows above); a log in the zk data folder's `logs/` | Yes, without you in the loop: a headless `claude -p` sends each capture, and then each held pair, to Anthropic. A step with no work makes no call |
| `zk hooks install`, `zk schedule install` (M10) | Claude Code's `settings.json` | One added hook entry and a backup next to it; a Windows scheduled task | No |
| `zk eval --context` (M11) | The index | Nothing | No |
| `zk ask`, `zk eval --answers` (M11, opt-in) | The index | Nothing | Yes: each question and the text of its top 5 notes (up to about 2,400 characters each) go to Anthropic through a headless `claude -p` with your login |
| `zk calibrate` (M11, opt-in) | The index and the vault's git log | Label files in `eval/calibration/` (note names, content hashes, labels, one-line reasons; no note text); one git commit | Yes: each sampled pair's titles and text (up to about 1,500 characters each) go to Anthropic through `claude -p`. `--no-judge` sends nothing |
| `zk import` (M7) | The legacy vault folder, read only | New notes in `notes/` and `inbox/held/`, an import report in `inbox/reports/`, and `eval/legacy-queries.yaml` if `--queries` is given; one git commit. `--dry-run` writes only to a temporary folder it deletes | No: conversion, dedup (local embeddings) and link repair all run on this machine |
| `zk serve --allow-writes`: resolving held notes (M8) | Held notes and the notes closest to them | Changes the note a held duplicate merges into, retargets links in other notes and labels in `eval/`, moves held notes to `notes/` or `inbox/merged/`; one git commit per decision | Yes: each held note and its closest note go in full to the model provider of the agent running `/zk-resolve`, which writes the merged note. A scheduled run sends them without you in the loop |
| `zk maps`, and the end of each `zk process` run that changed the vault (M12) | The index | Map notes in `maps/` (titles and summaries of other notes); retired maps moved to `maps/retired/`; one git commit | No |
| Session-start hook, `zk brief` (M12) | The index, read-only | A per-session record in the zk data folder's `hook-state/` | Yes, in every Claude Code session on this machine: the note count, map names and, inside a known project's repository, that project's map entries (titles and the start of each summary; about 1,000 tokens at most) go to Anthropic as part of the session's context |
| Search nudge hook (M12) | Nothing from the vault | A per-session record in `hook-state/` | One fixed sentence, no vault content |
| Failed-command hook (M12) | The error text Claude Code passes it, and the index, read-only | A per-session record in `hook-state/` | Yes: up to two matching notes' names, titles and summaries go to Anthropic as part of that session. The error text itself stays on this machine; zk only searches with it |
| Prompt hook (M12) | The prompt Claude Code passes it, and the index, read-only | A per-session record in `hook-state/` | Yes: up to three clearly matching notes' names, titles and summaries go to Anthropic as part of that session. The prompt already goes to Anthropic; zk only searches with it locally |
| `zk learn`, and the session-start hook in a repository zk hasn't read (M12) | A project folder: tree, README, manifests, recent commit subjects, and class and function names in source files; never dependency or build folders or likely secret files (`.env`, keys, `appsettings*.json` and similar) | A capture in `inbox/`, secrets redacted; `learn.json` (project paths and file hashes) in the zk data folder | Not by itself. The next `zk process` sends the capture to Anthropic like any capture (row above). `zk learn --skip FOLDER` keeps a project out, including from the automatic run |
| `zk docs`, and the session-start hook in a project whose docs changed (M14) | A project's `*.md` and `*.pdf` files (or the ones `--only` names), never dependency or build folders or likely secret files | Notes in `refs/<project>/` (the docs' own text, secrets redacted); sections that are gone moved to `refs/retired/`; one git commit; `docs.json` (project paths, options, file sizes and times) in the zk data folder | Not by itself. The vault's git remote gets them when you push, like any note. The sections an agent opens go to its model provider like any note |
| Read hook (M14) | The file path Claude Code passes it and `docs.json` | A per-session record in `hook-state/` | Yes: the doc's path, size, section count and contents-note name go to Anthropic as part of that session. No doc text |
| `~/.claude/rules/zk.md` (M12, added by `zk hooks install`) | Nothing | A new file in Claude Code's user rules folder, never overwritten | Its fixed text goes to Anthropic in every session; no vault content |
| `zk review`, weekly task (M12) | The index, the vault, its git log, `learn.json` | A report in `inbox/reports/`; one git commit | No |
| Headless runs' MCP config (M12) | Nothing | `zk-agents-mcp.json` in the zk data folder: the command for the write-enabled zk server | No |
| `zk.api.save_text`, used by tokenmaxxing (2026-11-12) | Text the calling program passes: a coding agent's note suggestions after a task | A new capture in `inbox/`, secrets redacted | Not by this call. The suggestions were written by the coding agent's model (OpenAI for Codex), and the next ingest run sends the capture to its agent's provider (Anthropic), like any capture |
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

- API embeddings: opt-in only, documented here before use.

The repository itself contains no vault content; tests use the synthetic vault in
`tests/fixtures/vault`.
