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
| M2 | Keyword search (SQLite FTS5, BM25) and evaluation harness | done |
| M3 | Read-only MCP server (`zk serve`) | done |
| M4 | Semantic search with local embeddings | done |
| M5 | Hybrid search (reciprocal rank fusion), the default mode | done |
| M6 | Ingest: session capture, automatic notes with duplicate check, checks that warn | done |
| M7-M9 | Import, maintenance, evaluation | planned |

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
uv run zk check path/to/vault --duplicates   # also list near-duplicate notes (embeds them)
uv run zk index path/to/vault          # just build or refresh the index
uv run zk search path/to/vault "how do I merge ranked lists"   # hybrid search (the default)
uv run zk search path/to/vault "hooks" --type pattern --tag claude-code -k 10
uv run zk search path/to/vault "older knowledge should count less" --mode semantic
uv run zk search path/to/vault "mergeable priority queue" --mode hybrid
uv run zk show path/to/vault reciprocal-rank-fusion             # one note with its links
uv run zk eval path/to/vault eval/fixture-queries.yaml         # Recall@k, MRR, latency
uv run zk eval path/to/vault                                   # the questions saved by ingest
uv run zk eval path/to/vault queries.yaml --mode keyword --mode semantic   # compare modes
uv run zk index path/to/vault --embed                          # precompute embeddings
uv run zk serve path/to/vault                                  # MCP server over stdio
```

Every command refreshes the index first (only changed files are re-parsed), so results are never
stale.

### Search

Words in the query are matched with Porter stemming ("merging" finds "merges") and combined with
OR, then ranked by BM25. Matches in the title count most, then aliases, tags, summary and body.

### Semantic search

`--mode semantic` ranks notes by meaning rather than shared words, using embeddings from a small
model that runs on your CPU (`BAAI/bge-small-en-v1.5` via fastembed; choose another with `--model`).
The model, about 70 MB, is downloaded from Hugging Face the first time it is used and kept in the zk
data folder. Note text is never sent anywhere. Vectors are stored in the index, keyed by each note's
content hash, so only new or edited notes are embedded and renamed notes keep their vectors.

### Hybrid search

Hybrid is the default mode. It runs both searches and fuses their rankings with reciprocal rank
fusion: each note scores the sum of 1 / (10 + rank) over the lists it appears in. Use `--mode keyword`
for a fast search that loads no model. Ranks are combined rather than
scores because BM25 scores and cosine similarities are on different scales.

### MCP server

`zk serve` lets Claude Code and Codex search the vault while you work. It runs over stdio (the client
starts it; no network port) and offers three read-only tools:

| Tool | Returns |
| --- | --- |
| `search_notes(query, k, type, tag, mode)` | ranked hits with summary (and a snippet when keyword search matched), no bodies |
| `get_note(note)` | one note in full, with the notes it links to and from |
| `get_related(note, k)` | linked, backlinking and same-tag notes, with summaries |

A note can be named by id, file name, vault path or title. Each call refreshes the index first, so
edits made in Obsidian are visible on the next call. `zk show` prints what `get_note` returns.

Claude Code (Windows paths shown; `--scope user` makes it available in every project):

```bash
claude mcp add zk --scope user -- uv --directory C:\path\to\zettel-engine run zk serve C:\path\to\vault
```

Codex, in `~/.codex/config.toml`:

```toml
[mcp_servers.zk]
command = "uv"
args = ["--directory", "C:\\path\\to\\zettel-engine", "run", "zk", "serve", "C:\\path\\to\\vault"]
```

Whatever a tool returns becomes part of the conversation and is sent to the client's model
provider; see [docs/data-flow.md](docs/data-flow.md).

### Capturing sessions

`zk capture` saves a conversation into the vault's `inbox/` folder, where later steps turn it into
notes. It only ever creates new files; it never changes or deletes one.

```bash
uv run zk capture path/to/vault --transcript session.jsonl   # a Claude Code transcript
uv run zk capture path/to/vault --text "an idea" --source codex
```

From a transcript it keeps your messages and the assistant's replies and drops tool calls, tool
output, thinking and subagents. Sessions with fewer than two of your messages are skipped, long ones
keep their first and last 20,000 characters, and common secret formats (API keys, tokens, private
keys, `password=...`) are replaced with `[REDACTED]`.

To capture every Claude Code session automatically, add a SessionEnd hook to
`%USERPROFILE%\.claude\settings.json` (`~/.claude/settings.json` elsewhere):

```json
{
  "hooks": {
    "SessionEnd": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "uv --directory C:\\path\\to\\zettel-engine run zk capture C:\\path\\to\\vault --hook"
          }
        ]
      }
    ]
  }
}
```

With `--hook`, `zk capture` reads the hook's event from stdin, and if anything goes wrong it reports
on stderr and exits 0, so a capture problem never interrupts Claude Code.

### Turning conversations into notes

Start the MCP server with `--allow-writes` to add the ingest tools:

```bash
claude mcp add zk --scope user -- uv --directory C:\path\to\zettel-engine run zk serve C:\path\to\vault --allow-writes
```

The agent decides which ideas are worth a note. zk does the rest, with no review step:

- `ingest_notes(notes, capture)` takes structured fields for each note (title, type, summary,
  details, tags, sources, links, and one to three test questions). It renders each note in the
  note format and refuses the whole call if any note would fail `zk check`.
- A note at cosine similarity 0.85 or more to an existing note, or to another note in the same call,
  is **held** in `inbox/held/` instead of written. You get a warning, and nothing is lost.
  `apply_held` releases a held note, and `discard` sets it aside in `inbox/rejected/`.
- Every other note goes straight into `notes/`. Its questions are saved in `eval/questions/`, and
  the capture moves to `inbox/processed/`. The whole run is **one git commit**, so `git revert`
  undoes it. Your own uncommitted changes are left alone.
- Then the checks run. They **only warn**, in the tool result and in `inbox/warnings/<run>.md`.
  Each saved test question is re-scored and compared with its score when it was added, and you're
  warned when the drop is beyond noise. About one question in five is a held-out sample, scored
  separately. The new notes are also checked for being too long to embed, linking to nothing,
  or having broken links.

Two ready-made commands go in `~/.claude/commands/` for Claude Code, or `~/.codex/prompts/` for
Codex, so that `/zk-ingest` and `/zk-inbox` work in the apps too:

- [docs/commands/zk-ingest.md](docs/commands/zk-ingest.md): saves what's worth keeping from the
  current conversation. Use it where no session-end hook runs, such as Codex or the desktop apps.
- [docs/commands/zk-inbox.md](docs/commands/zk-inbox.md): turns the captures waiting in the inbox
  into notes.

To process each captured session without being asked, let the hook start a headless agent after
the capture is saved. This is opt-in, because every captured session then costs a short model
run:

```json
"command": "uv --directory C:\\path\\to\\zettel-engine run zk capture C:\\path\\to\\vault --hook --then \"claude -p \\\"Process the zk inbox with the zk tools, following the zk server's instructions.\\\" --allowedTools mcp__zk\""
```

`--then` runs its command detached, logging to the zk data folder's `logs/`. It sets
`ZK_INGESTING=1`, so the headless session's own end isn't captured again. `--allowedTools
mcp__zk` limits that agent to the zk tools, which can only create and move files.

### Evaluation

A query set is a YAML file of queries with the notes that should be found for each:

```yaml
queries:
  - query: how do I merge two ranked lists
    relevant: [reciprocal-rank-fusion]   # note file names or frontmatter ids
    kind: paraphrase                      # optional; results are also broken down by kind
```

`zk eval` reports Recall@1/3/5/10, MRR and latency overall and per kind, and lists the queries whose
relevant note was not in the top 10. Keep query sets about private vaults out of this repository.
Without a query file it scores the test questions ingest saved in the vault's `eval/` folder, by
kind (`generated`, and the `holdout` sample).

Apart from `capture` and the ingest tools, `zk` only reads the vault. It skips hidden folders (`.obsidian`, `.git`) plus `inbox/**` and
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
src/zk/index.py   SQLite catalog: notes, links, issues, full-text index; incremental builds
src/zk/search.py  keyword (BM25), semantic (cosine) and hybrid (RRF) search with filters
src/zk/embed.py   local embedding model and the vector cache in the index
src/zk/lookup.py  one note, or its neighbours, by id, file name, path or title
src/zk/evaluate.py  retrieval metrics over a labelled query set
src/zk/server.py  MCP server: the same functions as read-only tools
src/zk/capture.py  session transcripts and text into inbox captures (pure)
src/zk/vault.py   the only writer of vault files: create or move, never overwrite, atomic
src/zk/proposals.py  render a note draft in the schema and validate it (pure)
src/zk/ingest.py  one ingest run: duplicate check, write, questions, commit, writing through vault.py
src/zk/checks.py  warnings after each ingest: score drops beyond noise, new-note health
src/zk/cli.py     the `zk` command
eval/             query sets for the fixture vault
tests/            pytest suite and a small synthetic fixture vault
docs/             decisions, data flow and the /zk-ingest and /zk-inbox commands
```

See [docs/decisions.md](docs/decisions.md) for why things are built this way and
[docs/data-flow.md](docs/data-flow.md) for where vault content goes.
