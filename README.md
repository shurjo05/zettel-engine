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
| M7 | Import the legacy notes: convert, dedup, repair links | done |
| M8 | Held duplicates judged by a model, then merged or released | done |
| M9 | Maintenance reports: link suggestions, orphans, map suggestions | done |
| M10 | Hooks that run on their own: install the session hook, catch up on sessions it missed, schedule ingest and the duplicate resolver, tested on Windows (Claude Code only for now) | done |
| M11 | Context efficiency, answer quality with citations (`zk ask`), threshold re-checks (`zk calibrate`) | done |
| M12 | Notes reach Claude without being asked: map notes, a session brief, a search nudge, notes for failed commands and matching prompts, reading past projects (`zk learn`), a weekly review | done |
| M13 | Vector search inside SQLite with sqlite-vec, NumPy as the fallback | in progress |

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
uv run zk suggest path/to/vault        # links to consider, orphans' likely parents, map ideas
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
uv run zk hooks status path/to/vault                           # what runs on its own
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

Every note's vector is scored against the query (exact search, no approximation). The scoring runs
inside SQLite through [sqlite-vec](https://github.com/asg017/sqlite-vec), a small extension
installed with zk, so vectors never leave the database. If this Python's `sqlite3` can't load
extensions, zk falls back to NumPy with the same ranking; `ZK_VECTORS=numpy` forces that.
`uv run python scripts/bench_vectors.py` times both at 100 to 100,000 notes.

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

To capture sessions without being asked, install the session-end hook (see
[Running on its own](#running-on-its-own)).

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
  `apply_held` releases a held note, `merge_held` merges it into the note it duplicates (see
  below), and `discard` sets it aside in `inbox/rejected/`.
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
  current conversation. Use it where no session-end hook runs, such as Codex.
- [docs/commands/zk-inbox.md](docs/commands/zk-inbox.md): turns the captures waiting in the inbox
  into notes.

### Resolving held duplicates

A held note is either the same idea as an existing note or only related to it, and similarity
alone can't tell which: on the legacy notes, 2 of 5 holds were only related. A model reads each
pair and decides, and zk does the writing:

- `get_held_pair(name)` returns the held note and its closest existing note in full, with the
  next closest notes.
- **Related**: `apply_held(name, reason)` moves it into `notes/`. Links to it start resolving.
- **Same idea**: `merge_held(name, into, note, reason)` rewrites the existing note from the merged
  draft the model writes. The note keeps its id, file name and created date, and gains the held
  note's sources and title (as an alias). Links and test questions that named the held note are
  pointed at it, and the held file moves to `inbox/merged/`. It is all one git commit, with the
  reason in the message, so `git revert` undoes a bad merge. If any file changed while merging,
  nothing is written.
- A merged note should stay under about 400 words, with the held note's point in its summary,
  because search by meaning reads only a note's first 512 tokens. A longer merge still goes
  through, with a warning.

[docs/commands/zk-resolve.md](docs/commands/zk-resolve.md) is the `/zk-resolve` command. Run it
with your strongest model by hand, or let `zk process` run it after every session (below). Each
run sends the held notes and their closest notes to the model's provider (see
[docs/data-flow.md](docs/data-flow.md)).

### Running on its own

These commands make the whole loop run without you: capture each session when it ends, turn
captures into notes, and resolve any notes held as duplicates. Run `zk hooks install` with `--dry-run` first to
see the change before making it.

```bat
cd C:\path\to\zettel-engine
claude mcp add zk --scope user -- uv --directory C:\path\to\zettel-engine run zk serve C:\path\to\vault
uv run zk hooks install C:\path\to\vault --dry-run
uv run zk hooks install C:\path\to\vault
uv run zk schedule install C:\path\to\vault --at 21:00
uv run zk hooks status C:\path\to\vault
```

- **`zk hooks install`** adds zk's hooks to Claude Code's `%USERPROFILE%\.claude\settings.json`
  (used by the CLI, the IDE extensions and the desktop app): SessionEnd captures the session, and
  three more bring notes in (see [Notes without asking](#notes-without-asking)). It only adds
  entries, one for each event that has no zk hook yet, never changes one, and backs the file up
  first. `--no-recall` installs only the capture. Every hook runs `python -m zk.hook`, which
  imports only what its event needs (never numpy), because Claude Code gives session-end hooks
  about a second and a half and the others run often.
- **After each captured session** the hook starts `zk process` in the background. It runs a
  headless `claude -p` with only the zk tools to ingest the inbox, then, if any note was held, a
  second one (with `--resolve-model`, default opus) to resolve the holds. A step with no work makes
  no model call. One run happens at a time; captures that arrive meanwhile get another round.
  Their prompts start with `[zk]` and run with `ZK_INGESTING=1`, so zk's own sessions are never
  captured. `--no-process` installs a hook that only captures.
- **`zk capture --catch-up`** captures recent sessions no hook did: an app where hooks don't run,
  a crash, a session that ended while zk wasn't installed. It reads Claude Code transcripts in
  `~/.claude/projects`, skips sessions used in the last 30 minutes, and keeps a record in the zk data folder (`sessions.json`) so nothing is captured
  twice. A resumed session yields a capture of its new turns only. `--dry-run` lists what it would
  capture.
- **Captures are ingested a few at a time** (three, or about 120,000 characters), oldest first,
  so a backlog never fills one agent's context. A run that changed the vault ends by bringing the
  map notes up to date.
- **`zk schedule install`** adds a daily Windows task, `zk catch-up`, that runs
  `zk process --catch-up`, and a weekly one, `zk review` (Friday 12:00 by default, when the PC
  is usually on; `--review-day`, `--review-at`, `--review-day never`). If the PC was off at that time, a task runs at the next
  chance. `zk schedule status` and `zk schedule remove` go with them. Elsewhere it prints cron lines.

Logs go to the zk data folder's `logs/`. A run whose agent processes none of its captures (for
example because its zk server didn't start) stops, exits with code 1, and is listed in the weekly
review; `process-status.json` beside the logs records the last run. The headless agents bring their own write-enabled zk
server (`--mcp-config`, and only that server), so the one registered for everyday sessions on the
first line stays read-only: sessions don't load the write tools or their instructions, and can't
write notes. Both installs warn when it's missing or registered with `--allow-writes`, and
`zk hooks status` says which. Sessions in the cloud (claude.ai/code) can't be
captured, since their transcripts aren't on your machine. Codex sessions aren't captured
automatically yet; `/zk-ingest` works there.

The older `zk capture --hook [--then COMMAND]` still works.

### Notes without asking

Until M12, notes reached Claude only when it chose to search. Now the vault shows up on its own,
cheaply, at a few points, and Claude opens more only when it needs it:

1. **Map notes**, `zk maps VAULT`. A map note lists the notes on one topic or one project, a line
   each. Topic maps come from the clusters `zk suggest` finds; project maps from notes'
   `project:`, `repo:` and `local-repo:` sources (spellings of one project share a map). They
   live in `maps/`, are rewritten after each `zk process` run that changed the vault, and keep their name while a topic keeps half its notes. A map edited by hand is left
   alone from then on, and a map whose topic is gone moves to `maps/retired/` (never deleted).
   Search leaves maps out unless asked (`type: map`). Patterns and decisions from one project stay
   in that project's map, and join a topic map once they hold for two projects.
2. **The session brief** (SessionStart). Each session is told how many notes there are, the map
   names (capped, so it stays small as the vault grows) and, in a project's repository, that
   project's notes. `zk brief VAULT --cwd FOLDER` prints it and its size. Target: under 1,000
   tokens.
3. **A nudge** (PreToolUse on Grep and Glob): one line, the first time a session searches files,
   saying the vault may already cover it.
4. **Notes for a failed command** (PostToolUseFailure on Bash or PowerShell): keyword search on
   the error lines, and up to two notes only when most of their words match. Each note is shown
   once per session; nothing is added otherwise.
5. **Pointers on each prompt** (UserPromptSubmit): keyword search (BM25) on the prompt's
   uncommon words, and up to three notes only when a note shares at least three of them. Slash
   commands get nothing, and a note already pointed to this session isn't repeated. It's silent
   for most prompts; when it fires it adds about 60 tokens (at most about 300).
6. **When to search**: `zk hooks install` adds `~/.claude/rules/zk.md`, which Claude Code loads in
   every project. Nothing is written into your repositories. It's a new file, never overwritten;
   edit it as you like.
7. **Past projects**, `zk learn VAULT FOLDER...`. A free first pass, with no model, reads the
   folder tree, README, manifests, recent commit subjects and the class and function names in each
   source file, skipping dependencies, build output and likely secret files, and saves it as an
   inbox capture. The next ingest turns it into notes on how the project is built, so reading a
   project costs no extra model call. File hashes are remembered, so a later run reads only what
   changed. A session that starts in a repository zk hasn't read starts this in the background.
   `--dry-run` prints the capture; `--skip` keeps a project out for good.
8. **The weekly review**, `zk review VAULT`: health, inbox, retrieval on the test questions, map
   changes pending, hold drift, the brief's size, and the projects read, as one page in
   `inbox/reports/`. No model.

Notes recorded from a project's sessions or `zk learn` get a `project:<name>` source, and
`search_notes` takes `project` to keep to one.

### Importing a legacy vault

`zk import` brings an old vault's notes into this one. It only reads the old folder.

```bash
uv run zk import "C:\Obsidian\AI Second Brain\AI Brain" C:\path\to\vault --queries old-queries.yaml --dry-run
uv run zk import "C:\Obsidian\AI Second Brain\AI Brain" C:\path\to\vault --queries old-queries.yaml
```

- **Every file is accounted for.** A file with a note `type` is imported. Everything else (system
  docs, maps, templates, code, empty files) is skipped, and the report says why.
- **Notes keep their history.** The old id (its timestamp part), the created and updated dates,
  tags, aliases and sources are kept, and a `legacy:<old path>` source is added. Summary, Core Idea,
  Example and the Links list become the new layout. `domain`, `confidence`, `status` and the empty
  `related` field are dropped; the old vault still has them.
- **Duplicates are held**, exactly as in ingest. Accepted notes are imported before inbox ones, so
  when two match it's the inbox copy that goes to `inbox/held/`. Links to a held note keep its
  name and are listed as waiting on it. They resolve again if `apply_held` releases it.
- **Links are repaired.** A link is rewritten to its note's new name, found by file name, id, title
  or alias. A drifted link (a shorter slug, an old id prefix, a missing "by" or "for") goes to the
  one note whose name holds at least 75% of its words, and only if no other note fits as well.
  Links that still don't resolve are listed.
- `--queries` carries a legacy test set over to `eval/legacy-queries.yaml`, with its answers renamed.
- `--dry-run` runs the same import into a throwaway copy and prints the report. A real import
  writes the report to `inbox/reports/` and is one git commit.

### Maintenance suggestions

`zk suggest VAULT` reports what would make the vault better connected. It changes nothing:

- **Links to consider**: pairs of notes at cosine similarity 0.78 or more that aren't linked
  either way. Pairs at 0.85 or more are left to the duplicate check, and no note appears in more
  than three suggestions.
- **Orphans**: notes nothing links to, each with up to three unlinked notes at 0.70 or more,
  the likeliest places to link from, or "no likely parent".
- **Clusters with no map note**: groups of four or more notes that are each among the others'
  five nearest neighbours (and at least 0.75 similar), where no member already links to three
  quarters of the rest. A cluster of more than 20 notes is split again at a stricter threshold,
  because big ones mixed topics. Each lists its shared tags as a hint for a title, and the member
  closest to being its map.

The MCP tool `suggest_links(note)` gives the same suggestions for one note, and ingest now adds
them to its "linked to no other note" warning.

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

Three more measurements answer "is this worth it, and is it still tuned right?":

```bash
uv run zk eval path/to/vault --context            # tokens per question: zk tools vs naive RAG vs whole vault
uv run zk ask path/to/vault "why fuse by rank?"   # an answer from the notes, citing them as [[name]]
uv run zk eval path/to/vault --answers --limit 20 # citation accuracy, grounding, abstention
uv run zk calibrate path/to/vault                 # re-check the similarity thresholds
```

- **`--context`** replays what an agent does through the MCP tools for each question
  (`search_notes`, then `get_note` on the first relevant hit) and counts the tokens of the JSON
  they return, next to two baselines: opening the top 5 notes in full, and the whole vault. Local,
  no model.
- **`zk ask`** gives the top 5 notes to a headless `claude -p` (your own login, no tools) and asks
  for an answer that cites each note it uses, or "NOT IN NOTES". **`--answers`** runs it over the
  labelled questions and scores the answers by their citations: does it cite a relevant note, does
  it cite only notes it was given, and does it abstain when retrieval missed.
- **`zk calibrate`** samples note pairs in a 0.03 band just below and just above each cut
  (duplicate 0.85, link 0.78, orphan parent 0.70), has Claude label each pair same, related or
  unrelated, and reports precision per band against the precision measured by hand. It warns when
  a cut has drifted and never changes one. Labels are saved in the vault's `eval/calibration/` with
  both notes' content hashes, so the next run only judges new pairs. `--no-judge` reports from
  saved labels only. It also reads the vault's git history for the hold rate and the share of
  holds the resolver released as "only related", which counts false holds directly.

`zk ask`, `--answers` and `zk calibrate` send the question and the notes they use to Anthropic (see
[docs/data-flow.md](docs/data-flow.md)).

Apart from `capture`, `calibrate` (its label files) and the ingest tools, `zk` only reads the vault. It skips hidden folders (`.obsidian`, `.git`) plus `inbox/**` and
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
src/zk/search.py  keyword (BM25), semantic (cosine, in SQLite via sqlite-vec) and hybrid (RRF) search
src/zk/embed.py   local embedding model and the vector cache in the index
src/zk/lookup.py  one note, or its neighbours, by id, file name, path or title
src/zk/evaluate.py  retrieval metrics over a labelled query set
src/zk/context.py   context tokens per question: zk tools vs naive RAG vs whole vault
src/zk/answer.py    answer from the notes with citations, and score the citations
src/zk/calibrate.py  re-check similarity thresholds on model-judged pairs
src/zk/agent.py     one-shot headless Claude, for answering and judging
src/zk/server.py  MCP server: the same functions as read-only tools
src/zk/capture.py  session transcripts and text into inbox captures (pure)
src/zk/vault.py   the only writer of vault files: create or move, never overwrite, atomic
src/zk/proposals.py  render a note draft in the schema and validate it (pure)
src/zk/ingest.py  one ingest run: duplicate check, write, questions, commit, writing through vault.py
src/zk/checks.py  warnings after each ingest: score drops beyond noise, new-note health
src/zk/migrate.py  import a legacy vault: convert, hold duplicates, repair links, report
src/zk/resolve.py  judge and merge held duplicates
src/zk/maintain.py  link, orphan and map suggestions
src/zk/maps.py    map notes for topics and projects, kept up to date
src/zk/recall.py  the session brief, the search nudge, notes for a failed command or a prompt (light)
src/zk/learn.py   read a project's structure into a capture, with no model
src/zk/review.py  the weekly vault health review
src/zk/sessions.py, hook.py, process.py, hooks.py  capture sessions, process the inbox, install hooks
src/zk/cli.py     the `zk` command
eval/             query sets for the fixture vault
tests/            pytest suite and a small synthetic fixture vault
docs/             decisions, data flow and the /zk-ingest, /zk-inbox and /zk-resolve commands
```

See [docs/decisions.md](docs/decisions.md) for why things are built this way and
[docs/data-flow.md](docs/data-flow.md) for where vault content goes.
