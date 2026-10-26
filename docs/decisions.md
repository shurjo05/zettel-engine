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

## 11. Local embeddings, exact search in NumPy, cached by content hash (2026-10-22)

Semantic search uses `BAAI/bge-small-en-v1.5` (384 dimensions) through fastembed, running on the
CPU. A local model keeps note text on this machine and needs no key or per-call cost; an API model
would only ever be opt-in. fastembed downloads its quantized ONNX build of the model (about 65 MB, from
the `Qdrant/bge-small-en-v1.5-onnx-Q` repository on Hugging Face) once into the zk data folder.

Each note is one vector (title plus body), with no chunking. The model reads at most 512 tokens and
fastembed silently drops the rest. On the legacy test copy (96 notes) the median note is 481 tokens and
33 notes run over, the longest at 887, so their tails are not embedded. The title and summary come
first and are always inside the window, and semantic search still reached an MRR of 0.92 on the eval
set, so chunking is deferred until the eval shows a miss caused by truncation.

Vectors live in the existing SQLite index (`embeddings`, keyed by content hash and model) and are
scored by brute force: one matrix-vector product over all notes. At a few thousand notes that is
exact and takes well under a millisecond, so a vector database (Chroma, Qdrant) or an approximate
index (HNSW) would add a second store to keep in sync for no measurable gain. If the vault ever
reaches roughly 100k notes, sqlite-vec is the next step and keeps everything in one file.

The content-hash key means a renamed note reuses its vector and an edited note is re-embedded;
the model key means switching models never mixes incompatible vectors. Keyword search stays the
default until the eval and M5 (hybrid) decide otherwise.

## 12. fastembed as a dependency (2026-10-22)

fastembed runs ONNX models with onnxruntime and brings tokenizers, huggingface-hub and NumPy.
The alternative, sentence-transformers, needs PyTorch, roughly ten times the install, for the same
models. fastembed is imported only when something is embedded, so commands that don't embed start
as fast as before.

## 13. Hybrid search by reciprocal rank fusion (2026-10-22)

On the legacy eval set semantic search beat keyword search overall (MRR 0.92 vs 0.82) but lost
exact-term queries that keyword search ranked first ("mergeable priority queue" fell out of the top
10). `--mode hybrid` reads the top 50 of each list and scores each note by the sum of
1 / (60 + rank) over the lists it appears in.

Measured on the legacy set (keyword / semantic / hybrid MRR 0.82 / 0.92 / 0.90), hybrid search is
the only mode with every query's note in the top 5 (R@5 1.00 vs 0.97 for semantic), at the cost of
some first-place hits: semantic's rank-1 paraphrase wins drop to ranks 1 to 3. Agents read the top
five summaries, so recall in the top 5 matters more than rank 1, and hybrid is the default for
`search_notes` and `zk search`.

The constant is 10, not the usual 60. For a paraphrased question keyword search still returns 40 to
50 weak OR-matches, and with k = 60 those flat ranks let mediocre-in-both notes outscore semantic's
#1; k = 10 trusts each list's top ranks more (MRR 0.90 vs 0.87; depth 50 beats 20 at either k).
This was chosen on the same 30 queries it is judged on, so it may be overfit: re-check it when the
eval set grows.

RRF uses ranks, not scores, because BM25 scores are unbounded and depend on the query's words
while cosine similarities sit in [-1, 1]; normalising and weighting them would add a tuning knob
per scale. RRF has one constant and needs no training data, which matters with a
30-query eval set that would overfit any learned weights.

## 14. Capture sessions into the inbox with a hook; no model in the capture path (2026-10-22)

A Claude Code SessionEnd hook runs `zk capture --hook`, which writes the conversation (your messages
and the assistant's text, without tool traffic) to `inbox/`. Turning captures into notes is done
later by the agent you are already talking to, through MCP tools with a review step (M6b), rather
than by an LLM call from the hook: that would need an API key in the core, spend tokens on every
session including trivial ones, and write notes nobody reviewed. A cheap local filter skips sessions
with fewer than two user messages.

`zk/vault.py` is the only code that writes to the vault, and it only creates files: the content is
written to a temporary file and hard-linked to the new name, which fails if the name exists, so a
crash never leaves a partial file and an existing note can never be replaced. The hook always exits
0 so a capture failure cannot disturb a Claude Code session; it reports on stderr instead.

## 15. Proposals wait in the inbox; apply is a scoped git commit (2026-10-22)

*Superseded in part by decision 16: notes are no longer reviewed one by one, and proposals became held duplicates.*

The agent never writes a note directly. `propose_note` takes structured fields and zk renders the
Markdown, so every note has the same frontmatter, H1, summary and sections regardless of which model
wrote it, and anything `zk check` would flag is refused before it is saved. Proposals live in
`inbox/proposals/`, which the index skips, so drafts never show up in search or mix with notes.
`apply_proposal` exists only behind `--allow-writes`, and its description tells the agent to call
it only after the person approves; the person can also read and edit proposals in Obsidian first.

Applying commits exactly the files it touched (`git commit --only`), so a note can be reverted on its
own and changes the person has not committed are never swept in. Editing existing notes is left out
for now: every tool creates or moves files, which keeps the worst case of a confused agent to an
unwanted new file.

The duplicate verdict thresholds are cosine 0.85 for likely duplicate and 0.80 for related,
calibrated on the legacy notes. bge-small similarities run high: the median note's nearest
neighbour is 0.80, and 143 pairs score at least 0.75, so the starting values (0.90 and 0.75) would
call almost every note related while missing most duplicates. Ten hand-reworded copies of real
notes scored 0.78 to 0.94 (median 0.86) and ranked their original first every time; ten pairs that
only share a topic scored 0.62 to 0.81. The ranges overlap, so no cut is clean. At 0.85 / 0.80, 7 of
the 10 copies are flagged and the other 3 show as related at rank 1, while no topic pair is flagged.
Recall wins because a person reviews every proposal and a wrong flag costs one look. The gap between
the top hit and the second looked like a useful extra signal (at least 0.08 for 9 of 10 copies)
but is untested on false positives, so it is left out.

## 16. Ingest is automatic; a likely duplicate is held, and a run is one commit (2026-10-26)

This replaces the review step in decision 15. Reviewing every note defeats the point of automatic
capture, so `ingest_notes` writes every note that passes the duplicate check straight into
`notes/`. Recall still beats precision at 0.85, because a flagged note is **held** in `inbox/held/`
with a warning instead of being thrown away: a false alarm costs one look, while a missed duplicate
is a note written twice, which wastes tokens every time both come back in search. Each draft is
also compared with the other drafts in the same call, since one session often states an idea
twice. The call is validated as a whole: if any note would fail `zk check`, nothing is written.

A run is one git commit (its notes, questions, warnings and the retired capture), not one per note.
The history reads as one entry per session, and `git revert` still undoes a bad run. Undoing a
single note from a run means deleting that file by hand, which is acceptable at this size.

Processing captures without being asked is opt-in through `zk capture --hook --then COMMAND`. zk
knows nothing about which agent runs. The command is the user's own (for example a headless
`claude -p` limited to the zk tools), which keeps the core provider-independent. `ZK_INGESTING=1`
stops the headless session's own end from being captured again.

## 17. Checks on every ingest warn and never fix (2026-10-26)

Every note brings one to three test questions, so the evaluation set grows with the vault instead
of going stale. The weakness is that the same model writes both the note and its question, so
questions can be easier than real ones. Two things limit that. A fixed one-in-five sample, chosen
by a hash of the question, is marked `holdout`, scored separately and never used for tuning. And
the hand-written question sets stay the reference for any retrieval change (`zk eval` with a file).

Each question is compared with its own score from when it was first saved, not with the previous
run, so slow drift shows up too. A single question moves MRR by about 0.03 at today's size, so a
plain threshold would cry wolf. Instead a paired bootstrap resamples the questions, and zk warns
only when the 5th percentile of the mean drop is above zero, with at least 5 questions and a drop
of at least 0.02. The first scores live in a JSON file next to the index. Losing it only resets
the baselines.

The health warnings cover only the notes added in that run (too long to embed, linked to nothing,
broken links, held as duplicates). Re-reporting the 33 long legacy notes on every run would bury
anything new; `zk check` (with `--duplicates`) shows the whole vault on demand. Note length is
estimated at 3.7 characters per token, the average for bge-small on the legacy notes, because the
tokenizer is only available once the model is downloaded. There's no scheduled report, and nothing
is re-tuned automatically. Both were options Shurjo chose against.

## 18. Legacy notes are imported by code, not by an agent (2026-10-26)

The plan said the old notes would go "through the M6 pipeline". The deterministic half of it is
used here: the schema renderer, the duplicate check at 0.85, held notes and one commit. The agent
half isn't. The legacy notes are already atomic and well structured, so a model rewriting them would
cost tokens, could change their wording, and couldn't be repeated exactly. `zk import` converts
each note's fields directly, keeps its old id, dates and sources, and adds `legacy:<path>` as
provenance. Sending the notes to a model provider is avoided entirely.

`domain`, `confidence`, `status` and `related` are dropped. The new schema doesn't have them,
every note had `status: seed` and an empty `related`, and the legacy vault is never changed, so
they can still be looked up there.

Most broken legacy links were slug drift, so link repair matters more than anything else here.
Exact matches come first: file name, the name without its id prefix, id, title, alias. Then comes
a word-overlap match: at least 75% of the link's words, ignoring "by", "for", "the" and the like,
must appear in exactly one note's name, and a tie leaves the link broken. On the legacy copy that
repaired 47 of 50 broken links, and the 4 still broken after import (one link appears twice) point
at notes that never existed. Every repair was checked by hand and none is wrong. A drifted link
that keeps a note's id prefix goes to that note when no other legacy note shares the id.

Links to a held note keep the held note's name. Pointing them at its twin was tried first, but
with the real model 2 of the 5 holds were notes that were only related (semantic edges against
Leiden clustering, skill files against agent files), and the redirect moved those links' meaning to
the wrong note. Kept as they are, the links are reported as waiting on a held note and resolve
again if `apply_held` releases it. A carried-over test question whose answer is held lists the
twin as well, so it still has an answer in the index. `zk check` lists these links apart from broken
ones.

Measured with bge-small on a copy of the legacy vault: 91 imported, 5 held, 34 skipped; 47 of the
50 broken links repaired, 4 still broken and 15 waiting on a held note. The legacy test set's MRR,
before (119 legacy notes) and after (91 imported), was 0.81 to 0.85 for keyword search, 0.88 to 0.93
for semantic search and 0.88 to 0.93 for hybrid search. On the paraphrase questions only it was
0.68 to 0.76, 0.83 to 0.88 and 0.81 to 0.89. No question got worse in any mode. Most of the gain
comes from the cleaner corpus: skipped maps and system docs no longer compete with notes.
