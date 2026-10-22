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

The agent never writes a note directly. `propose_note` takes structured fields and zk renders the
Markdown, so every note has the same frontmatter, H1, summary and sections regardless of which model
wrote it, and anything `zk check` would flag is refused before it is saved. Proposals live in
`inbox/proposals/`, which the index skips, so drafts never show up in search or mix with notes.
`apply_proposal` exists only behind `--allow-writes`, and its description tells the agent to call
it only after the person approves; the person can also read and edit proposals in Obsidian first.

Applying commits exactly the files it touched (`git commit --only`), so a note can be reverted on its
own and changes the person has not committed are never swept in. Editing existing notes is left out
for now: every tool creates or moves files, which keeps the worst case of a confused agent to an
unwanted new file. The duplicate verdict thresholds (cosine 0.90 likely duplicate, 0.75 related)
are starting values to be calibrated on real notes.
