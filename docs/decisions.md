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
