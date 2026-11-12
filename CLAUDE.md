# zettel-engine: notes for AI agents

Read README.md first. Then:

- Commands: `uv sync`, `uv run pytest -q`, `uv run ruff check .`, `uv run ruff format .`
- `notes.py` stays pure (no file access). `index.py` may only read the vault and write the
  SQLite index. `search.py`, `lookup.py` and `evaluate.py` only read the index; `cli.py` and
  `server.py` may refresh it through `index.build_index`, and `embed.py` writes only the
  `embeddings` table. MCP tools stay read-only unless
  `--allow-writes` is given. Only `vault.py` may write vault
  files. It creates or moves files and never deletes; the one write that changes a file,
  `vault.replace`, needs the content hash the caller read. `ingest.py`, `migrate.py` and
  `resolve.py` write only through it, and only `resolve.merge_held`, `maps.update` and `docs.import_docs` (on files zk generated and nobody has
  edited) use `replace`. `maps.py` writes only `maps/` and never deletes: it retires a map by moving it.
  `docs.py` reads a project's doc files and writes only `refs/` through `vault.py`, using
  `replace` only on sections zk generated and nobody edited, and retires a section by moving it.
  `maintain.py` only reads the index. `sessions.py` reads agent transcripts and writes only
  captures (through `vault.py`) and its record in the zk data folder. `hooks.py` writes the
  person's agent settings only by adding an entry after a backup, never changing or removing
  one. `process.py` runs headless agents limited to the zk tools. `hook.py` must stay light
  (no numpy): session-end hooks get one to three seconds. `recall.py` and `learn.py`, which hooks import, stay light too; `recall.py` only reads
  the index (opened read-only) and writes only per-session records in the zk data folder. `learn.py`
  reads the person's project folders and writes only inbox captures (through `vault.py`) and its
  record in the zk data folder. `review.py` reads the index, vault and git log and writes only its
  report in `inbox/reports/`, through `vault.py`. `hooks.py` may also add `~/.claude/rules/zk.md`,
  never overwriting it. `context.py` and `answer.py` only read
  the index. `calibrate.py` reads the index and the vault's git log and writes only its label
  files in `eval/calibration/`, through `vault.py`. `agent.py` runs a one-shot headless Claude
  with the `[zk]` marker; anything sent through it must have a row in `docs/data-flow.md`. `checks.py` reads the vault and the index and writes only the score
  history file next to the index. `api.py` is the public API (docs/api.md): it may refresh the
  index like `cli.py`, and its only vault write is a new inbox capture through `vault.py`; keep
  its names stable and pinned by `tests/test_api.py`.
- Retrieval changes must be judged with `zk eval`; report before and after numbers in the PR.
- Tests must use copies of `tests/fixtures/vault` (the `vault` fixture), never a real vault.
- Fixture notes are synthetic; never commit personal vault content.
- Record significant design choices in `docs/decisions.md` and any new place vault content goes
  in `docs/data-flow.md`.
- Keep dependencies minimal; justify any new one in `docs/decisions.md`.
