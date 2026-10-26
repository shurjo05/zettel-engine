# zettel-engine: notes for AI agents

Read README.md first. Then:

- Commands: `uv sync`, `uv run pytest -q`, `uv run ruff check .`, `uv run ruff format .`
- `notes.py` stays pure (no file access). `index.py` may only read the vault and write the
  SQLite index. `search.py`, `lookup.py` and `evaluate.py` only read the index; `cli.py` and
  `server.py` may refresh it through `index.build_index`, and `embed.py` writes only the
  `embeddings` table. MCP tools stay read-only unless
  `--allow-writes` is given. Only `vault.py` may write vault
  files, and it only creates new ones (never overwrites or deletes).
- Retrieval changes must be judged with `zk eval`; report before and after numbers in the PR.
- Tests must use copies of `tests/fixtures/vault` (the `vault` fixture), never a real vault.
- Fixture notes are synthetic; never commit personal vault content.
- Record significant design choices in `docs/decisions.md` and any new place vault content goes
  in `docs/data-flow.md`.
- Keep dependencies minimal; justify any new one in `docs/decisions.md`.
