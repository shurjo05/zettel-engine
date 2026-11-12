# zk's public API

Other programs may import these names from `zk.api`, and only these. Everything else in zk is
internal and can change in any release. `tests/test_api.py` pins how they behave. The first user
is tokenmaxxing, which mounts zk's tools in its own MCP server and turns a coding agent's note
suggestions into inbox captures.

Callers depend on the main branch (`zettel-engine @ git+https://github.com/shurjo05/zettel-engine`)
and record the exact commit in their lock file. That's safe because these names are the contract:
a change that would break them must change `tests/test_api.py` too, which makes it visible in review.

| Name | What it does |
| --- | --- |
| `add_knowledge_tools(server, vault, db, excludes, embedder=None)` | Adds zk's read-only tools (`search_notes`, `get_note`, `get_related`, and `suggest_links` when an embedder is given) to another `MCPServer`. They are the same code as `zk serve` and give the same answers. No write tool can be added this way. |
| `INSTRUCTIONS` | The read-only tools' instructions, to include in the host server's own. |
| `run_stdio(server)` | Serves an `MCPServer` over stdio the way `zk serve` does, including the fix for Claude Code versions that open with a newer protocol request. |
| `load_embedder(model=DEFAULT_MODEL)` | The local embedding model zk uses. |
| `find_duplicates(vault, embedder, title, summary, body="", *, db=None, excludes=DEFAULT_EXCLUDES, k=5)` | Existing notes most like a would-be note, each with `path`, `title`, `summary`, `similarity` and `verdict` (`likely-duplicate`, `related` or `new`, ingest's own thresholds). Refreshes the index and every note's vector first, so a note written a moment ago counts. |
| `save_text(vault, text, source, *, project=None, now=None)` | Saves text as a new inbox capture, like `zk capture --text`, and returns its path. `source` keeps `:`, so ingest can treat captures by kind (e.g. `codex:<repo>:<task>`). Secrets are redacted, and an existing file is never overwritten. |
| `data_dir()`, `default_db_path(vault)`, `DEFAULT_EXCLUDES`, `DEFAULT_MODEL` | Where zk keeps its index, and the defaults the CLI uses. |

Related: `zk hooks status --server NAME` checks that Claude Code has a read-only server under
that name. Use it when another program's server (one that mounts zk's tools) replaces the `zk` entry.

## Rules for callers

- The vault is written only by `save_text`, which creates a new file in `inbox/` and never
  changes or deletes one. Notes reach `notes/` only through zk's own ingest.
- A capture from `save_text` is read by the next ingest run and sent to that agent's model
  provider (see `docs/data-flow.md`).
