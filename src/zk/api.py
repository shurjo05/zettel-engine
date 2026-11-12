"""zk's public API: the names other programs may import. See docs/api.md.

Everything else in zk is internal and may change between versions. This module
holds no logic of its own beyond wiring: it re-exports the MCP pieces another
server mounts, and offers two calls that do in one step what the CLI does in
several (load the embedder with every vector up to date; save a text capture).

It keeps zk's write rules. The only vault write here is `save_text`, which
creates a new inbox capture through `zk.vault`; the knowledge tools it exports
are read-only, and no write tool can be reached through it.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

from zk import capture, vault
from zk.embed import DEFAULT_MODEL, Embedder, update_embeddings
from zk.index import DEFAULT_EXCLUDES, build_index, data_dir, default_db_path, open_db
from zk.ingest import check_duplicates
from zk.server import INSTRUCTIONS, add_knowledge_tools, run_stdio

__all__ = [
    "DEFAULT_EXCLUDES",
    "DEFAULT_MODEL",
    "INSTRUCTIONS",
    "add_knowledge_tools",
    "data_dir",
    "default_db_path",
    "find_duplicates",
    "load_embedder",
    "run_stdio",
    "save_text",
]


def load_embedder(model: str = DEFAULT_MODEL) -> Embedder:
    """The local embedding model zk uses (downloaded once, then run on this machine)."""
    from zk.embed import FastEmbedder

    return FastEmbedder(model)


def find_duplicates(
    vault_dir: Path,
    embedder: Embedder,
    title: str,
    summary: str,
    body: str = "",
    *,
    db: Path | None = None,
    excludes: Iterable[str] = DEFAULT_EXCLUDES,
    k: int = 5,
) -> list[dict]:
    """Existing notes most like a would-be note, each with zk's verdict.

    Refreshes the index and brings every note's vector up to date first, so a note
    written a moment ago is compared too. Verdicts use ingest's own thresholds:
    "likely-duplicate", "related" or "new". Each hit has path, title, summary,
    similarity and verdict.
    """
    conn = open_db(db or default_db_path(vault_dir))
    try:
        build_index(conn, vault_dir.resolve(), tuple(excludes))
        update_embeddings(conn, embedder)
        return check_duplicates(conn, embedder, title, summary, body, k=k)
    finally:
        conn.close()


def save_text(
    vault_dir: Path,
    text: str,
    source: str,
    *,
    project: str | None = None,
    now: datetime | None = None,
) -> Path:
    """Save text as a new inbox capture, as `zk capture --text` does. Returns its path.

    `source` is kept in the capture's frontmatter (letters, digits, "-" and ":"), so
    ingest can treat it by kind, e.g. "codex:<repo>:<task>". Secrets are redacted and
    an existing file is never overwritten.
    """
    source = _source(source)
    now = now or datetime.now().astimezone()
    captured = now.isoformat(timespec="seconds")
    body = capture.render_capture(
        [capture.Turn("user", text.strip())], source, captured, project=project
    )
    slug = source.replace(":", "-")[:60].strip("-") or "manual"
    name = f"inbox/{now:%Y-%m-%d-%H%M}-{slug}.md"
    return vault.write_new(vault_dir, vault.unique_path(vault_dir, name), body)


def _source(raw: str) -> str:
    parts = [re.sub(r"[^a-z0-9]+", "-", p.lower()).strip("-") for p in raw.split(":")]
    return ":".join(p for p in parts if p) or "manual"
