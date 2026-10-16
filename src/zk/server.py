"""MCP server: the vault's search and lookup, offered as tools to Claude Code and Codex.

Like the CLI, this module holds no retrieval logic; each tool refreshes the
index and calls the same library functions. It runs over stdio, so stdout is
the protocol and anything human-readable goes to stderr. Every tool is
read-only; writing to the vault arrives later and only behind a flag.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Callable, Iterable
from pathlib import Path

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

from zk import lookup
from zk.index import build_index, open_db
from zk.search import search

INSTRUCTIONS = """\
Tools for a personal Zettelkasten: short Markdown notes, one idea each, linked to each other.
Start with search_notes; its hits carry summaries, which often answer the question on their own.
Open a note with get_note only when you need its full text, and use get_related to follow links
and shared tags. Name notes by the id or path a tool returned. All tools are read-only.
"""

READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=False)


def _compact(value: object) -> str:
    """JSON without indentation: whatever a tool returns is paid for in context tokens."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def create_server(vault: Path, db: Path, excludes: Iterable[str]) -> MCPServer:
    vault = vault.resolve()
    excludes = tuple(excludes)
    lock = threading.Lock()  # tools run in worker threads; refresh one at a time

    def run(fn: Callable[[sqlite3.Connection], object]) -> str:
        with lock:
            conn = open_db(db)
            try:
                build_index(conn, vault, excludes)  # cheap when nothing changed
                return _compact(fn(conn))
            except LookupError as exc:  # NoteNotFound, AmbiguousNote
                raise ToolError(str(exc)) from exc
            finally:
                conn.close()

    server = MCPServer("zk", instructions=INSTRUCTIONS)

    @server.tool(annotations=READ_ONLY, structured_output=False)
    def search_notes(
        query: str, k: int = 5, type: str | None = None, tag: str | None = None
    ) -> str:
        """Keyword search (BM25, with stemming) over the notes.

        Returns up to k hits, best first, each with id, path, title, type, summary, score and a
        short snippet ([matched words] in brackets). Bodies are not included. Optional filters:
        type (concept, pattern, problem, decision) and tag.
        """
        k = max(1, min(k, 25))
        return run(lambda c: [h.to_dict() for h in search(c, query, k=k, type=type, tag=tag)])

    @server.tool(annotations=READ_ONLY, structured_output=False)
    def get_note(note: str) -> str:
        """One note in full: metadata, Markdown body, and the notes it links to and from.

        `note` is an id, a file name or vault path (with or without .md), or a title.
        """
        return run(lambda c: lookup.get_note(c, note))

    @server.tool(annotations=READ_ONLY, structured_output=False)
    def get_related(note: str, k: int = 10) -> str:
        """A note's neighbours with their summaries (no bodies).

        Lists notes it links to, then notes linking to it, then notes sharing its tags,
        up to k in total; `total` says how many there are.
        """
        k = max(1, min(k, 50))
        return run(lambda c: lookup.related(c, note, k=k))

    return server
