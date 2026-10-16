"""Read single notes and their neighbours from the index.

Like search, lookup only reads the index. Notes are named the way links name
them: by vault path, file name or frontmatter id (case-insensitive), with the
title as a last resort because that is what an agent most often has.
"""

from __future__ import annotations

import json
import sqlite3

from zk.search import search

MAX_BODY_CHARS = 20_000


class NoteNotFound(LookupError):
    def __init__(self, name: str, suggestions: list[str]):
        self.name = name
        self.suggestions = suggestions
        hint = f" Closest matches: {', '.join(suggestions)}." if suggestions else ""
        super().__init__(f"No note named {name!r}.{hint}")


class AmbiguousNote(LookupError):
    def __init__(self, name: str, paths: list[str]):
        self.name = name
        self.paths = paths
        super().__init__(
            f"{name!r} matches several notes: {', '.join(paths)}. Name it by its path."
        )


def find_paths(conn: sqlite3.Connection, name: str) -> list[str]:
    """Every note path that `name` could refer to, most specific rule first."""
    key = name.strip().replace("\\", "/").lower().removesuffix(".md").lstrip("/")
    if not key:
        return []
    rows = conn.execute("SELECT path, id, title FROM notes").fetchall()
    rules = [
        lambda r: r["path"][:-3].lower() == key,  # vault path
        lambda r: "/" not in key and r["path"][:-3].lower().rsplit("/", 1)[-1] == key,  # file name
        lambda r: (r["id"] or "").lower() == key,
        lambda r: r["title"].lower() == key,
    ]
    for rule in rules:
        paths = sorted(r["path"] for r in rows if rule(r))
        if paths:
            return paths
    return []


def resolve(conn: sqlite3.Connection, name: str) -> str:
    """The one note path `name` refers to; raises NoteNotFound or AmbiguousNote."""
    paths = find_paths(conn, name)
    if len(paths) == 1:
        return paths[0]
    if paths:
        raise AmbiguousNote(name, paths)
    raise NoteNotFound(name, [h.path for h in search(conn, name, k=3)])


def _brief(row: sqlite3.Row) -> dict:
    return {"path": row["path"], "id": row["id"], "title": row["title"], "type": row["type"]}


def get_note(conn: sqlite3.Connection, name: str, max_chars: int = MAX_BODY_CHARS) -> dict:
    """One note in full, with the titles of the notes it links to and from."""
    path = resolve(conn, name)
    row = conn.execute("SELECT * FROM notes WHERE path = ?", (path,)).fetchone()
    body = row["body"]
    truncated = len(body) > max_chars
    links_out = [
        _brief(r)
        for r in conn.execute(
            "SELECT DISTINCT n.path, n.id, n.title, n.type FROM links l "
            "JOIN notes n ON n.path = l.dst_path WHERE l.src_path = ? ORDER BY l.rowid",
            (path,),
        )
    ]
    broken = [
        r[0]
        for r in conn.execute(
            "SELECT DISTINCT target FROM links WHERE src_path = ? AND dst_path IS NULL", (path,)
        )
    ]
    links_in = [
        _brief(r)
        for r in conn.execute(
            "SELECT DISTINCT n.path, n.id, n.title, n.type FROM links l "
            "JOIN notes n ON n.path = l.src_path WHERE l.dst_path = ? ORDER BY n.path",
            (path,),
        )
    ]
    return {
        **_brief(row),
        "summary": row["summary"],
        "tags": json.loads(row["tags"]),
        "aliases": json.loads(row["aliases"]),
        "sources": json.loads(row["sources"]),
        "created": row["created"],
        "updated": row["updated"],
        "body": body[:max_chars],
        "truncated": truncated,
        "links_out": links_out,
        "broken_links": broken,
        "links_in": links_in,
    }


def related(conn: sqlite3.Connection, name: str, k: int = 10) -> dict:
    """A note's neighbours, with summaries but no bodies.

    Order: notes it links to, then notes linking to it, then notes sharing
    the most tags. A note reached several ways is listed once, with every
    relation.
    """
    path = resolve(conn, name)
    row = conn.execute("SELECT * FROM notes WHERE path = ?", (path,)).fetchone()
    found: dict[str, list[str]] = {}

    def add(other: str, relation: str) -> None:
        if other != path:
            found.setdefault(other, []).append(relation)

    for r in conn.execute(
        "SELECT dst_path FROM links WHERE src_path = ? AND dst_path IS NOT NULL ORDER BY rowid",
        (path,),
    ):
        if "links to" not in found.get(r[0], []):
            add(r[0], "links to")
    for r in conn.execute(
        "SELECT DISTINCT src_path FROM links WHERE dst_path = ? ORDER BY src_path", (path,)
    ):
        add(r[0], "linked from")

    tags = {t.lower() for t in json.loads(row["tags"])}
    if tags:
        shared: list[tuple[int, str, list[str]]] = []
        for other in conn.execute("SELECT path, tags FROM notes WHERE path != ?", (path,)):
            common = sorted(tags & {t.lower() for t in json.loads(other["tags"])})
            if common:
                shared.append((-len(common), other["path"], common))
        for _, other, common in sorted(shared):
            add(other, "shares tags: " + ", ".join(common))

    neighbours = []
    for other, relations in list(found.items())[: max(k, 0)]:
        n = conn.execute("SELECT * FROM notes WHERE path = ?", (other,)).fetchone()
        neighbours.append({**_brief(n), "summary": n["summary"], "relations": relations})
    return {**_brief(row), "related": neighbours, "total": len(found)}
