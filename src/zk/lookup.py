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
# An id named in more than this share of a project's sections (or a class used all over)
# links nothing: it is the project's subject, not a thread to follow.
MAX_REF_SHARE = 0.25
MIN_REF_LIMIT = 8


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


def compact(value: object) -> str:
    """JSON without indentation, as the MCP tools return it: every character costs context."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


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

    front = json.loads(row["frontmatter"] or "{}")
    if row["type"] == "reference" and front.get("refs"):
        for other, shared in _sharing_refs(conn, path, front):
            add(other, "shares " + ", ".join(shared))

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


def _ref_rows(
    conn: sqlite3.Connection, project: str | None, code: bool = False
) -> list[tuple[str, str]]:
    """(path, ref) for every doc section's refs, in one project if given. `code` adds the
    project's code--ids note, which isn't a section and so isn't counted as one."""
    sql = (
        "SELECT n.path, j.value FROM notes n, json_each(n.frontmatter, '$.refs') j "
        "WHERE n.type = 'reference'"
    )
    if not code:
        sql += " AND json_extract(n.frontmatter, '$.code_ids') IS NULL"
    params: list = []
    if project:
        sql += " AND json_extract(n.frontmatter, '$.project') = ?"
        params.append(project)
    return [(r[0], r[1]) for r in conn.execute(sql, params)]


def _sharing_refs(conn: sqlite3.Connection, path: str, front: dict) -> list[tuple[str, list]]:
    """Sections of the same project that name the same ids, rarest shared id first."""
    rows = _ref_rows(conn, front.get("project"))
    by_ref: dict[str, set[str]] = {}
    for p, ref in rows:
        by_ref.setdefault(ref, set()).add(p)
    sections = len({p for p, _ in rows})
    limit = max(MIN_REF_LIMIT, int(sections * MAX_REF_SHARE))
    mine = [r for r in front["refs"] if 1 < len(by_ref.get(r, ())) <= limit]
    shared: dict[str, list[str]] = {}
    for ref in sorted(mine, key=lambda r: (len(by_ref[r]), r)):
        for other in sorted(by_ref[ref] - {path}):
            shared.setdefault(other, []).append(ref)
    rarity = {r: len(by_ref[r]) for r in mine}
    return sorted(
        shared.items(), key=lambda kv: (min(rarity[r] for r in kv[1]), -len(kv[1]), kv[0])
    )


def refs(
    conn: sqlite3.Connection, ref: str, project: str | None = None, k: int = 30, code_k: int = 10
) -> dict:
    """Every doc section that names `ref` (an id like R-06, or a class name), grouped by
    doc, and where the code tags it. Sections whose code names may be out of date carry
    `stale`."""
    ref = ref.strip()
    paths = sorted({p for p, r in _ref_rows(conn, project, code=True) if r in (ref, ref.upper())})
    sections, code = [], []
    for p in paths:
        row = conn.execute("SELECT * FROM notes WHERE path = ?", (p,)).fetchone()
        front = json.loads(row["frontmatter"] or "{}")
        if front.get("code_ids") is not None:
            for spot in front["code_ids"].get(ref, front["code_ids"].get(ref.upper(), [])):
                code.append({"project": front.get("project"), "at": spot})
            continue
        entry = {
            "path": p,
            "title": row["title"],
            "doc": front.get("doc"),
            "project": front.get("project"),
        }
        if front.get("stale"):
            entry["stale"] = front["stale"]
        sections.append(entry)
    sections.sort(key=lambda e: (str(e["doc"]), e["path"]))
    return {
        "ref": ref,
        "sections": sections[: max(k, 0)],
        "total": len(sections),
        "code": code[: max(code_k, 0)],  # the spots naming fewest other ids first
        "code_total": len(code),
    }


def hubs(conn: sqlite3.Connection, project: str, n: int = 5) -> list[tuple[str, int]]:
    """A project's most-named ids: (id, sections naming it), most first. Class names and
    ids named almost everywhere are left out."""
    from zk.crossref import ids_in

    rows = _ref_rows(conn, project)
    by_ref: dict[str, set[str]] = {}
    for p, ref in rows:
        if ids_in(ref) == [ref]:
            by_ref.setdefault(ref, set()).add(p)
    sections = len({p for p, _ in rows})
    limit = max(MIN_REF_LIMIT, int(sections * MAX_REF_SHARE))
    ranked = sorted(
        ((r, len(ps)) for r, ps in by_ref.items() if 2 < len(ps) <= limit),
        key=lambda x: (-x[1], x[0]),
    )
    return ranked[:n]
