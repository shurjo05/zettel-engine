"""Maintenance reports: links worth adding, orphans and their likely parents, map suggestions.

Read-only on the index, like search. Everything here is a suggestion for a
person (or an agent) to act on; nothing is changed. All three reports come from
the same two inputs: note embeddings, and the links that already exist.
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter

import numpy as np

from zk.embed import Embedder
from zk.proposals import DUPLICATE_AT, RELATED_AT

NEIGHBOURS = 5  # a cluster joins notes that are each in the other's 5 nearest
CLUSTER_AT = 0.75  # ...and at least this similar
MIN_CLUSTER = 4  # a map is worth suggesting for four or more notes


def _notes(conn: sqlite3.Connection, embedder: Embedder):
    """(rows, matrix): every note with an embedding, in path order, and its unit vectors."""
    rows = conn.execute(
        "SELECT n.path, n.title, n.tags, e.vector FROM notes n JOIN embeddings e "
        "ON e.content_hash = n.content_hash AND e.model = ? ORDER BY n.path",
        (embedder.model,),
    ).fetchall()
    if not rows:
        return [], np.zeros((0, 0), dtype=np.float32)
    return rows, np.stack([np.frombuffer(r["vector"], dtype=np.float32) for r in rows])


def _linked(conn: sqlite3.Connection) -> set[frozenset[str]]:
    """Pairs of notes joined by a link in either direction."""
    return {
        frozenset((r["src_path"], r["dst_path"]))
        for r in conn.execute(
            "SELECT DISTINCT src_path, dst_path FROM links "
            "WHERE status = 'resolved' AND src_path != dst_path"
        )
    }


def link_suggestions(
    conn: sqlite3.Connection, embedder: Embedder, at: float = RELATED_AT, per_note: int = 3
) -> list[dict]:
    """Unlinked pairs at similarity `at` or more, below the duplicate cut, most similar first.

    Each note appears in at most `per_note` suggestions, so a dense topic doesn't
    fill the report.
    """
    rows, matrix = _notes(conn, embedder)
    if len(rows) < 2:
        return []
    scores = matrix @ matrix.T
    linked = _linked(conn)
    pairs = []
    for i, j in zip(*np.nonzero(np.triu(scores >= at, k=1)), strict=True):
        s = float(scores[i, j])
        a, b = rows[i]["path"], rows[j]["path"]
        if s < DUPLICATE_AT and frozenset((a, b)) not in linked:
            pairs.append((s, a, b))
    used: Counter[str] = Counter()
    out = []
    for s, a, b in sorted(pairs, key=lambda p: (-p[0], p[1], p[2])):
        if used[a] < per_note and used[b] < per_note:
            used.update((a, b))
            out.append({"a": a, "b": b, "similarity": round(s, 3)})
    return out


def suggest_for(conn: sqlite3.Connection, embedder: Embedder, path: str, k: int = 3) -> list[dict]:
    """The notes most similar to one note that it isn't linked with yet."""
    rows, matrix = _notes(conn, embedder)
    index = {r["path"]: i for i, r in enumerate(rows)}
    if path not in index:
        return []
    linked = _linked(conn)
    scores = matrix @ matrix[index[path]]
    out = []
    for i in np.argsort(-scores):
        other = rows[int(i)]["path"]
        if other != path and frozenset((path, other)) not in linked:
            out.append(
                {
                    "path": other,
                    "title": rows[int(i)]["title"],
                    "similarity": round(float(scores[i]), 3),
                }
            )
            if len(out) == k:
                break
    return out


def orphan_report(conn: sqlite3.Connection, embedder: Embedder, k: int = 3) -> list[dict]:
    """Notes nothing links to, each with the unlinked notes most likely to link to it."""
    orphans = [
        r["path"]
        for r in conn.execute(
            "SELECT path FROM notes n WHERE NOT EXISTS ("
            "  SELECT 1 FROM links l WHERE l.dst_path = n.path AND l.src_path != n.path"
            ") ORDER BY path"
        )
    ]
    return [{"path": p, "candidates": suggest_for(conn, embedder, p, k)} for p in orphans]


def clusters(
    conn: sqlite3.Connection,
    embedder: Embedder,
    k: int = NEIGHBOURS,
    at: float = CLUSTER_AT,
    min_size: int = MIN_CLUSTER,
) -> list[dict]:
    """Groups of closely related notes: connected components of the mutual k-nearest graph.

    Two notes are joined when each is among the other's k nearest and they are at
    least `at` similar. Requiring both directions stops one broad note from
    chaining unrelated topics together, the usual failure of threshold clustering.
    Each cluster says whether a member already acts as its map (links to at
    least half of the others) and which tags its members share.
    """
    rows, matrix = _notes(conn, embedder)
    n = len(rows)
    if n < min_size:
        return []
    scores = matrix @ matrix.T
    np.fill_diagonal(scores, -1.0)
    nearest = np.argsort(-scores, axis=1)[:, :k]
    near = np.zeros((n, n), dtype=bool)
    near[np.repeat(np.arange(n), k), nearest.ravel()] = True
    edges = near & near.T & (scores >= at)

    parent = list(range(n))  # union-find over the edges

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in zip(*np.nonzero(np.triu(edges, k=1)), strict=True):
        parent[find(int(i))] = find(int(j))
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    links = {
        (r["src_path"], r["dst_path"])
        for r in conn.execute("SELECT src_path, dst_path FROM links WHERE status = 'resolved'")
    }
    out = []
    for members in groups.values():
        if len(members) < min_size:
            continue
        paths = [rows[i]["path"] for i in members]
        hub = max(paths, key=lambda p: (sum((p, q) in links for q in paths if q != p), p))
        covers = sum((hub, q) in links for q in paths if q != hub)
        tags = Counter(t.lower() for i in members for t in json.loads(rows[i]["tags"]))
        sub = scores[np.ix_(members, members)]
        cohesion = float(sub[~np.eye(len(members), dtype=bool)].mean())
        out.append(
            {
                "notes": sorted(paths),
                "shared_tags": [t for t, c in tags.most_common(3) if c > 1],
                "cohesion": round(cohesion, 3),
                "map": hub if covers * 2 >= len(paths) - 1 else None,
            }
        )
    return sorted(out, key=lambda c: (-len(c["notes"]), c["notes"][0]))
