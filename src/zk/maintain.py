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
from zk.proposals import DUPLICATE_AT

# Calibrated by hand on the 93 imported legacy notes (bge-small), 2026-11-03.
LINK_AT = 0.78  # 12 of 18 pairs at 0.78 were clearly useful and 1 was noise; 0.80 kept only 9
PARENT_AT = 0.70  # every orphan candidate judged noise scored 0.74 or less, most under 0.70
NEIGHBOURS = (
    5  # a cluster joins notes that are each in the other's 5 nearest (4-6 made no difference)
)
CLUSTER_AT = 0.75  # ...and at least this similar
SPLIT_OVER = 20  # a 36-note cluster mixed topics; 17 at the next step did not: re-cluster stricter
SPLIT_STEP = 0.03
MIN_CLUSTER = 4  # a map is worth suggesting for four or more notes
MAP_COVERS = 0.75  # a note that links to 75% of its cluster already serves as its map


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
    conn: sqlite3.Connection, embedder: Embedder, at: float = LINK_AT, per_note: int = 3
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


def suggest_for(
    conn: sqlite3.Connection, embedder: Embedder, path: str, k: int = 3, at: float = -1.0
) -> list[dict]:
    """The notes most similar to one note (at least `at`) that it isn't linked with yet."""
    rows, matrix = _notes(conn, embedder)
    index = {r["path"]: i for i, r in enumerate(rows)}
    if path not in index:
        return []
    linked = _linked(conn)
    scores = matrix @ matrix[index[path]]
    out = []
    for i in np.argsort(-scores):
        other = rows[int(i)]["path"]
        if float(scores[i]) < at:
            break
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


def orphan_report(
    conn: sqlite3.Connection, embedder: Embedder, k: int = 3, at: float = PARENT_AT
) -> list[dict]:
    """Notes nothing links to, each with the unlinked notes (at least `at` similar) most
    likely to link to it. An empty list means no note is a likely parent."""
    orphans = [
        r["path"]
        for r in conn.execute(
            "SELECT path FROM notes n WHERE NOT EXISTS ("
            "  SELECT 1 FROM links l WHERE l.dst_path = n.path AND l.src_path != n.path"
            ") ORDER BY path"
        )
    ]
    return [{"path": p, "candidates": suggest_for(conn, embedder, p, k, at)} for p in orphans]


def clusters(
    conn: sqlite3.Connection,
    embedder: Embedder,
    k: int = NEIGHBOURS,
    at: float = CLUSTER_AT,
    min_size: int = MIN_CLUSTER,
    split_over: int = SPLIT_OVER,
) -> list[dict]:
    """Groups of closely related notes: connected components of the mutual k-nearest graph.

    Two notes are joined when each is among the other's k nearest and they are at
    least `at` similar. Requiring both directions stops one broad note from
    chaining unrelated topics together, the usual failure of threshold clustering.
    A component over `split_over` notes still mixed topics on real notes, so it is
    split again with a stricter threshold. Each cluster names the member closest
    to being its map (the one linking to most others) and says whether it
    already is one, and which tags its members share.
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

    def components(members: list[int], threshold: float) -> list[list[int]]:
        for i in members:
            parent[i] = i
        sub = np.ix_(members, members)
        local = np.triu(edges[sub] & (scores[sub] >= threshold), k=1)
        for a, b in zip(*np.nonzero(local), strict=True):
            parent[find(members[int(a)])] = find(members[int(b)])
        found: dict[int, list[int]] = {}
        for i in members:
            found.setdefault(find(i), []).append(i)
        out = []
        for group in found.values():
            if len(group) > split_over and threshold + SPLIT_STEP < DUPLICATE_AT:
                out += components(group, threshold + SPLIT_STEP)
            else:
                out.append(group)
        return out

    groups = components(list(range(n)), at)

    links = {
        (r["src_path"], r["dst_path"])
        for r in conn.execute("SELECT src_path, dst_path FROM links WHERE status = 'resolved'")
    }
    out = []
    for members in groups:
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
                "map": hub if covers >= MAP_COVERS * (len(paths) - 1) else None,
                "closest_to_map": {"path": hub, "links_to": covers},
            }
        )
    return sorted(out, key=lambda c: (-len(c["notes"]), c["notes"][0]))
