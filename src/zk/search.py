"""Search over the index: keyword (BM25 via SQLite FTS5), semantic (embeddings), or both fused.

Search only reads the index. Build or refresh it first with `index.build_index`,
and for semantic search `embed.update_embeddings`.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from zk.embed import Embedder

# bm25() weights for notes_fts columns, in table order:
# path (not searched), title, aliases, tags, summary, body.
# A hit in the title says the note is *about* the term; a hit in the body
# may be a passing mention. These are starting values; the eval set judges them.
COLUMN_WEIGHTS = (0.0, 10.0, 8.0, 5.0, 3.0, 1.0)

_WORD = re.compile(r"\w+", re.UNICODE)

MODES = ("keyword", "semantic", "hybrid")

# Reciprocal rank fusion: a note scores sum(1 / (RRF_K + rank)) over the lists
# it appears in. The literature's usual 60 (Cormack et al., 2009) flattens rank
# differences; on our eval set a smaller constant, which trusts each list's top
# ranks more, did better (MRR 0.90 vs 0.87), because keyword search returns many
# weak OR-matches for paraphrased questions. See docs/decisions.md (13).
RRF_K = 10

# What agents and `zk search` use unless told otherwise: on the eval set, the
# only mode that gets every query's note into the top 5.
DEFAULT_MODE = "hybrid"
# How deep each list is read before fusing. Deeper than k, so a note ranked
# modestly by both lists can still beat one ranked high by only one.
FUSION_DEPTH = 50


@dataclass
class Hit:
    rank: int
    path: str
    id: str | None
    title: str
    type: str | None
    summary: str
    score: float  # higher is better
    snippet: str

    def to_dict(self) -> dict:
        return asdict(self)


def fts_query(text: str) -> str | None:
    """Turn free text into an FTS5 query: every word quoted, joined with OR.

    Quoting stops user input from being read as FTS5 syntax (AND, NEAR, *, -),
    and OR lets a question match notes that contain only some of its words.
    BM25 then rewards notes that match more, and rarer, words.
    """
    words = _WORD.findall(text.lower())
    if not words:
        return None
    unique = dict.fromkeys(words)  # keep order, drop repeats
    return " OR ".join(f'"{w}"' for w in unique)


def _filters(type: str | None, tag: str | None) -> tuple[str, list]:
    sql, params = "", []
    if type:
        sql += " AND n.type = ?"
        params.append(type.lower())
    if tag:
        sql += " AND EXISTS (SELECT 1 FROM json_each(n.tags) WHERE lower(value) = ?)"
        params.append(tag.lower())
    return sql, params


def search(
    conn: sqlite3.Connection,
    query: str,
    k: int = 5,
    type: str | None = None,
    tag: str | None = None,
    mode: str = "keyword",
    embedder: Embedder | None = None,
) -> list[Hit]:
    """Return the top `k` notes for `query`, best first."""
    if mode in ("semantic", "hybrid"):
        if embedder is None:
            raise ValueError(f"{mode} search needs an embedder")
        if mode == "hybrid":
            return hybrid_search(conn, embedder, query, k=k, type=type, tag=tag)
        return semantic_search(conn, embedder, query, k=k, type=type, tag=tag)
    if mode != "keyword":
        raise ValueError(f"unknown search mode {mode!r}; expected one of {', '.join(MODES)}")
    return keyword_search(conn, query, k=k, type=type, tag=tag)


def keyword_search(
    conn: sqlite3.Connection,
    query: str,
    k: int = 5,
    type: str | None = None,
    tag: str | None = None,
) -> list[Hit]:
    """BM25 over the full-text index, with title and alias matches weighted up."""
    match = fts_query(query)
    if match is None or k <= 0:
        return []

    weights = ", ".join(str(w) for w in COLUMN_WEIGHTS)
    sql = f"""
        SELECT n.path, n.id, n.title, n.type, n.summary,
               -bm25(notes_fts, {weights}) AS score,
               snippet(notes_fts, 5, '[', ']', '...', 12) AS snippet
        FROM notes_fts
        JOIN notes n ON n.rowid = notes_fts.rowid
        WHERE notes_fts MATCH ?
    """
    where, extra = _filters(type, tag)
    sql += where + " ORDER BY score DESC, n.path LIMIT ?"
    params: list = [match, *extra, k]

    return [
        Hit(
            rank=i,
            path=row["path"],
            id=row["id"],
            title=row["title"],
            type=row["type"],
            summary=row["summary"],
            score=round(row["score"], 4),
            snippet=" ".join(row["snippet"].split()),
        )
        for i, row in enumerate(conn.execute(sql, params), start=1)
    ]


def semantic_search(
    conn: sqlite3.Connection,
    embedder: Embedder,
    query: str,
    k: int = 5,
    type: str | None = None,
    tag: str | None = None,
) -> list[Hit]:
    """Cosine similarity between the query's vector and every note's vector.

    Vectors are unit length, so cosine similarity is a dot product, and scoring
    every note is one matrix-vector product: exact, and fast at vault scale.
    """
    if not query.strip() or k <= 0:
        return []
    where, params = _filters(type, tag)
    rows = conn.execute(
        "SELECT n.path, n.id, n.title, n.type, n.summary, e.vector FROM notes n "
        "JOIN embeddings e ON e.content_hash = n.content_hash AND e.model = ? "
        "WHERE 1 = 1" + where,
        [embedder.model, *params],
    ).fetchall()
    if not rows:
        return []
    matrix = np.stack([np.frombuffer(r["vector"], dtype=np.float32) for r in rows])
    scores = matrix @ embedder.embed_query(query)
    # Highest score first; ties broken by path so results are reproducible.
    order = sorted(range(len(rows)), key=lambda i: (-scores[i], rows[i]["path"]))[:k]
    return [
        Hit(
            rank=rank,
            path=rows[i]["path"],
            id=rows[i]["id"],
            title=rows[i]["title"],
            type=rows[i]["type"],
            summary=rows[i]["summary"],
            score=round(float(scores[i]), 4),
            snippet="",
        )
        for rank, i in enumerate(order, start=1)
    ]


def hybrid_search(
    conn: sqlite3.Connection,
    embedder: Embedder,
    query: str,
    k: int = 5,
    type: str | None = None,
    tag: str | None = None,
) -> list[Hit]:
    """Keyword and semantic results fused with reciprocal rank fusion.

    BM25 scores and cosine similarities are on different scales, so they are not
    added; only ranks are. A note found by both lists rises; a note only one list
    finds still counts. The keyword snippet is kept when there is one.
    """
    if k <= 0:
        return []
    depth = max(k, FUSION_DEPTH)
    lists = [
        keyword_search(conn, query, k=depth, type=type, tag=tag),
        semantic_search(conn, embedder, query, k=depth, type=type, tag=tag),
    ]
    fused: dict[str, float] = {}
    best: dict[str, Hit] = {}
    for hits in lists:
        for hit in hits:
            fused[hit.path] = fused.get(hit.path, 0.0) + 1 / (RRF_K + hit.rank)
            if hit.path not in best or (hit.snippet and not best[hit.path].snippet):
                best[hit.path] = hit
    order = sorted(fused, key=lambda path: (-fused[path], path))[:k]
    return [
        Hit(**{**asdict(best[path]), "rank": rank, "score": round(fused[path], 6)})
        for rank, path in enumerate(order, start=1)
    ]
