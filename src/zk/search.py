"""Keyword search over the index with BM25 ranking (SQLite FTS5).

Search only reads the index. Build or refresh it first with `index.build_index`.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import asdict, dataclass

# bm25() weights for notes_fts columns, in table order:
# path (not searched), title, aliases, tags, summary, body.
# A hit in the title says the note is *about* the term; a hit in the body
# may be a passing mention. These are starting values; the eval set judges them.
COLUMN_WEIGHTS = (0.0, 10.0, 8.0, 5.0, 3.0, 1.0)

_WORD = re.compile(r"\w+", re.UNICODE)


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


def search(
    conn: sqlite3.Connection,
    query: str,
    k: int = 5,
    type: str | None = None,
    tag: str | None = None,
) -> list[Hit]:
    """Return the top `k` notes for `query`, best first."""
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
    params: list = [match]
    if type:
        sql += " AND n.type = ?"
        params.append(type.lower())
    if tag:
        sql += " AND EXISTS (SELECT 1 FROM json_each(n.tags) WHERE lower(value) = ?)"
        params.append(tag.lower())
    sql += " ORDER BY score DESC, n.path LIMIT ?"
    params.append(k)

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
