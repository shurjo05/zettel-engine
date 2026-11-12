"""Search over the index: keyword (BM25 via SQLite FTS5), semantic (embeddings), or both fused.

Search only reads the index. Build or refresh it first with `index.build_index`,
and for semantic search `embed.update_embeddings`.
"""

from __future__ import annotations

import os
import re
import sqlite3
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

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


def project_slug(name: str) -> str:
    """A project's name as notes record it in `sources` (project:<slug>): lower case, dashes."""
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


# Folders that hold projects rather than being one. A session started in one of them
# (C:\Users\me\source\repos, say) names no project.
CONTAINER_FOLDERS = frozenset(
    "repos repo source src projects project code dev documents desktop downloads users home "
    "workspace workspaces github git onedrive".split()
)


def is_container(slug: str | None) -> bool:
    """Whether a folder's slug is a folder that holds projects, not a project."""
    return bool(slug) and slug.replace("-", "") in CONTAINER_FOLDERS


_SOURCE_PROJECT = re.compile(r"^\s*(?:project|repo|local-repo):\s*([^/\\(+:]+)", re.I)


def source_project(source: str) -> str | None:
    """The project a source names, as a slug, or None.

    `project:<name>` (sessions), `repo:<name>` (zk learn) and `local-repo:<name>/<path>`
    (the legacy import) all name the folder the note came from.
    """
    m = _SOURCE_PROJECT.match(source)
    slug = project_slug(m.group(1)) if m else ""
    return slug if slug and not is_container(slug) else None


def same_project(a: str, b: str) -> bool:
    """Whether two project slugs name the same project.

    They do when they differ only in punctuation (aphiddetection, aphid-detection) or
    when one is the other plus words at the end (ruflo, ruflo-claude-flow;
    nautilustrx, nautilustrx-shurjo05, as GitHub Classroom names copies).
    """
    if not a or not b:
        return False
    if a.replace("-", "") == b.replace("-", ""):
        return True
    short, long = sorted((a, b), key=len)
    return long.startswith(short + "-")


def _source_in_project(source: str, slug: str) -> bool:
    found = source_project(source or "")
    return bool(found) and same_project(found, slug)


def use_project_filter(conn: sqlite3.Connection) -> None:
    """Make the SQL function the project filter needs available on this connection."""
    conn.create_function("zk_in_project", 2, _source_in_project, deterministic=True)


def _filters(
    type: str | None, tag: str | None, project: str | None = None, notes_only: bool = False
) -> tuple[str, list]:
    sql, params = "", []
    if type:
        sql += " AND n.type = ?"
        params.append(type.lower())
    elif notes_only:  # the duplicate check: a project's docs (references) aren't notes
        sql += " AND coalesce(n.type, '') NOT IN ('map', 'reference')"
    else:
        sql += " AND n.type IS NOT 'map'"  # map notes list other notes; open them by name
    if project:
        sql += " AND EXISTS (SELECT 1 FROM json_each(n.sources) WHERE zk_in_project(value, ?))"
        params.append(project_slug(project))
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
    project: str | None = None,
) -> list[Hit]:
    """Return the top `k` notes for `query`, best first."""
    if mode in ("semantic", "hybrid"):
        if embedder is None:
            raise ValueError(f"{mode} search needs an embedder")
        if mode == "hybrid":
            return hybrid_search(conn, embedder, query, k=k, type=type, tag=tag, project=project)
        return semantic_search(conn, embedder, query, k=k, type=type, tag=tag, project=project)
    if mode != "keyword":
        raise ValueError(f"unknown search mode {mode!r}; expected one of {', '.join(MODES)}")
    return keyword_search(conn, query, k=k, type=type, tag=tag, project=project)


def keyword_search(
    conn: sqlite3.Connection,
    query: str,
    k: int = 5,
    type: str | None = None,
    tag: str | None = None,
    project: str | None = None,
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
    where, extra = _filters(type, tag, project)
    if project:
        use_project_filter(conn)
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
    project: str | None = None,
) -> list[Hit]:
    """Cosine similarity between the query's vector and every note's vector.

    Every note is scored (exact search), inside SQLite with sqlite-vec when it
    loads, else as one NumPy matrix-vector product; both give the same ranking.
    """
    if not query.strip() or k <= 0:
        return []
    return _nearest(conn, embedder, embedder.embed_query(query), k, type, tag, project)


def similar_notes(conn: sqlite3.Connection, embedder: Embedder, text: str, k: int = 5) -> list[Hit]:
    """Notes closest to a piece of note text (as a document, not a query): the duplicate check."""
    if not text.strip() or k <= 0:
        return []
    return similar_to_vector(conn, embedder, embedder.embed_documents([text])[0], k)


def similar_to_vector(
    conn: sqlite3.Connection, embedder: Embedder, vector: np.ndarray, k: int = 5
) -> list[Hit]:
    """Notes closest to an already embedded document vector."""
    return _nearest(conn, embedder, vector, k, None, None, notes_only=True) if k > 0 else []


_VEC_MISSING = False  # set once this Python can't load sqlite-vec at all


def vector_backend(conn: sqlite3.Connection) -> str:
    """'sqlite-vec' if this connection can score vectors in SQL, else 'numpy'.

    sqlite-vec is a SQLite extension (one compiled file, installed with zk). It
    loads per connection; a Python whose sqlite3 can't load extensions, or
    ZK_VECTORS=numpy, falls back to NumPy.
    """
    global _VEC_MISSING
    if os.environ.get("ZK_VECTORS", "").lower() == "numpy" or _VEC_MISSING:
        return "numpy"
    try:
        conn.execute("SELECT vec_version()")
        return "sqlite-vec"
    except sqlite3.OperationalError:
        pass
    try:
        import sqlite_vec

        conn.enable_load_extension(True)
        try:
            sqlite_vec.load(conn)
        finally:
            conn.enable_load_extension(False)
        return "sqlite-vec"
    except (ImportError, AttributeError, sqlite3.Error):
        _VEC_MISSING = True
        return "numpy"


def _nearest(
    conn: sqlite3.Connection,
    embedder: Embedder,
    vector: np.ndarray,
    k: int,
    type: str | None,
    tag: str | None,
    project: str | None = None,
    notes_only: bool = False,
) -> list[Hit]:
    where, params = _filters(type, tag, project, notes_only)
    if project:
        use_project_filter(conn)
    if vector_backend(conn) == "sqlite-vec":
        # Unit vectors: cosine similarity = 1 - cosine distance. Scored in C, in SQL.
        rows = conn.execute(
            "SELECT n.path, n.id, n.title, n.type, n.summary, "
            "1 - vec_distance_cosine(e.vector, ?) AS score FROM notes n "
            "JOIN embeddings e ON e.content_hash = n.content_hash AND e.model = ? "
            "WHERE 1 = 1" + where + " ORDER BY score DESC, n.path LIMIT ?",
            [vector.astype("float32").tobytes(), embedder.model, *params, k],
        ).fetchall()
        return [
            Hit(
                rank=rank,
                path=r["path"],
                id=r["id"],
                title=r["title"],
                type=r["type"],
                summary=r["summary"],
                score=round(float(r["score"]), 4),
                snippet="",
            )
            for rank, r in enumerate(rows, start=1)
        ]
    import numpy as np

    rows = conn.execute(
        "SELECT n.path, n.id, n.title, n.type, n.summary, e.vector FROM notes n "
        "JOIN embeddings e ON e.content_hash = n.content_hash AND e.model = ? "
        "WHERE 1 = 1" + where,
        [embedder.model, *params],
    ).fetchall()
    if not rows:
        return []
    matrix = np.stack([np.frombuffer(r["vector"], dtype=np.float32) for r in rows])
    scores = matrix @ vector
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
    project: str | None = None,
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
        keyword_search(conn, query, k=depth, type=type, tag=tag, project=project),
        semantic_search(conn, embedder, query, k=depth, type=type, tag=tag, project=project),
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
