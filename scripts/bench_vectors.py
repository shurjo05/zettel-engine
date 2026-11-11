"""Time exact vector search at growing vault sizes: NumPy versus sqlite-vec.

    uv run python scripts/bench_vectors.py [sizes...]

Synthetic notes with random unit vectors (384 dimensions, like bge-small) go in
an in-memory index; each backend answers the same queries through
search._nearest, so both timings include reading vectors from SQLite.
Nothing touches a vault.
"""

from __future__ import annotations

import os
import sqlite3
import sys
import time

import numpy as np

from zk import search
from zk.index import open_db

DIM = 384
QUERIES = 20


class FixedEmbedder:
    model = "bench/random"


def build(n: int, rng: np.random.Generator) -> sqlite3.Connection:
    conn = open_db(":memory:")
    vectors = rng.standard_normal((n, DIM)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    conn.executemany(
        "INSERT INTO notes (path, id, title, type, summary, tags, aliases, sources, content_hash,"
        " body) VALUES (?, ?, ?, 'concept', '', '[]', '[]', '[]', ?, '')",
        [(f"notes/n{i}.md", f"n{i}", f"Note {i}", f"h{i}") for i in range(n)],
    )
    conn.executemany(
        "INSERT INTO embeddings (content_hash, model, vector) VALUES (?, ?, ?)",
        [(f"h{i}", FixedEmbedder.model, v.tobytes()) for i, v in enumerate(vectors)],
    )
    return conn


def timed(conn: sqlite3.Connection, queries: np.ndarray) -> tuple[float, list]:
    start = time.perf_counter()
    results = [search._nearest(conn, FixedEmbedder(), q, 10, None, None) for q in queries]
    return (time.perf_counter() - start) * 1000 / len(queries), results


def main(sizes: list[int]) -> None:
    rng = np.random.default_rng(0)
    print(f"{'notes':>8} {'numpy ms':>10} {'sqlite-vec ms':>14}  same top 10")
    for n in sizes:
        conn = build(n, rng)
        queries = rng.standard_normal((QUERIES, DIM)).astype(np.float32)
        queries /= np.linalg.norm(queries, axis=1, keepdims=True)
        os.environ["ZK_VECTORS"] = "numpy"
        np_ms, np_hits = timed(conn, queries)
        os.environ.pop("ZK_VECTORS")
        assert search.vector_backend(conn) == "sqlite-vec"
        vec_ms, vec_hits = timed(conn, queries)
        same = all(
            [h.path for h in a] == [h.path for h in b]
            for a, b in zip(np_hits, vec_hits, strict=True)
        )
        print(f"{n:>8} {np_ms:>10.2f} {vec_ms:>14.2f}  {same}")
        conn.close()


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]] or [100, 1_000, 10_000, 100_000])
