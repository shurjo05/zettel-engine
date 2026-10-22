"""Note embeddings: turn each note into a vector of its meaning, and keep them in the index.

The model runs locally (fastembed, ONNX on the CPU), so note text never leaves
this machine; only the model's weights are downloaded, once. Vectors are keyed
by the note's content hash and the model name: a renamed note reuses its
vector, an edited note gets a new one, and two models never mix.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from zk.index import data_dir

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

# bge v1.5 models were trained to match short queries to passages when the
# query carries this instruction; documents are embedded as they are.
_QUERY_PREFIXES = {
    "BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-base-en-v1.5": "Represent this sentence for searching relevant passages: ",
}


class Embedder(Protocol):
    """Anything that maps text to unit-length float32 vectors of a fixed size."""

    model: str

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray: ...  # shape (n, dim)

    def embed_query(self, text: str) -> np.ndarray: ...  # shape (dim,)


def normalize(vectors: np.ndarray) -> np.ndarray:
    """Scale each row to length 1, so a dot product is the cosine similarity."""
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.where(norms == 0, 1, norms)


class FastEmbedder:
    """A local ONNX embedding model via fastembed, loaded on first use."""

    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model
        self._engine = None

    def _load(self):
        if self._engine is None:
            from fastembed import TextEmbedding  # slow import; only when embedding

            cache = data_dir() / "models"
            cache.mkdir(parents=True, exist_ok=True)
            self._engine = TextEmbedding(self.model, cache_dir=str(cache))
        return self._engine

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        return normalize(np.stack(list(self._load().embed(list(texts), batch_size=32))))

    def embed_query(self, text: str) -> np.ndarray:
        prefix = _QUERY_PREFIXES.get(self.model, "")
        return normalize(next(iter(self._load().embed([prefix + text]))))


def note_text(title: str, body: str) -> str:
    """What gets embedded: the title, then the body without a repeated H1."""
    lines = body.lstrip().splitlines()
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
    return f"{title}\n\n" + "\n".join(lines).strip()


@dataclass
class EmbedStats:
    embedded: int = 0  # notes that needed a new vector
    reused: int = 0  # notes whose content already had one
    removed: int = 0  # vectors no note uses any more


def update_embeddings(
    conn: sqlite3.Connection, embedder: Embedder, batch_size: int = 64
) -> EmbedStats:
    """Give every note in the index a vector from `embedder`, embedding only what's new."""
    stats = EmbedStats()
    have = {
        r[0]
        for r in conn.execute(
            "SELECT content_hash FROM embeddings WHERE model = ?", (embedder.model,)
        )
    }
    todo: dict[str, str] = {}  # content_hash -> text; identical files share one vector
    for row in conn.execute("SELECT content_hash, title, body FROM notes"):
        if row["content_hash"] in have or row["content_hash"] in todo:
            stats.reused += 1
        else:
            todo[row["content_hash"]] = note_text(row["title"], row["body"])
    stats.embedded = len(todo)

    items = list(todo.items())
    with conn:
        for start in range(0, len(items), batch_size):
            batch = items[start : start + batch_size]
            vectors = embedder.embed_documents([text for _, text in batch])
            conn.executemany(
                "INSERT OR REPLACE INTO embeddings VALUES (?, ?, ?)",
                [
                    (h, embedder.model, v.astype(np.float32).tobytes())
                    for (h, _), v in zip(batch, vectors, strict=True)
                ],
            )
        cursor = conn.execute(
            "DELETE FROM embeddings WHERE content_hash NOT IN (SELECT content_hash FROM notes)"
        )
        stats.removed = cursor.rowcount
    return stats
