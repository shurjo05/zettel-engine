import numpy as np
import pytest
from conftest import WordEmbedder

from zk.embed import normalize, note_text, update_embeddings
from zk.index import build_index, open_db


@pytest.fixture
def conn(vault, tmp_path):
    c = open_db(tmp_path / "index.sqlite")
    build_index(c, vault)
    return c


def count(conn, model="test/word-hash") -> int:
    return conn.execute("SELECT count(*) FROM embeddings WHERE model = ?", (model,)).fetchone()[0]


def test_note_text_is_title_then_body_without_repeated_h1():
    assert note_text("BM25", "# BM25\n\nScores documents.\n") == "BM25\n\nScores documents."
    assert note_text("Legacy", "## Summary\nOld.") == "Legacy\n\n## Summary\nOld."


def test_normalize_gives_unit_rows_and_leaves_zero_rows_alone():
    rows = normalize(np.array([[3.0, 4.0], [0.0, 0.0]]))
    assert rows.dtype == np.float32
    assert np.allclose(rows, [[0.6, 0.8], [0.0, 0.0]])


def test_only_new_content_is_embedded(conn, embedder):
    first = update_embeddings(conn, embedder)
    assert (first.embedded, first.reused, first.removed) == (10, 0, 0)
    assert count(conn) == 10

    second = update_embeddings(conn, embedder)
    assert (second.embedded, second.reused, second.removed) == (0, 10, 0)
    assert len(embedder.calls) == 1  # the second run never called the model


def test_a_renamed_note_keeps_its_vector(conn, embedder, vault):
    update_embeddings(conn, embedder)
    (vault / "notes" / "embeddings.md").rename(vault / "notes" / "vector-embeddings.md")
    build_index(conn, vault)
    stats = update_embeddings(conn, embedder)
    assert (stats.embedded, stats.reused, stats.removed) == (0, 10, 0)


def test_edits_and_deletes_replace_and_drop_vectors(conn, embedder, vault):
    update_embeddings(conn, embedder)
    note = vault / "notes" / "bm25-ranking.md"
    note.write_text(note.read_text(encoding="utf-8") + "\nMore.\n", encoding="utf-8")
    (vault / "notes" / "orphan-note.md").unlink()
    build_index(conn, vault)
    stats = update_embeddings(conn, embedder)
    assert (stats.embedded, stats.reused, stats.removed) == (1, 8, 2)
    assert embedder.calls[-1][0].startswith("BM25 Ranking\n\n")
    assert count(conn) == 9


def test_each_model_keeps_its_own_vectors(conn, embedder):
    update_embeddings(conn, embedder)
    other = WordEmbedder(model="test/other", dim=32)
    assert update_embeddings(conn, other).embedded == 10
    assert count(conn) == 10 and count(conn, "test/other") == 10
