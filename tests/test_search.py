import pytest

from zk.index import build_index, open_db
from zk.search import fts_query, search


@pytest.fixture
def conn(vault):
    c = open_db(":memory:")
    build_index(c, vault)
    return c


def test_fts_query_quotes_words_and_ors_them():
    assert fts_query("How do I merge? merge!") == '"how" OR "do" OR "i" OR "merge"'
    assert fts_query("  ?! ") is None


def test_fts_syntax_in_user_input_is_harmless(conn):
    # Unquoted, these would be FTS5 operators or a syntax error.
    for q in ['NEAR(bm25 "', "bm25 AND -ranking*", "title:embeddings", '"""']:
        search(conn, q)


def test_title_match_ranks_first(conn):
    hits = search(conn, "BM25 ranking")
    assert hits[0].path == "notes/bm25-ranking.md"
    assert [h.rank for h in hits] == list(range(1, len(hits) + 1))
    assert hits[0].score >= hits[-1].score


def test_stemming_matches_word_forms(conn):
    # "merging" and "lists" stem to the same roots as "merges" and "list".
    assert search(conn, "merging lists")[0].path == "notes/reciprocal-rank-fusion.md"


def test_aliases_and_tags_are_searchable(conn):
    assert search(conn, "RRF")[0].path == "notes/reciprocal-rank-fusion.md"
    assert search(conn, "okapi")[0].path == "notes/bm25-ranking.md"


def test_filters(conn):
    assert {h.type for h in search(conn, "embeddings", k=10, type="concept")} == {"concept"}
    tagged = search(conn, "retrieval", k=10, tag="hybrid-search")
    assert [h.path for h in tagged] == ["notes/reciprocal-rank-fusion.md"]


def test_no_match_and_k_limit(conn):
    assert search(conn, "zzzzqqq") == []
    assert len(search(conn, "note", k=2)) == 2
    assert search(conn, "note", k=0) == []


def test_search_sees_edits_and_deletes(conn, vault):
    note = vault / "notes" / "orphan-note.md"
    note.write_text(note.read_text(encoding="utf-8") + "\nquasar\n", encoding="utf-8")
    build_index(conn, vault)
    assert search(conn, "quasar")[0].path == "notes/orphan-note.md"

    note.unlink()
    build_index(conn, vault)
    assert search(conn, "quasar") == []
