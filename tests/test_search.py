import pytest

from zk.embed import update_embeddings
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


def test_semantic_search_ranks_by_cosine_similarity(conn, embedder):
    assert search(conn, "merging lists", mode="semantic", embedder=embedder) == []  # no vectors yet
    update_embeddings(conn, embedder)
    hits = search(conn, "merging ranked lists", k=3, mode="semantic", embedder=embedder)
    assert hits[0].path == "notes/reciprocal-rank-fusion.md"
    assert [h.rank for h in hits] == [1, 2, 3]
    assert hits[0].score >= hits[1].score >= hits[2].score
    assert -1.0 <= hits[-1].score <= 1.0


def test_semantic_search_applies_filters(conn, embedder):
    update_embeddings(conn, embedder)
    hits = search(conn, "ranking", k=10, mode="semantic", embedder=embedder, tag="embeddings")
    assert [h.path for h in hits] == ["notes/embeddings.md"]
    hits = search(conn, "ranking", k=10, mode="semantic", embedder=embedder, type="pattern")
    assert {h.type for h in hits} == {"pattern"}


def test_bad_mode_or_missing_embedder_is_an_error(conn):
    with pytest.raises(ValueError, match="unknown search mode"):
        search(conn, "x", mode="fuzzy")
    with pytest.raises(ValueError, match="needs an embedder"):
        search(conn, "x", mode="semantic")


def test_hybrid_fuses_ranks_from_both_lists(conn, embedder):
    from zk.search import RRF_K, keyword_search, semantic_search

    update_embeddings(conn, embedder)
    query = "merging ranked lists"
    kw = {h.path: h.rank for h in keyword_search(conn, query, k=50)}
    sem = {h.path: h.rank for h in semantic_search(conn, embedder, query, k=50)}
    hits = search(conn, query, k=10, mode="hybrid", embedder=embedder)

    expected = {p: sum(1 / (RRF_K + r[p]) for r in (kw, sem) if p in r) for p in set(kw) | set(sem)}
    assert [h.path for h in hits] == sorted(expected, key=lambda p: (-expected[p], p))[:10]
    assert hits[0].path == "notes/reciprocal-rank-fusion.md"
    assert hits[0].score == round(expected[hits[0].path], 6)
    assert "[" in hits[0].snippet  # the keyword snippet survives fusion


def test_hybrid_keeps_notes_only_one_list_finds(conn, embedder):
    from zk.search import keyword_search

    update_embeddings(conn, embedder)
    # "zebra" matches no note's words, so only the semantic list has results.
    assert keyword_search(conn, "zebra") == []
    assert search(conn, "zebra", k=3, mode="hybrid", embedder=embedder)
    with pytest.raises(ValueError, match="hybrid search needs an embedder"):
        search(conn, "x", mode="hybrid")
