import pytest

from zk.index import build_index, open_db
from zk.lookup import AmbiguousNote, NoteNotFound, get_note, related, resolve


@pytest.fixture
def conn(vault):
    c = open_db(":memory:")
    build_index(c, vault)
    return c


@pytest.mark.parametrize(
    "name",
    [
        "notes/bm25-ranking.md",
        "notes/bm25-ranking",
        "BM25-Ranking",
        "bm25-ranking.md",
        "20261015100000",
        "bm25 ranking",
        "notes\\bm25-ranking.md",
    ],
)
def test_a_note_can_be_named_by_path_file_name_id_or_title(conn, name):
    assert resolve(conn, name) == "notes/bm25-ranking.md"


def test_ambiguous_and_unknown_names_raise_helpful_errors(conn):
    with pytest.raises(AmbiguousNote) as exc:
        resolve(conn, "20261015100500")  # two notes share this id
    assert exc.value.paths == ["notes/duplicate-id-a.md", "notes/duplicate-id-b.md"]

    with pytest.raises(NoteNotFound) as exc:
        resolve(conn, "fusion")
    assert exc.value.suggestions[0] == "notes/reciprocal-rank-fusion.md"
    assert "Closest matches" in str(exc.value)


def test_get_note_returns_body_metadata_and_links(conn):
    note = get_note(conn, "bm25-ranking")
    assert note["title"] == "BM25 Ranking"
    assert note["tags"] == ["retrieval", "keyword-search"]
    assert note["body"].startswith("# BM25 Ranking")
    assert not note["truncated"]
    assert [n["title"] for n in note["links_out"]] == ["Reciprocal Rank Fusion"]
    # code-block-note links here by id.
    assert [n["title"] for n in note["links_in"]] == ["Code Block Note", "Reciprocal Rank Fusion"]
    assert get_note(conn, "embeddings")["broken_links"] == ["hnsw-index"]


def test_long_bodies_are_truncated_and_flagged(conn):
    note = get_note(conn, "bm25-ranking", max_chars=20)
    assert len(note["body"]) == 20
    assert note["truncated"]


def test_related_lists_links_then_backlinks_then_shared_tags(conn):
    result = related(conn, "bm25-ranking")
    rows = [(n["path"], n["relations"]) for n in result["related"]]
    assert rows == [
        (
            "notes/reciprocal-rank-fusion.md",
            ["links to", "linked from", "shares tags: retrieval"],
        ),
        ("notes/code-block-note.md", ["linked from"]),
        ("notes/embeddings.md", ["shares tags: retrieval"]),
    ]
    assert result["total"] == 3
    assert "body" not in result["related"][0]
    assert result["related"][0]["summary"]


def test_related_respects_k(conn):
    result = related(conn, "bm25-ranking", k=1)
    assert len(result["related"]) == 1
    assert result["total"] == 3
