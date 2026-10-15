from pathlib import Path

import pytest

from zk.evaluate import format_report, load_queries, run_eval
from zk.index import build_index, open_db

FIXTURE_QUERIES = Path(__file__).parent.parent / "eval" / "fixture-queries.yaml"


@pytest.fixture
def conn(vault):
    c = open_db(":memory:")
    build_index(c, vault)
    return c


def test_metrics_on_known_ranks(conn):
    queries = [
        {"query": "BM25 ranking", "relevant": ["bm25-ranking"], "kind": "a"},
        # Two relevant notes, one of which is not retrievable for this query.
        {"query": "reciprocal rank fusion", "relevant": ["reciprocal-rank-fusion", "orphan-note"]},
        {"query": "zzzzqqq", "relevant": ["embeddings"], "kind": "a"},
    ]
    report = run_eval(conn, queries)
    first, second, third = report.results
    assert first.first_hit_rank == 1 and first.recall[1] == 1.0
    assert second.first_hit_rank == 1 and second.recall[10] == 0.5
    assert third.first_hit_rank is None and third.recall[10] == 0.0

    s = report.summary()
    assert s["queries"] == 3
    assert s["mrr"] == round((1 + 1 + 0) / 3, 3)
    assert s["recall@10"] == round((1 + 0.5 + 0) / 3, 3)
    assert report.summary("a")["queries"] == 2


def test_labels_resolve_by_id_and_unknown_labels_are_reported(conn):
    report = run_eval(conn, [{"query": "BM25", "relevant": ["20261015100000", "no-such-note"]}])
    r = report.results[0]
    assert r.relevant == ["notes/bm25-ranking.md"]
    assert r.unresolved == ["no-such-note"]


def test_fixture_query_set_passes(conn):
    report = run_eval(conn, load_queries(FIXTURE_QUERIES))
    assert report.summary()["mrr"] == 1.0
    assert "R@1" in format_report(report)


def test_bad_query_file_is_rejected(tmp_path):
    bad = tmp_path / "q.yaml"
    bad.write_text("queries:\n  - query: no labels\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_queries(bad)
