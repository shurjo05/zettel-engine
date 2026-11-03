import json
from collections import Counter

import pytest

from zk import maintain
from zk.cli import main
from zk.embed import update_embeddings
from zk.index import build_index, open_db


@pytest.fixture
def indexed(vault, tmp_path, embedder):
    def make():
        conn = open_db(tmp_path / "index.sqlite")
        build_index(conn, vault)
        update_embeddings(conn, embedder)
        return conn

    return vault, make, embedder


def write(vault, name, words, links=(), tags=()):
    body = " ".join(words) + "".join(f"\n\n- [[{target}]]" for target in links)
    (vault / "notes" / f"{name}.md").write_text(
        f"---\nid: '{name}'\ntitle: {name}\ntype: concept\ntags: {list(tags)}\n---\n"
        f"# {name}\n\n{body}\n",
        encoding="utf-8",
    )


def test_link_suggestions_skip_linked_pairs_and_duplicates(indexed):
    vault, make, embedder = indexed
    conn = make()
    pairs = maintain.link_suggestions(conn, embedder, at=0.3, per_note=2)
    found = {frozenset((p["a"], p["b"])) for p in pairs}
    # bm25-ranking links to reciprocal-rank-fusion; the duplicate-id pair is a duplicate.
    assert frozenset(("notes/bm25-ranking.md", "notes/reciprocal-rank-fusion.md")) not in found
    assert frozenset(("notes/duplicate-id-a.md", "notes/duplicate-id-b.md")) not in found
    counts = Counter(p for pair in pairs for p in (pair["a"], pair["b"]))
    assert max(counts.values()) <= 2
    assert [p["similarity"] for p in pairs] == sorted(
        (p["similarity"] for p in pairs), reverse=True
    )


def test_orphans_come_with_unlinked_candidates(indexed):
    vault, make, embedder = indexed
    conn = make()
    report = {o["path"]: o["candidates"] for o in maintain.orphan_report(conn, embedder)}
    assert "notes/orphan-note.md" in report
    assert "notes/reciprocal-rank-fusion.md" not in report  # bm25-ranking links to it
    candidates = report["notes/orphan-note.md"]
    assert len(candidates) == 3 and "notes/orphan-note.md" not in [c["path"] for c in candidates]


def test_clusters_split_topics_and_spot_an_existing_map(indexed):
    vault, make, embedder = indexed
    for f in (vault / "notes").glob("*.md"):
        f.unlink()
    garden = ["soil", "compost", "seedling", "watering", "mulch", "pruning", "harvest"]
    sailing = ["mainsail", "jib", "tacking", "mooring", "anchor", "keel", "rudder"]
    for i in range(5):
        write(vault, f"garden-{i}", garden[i:] + garden[:i][:2], tags=["garden"])
        write(vault, f"sail-{i}", sailing[i:] + sailing[:i][:2], tags=["sailing"])
    # sail-0 links to every other sailing note: that cluster already has its map.
    write(
        vault,
        "sail-0",
        sailing,
        links=[f"sail-{i}" for i in range(1, 5)],
        tags=["sailing"],
    )
    conn = make()
    groups = maintain.clusters(conn, embedder, k=4, at=0.3, min_size=4)
    by_tag = {g["shared_tags"][0]: g for g in groups}
    assert set(by_tag) == {"garden", "sailing"}
    assert by_tag["garden"]["notes"] == [f"notes/garden-{i}.md" for i in range(5)]
    assert by_tag["garden"]["map"] is None
    assert by_tag["sailing"]["map"] == "notes/sail-0.md"


def test_suggest_command(vault, tmp_path, capsys):
    db = str(tmp_path / "i.sqlite")
    assert main(["suggest", str(vault), "--db", db]) == 0
    out = capsys.readouterr().out
    assert "orphans (nothing links to them) and notes that might" in out
    assert "notes/orphan-note.md  <-  " in out
    main(["suggest", str(vault), "--db", db, "--json"])
    data = json.loads(capsys.readouterr().out)
    assert set(data) == {"links", "orphans", "maps"}
