import hashlib
from pathlib import Path

from zk.index import build_index, health_report, iter_note_files, open_db


def snapshot(folder: Path) -> dict[str, str]:
    return {
        p.relative_to(folder).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in folder.rglob("*")
        if p.is_file()
    }


def test_excludes_inbox_and_hidden_folders(vault):
    files = {p.relative_to(vault).as_posix() for p in iter_note_files(vault)}
    assert "inbox/2026-10-15-capture.md" not in files
    assert ".obsidian/hidden.md" not in files
    assert len(files) == 10


def test_build_then_rebuild_is_incremental(vault, tmp_path):
    conn = open_db(tmp_path / "index.sqlite")
    first = build_index(conn, vault)
    assert (first.added, first.updated, first.unchanged) == (10, 0, 0)

    second = build_index(conn, vault)
    assert (second.added, second.updated, second.removed, second.unchanged) == (0, 0, 0, 10)

    (vault / "notes" / "embeddings.md").write_text("---\nid: 1\n---\n# Changed\n", encoding="utf-8")
    (vault / "notes" / "orphan-note.md").unlink()
    third = build_index(conn, vault)
    assert (third.updated, third.removed, third.unchanged) == (1, 1, 8)


def test_indexing_never_writes_to_the_vault(vault, tmp_path):
    before = snapshot(vault)
    conn = open_db(tmp_path / "index.sqlite")
    build_index(conn, vault)
    health_report(conn)
    assert snapshot(vault) == before


def test_health_report_on_fixture_vault(vault):
    conn = open_db(":memory:")
    build_index(conn, vault)
    r = health_report(conn)

    assert r.notes == 10
    assert r.duplicate_ids == {
        "20261015100500": ["notes/duplicate-id-a.md", "notes/duplicate-id-b.md"]
    }
    assert r.broken_links == [{"path": "notes/embeddings.md", "target": "hnsw-index"}]
    assert r.ambiguous_links == []
    assert r.orphans == [
        "notes/bad-yaml.md",
        "notes/code-block-note.md",
        "notes/legacy-dialect.md",
        "notes/no-frontmatter.md",
        "notes/orphan-note.md",
    ]
    errors = {(i["path"], i["code"]) for i in r.issues if i["level"] == "error"}
    assert errors == {
        ("notes/bad-yaml.md", "invalid-yaml"),
        ("notes/duplicate-id-b.md", "invalid-type"),
        ("notes/no-frontmatter.md", "missing-frontmatter"),
    }
    assert r.errors == 4  # three note errors plus one duplicate id


def test_links_resolve_case_insensitively_and_by_id(vault):
    conn = open_db(":memory:")
    build_index(conn, vault)
    rows = {
        (r["src_path"], r["target"]): r["dst_path"]
        for r in conn.execute("SELECT src_path, target, dst_path FROM links")
    }
    assert rows[("notes/reciprocal-rank-fusion.md", "BM25-Ranking")] == "notes/bm25-ranking.md"
    assert rows[("notes/code-block-note.md", "20261015100000")] == "notes/bm25-ranking.md"


def test_same_file_name_in_two_folders_is_ambiguous(vault):
    (vault / "other").mkdir()
    (vault / "other" / "embeddings.md").write_text("---\nid: 9\n---\n# Other\n", encoding="utf-8")
    conn = open_db(":memory:")
    build_index(conn, vault)
    r = health_report(conn)
    assert {"path": "notes/orphan-note.md", "target": "embeddings"} in r.ambiguous_links


def test_switching_vaults_resets_the_index(vault, tmp_path):
    conn = open_db(tmp_path / "index.sqlite")
    build_index(conn, vault)
    empty = tmp_path / "empty-vault"
    empty.mkdir()
    stats = build_index(conn, empty)
    assert stats.total == 0
    assert health_report(conn).notes == 0
