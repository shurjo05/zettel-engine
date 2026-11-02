import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from zk import migrate
from zk.embed import update_embeddings
from zk.index import build_index, health_report, open_db

FIXTURES = Path(__file__).parent / "fixtures"
QUERIES = FIXTURES / "legacy-queries.yaml"


@pytest.fixture
def legacy(tmp_path):
    copy = tmp_path / "legacy"
    shutil.copytree(FIXTURES / "legacy", copy)
    return copy


@pytest.fixture
def target(vault, tmp_path, embedder):
    conn = open_db(tmp_path / "index.sqlite")
    build_index(conn, vault)
    update_embeddings(conn, embedder)
    return vault, conn, embedder


def snapshot(folder):
    return {p: p.read_bytes() for p in folder.rglob("*") if p.is_file()}


def test_every_legacy_file_is_accounted_for(legacy):
    items = {i.path: (i.status, i.reason) for i in migrate.plan(legacy)}
    assert len(items) == 8
    assert items["ai-workspace/4_system/dedupe.py"] == ("skipped", "not a note (not Markdown)")
    assert items["ai-workspace/4_system/docs/pipeline.md"][0] == "skipped"
    assert items["ai-workspace/3_resources/templates/atomic_note.md"] == ("skipped", "template")
    assert items["{{source_note}}.md"] == ("skipped", "empty file")
    assert sum(s == "imported" for s, _ in items.values()) == 4
    # Accepted notes come first, so the inbox copy of a duplicate is the one held.
    assert migrate.plan(legacy)[0].path.startswith("ai-workspace/1_atomic/")


def test_import_converts_holds_duplicates_and_repairs_links(legacy, target):
    vault, conn, embedder = target
    before = snapshot(legacy)
    report = migrate.import_legacy(legacy, vault, conn, embedder, queries=QUERIES)
    assert snapshot(legacy) == before  # the legacy vault is only read

    by_path = {i.path: i for i in report.items}
    cache = by_path[
        "ai-workspace/1_atomic/patterns/20260301100000-cache-embeddings-by-content-hash.md"
    ]
    assert cache.status == "imported"
    assert cache.target == "notes/cache-embeddings-by-content-hash.md"
    copy = by_path["ai-workspace/0_inbox/raw/20260506140002-batch-model-calls-copy.md"]
    assert copy.status == "held" and "batch-model-calls-for-throughput" in copy.reason
    assert (vault / copy.target).is_file()

    text = (vault / cache.target).read_text(encoding="utf-8")
    front = yaml.safe_load(text.split("---")[1])
    assert front["id"] == "20260301100000"
    assert (front["created"], front["updated"]) == ("2026-03-01", "2026-03-02")
    assert front["sources"] == [
        "conversation:2026-03-01",
        f"legacy:{cache.path}",
    ]
    assert "domain" not in front and "confidence" not in front
    assert "## Details" in text and "### Example" in text and "## Summary" not in text
    # Drifted links now point at the new names; the held copy's links go to its twin.
    assert "- [[event-handlers-in-c]]" in text
    assert "- [[batch-model-calls-for-throughput]]: embedding many notes at once" in text

    csharp = (vault / "notes" / "event-handlers-in-c.md").read_text(encoding="utf-8")
    assert "# Event Handlers in C#" in csharp and "type: pattern" in csharp  # snippet -> pattern

    # Five drifted links and two to a note that never existed (the held copy repeats two).
    assert report.links["legacy_broken"] == 7
    assert report.links["repaired"] == 5
    assert report.still_broken == [
        ("notes/batch-model-calls-for-throughput.md", "a-note-that-never-existed")
    ]

    build_index(conn, vault)
    imported = [i.target for i in report.items if i.status == "imported"]
    assert [
        i for i in health_report(conn).issues if i["path"] in imported and i["level"] == "error"
    ] == []

    saved = yaml.safe_load((vault / report.queries).read_text(encoding="utf-8"))
    assert saved["queries"][0]["relevant"] == ["cache-embeddings-by-content-hash"]
    report_text = (vault / migrate.REPORTS_DIR).glob("import-*.md")
    assert "8 legacy files: 3 imported, 1 held" in next(report_text).read_text(encoding="utf-8")


def test_import_is_one_commit(legacy, target):
    vault, conn, embedder = target

    def git(*args):
        return subprocess.run(
            ["git", "-C", str(vault), *args], capture_output=True, text=True, check=True
        )

    git("init", "-q")
    git("config", "user.email", "test@example.com")
    git("config", "user.name", "Test")
    git("add", "-A")
    git("commit", "-q", "-m", "start")
    report = migrate.import_legacy(legacy, vault, conn, embedder)
    assert report.commit
    assert git("log", "-1", "--format=%s").stdout.strip() == (
        "zk: import 3 legacy notes (1 held as duplicates)"
    )
    assert git("status", "--porcelain").stdout == ""


def test_dry_run_writes_nothing(legacy, vault, embedder):
    before = snapshot(vault)
    report = migrate.dry_run(legacy, vault, embedder, QUERIES)
    assert report.count("imported") == 3
    assert snapshot(vault) == before


def test_a_note_that_cannot_convert_stops_the_import(legacy, target):
    vault, conn, embedder = target
    bad = legacy / "ai-workspace" / "1_atomic" / "concepts" / "bad.md"
    bad.write_text(
        "---\ntitle: Bad\ntype: concept\n---\n# Bad\n\n## Summary\n# Heading\n", encoding="utf-8"
    )
    before = snapshot(vault)
    with pytest.raises(migrate.InvalidProposal, match="bad.md"):
        migrate.import_legacy(legacy, vault, conn, embedder)
    assert snapshot(vault) == before


def test_the_vault_and_legacy_folder_must_be_separate(vault, tmp_path, embedder):
    conn = open_db(tmp_path / "i.sqlite")
    with pytest.raises(ValueError, match="must not contain"):
        migrate.import_legacy(vault, vault / "notes", conn, embedder)


def test_links_to_a_held_note_keep_its_name(legacy, target, tmp_path):
    # A false hold must not move a link's meaning to the merely similar twin.
    vault, conn, embedder = target
    cache = (
        legacy / "ai-workspace/1_atomic/patterns/20260301100000-cache-embeddings-by-content-hash.md"
    )
    cache.write_text(
        cache.read_text(encoding="utf-8").replace(
            "## Source Note",
            "- [[20260506140002-batch-model-calls-copy]] - a copy\n\n## Source Note",
        ),
        encoding="utf-8",
    )
    queries = tmp_path / "queries.yaml"
    queries.write_text(
        yaml.safe_dump(
            {
                "queries": [
                    {
                        "query": "batching calls",
                        "relevant": [
                            "20260506140002-batch-model-calls-copy",
                            "20260302110000-batch-model-calls-for-throughput",
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    report = migrate.import_legacy(legacy, vault, conn, embedder, queries)

    text = (vault / "notes/cache-embeddings-by-content-hash.md").read_text(encoding="utf-8")
    assert "- [[batch-model-calls-copy]]: a copy" in text
    assert (vault / "inbox/held/batch-model-calls-copy.md").is_file()
    assert report.to_held == [
        ("notes/cache-embeddings-by-content-hash.md", "batch-model-calls-copy")
    ]
    assert "and 1 point at a held note" in migrate.summary(report)
    saved = yaml.safe_load((vault / report.queries).read_text(encoding="utf-8"))
    # The held note's twin answers the question meanwhile; the twin is listed once.
    assert saved["queries"][0]["relevant"] == [
        "batch-model-calls-copy",
        "batch-model-calls-for-throughput",
    ]
