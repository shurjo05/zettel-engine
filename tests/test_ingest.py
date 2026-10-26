import subprocess
from pathlib import Path

import pytest
import yaml

from zk import ingest
from zk.embed import update_embeddings
from zk.index import build_index, health_report, open_db
from zk.proposals import InvalidProposal, Proposal
from zk.vault import VaultWriteError

CAPTURE = "2026-10-22-1900-claude-abcd1234.md"


@pytest.fixture
def setup(vault, tmp_path, embedder):
    (vault / "inbox" / CAPTURE).write_text(
        '---\nkind: "capture"\nsource: "claude-code:abcd1234"\nproject: "demo"\n---\n'
        "# Capture\n\n## You\n\nHow do I fuse ranked lists?\n",
        encoding="utf-8",
    )
    conn = open_db(tmp_path / "index.sqlite")
    build_index(conn, vault)
    update_embeddings(conn, embedder)
    return vault, conn, embedder


def draft(**changes):
    fields = dict(
        title="Rank Fusion Needs Only Ranks",
        type="concept",
        summary="Fusing ranked lists by rank avoids comparing scores on different scales.",
        sources=["conversation:2026-10-22"],
        links={"reciprocal-rank-fusion": "the method this explains"},
        questions=["why fuse by position instead of by score?"],
    )
    fields.update(changes)
    return Proposal(**fields)


def refresher(vault, conn, embedder):
    def refresh():
        build_index(conn, vault)
        update_embeddings(conn, embedder)

    return refresh


def git(vault, *args):
    return subprocess.run(
        ["git", "-C", str(vault), *args], capture_output=True, text=True, check=True
    )


def init_repo(vault):
    git(vault, "init", "-q")
    git(vault, "config", "user.email", "test@example.com")
    git(vault, "config", "user.name", "Test")
    git(vault, "add", "-A")
    git(vault, "commit", "-q", "-m", "start")


def test_inbox_listing_lists_captures_only(setup):
    vault, conn, embedder = setup
    ingest.ingest(vault, conn, embedder, [draft()])
    (vault / "inbox" / "held").mkdir(exist_ok=True)
    names = [c["name"] for c in ingest.list_inbox(vault)]
    assert CAPTURE in names
    assert all(n.endswith("capture.md") or n == CAPTURE for n in names)  # no notes or warnings
    ours = next(c for c in ingest.list_inbox(vault) if c["name"] == CAPTURE)
    assert (ours["source"], ours["project"]) == ("claude-code:abcd1234", "demo")


def test_identical_text_is_flagged_as_a_duplicate(setup):
    vault, conn, embedder = setup
    row = conn.execute(
        "SELECT title, body FROM notes WHERE path = 'notes/bm25-ranking.md'"
    ).fetchone()
    top = ingest.check_duplicates(conn, embedder, row["title"], row["body"])[0]
    assert top["path"] == "notes/bm25-ranking.md"
    assert top["verdict"] == "likely-duplicate"


def test_ingest_writes_notes_questions_and_retires_the_capture(setup):
    vault, conn, embedder = setup
    other = draft(
        title="Keyword Search Finds Identifiers",
        summary="BM25 matches exact words, so names and error codes are found reliably.",
        links={},
        questions=["how do I look up an error code?", "what is keyword search good at?"],
    )
    result = ingest.ingest(
        vault,
        conn,
        embedder,
        [draft(), other],
        capture=CAPTURE,
        refresh=refresher(vault, conn, embedder),
    )
    assert result["notes"] == [
        "notes/rank-fusion-needs-only-ranks.md",
        "notes/keyword-search-finds-identifiers.md",
    ]
    assert result["held"] == [] and result["commit"] is None  # not a git repo
    assert (vault / "inbox" / "processed" / CAPTURE).is_file()
    assert not (vault / "inbox" / CAPTURE).exists()

    texts = [(vault / p).read_text(encoding="utf-8") for p in result["notes"]]
    assert all(f"capture:{CAPTURE}" in t for t in texts)
    ids = {
        conn.execute("SELECT id FROM notes WHERE path = ?", (p,)).fetchone()[0]
        for p in result["notes"]
    }
    assert len(ids) == 2  # one run, distinct ids
    assert [i for i in health_report(conn).issues if i["path"] in result["notes"]] == []

    saved = yaml.safe_load((vault / result["questions"]).read_text(encoding="utf-8"))
    assert saved["source"] == f"capture:{CAPTURE}"
    assert [q["relevant"] for q in saved["queries"]] == [
        ["rank-fusion-needs-only-ranks"],
        ["keyword-search-finds-identifiers"],
        ["keyword-search-finds-identifiers"],
    ]
    assert {q["kind"] for q in saved["queries"]} <= {"generated", "holdout"}
    # The second note links to nothing, so the run warns about it.
    assert any(
        "keyword-search-finds-identifiers.md is linked to no other note" in w
        for w in result["warnings"]
    )
    assert (vault / result["warnings_file"]).is_file()


def test_duplicates_are_held_not_written(setup):
    vault, conn, embedder = setup
    row = conn.execute(
        "SELECT title, summary, body FROM notes WHERE path = 'notes/bm25-ranking.md'"
    ).fetchone()
    copy = draft(
        title=row["title"] + " Again",
        summary=row["summary"],
        body=row["body"].split("\n", 2)[-1],
        links={},
    )
    twins = [draft(), draft(title="Rank Fusion Needs Only Ranks, Again")]
    result = ingest.ingest(vault, conn, embedder, [copy, *twins])
    assert result["notes"] == ["notes/rank-fusion-needs-only-ranks.md"]
    held = {h["path"]: h["duplicate_of"] for h in result["held"]}
    assert held == {
        "inbox/held/bm25-ranking-again.md": "notes/bm25-ranking.md",
        "inbox/held/rank-fusion-needs-only-ranks-again.md": "notes/rank-fusion-needs-only-ranks.md",
    }
    assert all((vault / p).is_file() for p in held)
    assert sum("was held, not added" in w for w in result["warnings"]) == 2
    assert [h["name"] for h in ingest.list_held(vault)] == sorted(Path(p).name for p in held)

    released = ingest.apply_held(vault, "bm25-ranking-again.md")
    assert released["note"] == "notes/bm25-ranking-again.md"
    assert ingest.discard(vault, "rank-fusion-needs-only-ranks-again.md") == (
        "inbox/rejected/rank-fusion-needs-only-ranks-again.md"
    )


def test_one_invalid_note_stops_the_whole_run(setup):
    vault, conn, embedder = setup
    before = sorted(p for p in vault.rglob("*") if p.is_file())
    with pytest.raises(InvalidProposal) as err:
        ingest.ingest(
            vault,
            conn,
            embedder,
            [draft(), draft(title="No Questions", questions=[]), draft(type="idea")],
            capture=CAPTURE,
        )
    assert "note 2: give one to 3 test questions" in str(err.value)
    assert "note 3" in str(err.value) and "type must be one of" in str(err.value)
    assert sorted(p for p in vault.rglob("*") if p.is_file()) == before


def test_a_run_is_one_commit_of_its_own_files(setup):
    vault, conn, embedder = setup
    init_repo(vault)
    (vault / "notes" / "embeddings.md").write_text("edited by hand\n", encoding="utf-8")

    other = draft(title="Second Idea", summary="Another idea from the same session.", links={})
    result = ingest.ingest(
        vault,
        conn,
        embedder,
        [draft(), other],
        capture=CAPTURE,
        refresh=refresher(vault, conn, embedder),
    )
    assert result["commit"]
    assert git(vault, "rev-list", "--count", "HEAD").stdout.strip() == "2"
    message = git(vault, "log", "-1", "--format=%s").stdout.strip()
    assert message == f"zk: ingest {CAPTURE}: 2 notes"
    changed = git(vault, "show", "--name-status", "--format=", "HEAD").stdout.split("\n")
    for path in [*result["notes"], result["questions"], result["warnings_file"]]:
        assert f"A\t{path}" in changed
    assert any(line.endswith(f"inbox/processed/{CAPTURE}") for line in changed)
    assert not any("embeddings.md" in line for line in changed)
    assert "notes/embeddings.md" in git(vault, "status", "--porcelain").stdout  # still yours


def test_discard_sets_a_capture_aside_without_deleting(setup):
    vault, conn, embedder = setup
    assert ingest.discard(vault, CAPTURE, kind="capture") == f"inbox/processed/{CAPTURE}"
    assert (vault / "inbox" / "processed" / CAPTURE).is_file()


def test_names_cannot_escape_the_inbox(setup):
    vault, conn, embedder = setup
    for bad in ("../notes/bm25-ranking.md", "notes/x.md", ".hidden"):
        with pytest.raises((ValueError, FileNotFoundError, VaultWriteError)):
            ingest.read_capture(vault, bad)
        with pytest.raises((ValueError, FileNotFoundError, VaultWriteError)):
            ingest.apply_held(vault, bad)
        with pytest.raises((ValueError, FileNotFoundError, VaultWriteError)):
            ingest.ingest(vault, conn, embedder, [draft()], capture=bad)
