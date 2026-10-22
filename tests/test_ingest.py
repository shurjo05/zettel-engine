import subprocess

import pytest

from zk import ingest
from zk.embed import update_embeddings
from zk.index import build_index, health_report, open_db
from zk.proposals import Proposal
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


def proposal(**changes):
    fields = dict(
        title="Rank Fusion Needs Only Ranks",
        type="concept",
        summary="Fusing ranked lists by rank avoids comparing scores on different scales.",
        sources=["conversation:2026-10-22"],
        links={"reciprocal-rank-fusion": "the method this explains"},
    )
    fields.update(changes)
    return Proposal(**fields)


def test_inbox_listing_skips_proposals_and_processed(setup):
    vault, conn, embedder = setup
    ingest.propose(vault, conn, embedder, proposal(), capture=CAPTURE)
    names = [c["name"] for c in ingest.list_inbox(vault)]
    assert CAPTURE in names
    assert all(not n.startswith("rank-fusion") for n in names)
    ours = next(c for c in ingest.list_inbox(vault) if c["name"] == CAPTURE)
    assert (ours["source"], ours["project"]) == ("claude-code:abcd1234", "demo")


def test_propose_saves_a_proposal_only(setup):
    vault, conn, embedder = setup
    result = ingest.propose(vault, conn, embedder, proposal(), capture=CAPTURE)
    assert result["proposal"] == "rank-fusion-needs-only-ranks.md"
    assert (vault / "inbox" / "proposals" / result["proposal"]).is_file()
    assert not (vault / "notes" / result["proposal"]).exists()
    assert f"capture:{CAPTURE}" in result["note"]
    assert {d["verdict"] for d in result["duplicates"]} <= {"likely-duplicate", "related", "new"}
    # A second proposal with the same title gets its own file instead of replacing the first.
    again = ingest.propose(vault, conn, embedder, proposal(), capture=CAPTURE)
    assert again["proposal"] == "rank-fusion-needs-only-ranks-2.md"


def test_identical_text_is_flagged_as_a_duplicate(setup):
    vault, conn, embedder = setup
    row = conn.execute(
        "SELECT title, body FROM notes WHERE path = 'notes/bm25-ranking.md'"
    ).fetchone()
    top = ingest.check_duplicates(conn, embedder, row["title"], row["body"])[0]
    assert top["path"] == "notes/bm25-ranking.md"
    assert top["verdict"] == "likely-duplicate"


def test_apply_moves_the_note_and_retires_the_capture(setup):
    vault, conn, embedder = setup
    name = ingest.propose(vault, conn, embedder, proposal(), capture=CAPTURE)["proposal"]
    result = ingest.apply(vault, name)
    assert result == {"note": f"notes/{name}", "commit": None}  # not a git repo
    assert (vault / "notes" / name).is_file()
    assert not (vault / "inbox" / "proposals" / name).exists()
    assert (vault / "inbox" / "processed" / CAPTURE).is_file()
    assert not (vault / "inbox" / CAPTURE).exists()

    build_index(conn, vault)
    issues = [i for i in health_report(conn).issues if i["path"] == f"notes/{name}"]
    assert issues == []
    with pytest.raises(FileNotFoundError):
        ingest.apply(vault, name)


def test_apply_commits_only_its_own_files(setup):
    vault, conn, embedder = setup

    def git(*args):
        return subprocess.run(
            ["git", "-C", str(vault), *args], capture_output=True, text=True, check=True
        )

    git("init", "-q")
    git("config", "user.email", "test@example.com")
    git("config", "user.name", "Test")
    git("add", "-A")
    git("commit", "-q", "-m", "start")
    (vault / "notes" / "embeddings.md").write_text("edited by hand\n", encoding="utf-8")

    name = ingest.propose(vault, conn, embedder, proposal(), capture=CAPTURE)["proposal"]
    result = ingest.apply(vault, name)
    assert result["commit"]
    changed = git("show", "--name-status", "--format=", "HEAD").stdout.split("\n")
    assert f"A\tnotes/{name}" in changed
    assert any(line.endswith(f"inbox/processed/{CAPTURE}") for line in changed)
    assert not any("embeddings.md" in line for line in changed)
    assert "notes/embeddings.md" in git("status", "--porcelain").stdout  # still yours to commit


def test_discard_sets_things_aside_without_deleting(setup):
    vault, conn, embedder = setup
    name = ingest.propose(vault, conn, embedder, proposal())["proposal"]
    assert ingest.discard(vault, name) == f"inbox/rejected/{name}"
    assert ingest.discard(vault, CAPTURE, kind="capture") == f"inbox/processed/{CAPTURE}"
    assert (vault / "inbox" / "rejected" / name).is_file()


def test_names_cannot_escape_the_inbox(setup):
    vault, conn, embedder = setup
    for bad in ("../notes/bm25-ranking.md", "notes/x.md", ".hidden"):
        with pytest.raises((ValueError, FileNotFoundError, VaultWriteError)):
            ingest.read_capture(vault, bad)
        with pytest.raises((ValueError, FileNotFoundError, VaultWriteError)):
            ingest.apply(vault, bad)
