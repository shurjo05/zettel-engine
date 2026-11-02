import subprocess

import pytest
import yaml

from zk import ingest, resolve, vault
from zk.embed import update_embeddings
from zk.index import build_index, health_report, open_db
from zk.proposals import InvalidProposal, Proposal

HELD = """\
---
id: '20261101090000'
title: Okapi Scoring
type: concept
tags: [retrieval]
sources: ['conversation:2026-11-01']
created: '2026-11-01'
updated: '2026-11-01'
---
# Okapi Scoring

BM25 scores a document by summing term weights that saturate with term frequency.

## Details

The saturation means the tenth mention of a word adds little.
"""


@pytest.fixture
def setup(vault, tmp_path, embedder):
    (vault / "inbox" / "held").mkdir(parents=True)
    # Bytes, so Windows doesn't turn the newlines into CRLF (vaults use LF).
    (vault / "inbox" / "held" / "okapi-scoring.md").write_bytes(HELD.encode("utf-8"))
    # A note linking to the held one, and a test question answered by it.
    (vault / "notes" / "embeddings.md").write_text(
        (vault / "notes" / "embeddings.md").read_text(encoding="utf-8")
        + "\nSee [[okapi-scoring|Okapi]] and [[okapi-scoring#Details]].\n",
        encoding="utf-8",
    )
    (vault / "eval" / "questions").mkdir(parents=True)
    (vault / "eval" / "questions" / "run.yaml").write_text(
        "# kept comment\nqueries:\n- query: what is okapi-scoring about\n  relevant:\n"
        "  - okapi-scoring\n  kind: generated\n",
        encoding="utf-8",
    )
    conn = open_db(tmp_path / "index.sqlite")
    build_index(conn, vault)
    update_embeddings(conn, embedder)
    return vault, conn, embedder


def merged():
    return Proposal(
        title="BM25 Ranking",
        type="concept",
        summary="BM25 scores a document for a query by summing term weights that saturate.",
        body="BM25 improves on TF-IDF. Saturation means the tenth mention of a word adds little.",
        links={"reciprocal-rank-fusion": "fuses BM25 results with vector results"},
    )


def test_held_pair_shows_both_notes(setup):
    vault_dir, conn, embedder = setup
    pair = resolve.held_pair(vault_dir, conn, embedder, "okapi-scoring")
    assert pair["held"].startswith("---\nid: '20261101090000'")
    assert pair["closest"]["path"] == "notes/bm25-ranking.md"
    assert "# BM25 Ranking" in pair["closest"]["text"]
    assert len(pair["others"]) == 2


def test_merge_keeps_identity_and_retargets_links_and_questions(setup):
    vault_dir, conn, embedder = setup
    result = resolve.merge_held(
        vault_dir,
        conn,
        "okapi-scoring",
        "notes/bm25-ranking.md",
        merged(),
        "Both state BM25's saturating term weights.",
        refresh=lambda: build_index(conn, vault_dir),
    )
    text = (vault_dir / "notes" / "bm25-ranking.md").read_text(encoding="utf-8")
    front = yaml.safe_load(text.split("---")[1])
    assert front["id"] == "20261015100000" and front["created"] == "2026-10-15"
    assert front["sources"] == ["external:robertson-zaragoza-2009", "conversation:2026-11-01"]
    assert front["aliases"] == ["Okapi BM25", "Okapi Scoring"]
    assert "tenth mention" in text

    links = (vault_dir / "notes" / "embeddings.md").read_text(encoding="utf-8")
    assert "[[bm25-ranking|Okapi]] and [[bm25-ranking#Details]]" in links
    questions = (vault_dir / "eval" / "questions" / "run.yaml").read_text(encoding="utf-8")
    assert questions.startswith("# kept comment")
    assert "- bm25-ranking" in questions and "what is okapi-scoring about" in questions

    assert not (vault_dir / "inbox" / "held" / "okapi-scoring.md").exists()
    assert (vault_dir / result["held_moved_to"]).read_text(encoding="utf-8") == HELD
    assert result["links_retargeted"] == ["notes/embeddings.md"]
    assert result["question_files_updated"] == ["eval/questions/run.yaml"]
    report = health_report(conn)
    assert not [b for b in report.broken_links if "okapi" in b["target"]]


def test_merge_writes_nothing_when_a_file_changed_meanwhile(setup, monkeypatch):
    vault_dir, conn, _ = setup
    real = resolve._retarget_labels

    def edit_in_obsidian(text, old, new):  # runs after the linking note was read
        (vault_dir / "notes" / "embeddings.md").write_text("edited in Obsidian\n", "utf-8")
        return real(text, old, new)

    monkeypatch.setattr(resolve, "_retarget_labels", edit_in_obsidian)
    before = (vault_dir / "notes" / "bm25-ranking.md").read_bytes()
    with pytest.raises(vault.VaultWriteError, match="nothing written"):
        resolve.merge_held(vault_dir, conn, "okapi-scoring", "bm25-ranking", merged(), "same")
    assert (vault_dir / "notes" / "bm25-ranking.md").read_bytes() == before
    assert (vault_dir / "inbox" / "held" / "okapi-scoring.md").exists()


def test_merge_refuses_bad_input(setup):
    vault_dir, conn, _ = setup
    with pytest.raises(InvalidProposal, match="why"):
        resolve.merge_held(vault_dir, conn, "okapi-scoring", "bm25-ranking", merged(), " ")
    with pytest.raises(FileNotFoundError):
        resolve.merge_held(vault_dir, conn, "okapi-scoring", "../inbox/x", merged(), "same")
    with pytest.raises(InvalidProposal, match="the merged note"):
        bad = merged()
        bad.summary = ""
        resolve.merge_held(vault_dir, conn, "okapi-scoring", "bm25-ranking", bad, "same")
    assert (vault_dir / "inbox" / "held" / "okapi-scoring.md").exists()


def test_replace_needs_the_content_it_read(tmp_path):
    (tmp_path / "a.md").write_bytes(b"one\n")
    with pytest.raises(vault.VaultWriteError, match="changed since"):
        vault.replace(tmp_path, "a.md", "two\n", "0" * 64)
    with pytest.raises(vault.VaultWriteError, match="does not exist"):
        vault.replace(tmp_path, "b.md", "two\n", "0" * 64)
    from zk.notes import content_hash

    vault.replace(tmp_path, "a.md", "two\n", content_hash(b"one\n"))
    assert (tmp_path / "a.md").read_bytes() == b"two\n"


def test_merge_and_release_are_one_commit_each(setup):
    vault_dir, conn, _ = setup

    def git(*args):
        return subprocess.run(
            ["git", "-C", str(vault_dir), *args], capture_output=True, text=True, check=True
        )

    git("init", "-q")
    git("config", "user.email", "test@example.com")
    git("config", "user.name", "Test")
    git("add", "-A")
    git("commit", "-qm", "start")
    (vault_dir / "inbox" / "held" / "other.md").write_text(HELD, encoding="utf-8")
    git("add", "-A")
    git("commit", "-qm", "held another")

    resolve.merge_held(
        vault_dir, conn, "okapi-scoring", "bm25-ranking", merged(), "Same idea, reworded."
    )
    ingest.apply_held(vault_dir, "other", "Related, not the same.")
    log = git("log", "--format=%B%x00", "-2").stdout.split("\x00")
    assert log[0].strip() == "zk: add held note Okapi Scoring\n\nRelated, not the same."
    assert (
        log[1].strip() == "zk: merge held okapi-scoring into bm25-ranking\n\nSame idea, reworded."
    )
    assert git("status", "--porcelain").stdout == ""


def test_merge_never_drops_tags_or_links(setup):
    vault_dir, conn, _ = setup
    bare = merged()
    bare.links = {}
    resolve.merge_held(vault_dir, conn, "okapi-scoring", "bm25-ranking", bare, "same idea")
    text = (vault_dir / "notes" / "bm25-ranking.md").read_text(encoding="utf-8")
    front = yaml.safe_load(text.split("---")[1])
    assert front["tags"] == ["retrieval", "keyword-search"]
    assert "- [[reciprocal-rank-fusion]]" in text
