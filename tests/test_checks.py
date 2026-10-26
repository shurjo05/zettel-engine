import json
from datetime import datetime

import yaml

from zk import checks
from zk.embed import update_embeddings
from zk.index import build_index, open_db

NOW = datetime(2026, 10, 26, 9, 0, 0)


def index(vault, tmp_path, embedder=None):
    conn = open_db(tmp_path / "index.sqlite")
    build_index(conn, vault)
    if embedder is not None:
        update_embeddings(conn, embedder)
    return conn


def save_questions(vault, queries):
    folder = vault / "eval" / "questions"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "run.yaml").write_text(yaml.safe_dump({"queries": queries}), encoding="utf-8")


def test_holdout_is_a_stable_one_in_five_sample():
    questions = [f"question number {i}" for i in range(1000)]
    held = [q for q in questions if checks.is_holdout(q)]
    assert 150 < len(held) < 250
    assert checks.is_holdout("  Question   NUMBER 7 ") == checks.is_holdout("question number 7")


def test_small_or_noisy_drops_are_not_warnings():
    assert checks.drop_beyond_noise([1.0] * 4, [0.0] * 4) is None  # too few questions
    assert checks.drop_beyond_noise([1.0] * 20, [1.0] * 20) is None
    mixed = [1.0, 0.5] * 10
    assert checks.drop_beyond_noise(mixed, [0.5, 1.0] * 10) is None  # ups and downs cancel
    one_bad = [1.0] * 20
    assert checks.drop_beyond_noise(one_bad, [1.0] * 19 + [0.5]) is None  # one question moved
    assert checks.drop_beyond_noise([1.0] * 20, [0.5] * 20) == 0.5


def test_scores_are_compared_with_when_each_question_was_added(vault, tmp_path, embedder):
    # The right answer for these questions is a note that ranks below the top.
    texts = ("rank fusion", "fusing ranked lists", "reciprocal rank", "rrf", "merging ranks")
    save_questions(
        vault, [{"query": q, "relevant": ["bm25-ranking"], "kind": "generated"} for q in texts]
    )
    conn = index(vault, tmp_path, embedder)
    history = tmp_path / "index-eval.json"
    assert checks.check_scores(conn, vault, history, embedder, NOW) == []  # first scores
    saved = json.loads(history.read_text(encoding="utf-8"))
    assert len(saved["queries"]) == 5 and len(saved["runs"]) == 1
    assert checks.check_scores(conn, vault, history, embedder, NOW) == []  # nothing changed

    # Pretend each question used to find its note first: today's scores are a real drop.
    for entry in saved["queries"].values():
        entry["rr"] = 1.0
    history.write_text(json.dumps(saved), encoding="utf-8")
    warnings = checks.check_scores(conn, vault, history, embedder, NOW)
    assert len(warnings) == 1
    assert warnings[0].startswith("Search got worse on the generated test questions: MRR 1.00")

    # Questions whose note is gone are skipped, not counted as misses.
    (vault / "notes" / "bm25-ranking.md").rename(vault / "notes" / "renamed.md")
    build_index(conn, vault)
    assert checks.check_scores(conn, vault, history, embedder, NOW) == []


def test_new_note_health(vault, tmp_path):
    notes = vault / "notes"
    (notes / "long-one.md").write_text(
        "---\nid: '1'\ntitle: Long One\ntype: concept\n---\n# Long One\n\n"
        + "word " * 600
        + "\n\nSee [[missing-note]] and [[bm25-ranking]].\n",
        encoding="utf-8",
    )
    (notes / "alone.md").write_text(
        "---\nid: '2'\ntitle: Alone\ntype: concept\n---\n# Alone\n\nNo links here.\n",
        encoding="utf-8",
    )
    conn = index(vault, tmp_path)
    warnings = checks.check_new_notes(conn, ["notes/long-one.md", "notes/alone.md"])
    assert any(w.startswith("notes/long-one.md is about") for w in warnings)
    assert "notes/long-one.md links to [[missing-note]], which doesn't exist." in warnings
    assert "notes/alone.md is linked to no other note." in warnings
    assert not any("long-one.md is linked to no" in w for w in warnings)
    assert checks.long_notes(conn)[0][0] == "notes/long-one.md"


def test_near_duplicates_across_the_vault(vault, tmp_path, embedder):
    src = vault / "notes" / "bm25-ranking.md"
    (vault / "notes" / "bm25-copy.md").write_text(
        src.read_text(encoding="utf-8").replace("id: ", "id: 9", 1), encoding="utf-8"
    )
    conn = index(vault, tmp_path, embedder)
    pairs = checks.near_duplicates(conn, embedder, block=3)  # small blocks cross boundaries
    assert ("notes/bm25-copy.md", "notes/bm25-ranking.md", 1.0) in pairs
    assert all(a < b for a, b, _ in pairs)
