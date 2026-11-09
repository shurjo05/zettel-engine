import json
import re
from pathlib import Path

import pytest

from zk import answer, calibrate, context
from zk.cli import main
from zk.embed import update_embeddings
from zk.index import build_index, open_db

QUERIES = Path(__file__).parents[1] / "eval" / "fixture-queries.yaml"


@pytest.fixture
def indexed(vault, tmp_path, embedder):
    conn = open_db(tmp_path / "index.sqlite")
    build_index(conn, vault)
    update_embeddings(conn, embedder)
    return vault, conn, embedder


def test_context_counts_the_tool_path_and_the_baselines(indexed):
    vault, conn, embedder = indexed
    queries = [
        {"query": "merging ranked lists", "relevant": ["reciprocal-rank-fusion"]},
        {"query": "zebra migration patterns", "relevant": ["no-such-note"]},
    ]
    report = context.run_context(conn, queries, "keyword", k=5)
    found, missed = report["results"]
    assert found["found_at"] == 1 and missed["found_at"] is None
    assert report["answered_in_top_k"] == 0.5
    # Search plus one note costs less than opening every hit in full.
    assert 0 < found["tool_tokens"] < found["naive_tokens"]
    assert report["whole_vault_tokens"] > 0
    text = context.format_context(report)
    assert "of the whole vault" in text and "zebra migration patterns" in text


def test_eval_context_command(vault, tmp_path, capsys):
    db = ["--db", str(tmp_path / "i.sqlite")]
    assert main(["eval", str(vault), str(QUERIES), *db, "--context", "--mode", "keyword"]) == 0
    assert "naive RAG" in capsys.readouterr().out


def fake_model(cite):
    """Answers by citing whatever `cite(names_given)` returns."""
    prompts = []

    def ask(prompt):
        prompts.append(prompt)
        names = re.findall(r"^name: (.+)$", prompt, re.M)
        chosen = cite(names)
        return answer.NOT_COVERED + ": nothing" if chosen is None else f"See [[{chosen}]]."

    return ask, prompts


def test_answers_are_scored_by_their_citations(indexed):
    vault, conn, embedder = indexed
    queries = [
        {"query": "merging ranked lists", "relevant": ["reciprocal-rank-fusion"]},
        {"query": "zebra migration patterns", "relevant": ["no-such-note"]},
    ]
    ask, prompts = fake_model(lambda names: names[0] if names else None)
    report = answer.run_answers(conn, queries, ask, None, mode="keyword")
    assert report["citation_accuracy"] == 1.0 and report["grounded"] == 1.0
    assert "Answer the question using only the notes below" in prompts[0]

    invent, _ = fake_model(lambda names: "made-up-note")
    report = answer.run_answers(conn, queries[:1], invent, None, mode="keyword")
    assert report["grounded"] == 0.0 and report["results"][0]["invented"] == ["made-up-note"]
    assert "cited notes it wasn't given: made-up-note" in answer.format_answers(report)

    abstain, _ = fake_model(lambda names: None)
    report = answer.run_answers(conn, queries, abstain, None, mode="keyword")
    assert report["abstained_when_retrieved"] == 1.0


def test_ask_command(vault, tmp_path, capsys, monkeypatch):
    from zk import cli

    ask, _ = fake_model(lambda names: names[0])
    monkeypatch.setattr(cli, "make_ask", lambda model=None: ask)
    db = ["--db", str(tmp_path / "i.sqlite")]
    assert main(["ask", str(vault), "how do I merge ranked lists?", *db]) == 0
    out = capsys.readouterr()
    assert out.out.startswith("See [[") and "tokens)" in out.err


def judge_by_name(prompt):
    """Labels a pair 'same' when both titles share their first word, else 'unrelated'."""
    titles = re.findall(r"^title: (.+)$", prompt, re.M)
    lines = []
    for n in range(len(titles) // 2):
        a, b = titles[2 * n], titles[2 * n + 1]
        label = "same" if a.split()[0] == b.split()[0] else "unrelated"
        lines.append(json.dumps({"pair": n + 1, "label": label, "why": "test"}))
    return "\n".join(lines)


def test_calibrate_judges_saves_and_reuses_labels(indexed, monkeypatch):
    vault, conn, embedder = indexed
    # The stand-in embedder's scores sit low; one wide band covers every pair.
    monkeypatch.setattr(calibrate, "BAND", 2.0)
    monkeypatch.setattr(
        calibrate, "CUTS", {"link": {"at": 0.0, "hit": {"same", "related"}, "by_hand": 0.9}}
    )
    calls = []

    def model(prompt):
        calls.append(prompt)
        return judge_by_name(prompt)

    first = calibrate.run(vault, conn, embedder, model, per_band=4)
    assert first["judged_now"] == first["pairs"] > 0 and first["file"].startswith(
        "eval/calibration/"
    )
    saved = (vault / first["file"]).read_text(encoding="utf-8")
    assert "text_a" not in saved and "hash_a" in saved  # labels only, no note text
    assert first["cuts"]["link"]["above"]["judged"] > 0

    before = len(calls)
    second = calibrate.run(vault, conn, embedder, model, per_band=4)
    assert len(calls) == before and second["reused"] == second["pairs"]  # nothing re-sent


def test_calibration_warns_when_precision_drops():
    report = {
        "duplicate": {
            "at": 0.85,
            "by_hand": 0.6,
            "above": {"judged": 6, "hits": 2, "precision": 0.333},
            "below": {"judged": 6, "hits": 4, "precision": 0.667},
        }
    }
    found = calibrate.warnings(report)
    assert any("Consider raising it" in w for w in found)
    assert any("consider lowering it" in w for w in found)


def test_drift_reads_the_vault_history(vault):
    import subprocess

    def git(*args):
        subprocess.run(["git", "-C", str(vault), *args], check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "T")
    for message in [
        "zk: ingest a.md: 3 notes",
        "zk: ingest b.md: 1 note, 1 held",
        "zk: merge held x into y",
        "zk: add held note Z",
        "zk: add held note W",
    ]:
        git("commit", "-q", "--allow-empty", "-m", message)
    d = calibrate.drift(vault)
    assert d["notes_written"] == 4 and d["notes_held"] == 1 and d["hold_rate"] == 0.2
    assert d["false_hold_share"] == 0.667


def test_calibrate_command_without_judging(vault, tmp_path, capsys):
    db = ["--db", str(tmp_path / "i.sqlite")]
    assert main(["calibrate", str(vault), *db, "--no-judge"]) == 0
    assert "pairs sampled" in capsys.readouterr().out
