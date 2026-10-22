import json
from pathlib import Path

import pytest

from zk.cli import main


def test_check_prints_report_and_fails_on_errors(vault, tmp_path, capsys):
    code = main(["check", str(vault), "--db", str(tmp_path / "i.sqlite")])
    out = capsys.readouterr().out
    assert code == 1
    assert "notes: 10" in out
    assert "notes/embeddings.md -> [[hnsw-index]]" in out


def test_check_json(vault, tmp_path, capsys):
    main(["check", str(vault), "--db", str(tmp_path / "i.sqlite"), "--json"])
    data = json.loads(capsys.readouterr().out)
    assert data["notes"] == 10
    assert data["errors"] == 4


def test_exclude_option(vault, tmp_path, capsys):
    main(["index", str(vault), "--db", str(tmp_path / "i.sqlite"), "--exclude", "notes/dup*"])
    assert "indexed 8 notes" in capsys.readouterr().err


def test_check_passes_on_a_clean_vault(tmp_path, capsys):
    clean = tmp_path / "clean"
    (clean / "notes").mkdir(parents=True)
    (clean / "notes" / "a.md").write_text(
        "---\nid: 1\ntitle: A\ntype: concept\nsources: [x]\ncreated: 2026-10-15\n---\n"
        "# A\n\nSummary.\n\n- [[b]]\n",
        encoding="utf-8",
    )
    (clean / "notes" / "b.md").write_text(
        "---\nid: 2\ntitle: B\ntype: concept\nsources: [x]\ncreated: 2026-10-15\n---\n"
        "# B\n\nSummary.\n\n- [[a]]\n",
        encoding="utf-8",
    )
    assert main(["check", str(clean), "--db", str(tmp_path / "i.sqlite")]) == 0


def test_search_command(vault, tmp_path, capsys):
    db = str(tmp_path / "i.sqlite")
    assert main(["search", str(vault), "rank fusion", "--db", db]) == 0
    assert "1. Reciprocal Rank Fusion" in capsys.readouterr().out
    main(["search", str(vault), "rank fusion", "--db", db, "--json", "-k", "1"])
    hits = json.loads(capsys.readouterr().out)
    assert [h["path"] for h in hits] == ["notes/reciprocal-rank-fusion.md"]


def test_eval_command(vault, tmp_path, capsys):
    queries = str(Path(__file__).parent.parent / "eval" / "fixture-queries.yaml")
    db = str(tmp_path / "i.sqlite")
    assert main(["eval", str(vault), queries, "--db", db, "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["summary"]["mrr"] == 1.0


def test_show_command(vault, tmp_path, capsys):
    db = str(tmp_path / "index.sqlite")
    assert main(["show", str(vault), "--db", db, "20261015100100"]) == 0
    out = capsys.readouterr().out
    assert "Reciprocal Rank Fusion  (pattern" in out
    assert "links to: BM25 Ranking, Embeddings" in out


def test_show_unknown_note_exits_with_a_message(vault, tmp_path):
    with pytest.raises(SystemExit) as exc:
        main(["show", str(vault), "--db", str(tmp_path / "index.sqlite"), "fusion"])
    assert "Closest matches" in str(exc.value.code)


def test_semantic_search_command(vault, tmp_path, capsys):
    db = str(tmp_path / "index.sqlite")
    assert (
        main(["search", str(vault), "--db", db, "--mode", "semantic", "merging ranked lists"]) == 0
    )
    captured = capsys.readouterr()
    assert captured.out.startswith(" 1. Reciprocal Rank Fusion")
    assert "10 new, 0 reused" in captured.err

    assert main(["index", str(vault), "--db", db, "--embed"]) == 0
    assert "0 new, 10 reused" in capsys.readouterr().err


def test_eval_compares_modes(vault, tmp_path, capsys):
    queries = str(Path(__file__).parents[1] / "eval" / "fixture-queries.yaml")
    args = ["eval", str(vault), queries, "--db", str(tmp_path / "i.sqlite")]
    assert main([*args, "--mode", "keyword", "--mode", "semantic"]) == 0
    out = capsys.readouterr().out
    assert "== keyword ==" in out and "== semantic ==" in out
    assert main([*args, "--mode", "keyword", "--mode", "semantic", "--json"]) == 0
    assert set(json.loads(capsys.readouterr().out)) == {"keyword", "semantic"}
