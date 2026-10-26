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


TRANSCRIPT = Path(__file__).parent / "fixtures" / "transcripts" / "session.jsonl"


def test_capture_from_a_claude_code_hook(vault, monkeypatch, capsys):
    import io

    event = {"session_id": "abcd1234-ef", "transcript_path": str(TRANSCRIPT), "cwd": "/w/myproj"}
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(event)))
    assert main(["capture", str(vault), "--hook"]) == 0
    saved = sorted((vault / "inbox").glob("*-claude-abcd1234.md"))
    assert len(saved) == 1
    text = saved[0].read_text(encoding="utf-8")
    assert 'source: "claude-code:abcd1234-ef"' in text and 'project: "myproj"' in text
    assert "sk-ant" not in text
    # Captures sit in the inbox, which the index skips.
    assert main(["index", str(vault), "--db", str(vault.parent / "i.sqlite")]) == 0
    assert "indexed 10 notes" in capsys.readouterr().err


def test_capture_skips_short_sessions_and_never_fails_the_hook(
    vault, monkeypatch, capsys, tmp_path
):
    import io

    short = tmp_path / "short.jsonl"
    short.write_text('{"type":"user","message":{"content":"hi"}}\n', encoding="utf-8")
    before = set((vault / "inbox").iterdir())
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps({"transcript_path": str(short)})))
    assert main(["capture", str(vault), "--hook"]) == 0
    assert "skipped" in capsys.readouterr().err
    monkeypatch.setattr("sys.stdin", io.StringIO("not json"))
    assert main(["capture", str(vault), "--hook"]) == 0
    assert "failed" in capsys.readouterr().err
    assert set((vault / "inbox").iterdir()) == before


def test_capture_text(vault, capsys):
    assert (
        main(["capture", str(vault), "--text", "RRF needs ranks only.", "--source", "Codex!"]) == 0
    )
    assert main(["capture", str(vault), "--text", "Second idea.", "--source", "codex"]) == 0
    saved = sorted((vault / "inbox").glob("*-codex*.md"))
    assert len(saved) == 2  # within one minute the second gets a -2 suffix instead of overwriting
    texts = [p.read_text(encoding="utf-8") for p in saved]
    assert any("RRF needs ranks only." in t for t in texts)
    assert any("Second idea." in t for t in texts)


def test_capture_then_starts_ingest_in_the_background(vault, monkeypatch, capsys, tmp_path):
    import io

    from zk import cli

    started = []

    class FakePopen:
        def __init__(self, command, **kwargs):
            started.append((command, kwargs))

    monkeypatch.setattr(cli.subprocess, "Popen", FakePopen)
    monkeypatch.setenv("ZK_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.delenv("ZK_INGESTING", raising=False)
    event = {"session_id": "abcd1234-ef", "transcript_path": str(TRANSCRIPT)}
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(event)))
    assert main(["capture", str(vault), "--hook", "--then", "agent --ingest"]) == 0
    assert len(started) == 1
    command, kwargs = started[0]
    assert command == "agent --ingest" and kwargs["env"]["ZK_INGESTING"] == "1"
    assert kwargs["cwd"] == vault
    assert "started ingest" in capsys.readouterr().err


def test_zks_own_ingest_session_is_not_captured(vault, monkeypatch, capsys):
    import io

    before = set((vault / "inbox").iterdir())
    monkeypatch.setenv("ZK_INGESTING", "1")
    event = {"session_id": "abcd1234-ef", "transcript_path": str(TRANSCRIPT)}
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(event)))
    assert main(["capture", str(vault), "--hook", "--then", "agent"]) == 0
    assert "zk's own ingest session" in capsys.readouterr().err
    assert set((vault / "inbox").iterdir()) == before


def test_eval_defaults_to_the_questions_saved_in_the_vault(vault, tmp_path, capsys):
    db = ["--db", str(tmp_path / "i.sqlite")]
    with pytest.raises(SystemExit, match="no test questions"):
        main(["eval", str(vault), *db])
    folder = vault / "eval" / "questions"
    folder.mkdir(parents=True)
    (folder / "run.yaml").write_text(
        "queries:\n  - query: merging ranked lists\n    relevant: [reciprocal-rank-fusion]\n"
        "    kind: generated\n",
        encoding="utf-8",
    )
    assert main(["eval", str(vault), *db, "--mode", "keyword"]) == 0
    assert "generated" in capsys.readouterr().out


def test_check_lists_long_notes_and_near_duplicates(vault, tmp_path, capsys):
    src = vault / "notes" / "bm25-ranking.md"
    (vault / "notes" / "bm25-copy.md").write_text(
        src.read_text(encoding="utf-8").replace("id: ", "id: 9", 1) + "\n" + "more words " * 300,
        encoding="utf-8",
    )
    main(["check", str(vault), "--db", str(tmp_path / "i.sqlite"), "--duplicates"])
    out = capsys.readouterr().out
    assert "over about 512 tokens" in out and "notes/bm25-copy.md (~" in out
    assert "near-duplicates" in out


def test_import_dry_run_then_import(vault, tmp_path, capsys):
    legacy = Path(__file__).parent / "fixtures" / "legacy"
    db = ["--db", str(tmp_path / "i.sqlite")]
    assert main(["import", str(legacy), str(vault), *db, "--dry-run"]) == 0
    out = capsys.readouterr()
    assert "8 legacy files: 3 imported, 1 held" in out.out and "nothing was written" in out.err
    assert not (vault / "notes" / "cache-embeddings-by-content-hash.md").exists()
    assert main(["import", str(legacy), str(vault), *db]) == 0
    assert "3 imported" in capsys.readouterr().out
    assert (vault / "notes" / "cache-embeddings-by-content-hash.md").is_file()
