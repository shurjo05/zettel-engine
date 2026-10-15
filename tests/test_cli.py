import json

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
