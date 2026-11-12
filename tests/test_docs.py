import io
import json
import os
import subprocess
from pathlib import Path

import pytest

from zk import context, docs, hook, maps, recall
from zk.cli import main
from zk.embed import update_embeddings
from zk.index import build_index, default_db_path, health_report, open_db
from zk.search import search, similar_notes

WORDS = "client server login account contact message history session token queue".split()


def para(n: int, seed: int) -> str:
    return " ".join(WORDS[(seed * 7 + i * 3) % len(WORDS)] for i in range(n)) + "."


def contract() -> str:
    """A synthetic class-contract doc: an intro, five classes with members, a long table."""
    parts = ["# Chat class contract", "Short intro."]
    for c in range(1, 6):
        parts += [f"## C{c} Chat{c}Controller", para(30, c)]
        for m in range(3):
            parts += [f"### Member {m}", para(70, c * 10 + m)]
    rows = "\n".join(f"| R{i} | {para(12, i)} |" for i in range(80))
    parts += ["## Rules", f"| id | rule |\n|---|---|\n{rows}"]
    parts += ["## Example", "```\n# a comment, not a heading\nsend()\n```"]
    return "\n\n".join(parts)


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path / "Chat501"
    (root / "docs" / "design").mkdir(parents=True)
    (root / "docs" / "uml").mkdir()
    (root / "bin").mkdir()
    (root / "docs" / "design" / "class-contract.md").write_text(contract(), encoding="utf-8")
    (root / "README.md").write_text(
        "# Chat501\n\nA chat app.\n\nPOSTGRES_PASSWORD: devpass123\n", encoding="utf-8"
    )
    (root / "AGENTS.md").write_text("# Agents\n\nRules for agents.\n", encoding="utf-8")
    (root / "docs" / "design" / "class-contract.pdf").write_bytes(b"%PDF rendered twin")
    (root / "docs" / "uml" / "diagram.pdf").write_bytes(b"%PDF labels")
    (root / "bin" / "notes.md").write_text("# build output\n", encoding="utf-8")
    (root / "docs" / "copy.md").write_text(
        (root / "README.md").read_text(encoding="utf-8"), encoding="utf-8"
    )
    return root


@pytest.fixture
def git_vault(vault: Path) -> Path:
    for args in (["init", "-q"], ["config", "user.email", "t@example.com"]):
        subprocess.run(["git", "-C", str(vault), *args], check=True)
    subprocess.run(["git", "-C", str(vault), "config", "user.name", "Test"], check=True)
    subprocess.run(["git", "-C", str(vault), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(vault), "commit", "-qm", "start"], check=True)
    return vault


def test_markdown_is_cut_at_headings_into_sections_that_fit():
    sections = docs.split_markdown(contract(), "class-contract")
    labels = [" / ".join(s.path[1:]) for s in sections]
    assert all(s.path[0] == "Chat class contract" for s in sections)  # the H1 is the title
    assert "C1 Chat1Controller" in labels and "C1 Chat1Controller / Member 2" in labels
    assert sections[0].text.startswith("Short intro.")  # a short intro joins what follows
    assert all(len(s.text) <= docs.MAX_CHARS + 200 for s in sections)
    rules = [s for s in sections if s.path[1] == "Rules"]
    assert len(rules) > 1 and all(s.text.count("| id | rule |") == 1 for s in rules)
    example = next(s for s in sections if s.path[1:] == ["Example"])
    assert "# a comment, not a heading" in example.text  # code fences aren't cut at '#'


def test_pdf_text_is_cut_by_page():
    sections = docs.split_pdf_text(f"{para(20, 1)}\f\f{para(20, 2)}", "Spec")
    assert [s.path for s in sections] == [["Spec", "page 1"], ["Spec", "page 3"]]


def test_doc_files_skip_build_output_agent_files_twins_and_skipped_paths(project):
    found = docs.find_docs(project, docs.INCLUDE, ["docs/uml/*.pdf"])
    assert found == ["README.md", "docs/copy.md", "docs/design/class-contract.md"]


def _import(vault, project, **kw):
    return docs.import_docs(vault, project, exclude=["docs/uml/*.pdf"], **kw)


def test_import_writes_sections_once_and_rewrites_only_what_changed(git_vault, project):
    result = _import(git_vault, project)
    assert result["docs"] == 2 and result["duplicates"] == ["docs/copy.md = README.md"]
    refs = git_vault / "refs" / "chat501"
    readme = (refs / "readme--intro.md").read_text(encoding="utf-8")
    assert "devpass123" not in readme and "[REDACTED]" in readme
    contents = (refs / "docs-design-class-contract--contents.md").read_text(encoding="utf-8")
    assert "[[docs-design-class-contract--c1-chat1controller-member-0]]" in contents
    log = subprocess.run(
        ["git", "-C", str(git_vault), "log", "-1", "--format=%s"], capture_output=True, text=True
    ).stdout
    assert log.startswith(f"zk: docs chat501, {len(result['new'])} new")
    assert _import(git_vault, project)["unchanged"] == len(result["new"])  # nothing to do

    doc = project / "docs" / "design" / "class-contract.md"
    text = doc.read_text(encoding="utf-8")
    c3 = text.index("## C3")
    member = text.index("### Member 1\n\n", c3) + len("### Member 1\n\n")
    text = text[:member] + "Rewritten. " + text[member:]
    text = text.replace("## Example", "## Usage example")
    doc.write_text(text, encoding="utf-8")
    edited = refs / "docs-design-class-contract--c1-chat1controller-member-0.md"
    edited.write_text(edited.read_text(encoding="utf-8") + "\nMy own note.\n", encoding="utf-8")
    again = _import(git_vault, project)
    assert again["updated"] == [
        "refs/chat501/docs-design-class-contract--contents.md",
        "refs/chat501/docs-design-class-contract--c3-chat3controller-member-1.md",
    ]
    assert "refs/chat501/docs-design-class-contract--usage-example.md" in again["new"]
    assert again["retired"] == ["refs/chat501/docs-design-class-contract--example.md"]
    retired = git_vault / "refs" / "retired" / "chat501"
    assert (retired / "docs-design-class-contract--example.md").exists()
    assert "My own note." in edited.read_text(encoding="utf-8")  # an edited section is left alone


def test_sections_are_searchable_but_not_notes(git_vault, project, embedder):
    _import(git_vault, project)
    conn = open_db(default_db_path(git_vault))
    build_index(conn, git_vault)
    update_embeddings(conn, embedder)
    health = health_report(conn)
    assert not any(i["path"].startswith("refs/") for i in health.issues)
    assert not any(p.startswith("refs/") for p in health.orphans)
    hits = search(conn, "Member 1 Chat3Controller", mode="keyword", project="chat501", k=3)
    assert hits[0].path.startswith("refs/chat501/docs-design-class-contract--c3")
    body = (git_vault / hits[0].path).read_text(encoding="utf-8")
    duplicates = similar_notes(conn, embedder, body)
    assert not any(h.path.startswith("refs/") for h in duplicates)  # dedup ignores docs
    assert not any(
        "chat501" in s["name"] or any(n.startswith("refs/") for n in s["notes"])
        for s in maps.plan(conn, embedder)
    )
    brief = recall.brief(conn, project)
    assert 'search_notes(query, project="chat501")' in brief
    assert "docs/design/class-contract.md:" in brief
    report = context.run_context(
        conn,
        [{"query": "Member 1 Chat3Controller", "relevant": [hits[0].path.rsplit("/", 1)[-1]]}],
        "keyword",
    )
    assert report["file_tokens_median"] > report["tool_tokens_median"] > 0


def _read(path: Path, sid: str = "s1", **extra) -> dict:
    return {
        "hook_event_name": "PreToolUse",
        "tool_name": "Read",
        "session_id": sid,
        "tool_input": {"file_path": str(path), **extra},
    }


def test_a_long_doc_is_read_through_its_sections_first(git_vault, project, monkeypatch, capsys):
    _import(git_vault, project)
    doc = project / "docs" / "design" / "class-contract.md"
    reason = recall.on_read(_read(doc))
    assert reason and "search_notes" in reason and 'project="chat501"' in reason
    assert recall.on_read(_read(doc)) is None  # the second read goes ahead
    assert recall.on_read(_read(doc, "s2", offset=10, limit=50)) is None  # a part goes ahead
    assert recall.on_read(_read(project / "README.md", "s3")) is None  # short docs too
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(_read(doc, "s4"))))
    assert hook.main([str(git_vault)]) == 0
    out = json.loads(capsys.readouterr().out)["hookSpecificOutput"]
    assert out["permissionDecision"] == "deny" and "sections" in out["permissionDecisionReason"]
    doc.write_text(doc.read_text(encoding="utf-8") + "\nNew line.\n", encoding="utf-8")
    os.utime(doc, (1, 1))
    assert docs.stale(project)
    assert recall.on_read(_read(doc, "s5")) is None  # changed: the vault's copy may be behind
    monkeypatch.setenv("ZK_READ_REDIRECT", "0")
    assert recall.on_read(_read(doc, "s6")) is None


def test_cli_imports_indexes_and_forgets(git_vault, project, capsys):
    args = ["docs", str(git_vault), str(project), "--skip", "docs/uml/*.pdf"]
    assert main([*args, "--dry-run"]) == 0
    assert not (git_vault / "refs").exists()
    assert main(args) == 0
    assert "docs for chat501: 2 docs" in capsys.readouterr().out
    conn = open_db(default_db_path(git_vault))
    count = conn.execute("SELECT count(*) FROM notes WHERE type = 'reference'").fetchone()[0]
    assert count > 10  # indexed by the command itself
    assert docs.load_record()["projects"]
    assert main(["docs", str(git_vault), str(project), "--forget"]) == 0
    assert not docs.load_record()["projects"]


def test_a_session_starting_in_a_project_re_syncs_docs_that_changed(
    git_vault, project, monkeypatch, no_background_work
):
    subprocess.run(["git", "init", "-q", str(project)], check=True)
    _import(git_vault, project)
    event = {"hook_event_name": "SessionStart", "source": "startup", "cwd": str(project)}
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(event)))
    hook.main([str(git_vault)])
    assert ("docs", project) not in no_background_work  # unchanged: nothing to do
    readme = project / "README.md"
    readme.write_text("# Chat501\n\nA chat app, updated.\n", encoding="utf-8")
    os.utime(readme, (2, 2))
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(event)))
    hook.main([str(git_vault)])
    assert ("docs", project.resolve()) in no_background_work or ("docs", project) in (
        no_background_work
    )
