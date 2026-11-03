import io
import json
import os
import shutil
import time
from pathlib import Path

from zk import capture, hook, sessions
from zk.cli import main

FIXTURES = Path(__file__).parent / "fixtures" / "transcripts"
CLAUDE = FIXTURES / "session.jsonl"
CODEX = FIXTURES / "codex-rollout.jsonl"
CODEX_ID = "0199a1b2-c3d4-7e5f-8a9b-0c1d2e3f4a5b"


def lines(path):
    return path.read_text(encoding="utf-8").splitlines()


def test_codex_rollouts_keep_only_the_conversation():
    turns = capture.parse_codex(lines(CODEX))
    assert [t.role for t in turns] == ["user", "assistant", "user", "assistant"]
    text = "\n".join(t.text for t in turns)
    assert "environment_context" not in text and "Sandbox rules" not in text
    assert "file-list" not in text and "thinking" not in text
    assert text.count("Why does the SQLite index") == 1  # the event_msg echo isn't doubled
    assert capture.is_codex(lines(CODEX)) and not capture.is_codex(lines(CLAUDE))
    assert capture.codex_meta(lines(CODEX)) == {"id": CODEX_ID, "cwd": "/w/codexproj"}


def test_codex_event_messages_are_the_fallback():
    events = [
        {"type": "event_msg", "payload": {"type": "user_message", "message": "first question"}},
        {"type": "event_msg", "payload": {"type": "agent_message", "message": "an answer"}},
    ]
    turns = capture.parse_codex(json.dumps(e) for e in events)
    assert [(t.role, t.text) for t in turns] == [
        ("user", "first question"),
        ("assistant", "an answer"),
    ]


def test_zks_own_sessions_are_recognised():
    assert capture.is_zk_session([capture.Turn("user", "[zk] Process the zk inbox")])
    assert not capture.is_zk_session([capture.Turn("user", "Process [zk] later")])


def place(home: Path, kind: str, name: str, source: Path, age_minutes: float = 120) -> Path:
    if kind == "claude":
        target = home / "claude" / "projects" / "C--w-proj" / name
    else:
        target = home / "codex" / "sessions" / "2026" / "11" / "03" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(source, target)
    when = time.time() - age_minutes * 60
    os.utime(target, (when, when))
    return target


def inbox(vault):
    return sorted(p.name for p in (vault / "inbox").glob("*.md"))


def test_catch_up_captures_each_session_once(vault, isolated_homes):
    place(isolated_homes, "claude", "abcd1234-0000.jsonl", CLAUDE)
    place(isolated_homes, "codex", f"rollout-2026-11-03T10-00-00-{CODEX_ID}.jsonl", CODEX)
    place(isolated_homes, "claude", "busy0000-0000.jsonl", CLAUDE, age_minutes=5)  # in use
    place(isolated_homes, "claude", "old00000-0000.jsonl", CLAUDE, age_minutes=10 * 24 * 60)
    before = inbox(vault)
    assert [f["saved"] for f in sessions.catch_up(vault, dry_run=True)] == ["(dry run)"] * 2
    assert inbox(vault) == before
    found = sessions.catch_up(vault)
    assert sorted(f["session"] for f in found if f["saved"]) == [
        "claude:abcd1234-0000",
        f"codex:{CODEX_ID}",
    ]
    new = sorted(set(inbox(vault)) - set(before))
    assert [n.split("-", 4)[-1] for n in new] == ["claude-abcd1234.md", "codex-0199a1b2.md"]
    codex_text = next((vault / "inbox").glob("*-codex-*.md")).read_text(encoding="utf-8")
    assert 'source: "codex:' in codex_text and 'project: "codexproj"' in codex_text
    assert "abcdefghijklmnop" not in codex_text  # redacted like any capture
    # A second pass finds nothing new.
    assert not [f for f in sessions.catch_up(vault) if f["saved"]]


def test_a_resumed_session_captures_only_its_new_turns(vault, isolated_homes):
    path = place(isolated_homes, "claude", "abcd1234-0000.jsonl", CLAUDE)
    session = sessions.describe(path)
    assert sessions.capture_session(vault, session)[0]
    more = [
        {"type": "user", "message": {"content": "A follow-up about BM25 saturation " * 10}},
        {
            "type": "assistant",
            "message": {"content": [{"type": "text", "text": "k1 caps it." * 20}]},
        },
        {"type": "user", "message": {"content": "And the b parameter? " * 10}},
    ]
    with path.open("a", encoding="utf-8") as f:
        f.writelines(json.dumps(e) + "\n" for e in more)
    os.utime(path, (time.time() - 3600, time.time() - 3600))
    saved, _ = sessions.capture_session(vault, session)
    text = Path(saved).read_text(encoding="utf-8")
    assert "BM25 saturation" in text and "reciprocal rank fusion" not in text.lower()


def test_hook_entry_captures_codex_and_starts_processing(vault, isolated_homes, monkeypatch):
    rollout = place(isolated_homes, "codex", f"rollout-x-{CODEX_ID}.jsonl", CODEX)
    started = []
    monkeypatch.setattr(sessions, "start_process", lambda v: started.append(v))
    monkeypatch.delenv("ZK_INGESTING", raising=False)
    # Codex may send no transcript path; the rollout is found by session id.
    event = {"session_id": CODEX_ID, "transcript_path": None, "cwd": "/w/codexproj"}
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(event)))
    assert hook.main([str(vault), "--process"]) == 0
    assert started == [vault] and any("-codex-" in n for n in inbox(vault))
    # The same session ending again (nothing new) captures nothing and starts nothing.
    os.utime(rollout, None)
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(event)))
    assert hook.main([str(vault), "--process"]) == 0
    assert started == [vault]


def test_hook_entry_never_fails(vault, monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO("not json"))
    assert hook.main([str(vault)]) == 0
    assert "failed" in capsys.readouterr().err


def test_capture_catch_up_command(vault, isolated_homes, capsys):
    place(isolated_homes, "claude", "abcd1234-0000.jsonl", CLAUDE)
    assert main(["capture", str(vault), "--catch-up", "--dry-run"]) == 0
    assert "would capture 1 of 1 recent sessions" in capsys.readouterr().err
    assert main(["capture", str(vault), "--catch-up"]) == 0
    assert "captured 1 of 1" in capsys.readouterr().err


def test_capture_a_codex_transcript_by_path(vault, capsys):
    assert main(["capture", str(vault), "--transcript", str(CODEX)]) == 0
    assert any("-codex-0199a1b2" in n for n in inbox(vault))


def test_a_turn_that_grew_after_capture_keeps_only_its_new_end():
    turns = [capture.Turn("user", "q " * 300), capture.Turn("assistant", "first answer")]
    seen = {"turns": 2, "last_chars": len("first answer")}
    grown = [*turns[:1], capture.Turn("assistant", "first answer\n\nmore detail")]
    assert [(t.role, t.text) for t in sessions._new_turns(grown, seen)] == [
        ("assistant", "more detail")
    ]


def test_hook_entry_accepts_a_byte_order_mark(vault, monkeypatch, capsys):
    class Stdin:
        buffer = io.BytesIO(b"\xef\xbb\xbf" + json.dumps({"transcript_path": str(CLAUDE)}).encode())

    monkeypatch.setattr("sys.stdin", Stdin())
    monkeypatch.delenv("ZK_INGESTING", raising=False)
    assert hook.main([str(vault)]) == 0
    assert "saved" in capsys.readouterr().err
