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


def lines(path):
    return path.read_text(encoding="utf-8").splitlines()


def test_zks_own_sessions_are_recognised():
    assert capture.is_zk_session([capture.Turn("user", "[zk] Process the zk inbox")])
    assert not capture.is_zk_session([capture.Turn("user", "Process [zk] later")])


def place(home: Path, kind: str, name: str, source: Path, age_minutes: float = 120) -> Path:
    target = home / kind / "projects" / "C--w-proj" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(source, target)
    when = time.time() - age_minutes * 60
    os.utime(target, (when, when))
    return target


def inbox(vault):
    return sorted(p.name for p in (vault / "inbox").glob("*.md"))


def test_catch_up_captures_each_session_once(vault, isolated_homes):
    place(isolated_homes, "claude", "abcd1234-0000.jsonl", CLAUDE)
    place(isolated_homes, "claude", "ef567890-0000.jsonl", CLAUDE)
    place(isolated_homes, "claude", "busy0000-0000.jsonl", CLAUDE, age_minutes=5)  # in use
    place(isolated_homes, "claude", "old00000-0000.jsonl", CLAUDE, age_minutes=10 * 24 * 60)
    before = inbox(vault)
    assert [f["saved"] for f in sessions.catch_up(vault, dry_run=True)] == ["(dry run)"] * 2
    assert inbox(vault) == before
    found = sessions.catch_up(vault)
    assert sorted(f["session"] for f in found if f["saved"]) == [
        "claude:abcd1234-0000",
        "claude:ef567890-0000",
    ]
    new = sorted(set(inbox(vault)) - set(before))
    assert [n.split("-", 4)[-1] for n in new] == ["claude-abcd1234.md", "claude-ef567890.md"]
    text = next((vault / "inbox").glob("*-claude-abcd1234.md")).read_text(encoding="utf-8")
    assert 'source: "claude-code:abcd1234-0000"' in text
    assert "sk-ant" not in text  # redacted like any capture
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


def test_hook_entry_captures_and_starts_processing(vault, isolated_homes, monkeypatch):
    path = place(isolated_homes, "claude", "abcd1234-0000.jsonl", CLAUDE)
    started = []
    monkeypatch.setattr(sessions, "start_process", lambda v: started.append(v))
    monkeypatch.delenv("ZK_INGESTING", raising=False)
    event = {"session_id": "abcd1234-0000", "transcript_path": str(path), "cwd": "/w/proj"}
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(event)))
    assert hook.main([str(vault), "--process"]) == 0
    assert started == [vault] and any("-claude-abcd1234" in n for n in inbox(vault))
    # The same session ending again (nothing new) captures nothing and starts nothing.
    os.utime(path, None)
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
