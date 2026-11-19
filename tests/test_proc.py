"""zk never opens a window on Windows: every child process goes through zk.proc."""

import re
from pathlib import Path

from zk import proc

SRC = Path(__file__).resolve().parents[1] / "src" / "zk"
DETACHED_PROCESS = 0x8


def test_every_child_process_is_started_through_proc():
    direct = re.compile(r"\bsubprocess\.(run|Popen|call|check_call|check_output|getoutput)\b")
    offenders = [
        f"{path.name}:{n}"
        for path in sorted(SRC.glob("*.py"))
        if path.name != "proc.py"
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if direct.search(line) or "DETACHED_PROCESS" in line or "os.system(" in line
    ]
    assert offenders == []


def test_children_are_hidden_on_windows(monkeypatch):
    monkeypatch.setattr(proc.os, "name", "nt")
    assert proc.hidden({})["creationflags"] == proc.CREATE_NO_WINDOW
    assert proc.hidden({"creationflags": 0x200})["creationflags"] == 0x200 | proc.CREATE_NO_WINDOW
    for breakaway in (False, True):
        flags = proc.background_flags(breakaway)["creationflags"]
        assert flags & proc.CREATE_NO_WINDOW and flags & proc.CREATE_NEW_PROCESS_GROUP
        assert not flags & DETACHED_PROCESS  # it would make CREATE_NO_WINDOW ignored


def test_elsewhere_nothing_changes(monkeypatch):
    monkeypatch.setattr(proc.os, "name", "posix")
    assert proc.hidden({"cwd": "x"}) == {"cwd": "x"}
    assert proc.background_flags(True) == {"start_new_session": True}


def test_run_passes_the_flag(monkeypatch):
    seen = {}
    monkeypatch.setattr(proc.os, "name", "nt")
    monkeypatch.setattr(proc.subprocess, "run", lambda args, **kw: seen.update(kw))
    proc.run(["git", "status"], capture_output=True)
    assert seen == {"capture_output": True, "creationflags": proc.CREATE_NO_WINDOW}
