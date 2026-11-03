"""Find agent session transcripts on this machine and capture them into the vault inbox.

Two ways in, sharing one record of what was captured:
- the session-end hook (`python -m zk.hook`) captures the session that just ended;
- `catch_up` scans recent Claude Code and Codex transcripts for sessions no hook
  captured (an app where the hook didn't run, a crash, a machine that was off).

The record (`sessions.json` in the zk data folder) keeps, per session, how many
turns were captured. A session that is resumed and ends again yields a capture of
the new turns only, so nothing is captured twice.

Reads transcripts; writes captures only through `zk.vault`, and its own record in
the data folder. Imports stay light (no numpy), because a session-end hook has
about a second to finish.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from zk import capture, vault
from zk.index import data_dir

INGESTING = "ZK_INGESTING"  # set for zk's own agents, so their sessions aren't captured
IDLE_MINUTES = 30  # a transcript touched more recently may still be in use


@dataclass
class Session:
    kind: str  # "claude" or "codex"
    path: Path
    session_id: str
    project: str | None = None


def claude_home() -> Path:
    return Path(os.environ.get("CLAUDE_CONFIG_DIR") or Path.home() / ".claude")


def codex_home() -> Path:
    return Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex")


def _clean_id(raw: str) -> str:
    return re.sub(r"[^A-Za-z0-9-]", "", raw)


def _head(path: Path, n: int = 50) -> list[str]:
    with path.open(encoding="utf-8", errors="replace") as f:
        return [line for _, line in zip(range(n), f, strict=False)]


def describe(path: Path, session_id: str | None = None, cwd: str | None = None) -> Session:
    """What a transcript file is: which agent wrote it, its session id and project."""
    head = _head(path)
    if capture.is_codex(head):
        meta = capture.codex_meta(head)
        sid = session_id or meta.get("id") or path.stem[-36:]
        cwd = cwd or meta.get("cwd")
        kind = "codex"
    else:
        sid = session_id or path.stem
        cwd = cwd or capture.claude_cwd(head)
        kind = "claude"
    project = Path(cwd).name if cwd else None
    return Session(kind, path, _clean_id(str(sid)), project or None)


def transcripts(days: float, now: float | None = None) -> list[Path]:
    """Claude Code and Codex transcripts changed in the last `days` days, oldest first."""
    now = time.time() if now is None else now
    found = list((claude_home() / "projects").glob("*/*.jsonl"))  # subagents sit deeper
    found += (codex_home() / "sessions").glob("**/rollout-*.jsonl")
    recent = [(p.stat().st_mtime, p) for p in found if now - p.stat().st_mtime <= days * 86400]
    return [p for _, p in sorted(recent)]


def find_codex(session_id: str) -> Path | None:
    """A Codex rollout by session id, for hooks that send no transcript path."""
    matches = sorted((codex_home() / "sessions").glob(f"**/rollout-*{session_id}*.jsonl"))
    return matches[-1] if matches else None


# The record of captured sessions.


def record_path() -> Path:
    return data_dir() / "sessions.json"


def load_record() -> dict:
    try:
        data = json.loads(record_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def save_record(record: dict) -> None:
    path = record_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(record, indent=1, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def _new_turns(turns: list[capture.Turn], seen: dict) -> list[capture.Turn]:
    """The turns after those captured before, including the new end of a turn that grew."""
    n = seen.get("turns", 0)
    new = turns[n:]
    if 0 < n <= len(turns) and len(turns[n - 1].text) > seen.get("last_chars", 1 << 60):
        last = turns[n - 1]
        new.insert(0, capture.Turn(last.role, last.text[seen["last_chars"] :].strip()))
    return new


def capture_session(
    vault_dir: Path, session: Session, now: datetime | None = None, dry_run: bool = False
) -> tuple[str | None, str]:
    """Capture what's new in one session. Returns (saved path or None, why)."""
    now = now or datetime.now().astimezone()
    key = f"{session.kind}:{session.session_id}"
    record = load_record()
    seen = record.get(key, {})
    mtime = session.path.stat().st_mtime
    if seen.get("mtime") == mtime:
        return None, "unchanged since last time"
    with session.path.open(encoding="utf-8", errors="replace") as f:
        lines = list(f)
    turns = (capture.parse_codex if session.kind == "codex" else capture.parse_transcript)(lines)
    if capture.is_zk_session(turns):
        why, saved = "zk's own session", None
    else:
        new = _new_turns(turns, seen)
        keep, why = capture.worth_keeping(new)
        saved = None
        if keep and not dry_run:
            source = "claude-code" if session.kind == "claude" else "codex"
            text = capture.render_capture(
                new,
                f"{source}:{session.session_id}",
                now.isoformat(timespec="seconds"),
                session.project,
            )
            name = f"inbox/{now:%Y-%m-%d-%H%M}-{session.kind}-{session.session_id[:8]}.md"
            saved = vault.write_new(vault_dir, vault.unique_path(vault_dir, name), text)
            seen = {
                "turns": len(turns),
                "last_chars": len(turns[-1].text),
                "capture": saved.relative_to(vault_dir).as_posix(),
            }
        elif keep:
            return "(dry run)", why
    if not dry_run:
        # Remember the file's state, so an unchanged transcript isn't parsed again.
        record = load_record()  # re-read: another capture may have finished meanwhile
        record[key] = {**seen, "mtime": mtime, "checked": now.isoformat(timespec="seconds")}
        save_record(record)
    return (str(saved) if saved else None), why


def catch_up(
    vault_dir: Path,
    days: float = 3,
    idle_minutes: float = IDLE_MINUTES,
    dry_run: bool = False,
    now: float | None = None,
) -> list[dict]:
    """Capture every recent session not captured yet. Sessions still in use are left for later."""
    now = time.time() if now is None else now
    out = []
    for path in transcripts(days, now):
        if now - path.stat().st_mtime < idle_minutes * 60:
            continue
        try:
            session = describe(path)
            saved, why = capture_session(vault_dir, session, dry_run=dry_run)
        except Exception as exc:  # one unreadable transcript shouldn't stop the rest
            saved, why, session = None, f"failed: {exc}", None
        out.append(
            {
                "transcript": str(path),
                "session": f"{session.kind}:{session.session_id}" if session else None,
                "saved": saved,
                "why": why,
            }
        )
    return out


def start_process(vault_dir: Path) -> None:
    """Start `zk process` detached, so the session that is ending doesn't wait for it."""
    flags = (
        {
            "creationflags": subprocess.DETACHED_PROCESS
            | subprocess.CREATE_NEW_PROCESS_GROUP
            | subprocess.CREATE_NO_WINDOW
        }
        if os.name == "nt"
        else {"start_new_session": True}
    )
    subprocess.Popen(
        [sys.executable, "-m", "zk", "process", str(vault_dir)],
        cwd=vault_dir,
        env={**os.environ, INGESTING: "1"},
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        **flags,
    )


def from_hook(vault_dir: Path, event: dict, now: datetime | None = None) -> tuple[str | None, str]:
    """Capture the session a Claude Code or Codex session-end hook reports."""
    if os.environ.get(INGESTING):
        return None, "zk's own session"
    sid = str(event.get("session_id") or "")
    path = Path(event["transcript_path"]) if event.get("transcript_path") else None
    if path is None and sid:
        path = find_codex(sid)
    if path is None or not path.is_file():
        return None, "no transcript for this session"
    session = describe(path, sid or None, event.get("cwd"))
    return capture_session(vault_dir, session, now)
