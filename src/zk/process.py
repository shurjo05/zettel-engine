"""What runs after a session: ingest the inbox, then resolve held notes. `zk process VAULT`.

Each step is a headless agent with the zk tools, and each runs only when it has
work: no captures in the inbox means no ingest run, and no held notes means no
resolve run, so a quiet session costs no model call. A lock in the zk data
folder lets one run at a time; captures that arrive meanwhile are picked up by
another round before the running one exits. Everything is logged to the data
folder's logs/.

The agents run with ZK_INGESTING=1 and a prompt starting with "[zk]", so their
own sessions are never captured. They write to the vault only through the zk
tools (`--allowedTools mcp__zk`), that is, through `zk.vault`.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

from zk import sessions
from zk.capture import ZK_MARKER
from zk.index import data_dir
from zk.ingest import list_held, list_inbox

INGEST_PROMPT = (
    f"{ZK_MARKER} Process the zk inbox with the zk tools, following the zk server's instructions."
)
RESOLVE_PROMPT = (
    f"{ZK_MARKER} Resolve the notes held in the zk inbox with the zk tools, "
    "following the zk server's instructions."
)
AGENT_TIMEOUT = 30 * 60
STALE_LOCK = 2 * 60 * 60  # a lock older than this was left by a run that died
MAX_ROUNDS = 3


def default_commands(resolve_model: str | None = "opus") -> dict[str, list[str]]:
    """Headless Claude Code runs, limited to the zk tools."""
    claude = shutil.which("claude") or "claude"
    base = ["--allowedTools", "mcp__zk"]
    resolve = [claude, "-p", RESOLVE_PROMPT, *base]
    if resolve_model:
        resolve += ["--model", resolve_model]
    return {"ingest": [claude, "-p", INGEST_PROMPT, *base], "resolve": resolve}


class Busy(Exception):
    pass


class Lock:
    """One `zk process` at a time, across hooks and the scheduled task."""

    def __init__(self, path: Path):
        self.path = path

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(2):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                if time.time() - self.path.stat().st_mtime > STALE_LOCK:
                    self.path.unlink(missing_ok=True)  # zk's own lock file, not vault content
                    continue
                raise Busy(f"another zk process is running ({self.path})") from None
            os.write(fd, f"{os.getpid()}\n".encode())
            os.close(fd)
            return self
        raise Busy(f"could not take the lock {self.path}")

    def __exit__(self, *exc):
        self.path.unlink(missing_ok=True)


def lock_path() -> Path:
    return data_dir() / "process.lock"


def _run(command: list[str] | str, vault_dir: Path, log) -> int:
    log.write(f"$ {command if isinstance(command, str) else subprocess.list2cmdline(command)}\n")
    log.flush()
    flags = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    try:
        done = subprocess.run(
            command,
            shell=isinstance(command, str),
            cwd=vault_dir,
            env={**os.environ, sessions.INGESTING: "1"},
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=AGENT_TIMEOUT,
            **flags,
        )
    except subprocess.TimeoutExpired:
        log.write(f"timed out after {AGENT_TIMEOUT // 60} minutes\n")
        return -1
    except OSError as exc:
        log.write(f"could not start: {exc}\n")
        return -1
    log.write(f"exit code {done.returncode}\n")
    return done.returncode


def run(
    vault_dir: Path,
    catch_up: bool = False,
    days: float = 3,
    commands: dict | None = None,
    log_path: Path | None = None,
) -> dict:
    """Catch up (optionally), then ingest and resolve while there is work. Returns a summary."""
    commands = commands or default_commands()
    now = datetime.now()
    log_path = log_path or data_dir() / "logs" / f"process-{now:%Y-%m-%d-%H%M%S}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {"log": str(log_path), "caught_up": 0, "ingest_runs": 0, "resolve_runs": 0}
    with Lock(lock_path()), log_path.open("a", encoding="utf-8") as log:
        log.write(f"zk process {vault_dir} at {now.isoformat(timespec='seconds')}\n")
        if catch_up:
            found = sessions.catch_up(vault_dir, days)
            saved = [f for f in found if f["saved"]]
            summary["caught_up"] = len(saved)
            for f in found:
                log.write(f"catch-up: {f['session']}: {f['saved'] or 'skipped'} ({f['why']})\n")
        tried: set[str] = set()
        for _ in range(MAX_ROUNDS):
            waiting = {c["name"] for c in list_inbox(vault_dir)}
            if not waiting - tried:  # nothing new; a capture the agent left is not retried
                break
            tried |= waiting
            log.write(f"ingest: {len(waiting)} capture(s) in the inbox\n")
            _run(commands["ingest"], vault_dir, log)
            summary["ingest_runs"] += 1
            if list_held(vault_dir):
                log.write(f"resolve: {len(list_held(vault_dir))} held note(s)\n")
                _run(commands["resolve"], vault_dir, log)
                summary["resolve_runs"] += 1
        else:
            log.write("stopped after the last round; the next run picks up the rest\n")
        if not summary["ingest_runs"] and list_held(vault_dir):
            # Notes held earlier (or a resolve that failed) still get resolved.
            _run(commands["resolve"], vault_dir, log)
            summary["resolve_runs"] += 1
        log.write(f"done: {summary}\n")
    return summary
