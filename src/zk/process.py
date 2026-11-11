"""What runs after a session: ingest the inbox, then resolve held notes. `zk process VAULT`.

Each step is a headless agent with the zk tools, and each runs only when it has
work: no captures in the inbox means no ingest run, and no held notes means no
resolve run, so a quiet session costs no model call. A lock in the zk data
folder lets one run at a time; captures that arrive meanwhile are picked up by
another round before the running one exits. Everything is logged to the data
folder's logs/.

The agents run with ZK_INGESTING=1 and a prompt starting with "[zk]", so their
own sessions are never captured. They write to the vault only through the zk
tools (`--allowedTools mcp__zk`), that is, through `zk.vault`. They get their own
write-enabled zk server (`--mcp-config`, only that server), so the one registered
for everyday sessions can stay read-only.

Captures are ingested a few at a time (BATCH_CAPTURES, BATCH_CHARS), oldest
first, so a backlog never lands in one agent's context. When a run changed the
vault, the map notes are brought up to date (zk.maps) at the end.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from zk import sessions
from zk.capture import ZK_MARKER
from zk.index import data_dir
from zk.ingest import list_held, list_inbox

CAPTURES = "{captures}"  # replaced with this round's capture names
INGEST_PROMPT = (
    f"{ZK_MARKER} Process these captures in the zk inbox with the zk tools, following the zk "
    f"server's instructions, and leave any others for later: {CAPTURES}"
)
RESOLVE_PROMPT = (
    f"{ZK_MARKER} Resolve the notes held in the zk inbox with the zk tools, "
    "following the zk server's instructions."
)
AGENT_TIMEOUT = 30 * 60
STALE_LOCK = 2 * 60 * 60  # a lock older than this was left by a run that died
MAX_ROUNDS = 10
# Claude Code waits 30 s for an MCP server to start by default. A cold Python start on
# Windows, or one racing a `uv sync`, can take longer, and the agent then runs without
# its zk tools, so headless runs allow two minutes (MCP_TIMEOUT, in milliseconds).
MCP_START_MS = "120000"
BATCH_CAPTURES = 3  # captures per ingest run, so a backlog is read a few at a time
BATCH_CHARS = 120_000  # ...and at most this much text per run (one capture always goes)


def mcp_config(vault_dir: Path) -> Path:
    """A config with the write-enabled zk server, just for the headless runs."""
    path = data_dir() / "zk-agents-mcp.json"
    server = {
        "command": sys.executable,
        "args": ["-m", "zk", "serve", str(vault_dir.resolve()), "--allow-writes"],
    }
    text = json.dumps({"mcpServers": {"zk": server}}, indent=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.read_text(encoding="utf-8") != text:
        path.write_text(text, encoding="utf-8")
    return path


def default_commands(
    resolve_model: str | None = "opus", vault_dir: Path | None = None
) -> dict[str, list[str]]:
    """Headless Claude Code runs, limited to the zk tools of their own write-enabled server."""
    claude = shutil.which("claude") or "claude"
    base = ["--allowedTools", "mcp__zk"]
    if vault_dir is not None:
        base += ["--mcp-config", str(mcp_config(vault_dir)), "--strict-mcp-config"]
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


def batches(captures: list[dict], tried: set[str]) -> list[str]:
    """The next captures to ingest: oldest untried first, within the batch limits."""
    out, size = [], 0
    for c in captures:
        if c["name"] in tried:
            continue
        if out and (len(out) >= BATCH_CAPTURES or size + c["chars"] > BATCH_CHARS):
            break
        out.append(c["name"])
        size += c["chars"]
    return out


def _with_captures(command: list[str] | str, names: list[str]) -> list[str] | str:
    listed = ", ".join(names)
    if isinstance(command, str):
        return command.replace(CAPTURES, listed)
    return [part.replace(CAPTURES, listed) for part in command]


def refresh_maps(vault_dir: Path, log) -> None:
    """Rebuild the index and embeddings, then bring the map notes up to date."""
    from zk import maps
    from zk.embed import FastEmbedder, update_embeddings
    from zk.index import DEFAULT_EXCLUDES, build_index, default_db_path, open_db

    conn = open_db(default_db_path(vault_dir))
    try:
        build_index(conn, vault_dir, DEFAULT_EXCLUDES)
        embedder = FastEmbedder()
        update_embeddings(conn, embedder)
        build_index(conn, vault_dir, DEFAULT_EXCLUDES)
        result = maps.update(vault_dir, conn, embedder)
        build_index(conn, vault_dir, DEFAULT_EXCLUDES)  # so the next brief lists them
    finally:
        conn.close()
    log.write(maps.format_update(result) + "\n")


def _run(command: list[str] | str, vault_dir: Path, log) -> int:
    log.write(f"$ {command if isinstance(command, str) else subprocess.list2cmdline(command)}\n")
    log.flush()
    flags = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    try:
        done = subprocess.run(
            command,
            shell=isinstance(command, str),
            cwd=vault_dir,
            env={"MCP_TIMEOUT": MCP_START_MS, **os.environ, sessions.INGESTING: "1"},
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


def status_path() -> Path:
    """The last `zk process` run's outcome, for `zk review` and anyone checking by hand."""
    return data_dir() / "process-status.json"


def last_status() -> dict:
    try:
        return json.loads(status_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def run(
    vault_dir: Path,
    catch_up: bool = False,
    days: float = 3,
    commands: dict | None = None,
    log_path: Path | None = None,
) -> dict:
    """Catch up (optionally), then ingest and resolve while there is work. Returns a summary."""
    commands = commands or default_commands(vault_dir=vault_dir)
    now = datetime.now()
    log_path = log_path or data_dir() / "logs" / f"process-{now:%Y-%m-%d-%H%M%S}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "log": str(log_path),
        "caught_up": 0,
        "ingest_runs": 0,
        "resolve_runs": 0,
        "stalled": [],
    }
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
            waiting = list_inbox(vault_dir)
            batch = batches(waiting, tried)
            if not batch:  # nothing new; a capture the agent left is not retried
                break
            tried |= set(batch)
            os.utime(lock_path())  # still running: keep the lock from looking stale
            log.write(f"ingest: {len(batch)} of {len(waiting)} capture(s) in the inbox\n")
            _run(_with_captures(commands["ingest"], batch), vault_dir, log)
            summary["ingest_runs"] += 1
            left = sorted(set(batch) & {c["name"] for c in list_inbox(vault_dir)})
            if left == sorted(batch):
                # Not one capture was processed: the agent couldn't work (its zk tools
                # didn't start, a login expired). Stop rather than spend more runs.
                summary["stalled"] = left
                log.write(
                    f"ingest made no progress: {len(left)} capture(s) still in the inbox; "
                    "stopping (see the agent's output above)\n"
                )
                break
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
        if summary["ingest_runs"] or summary["resolve_runs"]:
            try:
                refresh_maps(vault_dir, log)
                summary["maps"] = True
            except Exception as exc:  # maps can wait for the next run; the notes are in
                log.write(f"maps not updated: {exc}\n")
        log.write(f"done: {summary}\n")
    status = {
        "at": now.isoformat(timespec="seconds"),
        "ok": not summary["stalled"],
        "stalled": len(summary["stalled"]),
        "waiting": len(list_inbox(vault_dir)),
        "log": str(log_path),
    }
    try:
        status_path().write_text(json.dumps(status, indent=2), encoding="utf-8")
    except OSError:
        pass
    return summary
