"""zk's Claude Code hooks: `python -m zk.hook VAULT [--process]`.

Claude Code runs this with the event as JSON on stdin, and the event's
hook_event_name says which hook it is:
- SessionEnd captures the session into the inbox and, with --process, starts
  `zk process` in the background to turn the inbox into notes.
- SessionStart prints the vault brief (recall.brief) as context, and starts
  `zk learn` in the background when the session is in a repository zk hasn't read.
- PreToolUse on Grep or Glob prints a one-line nudge, once per session. On Read of a
  long doc that `zk docs` imported, it turns the read back once, pointing at the doc's
  sections in the vault.
- PostToolUseFailure on a shell command prints notes that clearly match the error.
- UserPromptSubmit prints pointers to notes that clearly match the prompt, if any.

Hooks run on every session and failed command, so this imports only what the
event needs (never numpy), and it never fails: problems go to stderr, exit code 0.
zk's own headless runs (ZK_INGESTING) get no brief, nudge or notes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def read_event() -> dict:
    """The hook's JSON event from stdin, with or without a UTF-8 byte-order mark.

    PowerShell pipes add a BOM, which `json.load(sys.stdin)` rejects.
    """
    data = sys.stdin.buffer.read() if hasattr(sys.stdin, "buffer") else sys.stdin.read()
    text = data.decode("utf-8-sig") if isinstance(data, bytes) else data.lstrip("\ufeff")
    return json.loads(text)


def _context(event_name: str, text: str | None) -> None:
    if text:
        out = {"hookSpecificOutput": {"hookEventName": event_name, "additionalContext": text}}
        print(json.dumps(out, ensure_ascii=False))


def session_end(vault_dir: Path, event: dict, process: bool) -> None:
    from zk import sessions

    saved, why = sessions.from_hook(vault_dir, event)
    print(f"zk hook: {'saved ' + saved if saved else 'skipped'} ({why})", file=sys.stderr)
    if saved and process:
        sessions.start_process(vault_dir)


def session_start(vault_dir: Path, event: dict) -> None:
    from zk import learn, recall

    recall.clear_old_state()
    conn = recall.open_index(vault_dir)
    if conn is not None:
        try:
            _context("SessionStart", recall.brief(conn, event.get("cwd")))
        finally:
            conn.close()
    root = recall.project_root(event.get("cwd"))
    if (
        event.get("source", "startup") == "startup"
        and root is not None
        and root.resolve() != vault_dir.resolve()
    ):
        from zk import docs

        if not learn.is_known(root):
            learn.start(vault_dir, root)
        if docs.stale(root):  # its docs changed since they were put in the vault
            docs.start(vault_dir, root)


def pre_tool(event: dict) -> None:
    from zk import recall

    if event.get("tool_name") == "Read":
        reason = recall.on_read(event)
        if reason:
            decision = {"permissionDecision": "deny", "permissionDecisionReason": reason}
            out = {"hookSpecificOutput": {"hookEventName": "PreToolUse", **decision}}
            print(json.dumps(out, ensure_ascii=False))
        return
    _context("PreToolUse", recall.nudge(event))


def tool_failed(vault_dir: Path, event: dict) -> None:
    from zk import recall

    conn = recall.open_index(vault_dir)
    if conn is None:
        return
    try:
        _context("PostToolUseFailure", recall.on_failure(conn, event))
    finally:
        conn.close()


def prompt_submitted(vault_dir: Path, event: dict) -> None:
    from zk import recall

    conn = recall.open_index(vault_dir)
    if conn is None:
        return
    try:
        _context("UserPromptSubmit", recall.on_prompt(conn, event))
    finally:
        conn.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m zk.hook")
    parser.add_argument("vault", type=Path)
    parser.add_argument(
        "--process", action="store_true", help="at session end, start `zk process` too"
    )
    args = parser.parse_args(argv)
    try:
        event = read_event()
        name = event.get("hook_event_name") or "SessionEnd"
        if name == "SessionEnd":
            session_end(args.vault, event, args.process)
        elif os.environ.get("ZK_INGESTING"):
            pass  # zk's own headless runs: no brief, nudge or notes
        elif name == "SessionStart":
            session_start(args.vault, event)
        elif name == "PreToolUse":
            pre_tool(event)
        elif name == "PostToolUseFailure":
            tool_failed(args.vault, event)
        elif name == "UserPromptSubmit":
            prompt_submitted(args.vault, event)
    except Exception as exc:  # a broken hook must never break the agent that ran it
        print(f"zk hook: failed: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
