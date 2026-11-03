"""The session-end hook: `python -m zk.hook VAULT [--process]`.

Claude Code and Codex run this when a session ends, with the event as JSON on
stdin. It captures the session into the inbox and, with --process, starts
`zk process` in the background to turn the inbox into notes. Agents give
session-end hooks one to a few seconds, so this imports only what capturing
needs, and it never fails: any problem is reported on stderr with exit code 0.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def read_event() -> dict:
    """The hook's JSON event from stdin, with or without a UTF-8 byte-order mark.

    PowerShell pipes add a BOM, which `json.load(sys.stdin)` rejects.
    """
    data = sys.stdin.buffer.read() if hasattr(sys.stdin, "buffer") else sys.stdin.read()
    return json.loads(
        data.decode("utf-8-sig") if isinstance(data, bytes) else data.lstrip("\ufeff")
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m zk.hook")
    parser.add_argument("vault", type=Path)
    parser.add_argument(
        "--process", action="store_true", help="then start `zk process` in the background"
    )
    args = parser.parse_args(argv)
    try:
        from zk import sessions

        saved, why = sessions.from_hook(args.vault, read_event())
        print(f"zk hook: {'saved ' + saved if saved else 'skipped'} ({why})", file=sys.stderr)
        if saved and args.process:
            sessions.start_process(args.vault)
    except Exception as exc:  # a broken capture must never break the agent that ran the hook
        print(f"zk hook: failed: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
