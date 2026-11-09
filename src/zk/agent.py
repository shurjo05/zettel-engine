"""One-shot questions to a headless Claude Code (`claude -p`), for answering and judging.

zk has no API client: these runs use the person's own Claude Code login, like
`zk process`. Each prompt starts with the `[zk]` marker and runs with
ZK_INGESTING=1, so the session is never captured as a conversation. It runs in
an empty temporary folder with one turn, so it can only answer from the prompt.
Everything in the prompt goes to Anthropic (docs/data-flow.md).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable

from zk.capture import ZK_MARKER
from zk.sessions import INGESTING

TIMEOUT = 10 * 60

Ask = Callable[[str], str]  # prompt in, reply text out


class AgentError(RuntimeError):
    pass


def claude(model: str | None = None) -> Ask:
    exe = shutil.which("claude") or "claude"

    def ask(prompt: str) -> str:
        command = [exe, "-p", "--output-format", "text", "--max-turns", "1"]
        if model:
            command += ["--model", model]
        flags = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
        with tempfile.TemporaryDirectory(prefix="zk-ask-") as empty:
            try:
                done = subprocess.run(
                    command,
                    input=f"{ZK_MARKER} {prompt}",
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    cwd=empty,
                    env={**os.environ, INGESTING: "1"},
                    timeout=TIMEOUT,
                    **flags,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise AgentError(f"claude didn't run: {exc}") from exc
        if done.returncode != 0:
            raise AgentError(f"claude failed: {(done.stderr or done.stdout).strip()[:500]}")
        return done.stdout.strip()

    return ask
