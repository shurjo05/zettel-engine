"""Every child process zk starts, started so it never opens a window on Windows.

zk runs in the background: from hooks (often through pythonw.exe, which has no console),
from the scheduled task, and as detached runs started at session end. A console program
started from a process with no console opens a new, visible console window unless it is
started with CREATE_NO_WINDOW, so each git, claude, schtasks or pdftotext call would flash
a window on the person's screen. DETACHED_PROCESS is never used: it also leaves the child
without a console, and Windows ignores CREATE_NO_WINDOW when both are given, so every
program that child starts would open a window again. With CREATE_NO_WINDOW alone the
child gets a hidden console, and what it starts shares it.

Light on purpose (hooks import it): standard library only.
"""

from __future__ import annotations

import os
import subprocess

# The values from the Windows SDK, so the flags can be tested on any system.
CREATE_NEW_PROCESS_GROUP = 0x00000200
CREATE_BREAKAWAY_FROM_JOB = 0x01000000
CREATE_NO_WINDOW = 0x08000000


def _windows() -> bool:
    return os.name == "nt"


def hidden(kwargs: dict) -> dict:
    """`kwargs` for subprocess with CREATE_NO_WINDOW added on Windows."""
    if not _windows():
        return kwargs
    return {**kwargs, "creationflags": kwargs.get("creationflags", 0) | CREATE_NO_WINDOW}


def run(args, **kwargs) -> subprocess.CompletedProcess:
    """subprocess.run, with no window on Windows."""
    return subprocess.run(args, **hidden(kwargs))


def background_flags(breakaway: bool = False) -> dict:
    """How to start a run that outlives the process starting it, hidden: its own process
    group on Windows (out of the app's job object too, with `breakaway`), its own session
    elsewhere."""
    if not _windows():
        return {"start_new_session": True}
    flags = CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW
    return {"creationflags": flags | (CREATE_BREAKAWAY_FROM_JOB if breakaway else 0)}


def start(args, breakaway: bool = False, **kwargs) -> subprocess.Popen:
    """Start a background run that doesn't wait for or tie up its parent, with no window.
    With `breakaway`, it also leaves the app's job object, so closing the app doesn't end
    it, when the job allows that."""
    try:
        return subprocess.Popen(args, **background_flags(breakaway), **kwargs)
    except OSError:
        if not (breakaway and _windows()):
            raise
        # The job doesn't allow breaking away: a run in its own group is the best we can do.
        return subprocess.Popen(args, **background_flags(), **kwargs)
