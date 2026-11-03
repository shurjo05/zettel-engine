"""Install zk's session-end hook for Claude Code and Codex, and its scheduled catch-up.

These write files that belong to the person, not to zk: Claude Code's
settings.json, Codex's hooks.json and a Windows scheduled task. So installing
only ever adds an entry. It never changes or removes one already there, it is a
no-op when zk's hook is already installed, it backs the file up first, and
`dry_run` shows the change without making it. `remove_schedule` removes only
zk's own task.
"""

from __future__ import annotations

import difflib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape

from zk.sessions import claude_home, codex_home

TASK_NAME = "zk catch-up"
# Claude Code lets a hook set its own timeout; Codex caps session-end hooks at 3 seconds.
TIMEOUTS = {"claude": 10, "codex": 3}


def settings_path(agent: str) -> Path:
    return claude_home() / "settings.json" if agent == "claude" else codex_home() / "hooks.json"


def python_exe(windowless: bool = False) -> str:
    """This environment's python, so hooks run zk without uv (and fast)."""
    exe = Path(sys.executable)
    if windowless and os.name == "nt":
        quiet = exe.with_name("pythonw.exe")
        if quiet.exists():
            return str(quiet)
    return str(exe)


def _arg(value: str) -> str:
    """A path for a command line that cmd, PowerShell and Git Bash all read the same way.

    Forward slashes need no escaping in bash, and an unquoted path needs no `&` in
    PowerShell, so a path is quoted only when it has a space.
    """
    value = value.replace("\\", "/")
    return f'"{value}"' if " " in value else value


def hook_command(vault_dir: Path, process: bool = True) -> str:
    command = f"{_arg(python_exe())} -m zk.hook {_arg(str(vault_dir.resolve()))}"
    return command + " --process" if process else command


def _is_zk(command: str) -> bool:
    return "zk.hook" in command or ("zk capture" in command and "--hook" in command)


def installed(agent: str) -> list[str]:
    """The zk hook commands already in an agent's settings."""
    data, _ = _load(settings_path(agent))
    hooks = data.get("hooks")
    groups = hooks.get("SessionEnd") if isinstance(hooks, dict) else None
    groups = groups if isinstance(groups, list) else []
    return [
        h.get("command", "")
        for g in groups
        if isinstance(g, dict)
        for h in g.get("hooks") or []
        if isinstance(h, dict) and _is_zk(h.get("command", ""))
    ]


def _load(path: Path) -> tuple[dict, str]:
    if not path.exists():
        return {}, ""
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text) if text.strip() else {}
    except ValueError as exc:
        raise ValueError(f"{path} is not valid JSON ({exc}); not changing it") from None
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a JSON object; not changing it")
    return data, text


def install(agent: str, vault_dir: Path, process: bool = True, dry_run: bool = False) -> dict:
    """Add zk's SessionEnd hook to an agent's user settings, if no zk hook is there yet."""
    path = settings_path(agent)
    data, before = _load(path)
    existing = installed(agent)
    if existing:
        return {
            "agent": agent,
            "path": str(path),
            "action": "already installed",
            "commands": existing,
        }
    hooks = data.setdefault("hooks", {})
    if not isinstance(hooks, dict) or not isinstance(hooks.setdefault("SessionEnd", []), list):
        raise ValueError(f"{path} has an unexpected 'hooks' shape; not changing it")
    command = hook_command(vault_dir, process)
    hooks["SessionEnd"].append(
        {"hooks": [{"type": "command", "command": command, "timeout": TIMEOUTS[agent]}]}
    )
    after = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    diff = "".join(
        difflib.unified_diff(
            before.splitlines(True), after.splitlines(True), str(path), f"{path} (with zk)"
        )
    )
    result = {
        "agent": agent,
        "path": str(path),
        "action": "would add",
        "diff": diff,
        "commands": [command],
    }
    if dry_run:
        return result
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        backup = path.with_name(f"{path.name}.zk-backup-{datetime.now():%Y%m%d-%H%M%S}")
        shutil.copy2(path, backup)
        result["backup"] = str(backup)
    tmp = path.with_name(path.name + ".zk-tmp")
    tmp.write_text(after, encoding="utf-8")
    os.replace(tmp, path)
    result["action"] = "added"
    return result


# The scheduled catch-up (Windows Task Scheduler).


def schedule_args(vault_dir: Path) -> list[str]:
    return [
        python_exe(windowless=True),
        "-m",
        "zk",
        "process",
        str(vault_dir.resolve()),
        "--catch-up",
    ]


def task_xml(vault_dir: Path, at: str, today: str | None = None) -> str:
    """A daily task that also runs at the next chance when the PC was off at `at`."""
    exe, *args = schedule_args(vault_dir)
    arguments = subprocess.list2cmdline(args)
    today = today or datetime.now().strftime("%Y-%m-%d")
    return f"""<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Description>zk: capture sessions the hooks missed, then ingest and resolve</Description>
  </RegistrationInfo>
  <Triggers>
    <CalendarTrigger>
      <StartBoundary>{today}T{at}:00</StartBoundary>
      <ScheduleByDay><DaysInterval>1</DaysInterval></ScheduleByDay>
    </CalendarTrigger>
  </Triggers>
  <Principals>
    <Principal id="Author"><LogonType>InteractiveToken</LogonType></Principal>
  </Principals>
  <Settings>
    <StartWhenAvailable>true</StartWhenAvailable>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <ExecutionTimeLimit>PT2H</ExecutionTimeLimit>
  </Settings>
  <Actions Context="Author">
    <Exec>
      <Command>{escape(exe)}</Command>
      <Arguments>{escape(arguments)}</Arguments>
      <WorkingDirectory>{escape(str(vault_dir.resolve()))}</WorkingDirectory>
    </Exec>
  </Actions>
</Task>
"""


def _schtasks(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["schtasks", *args], capture_output=True, text=True)


def install_schedule(vault_dir: Path, at: str, xml_dir: Path, dry_run: bool = False) -> dict:
    xml = task_xml(vault_dir, at)
    if dry_run or os.name != "nt":
        return {
            "action": "would add" if os.name == "nt" else "not on Windows",
            "xml": xml,
            "cron": f"{at[3:]} {at[:2]} * * * " + subprocess.list2cmdline(schedule_args(vault_dir)),
        }
    xml_dir.mkdir(parents=True, exist_ok=True)
    path = xml_dir / "zk-catch-up.xml"
    path.write_text(xml, encoding="utf-16")
    done = _schtasks("/Create", "/TN", TASK_NAME, "/XML", str(path), "/F")  # /F: zk's own task
    if done.returncode:
        raise RuntimeError(f"schtasks failed: {(done.stderr or done.stdout).strip()}")
    return {"action": "added", "task": TASK_NAME, "at": at}


def schedule_status() -> str | None:
    if os.name != "nt":
        return None
    done = _schtasks("/Query", "/TN", TASK_NAME, "/FO", "LIST", "/V")
    return done.stdout.strip() if done.returncode == 0 else None


def remove_schedule() -> bool:
    if os.name != "nt" or schedule_status() is None:
        return False
    return _schtasks("/Delete", "/TN", TASK_NAME, "/F").returncode == 0
