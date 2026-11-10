"""Install zk's Claude Code hooks, its rules file, and its scheduled tasks.

The hooks (see hook.py): SessionEnd captures the session; SessionStart gives
the vault brief; PreToolUse on Grep and Glob nudges once; PostToolUseFailure on
a shell command shows notes that match the error. The rules file,
~/.claude/rules/zk.md, says when searching the vault is worth it; Claude Code
loads it in every project, so nothing is written into the person's repositories.
The scheduled tasks are the daily catch-up and the weekly health review.

These write files that belong to the person, not to zk: Claude Code's
settings.json, its rules folder and Windows scheduled tasks. So installing
only ever adds. It never changes or removes an entry already there, adds a hook
only for an event that has no zk hook yet, never overwrites a rules file, backs
settings.json up first, and `dry_run` shows the change without making it.
`remove_schedule` removes only zk's own tasks.
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

from zk.sessions import claude_home

TASK_NAME = "zk catch-up"
REVIEW_TASK = "zk review"
HOOK_TIMEOUT = 10  # seconds; Claude Code lets a hook set its own
NUDGE_TIMEOUT = 5
RULES_NAME = "zk.md"
RULES = """# zk vault

The zk tools (search_notes, get_note) search my personal notes: how I built past projects,
errors I've solved, concepts I've learned, and my style rules for school work. Search them
before other research when:
- I'm doing school work: find my style rules for that class and follow them.
- I'm building something: find how I built similar things and reuse what fits.
- I hit an error, or a concept I've struggled with.
Skip it for quick, self-contained questions. Summaries come first; open a note or map with
get_note only when the summary isn't enough. If nothing turns up, say so and carry on.
"""


def settings_path() -> Path:
    return claude_home() / "settings.json"


def rules_path() -> Path:
    return claude_home() / "rules" / RULES_NAME


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


def server_command(vault_dir: Path) -> str:
    """How to register the read-only zk MCP server for everyday sessions.

    zk's own headless runs bring their own write-enabled server (process.mcp_config).
    """
    return (
        f"claude mcp add zk --scope user -- {_arg(python_exe())} -m zk serve "
        f"{_arg(str(vault_dir.resolve()))}"
    )


def server_ready() -> tuple[bool | None, str]:
    """Whether everyday sessions have a read-only zk server; None if Claude can't be asked."""
    claude = shutil.which("claude")
    if not claude:
        return None, "claude is not on PATH, so it couldn't be checked"
    try:
        done = subprocess.run(
            [claude, "mcp", "get", "zk"], capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, f"couldn't ask claude ({exc})"
    if done.returncode != 0:
        return False, "no MCP server named zk is registered, so sessions can't search the vault"
    if "--allow-writes" in done.stdout:
        return False, (
            "the zk server is registered with --allow-writes, so every session loads the write "
            "tools and their instructions; zk's own runs no longer need it"
        )
    return True, "registered read-only"


def _is_zk(command: str) -> bool:
    return "zk.hook" in command or ("zk capture" in command and "--hook" in command)


def _zk_commands(data: dict) -> dict[str, list[str]]:
    hooks = data.get("hooks")
    out: dict[str, list[str]] = {}
    for event, groups in hooks.items() if isinstance(hooks, dict) else []:
        found = [
            h.get("command", "")
            for g in (groups if isinstance(groups, list) else [])
            if isinstance(g, dict)
            for h in g.get("hooks") or []
            if isinstance(h, dict) and _is_zk(h.get("command", ""))
        ]
        if found:
            out[event] = found
    return out


def installed() -> dict[str, list[str]]:
    """The zk hook commands already in Claude Code's user settings, by event."""
    data, _ = _load(settings_path())
    return _zk_commands(data)


def hook_groups(vault_dir: Path, process: bool = True, recall: bool = True) -> dict[str, dict]:
    """The settings entry zk adds for each hook event."""
    command = hook_command(vault_dir, process=False)

    def group(cmd: str, timeout: int, matcher: str | None = None) -> dict:
        entry = {"hooks": [{"type": "command", "command": cmd, "timeout": timeout}]}
        return {"matcher": matcher, **entry} if matcher else entry

    out = {"SessionEnd": group(hook_command(vault_dir, process), HOOK_TIMEOUT)}
    if recall:
        out["SessionStart"] = group(command, HOOK_TIMEOUT)
        out["PreToolUse"] = group(command, NUDGE_TIMEOUT, "Grep|Glob")
        out["PostToolUseFailure"] = group(command, HOOK_TIMEOUT, "Bash|PowerShell")
    return out


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


def install(
    vault_dir: Path, process: bool = True, dry_run: bool = False, recall: bool = True
) -> dict:
    """Add zk's hooks to Claude Code's user settings, for each event that has none yet,
    and the rules file if there is none. With recall=False, only the SessionEnd capture."""
    path = settings_path()
    data, before = _load(path)
    existing = _zk_commands(data)
    hooks = data.setdefault("hooks", {})
    if not isinstance(hooks, dict):
        raise ValueError(f"{path} has an unexpected 'hooks' shape; not changing it")
    added: list[str] = []
    for event, group in hook_groups(vault_dir, process, recall).items():
        if existing.get(event):
            continue
        if not isinstance(hooks.setdefault(event, []), list):
            raise ValueError(f"{path} has an unexpected 'hooks.{event}' shape; not changing it")
        hooks[event].append(group)
        added.append(event)
    rules = rules_path()
    add_rules = recall and not rules.exists()
    result: dict = {"path": str(path), "added": added, "existing": existing}
    if not added and not add_rules:
        result["action"] = "already installed"
        return result
    after = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    result["diff"] = "".join(
        difflib.unified_diff(
            before.splitlines(True), after.splitlines(True), str(path), f"{path} (with zk)"
        )
    )
    if add_rules:
        result["rules"] = str(rules)
    result["action"] = "would add"
    if dry_run:
        return result
    if added:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            backup = path.with_name(f"{path.name}.zk-backup-{datetime.now():%Y%m%d-%H%M%S}")
            shutil.copy2(path, backup)
            result["backup"] = str(backup)
        tmp = path.with_name(path.name + ".zk-tmp")
        tmp.write_text(after, encoding="utf-8")
        os.replace(tmp, path)
    if add_rules:
        rules.parent.mkdir(parents=True, exist_ok=True)
        with rules.open("x", encoding="utf-8") as f:  # "x": never overwrite a rules file
            f.write(RULES)
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


def review_args(vault_dir: Path) -> list[str]:
    return [python_exe(windowless=True), "-m", "zk", "review", str(vault_dir.resolve())]


DAYS = {
    "mon": "Monday", "tue": "Tuesday", "wed": "Wednesday", "thu": "Thursday",
    "fri": "Friday", "sat": "Saturday", "sun": "Sunday",
}  # fmt: skip


def task_xml(
    vault_dir: Path, at: str, today: str | None = None, weekly_on: str | None = None
) -> str:
    """A daily catch-up task, or with `weekly_on` (mon..sun) the weekly review. Either also
    runs at the next chance when the PC was off at `at`."""
    exe, *args = review_args(vault_dir) if weekly_on else schedule_args(vault_dir)
    arguments = subprocess.list2cmdline(args)
    today = today or datetime.now().strftime("%Y-%m-%d")
    if weekly_on:
        what = "zk: weekly vault health review (no model)"
        schedule = (
            "<ScheduleByWeek><WeeksInterval>1</WeeksInterval><DaysOfWeek>"
            f"<{DAYS[weekly_on]} /></DaysOfWeek></ScheduleByWeek>"
        )
    else:
        what = "zk: capture sessions the hooks missed, then ingest and resolve"
        schedule = "<ScheduleByDay><DaysInterval>1</DaysInterval></ScheduleByDay>"
    return f"""<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Description>{what}</Description>
  </RegistrationInfo>
  <Triggers>
    <CalendarTrigger>
      <StartBoundary>{today}T{at}:00</StartBoundary>
      {schedule}
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


def install_schedule(
    vault_dir: Path,
    at: str,
    xml_dir: Path,
    dry_run: bool = False,
    review_on: str | None = "sun",
    review_at: str = "20:00",
) -> dict:
    """The daily catch-up at `at`, and (unless review_on is None) the weekly review."""
    tasks = [(TASK_NAME, "zk-catch-up.xml", task_xml(vault_dir, at))]
    if review_on:
        tasks.append(
            (REVIEW_TASK, "zk-review.xml", task_xml(vault_dir, review_at, weekly_on=review_on))
        )
    if dry_run or os.name != "nt":
        cron = []
        if review_on:
            weekday = list(DAYS).index(review_on) + 1 if review_on != "sun" else 0
            cron.append(
                f"{review_at[3:]} {review_at[:2]} * * {weekday} "
                + subprocess.list2cmdline(review_args(vault_dir))
            )
        cron.append(f"{at[3:]} {at[:2]} * * * " + subprocess.list2cmdline(schedule_args(vault_dir)))
        return {
            "action": "would add" if os.name == "nt" else "not on Windows",
            "xml": "\n".join(x for _, _, x in tasks),
            "cron": "\n".join(cron),
        }
    xml_dir.mkdir(parents=True, exist_ok=True)
    for name, file, xml in tasks:
        path = xml_dir / file
        path.write_text(xml, encoding="utf-16")
        done = _schtasks("/Create", "/TN", name, "/XML", str(path), "/F")  # /F: zk's own task
        if done.returncode:
            raise RuntimeError(
                f"schtasks failed for {name}: {(done.stderr or done.stdout).strip()}"
            )
    return {
        "action": "added",
        "task": TASK_NAME,
        "at": at,
        "review": f"{DAYS[review_on]} {review_at}" if review_on else None,
    }


def schedule_status(name: str = TASK_NAME) -> str | None:
    if os.name != "nt":
        return None
    done = _schtasks("/Query", "/TN", name, "/FO", "LIST", "/V")
    return done.stdout.strip() if done.returncode == 0 else None


def remove_schedule() -> bool:
    if os.name != "nt":
        return False
    removed = False
    for name in (TASK_NAME, REVIEW_TASK):
        if schedule_status(name) is not None:
            removed |= _schtasks("/Delete", "/TN", name, "/F").returncode == 0
    return removed
