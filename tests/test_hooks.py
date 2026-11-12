import json
import os
import sys

import pytest

from zk import hooks, process
from zk.cli import main


def fake_agent(script: str) -> str:
    """A shell command that runs a small Python script in place of a headless agent."""
    return f'"{sys.executable}" -c "{script}"'


def test_install_only_adds_and_backs_up(vault, isolated_homes, capsys):
    settings = isolated_homes / "claude" / "settings.json"
    settings.parent.mkdir(parents=True)
    mine = {
        "model": "opus",
        "hooks": {"SessionEnd": [{"hooks": [{"type": "command", "command": "my-own-hook"}]}]},
    }
    settings.write_text(json.dumps(mine), encoding="utf-8")
    assert main(["hooks", "install", str(vault), "--dry-run"]) == 0
    assert "+" in capsys.readouterr().out and json.loads(settings.read_text()) == mine

    assert main(["hooks", "install", str(vault)]) == 0
    data = json.loads(settings.read_text())
    assert data["model"] == "opus"
    commands = [h["command"] for g in data["hooks"]["SessionEnd"] for h in g["hooks"]]
    assert commands[0] == "my-own-hook" and "-m zk.hook" in commands[1]
    assert commands[1].endswith("--process")
    backups = list(settings.parent.glob("settings.json.zk-backup-*"))
    assert len(backups) == 1 and json.loads(backups[0].read_text()) == mine

    # A second install changes nothing.
    assert main(["hooks", "install", str(vault)]) == 0
    assert "already installed" in capsys.readouterr().out
    assert json.loads(settings.read_text()) == data


def test_install_without_processing_and_status(vault, isolated_homes, capsys):
    assert main(["hooks", "install", str(vault), "--no-process"]) == 0
    data = json.loads((isolated_homes / "claude" / "settings.json").read_text())
    entry = data["hooks"]["SessionEnd"][0]["hooks"][0]
    assert entry["timeout"] == 10 and not entry["command"].endswith("--process")
    assert main(["hooks", "status", str(vault)]) == 0
    out = capsys.readouterr().out
    assert "SessionEnd: " in out and "codex" not in out


def test_install_refuses_a_settings_file_it_cannot_read(vault, isolated_homes):
    settings = isolated_homes / "claude" / "settings.json"
    settings.parent.mkdir(parents=True)
    settings.write_text("{ not json", encoding="utf-8")
    assert main(["hooks", "install", str(vault)]) == 1
    assert settings.read_text() == "{ not json"


def test_hook_command_survives_any_shell():
    command = hooks.hook_command(hooks.Path("/vault"))
    assert "\\" not in command


def test_schedule_task_runs_catch_up_and_starts_when_available(vault):
    xml = hooks.task_xml(vault, "21:30", today="2026-11-03")
    assert "<StartBoundary>2026-11-03T21:30:00</StartBoundary>" in xml
    assert "<StartWhenAvailable>true</StartWhenAvailable>" in xml
    assert "process" in xml and "--catch-up" in xml


def test_schedule_off_windows_prints_a_cron_line(vault, capsys):
    if os.name == "nt":
        pytest.skip("Windows installs the task")
    assert main(["schedule", "install", str(vault), "--at", "07:05"]) == 0
    assert capsys.readouterr().out.splitlines()[-1].startswith("05 07 * * * ")


def write_capture(vault, name):
    (vault / "inbox" / name).write_text("---\nkind: capture\n---\n# Capture\n", encoding="utf-8")


def test_process_skips_the_model_when_there_is_no_work(vault, tmp_path):
    for f in (vault / "inbox").glob("*.md"):
        f.unlink()
    marker = tmp_path / "ran"
    touch = fake_agent(f"open(r'{marker}', 'a').write('x')")
    summary = process.run(vault, commands={"ingest": touch, "resolve": touch})
    assert summary["ingest_runs"] == summary["resolve_runs"] == 0 and not marker.exists()


def test_process_ingests_then_resolves_holds(vault, tmp_path):
    for f in (vault / "inbox").glob("*.md"):
        f.unlink()
    write_capture(vault, "2026-11-03-1000-claude-abcd.md")
    calls = tmp_path / "calls"
    # The stand-in ingest agent holds one note and processes the capture.
    ingest = fake_agent(
        "import os, pathlib; "
        f"open(r'{calls}', 'a').write('ingest ' + os.environ['ZK_INGESTING'] + chr(10)); "
        "pathlib.Path('inbox/held').mkdir(exist_ok=True); "
        "pathlib.Path('inbox/held/dup.md').write_text('---\\ntitle: dup\\n---\\n'); "
        "pathlib.Path('inbox/2026-11-03-1000-claude-abcd.md').rename('inbox/done.txt')"
    )
    resolve = fake_agent(f"open(r'{calls}', 'a').write('resolve' + chr(10))")
    summary = process.run(vault, commands={"ingest": ingest, "resolve": resolve})
    assert calls.read_text().splitlines() == ["ingest 1", "resolve"]
    assert summary["ingest_runs"] == 1 and summary["resolve_runs"] == 1
    log = open(summary["log"], encoding="utf-8").read()
    assert "1 capture(s)" in log and "1 held note(s)" in log
    assert not process.lock_path().exists()


def test_a_capture_the_agent_leaves_is_not_retried_in_the_same_run(vault, tmp_path):
    write_capture(vault, "2026-11-03-1000-claude-stuck.md")
    calls = tmp_path / "calls"
    noop = fake_agent(f"open(r'{calls}', 'a').write('x')")
    assert process.run(vault, commands={"ingest": noop, "resolve": noop})["ingest_runs"] == 1


def test_only_one_process_runs_at_a_time(vault, capsys):
    process.lock_path().parent.mkdir(parents=True, exist_ok=True)
    process.lock_path().write_text(f"{os.getpid()}\n")  # held by a process that's running
    assert main(["process", str(vault)]) == 0
    assert "another zk process is running" in capsys.readouterr().err
    # A lock left by a run that died long ago is taken over.
    old = process.STALE_LOCK + 60
    os.utime(process.lock_path(), (os.path.getmtime(process.lock_path()) - old,) * 2)
    for f in (vault / "inbox").glob("*.md"):
        f.unlink()
    with process.Lock(process.lock_path()):
        pass
    assert not process.lock_path().exists()


def test_a_lock_whose_process_has_ended_is_taken_over_at_once(vault):
    import subprocess
    import sys

    ended = subprocess.Popen([sys.executable, "-c", "pass"])
    ended.wait()
    assert not process.pid_running(ended.pid) and process.pid_running(os.getpid())
    process.lock_path().parent.mkdir(parents=True, exist_ok=True)
    process.lock_path().write_text(f"{ended.pid}\n")  # fresh, but its holder crashed
    with process.Lock(process.lock_path()):
        assert process.lock_path().read_text().strip() == str(os.getpid())


def test_default_agents_are_marked_and_bring_their_own_write_server(vault):
    commands = process.default_commands("opus", vault)
    assert commands["ingest"][2].startswith("[zk]") and "mcp__zk" in commands["ingest"]
    assert commands["resolve"][-2:] == ["--model", "opus"]
    assert "--strict-mcp-config" in commands["ingest"]
    config_file = commands["ingest"][commands["ingest"].index("--mcp-config") + 1]
    config = json.loads(open(config_file).read())
    assert config["mcpServers"]["zk"]["args"][-1] == "--allow-writes"


def test_installs_warn_when_sessions_have_no_read_only_zk_server(vault, monkeypatch, capsys):
    monkeypatch.setattr(hooks, "server_ready", lambda name="zk": (False, "no MCP server named zk"))
    assert main(["hooks", "install", str(vault), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "warning: no MCP server named zk" in out
    # Everyday sessions get a read-only server; zk's own runs bring a write-enabled one.
    assert "claude mcp add zk --scope user --" in out and "--allow-writes" not in out
    monkeypatch.setattr(hooks, "server_ready", lambda name="zk": (True, "registered read-only"))
    assert main(["hooks", "install", str(vault), "--dry-run"]) == 0
    assert "warning" not in capsys.readouterr().out
