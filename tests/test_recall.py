import io
import json
import subprocess
from pathlib import Path

import pytest

from zk import hook, hooks, ingest, learn, maps, process, recall, review
from zk.embed import update_embeddings
from zk.index import build_index, default_db_path, open_db
from zk.proposals import Proposal
from zk.search import search

TOPIC = "Reciprocal rank fusion merges ranked lists from keyword and vector search"


def note(vault, name, title, summary, type="concept", sources=("conversation:2026-11-10",)):
    tags = '["retrieval", "fusion"]'
    text = (
        f'---\nid: "{name}"\ntitle: "{title}"\ntype: {type}\ntags: {tags}\n'
        f"sources: {json.dumps(list(sources))}\ncreated: 2026-11-10\n---\n"
        f"# {title}\n\n## Summary\n\n{summary}\n"
    )
    (vault / "notes" / f"{name}.md").write_text(text, encoding="utf-8")


@pytest.fixture
def mapped(vault, embedder):
    """The fixture vault plus a tight topic (five near-identical notes) and a project."""
    for i in range(5):
        note(vault, f"fusion-{i}", f"Rank fusion {i}", f"{TOPIC}, variant {i}.")
    note(vault, "robot-arm", "Arm kinematics", "Joint angles from the end position.",
         type="pattern", sources=["project:Robot-Arm:2026-11-01"])  # fmt: skip
    note(vault, "robot-camera", "Camera sync", "Trigger both cameras from one clock.",
         type="problem", sources=["project:robot-arm"])  # fmt: skip
    db = default_db_path(vault)
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = open_db(db)
    build_index(conn, vault)
    update_embeddings(conn, embedder)
    yield vault, conn, embedder
    conn.close()


def refresh(vault, conn, embedder):
    build_index(conn, vault)
    update_embeddings(conn, embedder)


def test_maps_are_written_kept_and_left_alone_once_edited(mapped):
    vault, conn, embedder = mapped
    first = maps.update(vault, conn, embedder)
    assert sorted(first["new"]) == ["maps/map-retrieval-fusion.md", "maps/project-robot-arm.md"]
    topic = (vault / "maps" / "map-retrieval-fusion.md").read_text(encoding="utf-8")
    assert 'type: "map"' in topic and "[[fusion-0]]" in topic and "[[robot-arm]]" not in topic
    project = (vault / "maps" / "project-robot-arm.md").read_text(encoding="utf-8")
    assert "[[robot-arm]]" in project and "[[robot-camera]]" in project

    refresh(vault, conn, embedder)
    again = maps.update(vault, conn, embedder)
    assert again["new"] == again["updated"] == again["retired"] == [] and again["unchanged"] == 2

    # A new note in the project rewrites its map.
    note(vault, "robot-gripper", "Gripper", "Limit force.", sources=["project:robot-arm"])
    refresh(vault, conn, embedder)
    assert maps.update(vault, conn, embedder)["updated"] == ["maps/project-robot-arm.md"]

    # Once edited by hand, a map is never rewritten or retired.
    path = vault / "maps" / "project-robot-arm.md"
    path.write_text(path.read_text(encoding="utf-8") + "\nMy own line.\n", encoding="utf-8")
    note(vault, "robot-base", "Base plate", "Bolt pattern.", sources=["project:robot-arm"])
    refresh(vault, conn, embedder)
    result = maps.update(vault, conn, embedder)
    assert result["edited"] == ["maps/project-robot-arm.md"] and "My own line" in path.read_text()


def test_a_map_whose_topic_is_gone_is_retired_not_deleted(mapped):
    vault, conn, embedder = mapped
    maps.update(vault, conn, embedder)
    for i in range(5):
        (vault / "notes" / f"fusion-{i}.md").unlink()  # the test removes notes; zk never does
    refresh(vault, conn, embedder)
    result = maps.update(vault, conn, embedder)
    assert result["retired"] == ["maps/map-retrieval-fusion.md"]
    assert (vault / "maps" / "retired" / "map-retrieval-fusion.md").exists()
    refresh(vault, conn, embedder)  # maps/retired/ is not indexed
    assert not conn.execute("SELECT 1 FROM notes WHERE path LIKE 'maps/retired/%'").fetchall()


def test_project_names_are_normalised():
    assert maps.projects_of(["project:Robot Arm:2026", "capture:x", "project:robot-arm"]) == [
        "robot-arm"
    ]
    assert maps.projects_of(
        ["local-repo:rpi5_baseline/launch/x.py", "repo:NautilusTrx", "local-repo:ksu-cis300 (all)"]
    ) == ["rpi5-baseline", "nautilustrx", "ksu-cis300"]
    groups = maps.group_projects(
        ["ruflo-claude-flow", "ruflo", "chat501", "chat", "aphiddetection", "aphid-detection"]
    )
    assert groups == [
        ["chat"],
        ["ruflo", "ruflo-claude-flow"],
        ["chat501"],
        ["aphid-detection", "aphiddetection"],
    ]


def test_a_project_pattern_reaches_topic_maps_only_from_two_projects(mapped):
    vault, conn, embedder = mapped
    note(vault, "fusion-alpha", "Rank fusion alpha", f"{TOPIC}, variant 0.", type="pattern",
         sources=["project:alpha"])  # fmt: skip
    note(vault, "fusion-both", "Rank fusion both", f"{TOPIC}, variant 1.", type="pattern",
         sources=["project:alpha", "project:beta"])  # fmt: skip
    refresh(vault, conn, embedder)
    members = {p for spec in maps.plan(conn, embedder) if spec["kind"] == "topic"
               for p in spec["notes"]}  # fmt: skip
    assert "notes/fusion-both.md" in members and "notes/fusion-alpha.md" not in members


def test_search_leaves_maps_out_unless_asked_and_keeps_to_a_project(mapped):
    vault, conn, embedder = mapped
    maps.update(vault, conn, embedder)
    refresh(vault, conn, embedder)
    hits = search(conn, "rank fusion keyword vector", k=10)
    assert hits and all(not h.path.startswith("maps/") for h in hits)
    assert [h.path for h in search(conn, "fusion", type="map")] == ["maps/map-retrieval-fusion.md"]
    found = {h.path for h in search(conn, "joint cameras clock", k=10, project="Robot Arm")}
    assert found == {"notes/robot-arm.md", "notes/robot-camera.md"}


def test_the_brief_names_maps_and_lists_the_current_project(mapped, tmp_path):
    vault, conn, embedder = mapped
    maps.update(vault, conn, embedder)
    refresh(vault, conn, embedder)
    repo = tmp_path / "Robot-Arm"
    (repo / ".git").mkdir(parents=True)
    (repo / "src").mkdir()
    text = recall.brief(conn, repo / "src")
    assert "Topic maps: map-retrieval-fusion (retrieval, fusion; 5)" in text
    assert "Project maps: project-robot-arm (2)" in text
    assert "This folder is the project robot-arm" in text and "- [[robot-camera]]" in text
    entries = [line for line in text.splitlines() if line.startswith("- [[")]
    assert entries and all(len(line) <= recall.ENTRY_CHARS for line in entries)
    assert recall.brief(conn, project="robot-arm") == text
    elsewhere = recall.brief(conn, tmp_path)
    assert "This folder" not in elsewhere and len(elsewhere) < recall.BRIEF_CHARS


def test_one_project_under_several_names_gets_one_map_and_is_found_from_its_folder(
    mapped, tmp_path
):
    vault, conn, embedder = mapped
    note(vault, "robot-servo", "Servo limits", "Clamp the duty cycle.",
         sources=["local-repo:robot_arm-shurjo05/src/servo.cpp + pwm.cpp"])  # fmt: skip
    for name, source in (("a", "project:ruflo"), ("b", "project:ruflo-claude-flow"),
                         ("c", "repo:Ruflo-Claude-Flow")):  # fmt: skip
        note(vault, f"swarm-{name}", f"Swarm {name}", "Queen agent assigns work.", sources=[source])
    refresh(vault, conn, embedder)
    result = maps.update(vault, conn, embedder)
    assert "maps/project-ruflo.md" in result["new"]
    assert not (vault / "maps" / "project-ruflo-claude-flow.md").exists()
    ruflo = (vault / "maps" / "project-ruflo.md").read_text(encoding="utf-8")
    assert all(f"[[swarm-{n}]]" in ruflo for n in "abc") and "ruflo-claude-flow" in ruflo
    arm = (vault / "maps" / "project-robot-arm.md").read_text(encoding="utf-8")
    assert "[[robot-servo]]" in arm and "robot-arm-shurjo05" in arm
    refresh(vault, conn, embedder)

    # The session's folder may be spelt like any of the project's names.
    for folder, project in (("RobotArm", "robot-arm"), ("ruflo-claude-flow", "ruflo")):
        repo = tmp_path / folder
        (repo / ".git").mkdir(parents=True)
        assert f"This folder is the project {project};" in recall.brief(conn, repo)
    found = {h.path for h in search(conn, "servo duty cycle", k=10, project="robot-arm")}
    assert "notes/robot-servo.md" in found


def test_a_topic_map_is_named_only_by_tags_most_of_its_notes_carry():
    rows = [
        {"path": "a", "tags": '["csharp", "xunit", "testing"]'},
        {"path": "b", "tags": '["csharp", "xunit", "testing"]'},
        {"path": "c", "tags": '["csharp", "linq"]'},
        {"path": "d", "tags": '["csharp", "razor"]'},
    ]
    assert maps._label({"a", "b", "c", "d"}, rows, "Hub title") == "csharp"
    assert maps._label({"c", "d"}, rows[2:], "Hub title") == "csharp"
    assert maps._label({"a", "c"}, [rows[0], {"path": "c", "tags": "[]"}], "Hub") == "Hub"


def test_session_start_hook_prints_the_brief_and_reads_new_repos(
    mapped, tmp_path, monkeypatch, capsys, no_background_work
):
    vault, conn, embedder = mapped
    repo = tmp_path / "new-project"
    (repo / ".git").mkdir(parents=True)
    monkeypatch.delenv("ZK_INGESTING", raising=False)
    event = {"hook_event_name": "SessionStart", "source": "startup", "cwd": str(repo)}
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(event)))
    assert hook.main([str(vault)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["hookSpecificOutput"]["hookEventName"] == "SessionStart"
    assert "notes from the user's past work" in out["hookSpecificOutput"]["additionalContext"]
    assert no_background_work == [repo]
    # zk's own headless runs get nothing.
    monkeypatch.setenv("ZK_INGESTING", "1")
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(event)))
    assert hook.main([str(vault)]) == 0
    assert capsys.readouterr().out == "" and no_background_work == [repo]


def test_the_nudge_comes_once_per_session(monkeypatch, capsys):
    monkeypatch.delenv("ZK_INGESTING", raising=False)
    event = {"hook_event_name": "PreToolUse", "tool_name": "Grep", "session_id": "s1"}
    for expected in (True, False):
        monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(event)))
        assert hook.main(["/vault"]) == 0
        assert bool(capsys.readouterr().out) is expected
    assert recall.nudge({**event, "tool_name": "Read", "session_id": "s2"}) is None
    assert len(recall.NUDGE) < 200  # about 40 tokens


def test_a_failed_command_shows_matching_notes_once(mapped, monkeypatch, capsys):
    vault, conn, embedder = mapped
    note(vault, "colcon-venv", "Colcon build fails in a virtualenv",
         "colcon picks up the venv python and cannot find catkin_pkg; build outside the venv.",
         type="problem")  # fmt: skip
    refresh(vault, conn, embedder)
    monkeypatch.delenv("ZK_INGESTING", raising=False)
    error = (
        "Exit code 1\nStarting >>> my_pkg\n--- stderr: my_pkg\n"
        "ModuleNotFoundError: No module named 'catkin_pkg'\n"
        "Failed   <<< my_pkg [0.4s, exited with code 1]\ncolcon build failed /home/u/ws/src"
    )
    event = {
        "hook_event_name": "PostToolUseFailure",
        "tool_name": "Bash",
        "session_id": "s3",
        "error": error,
    }
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(event)))
    assert hook.main([str(vault)]) == 0
    text = json.loads(capsys.readouterr().out)["hookSpecificOutput"]["additionalContext"]
    assert "colcon-venv: Colcon build fails in a virtualenv" in text
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(event)))
    assert hook.main([str(vault)]) == 0
    assert capsys.readouterr().out == ""  # shown once per session
    unrelated = {**event, "session_id": "s4", "error": "Exit code 2\nzsh: permission denied: ./x"}
    assert recall.on_failure(conn, unrelated) is None


def test_a_prompt_gets_pointers_only_when_notes_clearly_match(mapped, monkeypatch, capsys):
    vault, conn, embedder = mapped
    note(vault, "colcon-venv", "Colcon build fails in a virtualenv",
         "colcon picks up the venv python and cannot find catkin_pkg; build outside the venv.",
         type="problem")  # fmt: skip
    refresh(vault, conn, embedder)
    monkeypatch.delenv("ZK_INGESTING", raising=False)
    event = {
        "hook_event_name": "UserPromptSubmit",
        "session_id": "p1",
        "cwd": str(vault),
        "prompt": "my colcon build can't find catkin_pkg inside the virtualenv, what's wrong?",
    }
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(event)))
    assert hook.main([str(vault)]) == 0
    out = json.loads(capsys.readouterr().out)["hookSpecificOutput"]
    assert out["hookEventName"] == "UserPromptSubmit"
    text = out["additionalContext"]
    assert text.startswith("Notes in the zk vault match this prompt")
    assert "colcon-venv: Colcon build fails" in text and len(text) <= recall.PROMPT_CHARS

    # Each note once per session; nothing for unrelated prompts, commands or chit-chat.
    assert recall.on_prompt(conn, event) is None
    fresh = {**event, "session_id": "p2"}
    assert recall.on_prompt(conn, {**fresh, "prompt": "rename the login button to Sign in"}) is None
    assert recall.on_prompt(conn, {**fresh, "prompt": "/review colcon catkin_pkg venv"}) is None
    assert recall.on_prompt(conn, {**fresh, "prompt": "thanks, looks good"}) is None
    # zk's own headless runs get nothing.
    monkeypatch.setenv("ZK_INGESTING", "1")
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps({**event, "session_id": "p3"})))
    assert hook.main([str(vault)]) == 0
    assert capsys.readouterr().out == ""


def test_prompt_query_drops_common_words_and_paths():
    words = recall.prompt_query("Can you please fix the RRF fusion in src/zk/search.py for me?")
    assert words == ["rrf", "fusion"]


def test_error_query_keeps_the_error_words_only():
    words = recall.error_query(
        "Exit code 1\nbuilding...\nTraceback (most recent call last):\n"
        '  File "/home/u/app.py", line 3\nImportError: cannot import name Foo from bar 0xdeadbeef1'
    )
    assert "importerror" in words and "foo" in words and "deadbeef1" not in words
    assert not any("/" in w for w in words)


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "MaizeDetector"
    (root / "src" / "detect").mkdir(parents=True)
    (root / "node_modules" / "x").mkdir(parents=True)
    (root / "README.md").write_text("# Maize detector\n\nFinds leaf disease in photos.\n")
    (root / "pyproject.toml").write_text('[project]\nname = "maize"\ndependencies = ["torch"]\n')
    (root / ".env").write_text("API_KEY=supersecretvalue123\n")
    (root / "src" / "detect" / "model.py").write_text(
        "class LeafNet:\n    def forward(self, x):\n        return x\n\ndef train(): pass\n"
    )
    (root / "src" / "Viewer.cs").write_text("public class PhotoViewer { void Show() {} }\n")
    (root / "node_modules" / "x" / "index.js").write_text("function vendored() {}\n")
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    return root


def test_learn_reads_structure_for_free_and_only_changes_later(vault, repo):
    result = learn.learn(vault, repo)
    text = open(result["saved"], encoding="utf-8").read()
    assert 'source: "repo:maizedetector"' in text and 'project: "maizedetector"' in text
    assert "Finds leaf disease" in text and 'dependencies = ["torch"]' in text
    assert "class LeafNet, LeafNet.forward, def train" in text and "type PhotoViewer" in text
    assert "supersecret" not in text and "vendored" not in text
    assert learn.is_known(repo)

    assert learn.learn(vault, repo)["why"] == "unchanged since last time"
    (repo / "src" / "detect" / "model.py").write_text("class LeafNet2:\n    pass\n")
    changed = open(learn.learn(vault, repo)["saved"], encoding="utf-8").read()
    assert "class LeafNet2" in changed and "PhotoViewer" not in changed


def test_learn_dry_run_shows_the_redacted_text(vault, repo):
    (repo / "docker-compose.yml").write_text(
        "services:\n  db:\n    environment:\n      POSTGRES_PASSWORD: devpass123\n"
        "      DATABASE_URL: postgresql://app:devpass123@db:5432/app\n"
    )
    text = learn.learn(vault, repo, dry_run=True)["text"]
    assert "docker-compose.yml" in text and "devpass123" not in text


def test_learn_skips_the_vault_and_kept_out_projects(vault, repo):
    assert learn.learn(vault, vault)["saved"] is None
    learn.skip(repo)
    assert learn.learn(vault, repo)["why"] == "kept out with --skip" and learn.is_known(repo)


def test_ingested_notes_record_the_captures_project(vault, embedder, tmp_path):
    (vault / "inbox" / "c.md").write_text(
        '---\nkind: "capture"\nsource: "repo:maize"\nproject: "Maize Detector"\n---\n# C\n',
        encoding="utf-8",
    )
    conn = open_db(tmp_path / "i.sqlite")
    build_index(conn, vault)
    update_embeddings(conn, embedder)
    draft = Proposal(
        title="Two-stage leaf detection",
        type="pattern",
        summary="Find leaves first, then classify disease per leaf crop.",
        sources=["conversation:2026-11-10"],
        questions=["how does the maize detector find disease?"],
    )
    result = ingest.ingest(vault, conn, embedder, [draft], capture="c.md")
    text = (vault / result["notes"][0]).read_text(encoding="utf-8")
    assert "project:maize-detector" in text


def test_process_ingests_a_backlog_a_few_captures_at_a_time(vault, tmp_path):
    for f in (vault / "inbox").glob("*.md"):
        f.unlink()
    for i in range(5):
        (vault / "inbox" / f"2026-11-0{i + 1}-c{i}.md").write_text(
            "---\nkind: capture\n---\n# C\n", encoding="utf-8"
        )
    calls = tmp_path / "calls"
    script = tmp_path / "agent.py"  # a stand-in agent: records its captures and files them
    script.write_text(
        "import os, sys\n"
        f"open(r'{calls}', 'a').write(sys.argv[1] + chr(10))\n"
        "os.makedirs('inbox/processed', exist_ok=True)\n"
        "for n in sys.argv[1].split(', '):\n"
        "    os.replace('inbox/' + n, 'inbox/processed/' + n)\n"
        "print(os.environ.get('MCP_TIMEOUT'))\n",
        encoding="utf-8",
    )
    agent = f'"{process.sys.executable}" "{script}" "{{captures}}"'
    summary = process.run(vault, commands={"ingest": agent, "resolve": agent})
    rounds = calls.read_text().splitlines()
    assert summary["ingest_runs"] == 2 and len(rounds) == 2 and summary["stalled"] == []
    assert rounds[0].split(", ") == [f"2026-11-0{i + 1}-c{i}.md" for i in range(3)]
    assert process.MCP_START_MS in Path(summary["log"]).read_text()  # MCP servers get time
    assert process.last_status()["ok"] is True


def test_a_run_that_processes_nothing_stops_and_says_so(vault, tmp_path, capsys):
    for f in (vault / "inbox").glob("*.md"):
        f.unlink()
    for i in range(5):
        (vault / "inbox" / f"2026-11-0{i + 1}-c{i}.md").write_text(
            "---\nkind: capture\n---\n# C\n", encoding="utf-8"
        )
    noop = f'"{process.sys.executable}" -c "print(1)"'  # like an agent whose tools never started
    summary = process.run(vault, commands={"ingest": noop, "resolve": noop})
    assert summary["ingest_runs"] == 1 and len(summary["stalled"]) == 3
    assert "ingest made no progress" in Path(summary["log"]).read_text()
    status = process.last_status()
    assert status["ok"] is False and status["waiting"] == 5
    assert any("processed none of its captures" in line for line in review.advice(
        {**_review_numbers(), "last_process": status}
    ))  # fmt: skip
    from zk.cli import main

    args = ["process", str(vault), "--ingest-command", noop, "--resolve-command", noop]
    assert main(args) == 1
    assert "processed none of 3" in capsys.readouterr().err


def _review_numbers() -> dict:
    return {
        "errors": 0, "broken_links": 0, "held": 0, "inbox": 0, "questions": 50,
        "maps_pending": 0, "drift": None, "brief_tokens_in_project": 100,
    }  # fmt: skip


def test_batches_stop_at_the_size_limit():
    big = [{"name": n, "chars": process.BATCH_CHARS - 10} for n in ("a", "b")]
    assert process.batches(big, set()) == ["a"] and process.batches(big, {"a"}) == ["b"]


def test_install_adds_each_missing_hook_and_the_rules_file_once(vault, isolated_homes):
    settings = isolated_homes / "claude" / "settings.json"
    hooks.install(vault, recall=False)  # an earlier install: only the session-end capture
    assert list(json.loads(settings.read_text())["hooks"]) == ["SessionEnd"]
    result = hooks.install(vault)
    assert result["added"] == [
        "SessionStart", "PreToolUse (Grep|Glob)", "PreToolUse (Read)",
        "PostToolUseFailure (Bash|PowerShell)", "UserPromptSubmit",
    ]  # fmt: skip
    assert "matcher" not in json.loads(settings.read_text())["hooks"]["UserPromptSubmit"][0]
    data = json.loads(settings.read_text())
    assert [g["matcher"] for g in data["hooks"]["PreToolUse"]] == ["Grep|Glob", "Read"]
    assert data["hooks"]["PostToolUseFailure"][0]["matcher"] == "Bash|PowerShell"
    assert "zk vault" in hooks.rules_path().read_text()
    # An install from before the Read hook gets just that entry added.
    data["hooks"]["PreToolUse"].pop()
    settings.write_text(json.dumps(data))
    assert hooks.install(vault)["added"] == ["PreToolUse (Read)"]
    hooks.rules_path().write_text("my own rules\n")
    assert hooks.install(vault)["action"] == "already installed"
    assert hooks.rules_path().read_text() == "my own rules\n"


def test_weekly_review_task_and_report(vault, embedder, tmp_path):
    xml = hooks.task_xml(vault, "20:00", today="2026-11-15", weekly_on="sun")
    assert "<Sunday />" in xml and "review" in xml and "--catch-up" not in xml
    conn = open_db(tmp_path / "r.sqlite")
    build_index(conn, vault)
    update_embeddings(conn, embedder)
    result = review.run(vault, conn, embedder)
    assert result["file"].startswith("inbox/reports/review-")
    assert "too few to tell a retrieval drop from noise" in result["text"]
    assert "Session brief: about" in result["text"] and "in a project's folder" in result["text"]


def test_a_run_is_recorded_as_unfinished_until_it_ends(vault, tmp_path):
    for f in (vault / "inbox").glob("*.md"):
        f.unlink()
    (vault / "inbox" / "2026-11-01-c0.md").write_text(
        "---\nkind: capture\n---\n# C\n", encoding="utf-8"
    )
    # The stand-in agent reads the status file mid-run, as a killed run would leave it.
    peek = f'"{process.sys.executable}" -c "print(open(r\'{process.status_path()}\').read())"'
    summary = process.run(vault, commands={"ingest": peek, "resolve": peek})
    assert '"finished": false' in Path(summary["log"]).read_text()
    assert process.last_status()["finished"] is True
    killed = {"at": "2026-11-11T12:38:00", "finished": False, "log": "process.log"}
    advice = review.advice({**_review_numbers(), "last_process": killed})
    assert any("never finished" in line for line in advice)
    assert "discard" in process.INGEST_PROMPT and "in full" in process.INGEST_PROMPT


def test_the_background_run_breaks_away_from_the_apps_job_on_windows(vault, monkeypatch):
    from zk import sessions

    calls = []

    def popen(command, creationflags=0, **kwargs):
        calls.append(creationflags)
        if creationflags & 0x01000000:
            raise PermissionError("the job forbids breaking away")

    for name, value in (("DETACHED_PROCESS", 0x8), ("CREATE_NEW_PROCESS_GROUP", 0x200),
                        ("CREATE_NO_WINDOW", 0x08000000),
                        ("CREATE_BREAKAWAY_FROM_JOB", 0x01000000)):  # fmt: skip
        monkeypatch.setattr(sessions.subprocess, name, value, raising=False)
    monkeypatch.setattr(sessions.subprocess, "Popen", popen)
    monkeypatch.setattr(sessions.os, "name", "nt")
    sessions.start_process(vault)
    assert calls[0] & 0x01000000 and not calls[1] & 0x01000000 and calls[1] & 0x8


def test_learn_skips_datasets_and_media(vault, repo):
    images = repo / "data" / "train"
    images.mkdir(parents=True)
    for i in range(120):
        (images / f"leaf_{i}.jpg").write_bytes(b"x")
    (repo / "data" / "labels.npy").write_bytes(b"x")
    (repo / "src" / "detect" / "weights.pt").write_bytes(b"x")
    text = open(learn.learn(vault, repo)["saved"], encoding="utf-8").read()
    assert "leaf_0.jpg" not in text and "weights.pt" not in text
    assert "Data files not read: 122" in text
    assert "LeafNet" in text  # the code is still read


def _five_captures(vault):
    for f in (vault / "inbox").glob("*.md"):
        f.unlink()
    for i in range(5):
        (vault / "inbox" / f"2026-11-0{i + 1}-c{i}.md").write_text(
            "---\nkind: capture\n---\n# C\n", encoding="utf-8"
        )


def test_a_run_stopped_by_the_usage_limit_says_so_and_leaves_maps_alone(
    vault, tmp_path, monkeypatch
):
    _five_captures(vault)
    refreshed = []
    monkeypatch.setattr(maps, "update", lambda *a, **k: refreshed.append(1) or {})
    said = "print('You' + chr(39) + 've hit your session limit')"
    limit = f'"{process.sys.executable}" -c "{said}"'
    summary = process.run(vault, commands={"ingest": limit, "resolve": limit})
    assert summary["reason"] == "usage limit" and summary["ingest_runs"] == 1
    assert summary["processed"] == 0 and refreshed == []  # nothing changed, so no maps
    status = process.last_status()
    assert status["ok"] is False and status["reason"] == "usage limit"
    advice = review.advice({**_review_numbers(), "last_process": status})
    assert any("usage limit" in line for line in advice)


def test_container_folders_are_no_project(tmp_path):
    from zk import sessions
    from zk.search import source_project

    assert source_project("project:repos") is None and source_project("repo:Source") is None
    assert source_project("project:nautilustrx") == "nautilustrx"
    repos = tmp_path / "repos"
    repos.mkdir()
    subprocess.run(["git", "init", "-q", str(repos)], check=True)
    assert recall.project_of(repos) is None and sessions._project(str(repos)) is None
    assert sessions._project(str(tmp_path / "gone" / "source")) is None
    assert sessions._project(str(tmp_path / "gone" / "AphidDetection")) == "AphidDetection"
