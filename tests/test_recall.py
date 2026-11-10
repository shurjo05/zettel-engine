import io
import json
import subprocess

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
    agent = f"\"{process.sys.executable}\" -c \"open(r'{calls}', 'a').write('{{captures}}\\n')\""
    summary = process.run(vault, commands={"ingest": agent, "resolve": agent})
    rounds = calls.read_text().splitlines()
    assert summary["ingest_runs"] == 2 and len(rounds) == 2
    assert rounds[0].split(", ") == [f"2026-11-0{i + 1}-c{i}.md" for i in range(3)]


def test_batches_stop_at_the_size_limit():
    big = [{"name": n, "chars": process.BATCH_CHARS - 10} for n in ("a", "b")]
    assert process.batches(big, set()) == ["a"] and process.batches(big, {"a"}) == ["b"]


def test_install_adds_each_missing_hook_and_the_rules_file_once(vault, isolated_homes):
    settings = isolated_homes / "claude" / "settings.json"
    hooks.install(vault, recall=False)  # an earlier install: only the session-end capture
    assert list(json.loads(settings.read_text())["hooks"]) == ["SessionEnd"]
    result = hooks.install(vault)
    assert result["added"] == ["SessionStart", "PreToolUse", "PostToolUseFailure"]
    data = json.loads(settings.read_text())
    assert data["hooks"]["PreToolUse"][0]["matcher"] == "Grep|Glob"
    assert data["hooks"]["PostToolUseFailure"][0]["matcher"] == "Bash|PowerShell"
    assert "zk vault" in hooks.rules_path().read_text()
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
    assert "Session brief: about" in result["text"]
