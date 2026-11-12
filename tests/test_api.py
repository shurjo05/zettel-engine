"""zk.api is what other programs import (docs/api.md); these tests pin its behaviour."""

import json
from datetime import UTC, datetime

import anyio
from mcp import Client
from mcp.server.mcpserver import MCPServer

from zk import api
from zk.index import DEFAULT_EXCLUDES
from zk.ingest import list_inbox
from zk.server import create_server


def _call(server, tool, arguments=None):
    async def go():
        async with Client(server) as client:
            if tool is None:
                return (await client.list_tools()).tools
            return await client.call_tool(tool, arguments)

    return anyio.run(go)


def test_every_documented_name_exists():
    for name in api.__all__:
        assert hasattr(api, name), name


def test_mounted_tools_are_zks_read_only_tools(vault, tmp_path):
    host = MCPServer("other")

    @host.tool()
    def ping() -> str:
        return "pong"

    api.add_knowledge_tools(host, vault, tmp_path / "a.sqlite", DEFAULT_EXCLUDES)
    offered = {t.name: t for t in _call(host, None)}
    assert sorted(offered) == ["get_note", "get_related", "ping", "search_notes"]
    assert all(offered[n].annotations.read_only_hint for n in offered if n != "ping")


def test_mounted_tools_answer_exactly_as_zk_serve(vault, tmp_path, embedder):
    host = MCPServer("other")
    api.add_knowledge_tools(host, vault, tmp_path / "a.sqlite", DEFAULT_EXCLUDES, embedder)
    zk = create_server(vault, tmp_path / "b.sqlite", DEFAULT_EXCLUDES, embedder)
    for tool, args in [
        ("search_notes", {"query": "merging ranked lists", "k": 3}),
        ("get_note", {"note": "20261015100100"}),
        ("get_related", {"note": "reciprocal-rank-fusion"}),
        ("suggest_links", {"note": "reciprocal-rank-fusion"}),
    ]:
        mine = _call(host, tool, args).content[0].text
        theirs = _call(zk, tool, args).content[0].text
        assert json.loads(mine) == json.loads(theirs), tool


def test_find_duplicates_sees_a_note_written_a_moment_ago(vault, tmp_path, embedder):
    db = tmp_path / "i.sqlite"
    (vault / "notes" / "fresh-idea.md").write_text(
        "---\nid: 20261112120000\ntitle: Worktree per agent task\ntype: pattern\n---\n"
        "# Worktree per agent task\n\n"
        "Give each coding agent its own git worktree so parallel edits never collide.\n",
        encoding="utf-8",
    )
    hits = api.find_duplicates(
        vault,
        embedder,
        "Worktree per agent task",
        "Give each coding agent its own git worktree so parallel edits never collide.",
        db=db,
    )
    assert hits[0]["path"] == "notes/fresh-idea.md"
    assert hits[0]["verdict"] == "likely-duplicate"


def test_save_text_writes_a_new_capture_and_never_overwrites(vault):
    when = datetime(2026, 11, 12, 9, 30, tzinfo=UTC)
    first = api.save_text(vault, "a suggestion", "codex:My Repo:task-7", now=when)
    second = api.save_text(vault, "another", "codex:My Repo:task-7", now=when)
    assert first != second
    text = first.read_text(encoding="utf-8")
    assert 'source: "codex:my-repo:task-7"' in text
    assert "a suggestion" in text
    assert first.parent == vault / "inbox"
    assert {c["name"] for c in list_inbox(vault)} >= {first.name, second.name}


def test_save_text_redacts_secrets(vault):
    path = api.save_text(vault, "key sk-abcdefghijklmnopqrstuvwxyz123456", "codex:r:t")
    assert "sk-abcdef" not in path.read_text(encoding="utf-8")
