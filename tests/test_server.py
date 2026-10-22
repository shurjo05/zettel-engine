import json

import anyio
from mcp import Client

from zk.index import DEFAULT_EXCLUDES
from zk.server import create_server


def call(server, tool: str, arguments: dict):
    async def go():
        async with Client(server) as client:
            return await client.call_tool(tool, arguments)

    return anyio.run(go)


def tools(server):
    async def go():
        async with Client(server) as client:
            return (await client.list_tools()).tools

    return anyio.run(go)


def make(vault, tmp_path):
    return create_server(vault, tmp_path / "index.sqlite", DEFAULT_EXCLUDES)


def test_only_read_only_tools_are_offered(vault, tmp_path):
    offered = tools(make(vault, tmp_path))
    assert sorted(t.name for t in offered) == ["get_note", "get_related", "search_notes"]
    assert all(t.annotations.read_only_hint for t in offered)


def test_search_notes_returns_summaries_not_bodies(vault, tmp_path):
    result = call(make(vault, tmp_path), "search_notes", {"query": "merging lists", "k": 2})
    assert not result.is_error
    hits = json.loads(result.content[0].text)
    assert hits[0]["path"] == "notes/reciprocal-rank-fusion.md"
    assert "summary" in hits[0] and "body" not in hits[0]
    assert len(hits) <= 2


def test_get_note_and_get_related(vault, tmp_path):
    server = make(vault, tmp_path)
    note = json.loads(call(server, "get_note", {"note": "20261015100100"}).content[0].text)
    assert note["title"] == "Reciprocal Rank Fusion"
    near = json.loads(call(server, "get_related", {"note": note["path"]}).content[0].text)
    assert [n["title"] for n in near["related"]][:2] == ["BM25 Ranking", "Embeddings"]


def test_unknown_note_is_a_tool_error_the_model_can_read(vault, tmp_path):
    result = call(make(vault, tmp_path), "get_note", {"note": "fusion"})
    assert result.is_error
    assert "reciprocal-rank-fusion" in result.content[0].text


def test_each_call_sees_the_vault_as_it_is_now(vault, tmp_path):
    server = make(vault, tmp_path)
    assert json.loads(call(server, "search_notes", {"query": "zebra"}).content[0].text) == []
    note = vault / "notes" / "bm25-ranking.md"
    note.write_text(note.read_text(encoding="utf-8") + "\nA zebra appears.\n", encoding="utf-8")
    hits = json.loads(call(server, "search_notes", {"query": "zebra"}).content[0].text)
    assert [h["path"] for h in hits] == ["notes/bm25-ranking.md"]


def test_serving_never_writes_to_the_vault(vault, tmp_path):
    before = {p: p.read_bytes() for p in vault.rglob("*") if p.is_file()}
    server = make(vault, tmp_path)
    call(server, "search_notes", {"query": "bm25"})
    call(server, "get_note", {"note": "bm25-ranking"})
    call(server, "get_related", {"note": "bm25-ranking"})
    after = {p: p.read_bytes() for p in vault.rglob("*") if p.is_file()}
    assert before == after


def test_semantic_mode(vault, tmp_path, embedder):
    server = create_server(vault, tmp_path / "index.sqlite", DEFAULT_EXCLUDES, embedder=embedder)
    result = call(server, "search_notes", {"query": "merging ranked lists", "mode": "semantic"})
    hits = json.loads(result.content[0].text)
    assert hits[0]["path"] == "notes/reciprocal-rank-fusion.md"

    plain = make(vault, tmp_path / "other")
    result = call(plain, "search_notes", {"query": "x", "mode": "semantic"})
    assert result.is_error and "not enabled" in result.content[0].text
    result = call(plain, "search_notes", {"query": "x", "mode": "fuzzy"})
    assert result.is_error


def test_hybrid_mode(vault, tmp_path, embedder):
    server = create_server(vault, tmp_path / "index.sqlite", DEFAULT_EXCLUDES, embedder=embedder)
    result = call(server, "search_notes", {"query": "merging ranked lists", "mode": "hybrid"})
    assert json.loads(result.content[0].text)[0]["path"] == "notes/reciprocal-rank-fusion.md"


def test_default_mode_is_hybrid_when_semantic_search_is_available(vault, tmp_path, embedder):
    server = create_server(vault, tmp_path / "index.sqlite", DEFAULT_EXCLUDES, embedder=embedder)
    args = {"query": "zebra", "k": 3}  # no word matches, so only hybrid finds anything
    assert json.loads(call(server, "search_notes", args).content[0].text)
    assert (
        json.loads(call(make(vault, tmp_path / "plain"), "search_notes", args).content[0].text)
        == []
    )
