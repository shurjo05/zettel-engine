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


def test_ingest_tools_exist_only_with_allow_writes(vault, tmp_path, embedder):
    db = tmp_path / "index.sqlite"
    plain = {t.name for t in tools(create_server(vault, db, DEFAULT_EXCLUDES, embedder=embedder))}
    assert "ingest_notes" not in plain and "apply_held" not in plain
    server = create_server(vault, db, DEFAULT_EXCLUDES, embedder=embedder, allow_writes=True)
    offered = {t.name: t for t in tools(server)}
    assert {"list_inbox", "read_capture", "check_duplicates", "ingest_notes"} <= set(offered)
    assert {"list_held", "apply_held", "discard"} <= set(offered)
    assert offered["ingest_notes"].annotations.destructive_hint is False
    schema = offered["ingest_notes"].input_schema
    assert "questions" in json.dumps(schema)  # the note fields are described to the model


def test_ingest_over_mcp_writes_and_is_searchable_at_once(vault, tmp_path, embedder):
    server = create_server(
        vault, tmp_path / "index.sqlite", DEFAULT_EXCLUDES, embedder=embedder, allow_writes=True
    )
    note = {
        "title": "Rank Fusion Needs Only Ranks",
        "type": "concept",
        "summary": "Fusing by rank avoids comparing scores on different scales.",
        "sources": ["conversation:2026-10-26"],
        "links": {"reciprocal-rank-fusion": "the method this explains"},
        "questions": ["why fuse by position rather than by score?"],
    }
    result = call(server, "ingest_notes", {"notes": [note]})
    assert not result.is_error, result.content[0].text
    out = json.loads(result.content[0].text)
    assert out["notes"] == ["notes/rank-fusion-needs-only-ranks.md"]
    assert (vault / out["questions"]).is_file()
    hits = json.loads(
        call(server, "search_notes", {"query": "scales", "mode": "keyword"}).content[0].text
    )
    assert hits[0]["path"] == "notes/rank-fusion-needs-only-ranks.md"

    bad = call(server, "ingest_notes", {"notes": [{**note, "type": "idea"}]})
    assert bad.is_error and "type must be one of" in bad.content[0].text
    missing = call(
        server, "ingest_notes", {"notes": [{k: v for k, v in note.items() if k != "questions"}]}
    )
    assert missing.is_error
    assert sorted((vault / "notes").glob("rank-fusion*")) == [
        vault / "notes" / "rank-fusion-needs-only-ranks.md"
    ]


def test_held_pair_judged_and_merged_over_mcp(vault, tmp_path, embedder):
    held = vault / "inbox" / "held"
    held.mkdir(parents=True)
    (held / "okapi.md").write_text(
        (vault / "notes" / "bm25-ranking.md")
        .read_text(encoding="utf-8")
        .replace("20261015100000", "20261101090000")
        .replace("BM25 Ranking", "Okapi"),
        encoding="utf-8",
    )
    server = create_server(
        vault, tmp_path / "index.sqlite", DEFAULT_EXCLUDES, embedder=embedder, allow_writes=True
    )
    pair = json.loads(call(server, "get_held_pair", {"name": "okapi"}).content[0].text)
    assert pair["closest"]["path"] == "notes/bm25-ranking.md"
    note = {
        "title": "BM25 Ranking",
        "type": "concept",
        "summary": "BM25 sums saturating term weights, normalized by document length.",
    }
    args = {"name": "okapi", "into": pair["closest"]["path"], "note": note, "reason": "reworded"}
    result = call(server, "merge_held", args)
    assert not result.is_error, result.content[0].text
    assert json.loads(result.content[0].text)["held_moved_to"] == "inbox/merged/okapi.md"
    hits = json.loads(
        call(server, "search_notes", {"query": "saturating", "mode": "keyword"}).content[0].text
    )
    assert hits[0]["path"] == "notes/bm25-ranking.md"


def test_suggest_links_lists_unlinked_similar_notes(vault, tmp_path, embedder):
    server = create_server(vault, tmp_path / "index.sqlite", DEFAULT_EXCLUDES, embedder=embedder)
    result = call(server, "suggest_links", {"note": "orphan-note", "k": 2})
    assert not result.is_error, result.content[0].text
    found = json.loads(result.content[0].text)
    assert len(found) == 2 and found[0]["similarity"] >= found[1]["similarity"]
    plain = {t.name for t in tools(make(vault, tmp_path))}
    assert "suggest_links" not in plain  # needs semantic search


def test_stdio_server_accepts_initialize_after_a_newer_protocol_probe(vault, tmp_path):
    """Claude Code 2.1.280 opens with a 2026-07-28 request, then falls back to `initialize`.

    The SDK's default loop locked the connection to the newer protocol and
    refused the fallback, so the server never connected on Shurjo's PC.
    """
    import subprocess
    import sys

    proc = subprocess.Popen(
        [sys.executable, "-m", "zk", "serve", str(vault), "--db", str(tmp_path / "i.sqlite")],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )

    def send(message, reply=True):
        proc.stdin.write(json.dumps(message) + "\n")
        proc.stdin.flush()
        return json.loads(proc.stdout.readline()) if reply else None

    try:
        meta = {
            "io.modelcontextprotocol/protocolVersion": "2026-07-28",
            "io.modelcontextprotocol/clientInfo": {"name": "probe", "version": "1"},
            "io.modelcontextprotocol/clientCapabilities": {},
        }
        probe = send(
            {"jsonrpc": "2.0", "id": 1, "method": "server/discover", "params": {"_meta": meta}}
        )
        assert "error" in probe
        init = send(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "probe", "version": "1"},
                },
            }
        )
        assert "result" in init, init
        send({"jsonrpc": "2.0", "method": "notifications/initialized"}, reply=False)
        tools = send({"jsonrpc": "2.0", "id": 3, "method": "tools/list", "params": {}})
        assert "search_notes" in {t["name"] for t in tools["result"]["tools"]}
    finally:
        proc.kill()
        proc.wait()
