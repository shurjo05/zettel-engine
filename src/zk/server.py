"""MCP server: the vault's search and lookup, offered as tools to Claude Code and Codex.

Like the CLI, this module holds no retrieval logic; each tool refreshes the
index and calls the same library functions. It runs over stdio, so stdout is
the protocol and anything human-readable goes to stderr. The search tools are
read-only; the ingest tools, which create files in the vault, exist only when
the server is started with allow_writes (`zk serve --allow-writes`).
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Callable, Iterable
from pathlib import Path

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field

from zk import ingest, lookup, maintain, resolve
from zk.checks import history_path
from zk.embed import Embedder, update_embeddings
from zk.index import build_index, open_db
from zk.lookup import compact as _compact
from zk.proposals import InvalidProposal, Proposal
from zk.search import DEFAULT_MODE, MODES, search
from zk.vault import VaultWriteError

INSTRUCTIONS = """\
Tools for a personal Zettelkasten: short Markdown notes, one idea each, linked to each other.
Start with search_notes; its hits carry summaries, which often answer the question on their own.
Open a note with get_note only when you need its full text, and use get_related to follow links
and shared tags. Map notes (type map) list the notes on one topic or one project with a line each;
get_note opens a map by name, and search_notes can keep to one project. Name notes by the id or
path a tool returned. All tools are read-only.
"""

WRITE_INSTRUCTIONS = """
Ingest tools are enabled, and notes are written without review. To ingest a conversation or an
inbox capture (list_inbox, then read_capture): find the atomic ideas worth finding again in a
month (one idea per note; most conversations hold zero to three) and skip chit-chat. An error
whose cause and fix would help the next time it appears is worth a problem note; a typo is not.
For each idea, search_notes and check_duplicates first; if an existing note already
covers it, leave it out. Then call ingest_notes once with all of them, each with one to three test
questions it answers, worded the way someone would ask rather than copied from the note. Pass
`capture` when the notes come from an inbox capture. Report the notes written and any warnings.
A capture with nothing worth keeping is set aside with discard(kind="capture", reason=...);
every capture ends one way or the other. Discarding is for chit-chat, setup noise and work with
nothing reusable, not for a session that is merely about one assignment. School work (a course
code such as CIS505 in the folder or text is the sign) is worth keeping when it shows how a class
wants work done (style, format, naming, citation or submission rules), a concept learned, or a
mistake to avoid; before discarding one, check it for each of those, and say in the reason which
you looked for. Skip only the assignment's routine content. When two captures hold the same ideas
(copies of one session, say), ingest the first and discard the second with a reason naming the
notes that already cover it.

A capture whose source starts with repo: is not a conversation: it is the structure of one of the
person's own projects (folders, README, manifests, class and function names, commit subjects).
Write zero to five notes on what is reusable from it: how the project is built (its architecture
and the patterns it uses), decisions it shows, and problems its commits show were solved. Skip
what any project of that kind has. Its notes are recorded under that project automatically.

Notes held as likely duplicates are resolved one at a time: list_held, then get_held_pair. If the
held note states the same idea as the closest note (a reworded or narrower version of it), call
merge_held with a merged note that keeps everything the existing note says and adds only what the
held note says beyond it, woven in and kept under about 400 words. If it is a different idea that
is merely related, call apply_held. Give the reason either way.
"""

READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=False)
WRITES = ToolAnnotations(
    read_only_hint=False, destructive_hint=False, idempotent_hint=False, open_world_hint=False
)
_EXPECTED = (LookupError, ValueError, FileNotFoundError, InvalidProposal, VaultWriteError)


def create_server(
    vault: Path,
    db: Path,
    excludes: Iterable[str],
    embedder: Embedder | None = None,
    allow_writes: bool = False,
) -> MCPServer:
    vault = vault.resolve()
    excludes = tuple(excludes)
    lock = threading.Lock()  # tools run in worker threads; refresh one at a time

    def run(fn: Callable[[sqlite3.Connection], object], embed: bool = False) -> str:
        with lock:
            conn = open_db(db)
            try:
                build_index(conn, vault, excludes)  # cheap when nothing changed
                if embed:
                    update_embeddings(conn, embedder)  # only new or edited notes
                return _compact(fn(conn))
            except _EXPECTED as exc:  # not found, ambiguous, invalid, would overwrite
                raise ToolError(str(exc)) from exc
            finally:
                conn.close()

    server = MCPServer(
        "zk", instructions=INSTRUCTIONS + (WRITE_INSTRUCTIONS if allow_writes else "")
    )

    @server.tool(annotations=READ_ONLY, structured_output=False)
    def search_notes(
        query: str,
        k: int = 5,
        type: str | None = None,
        tag: str | None = None,
        mode: str | None = None,
        project: str | None = None,
    ) -> str:
        """Search the notes. Returns up to k hits, best first, without bodies.

        mode "hybrid" (default) fuses keyword and semantic rankings and suits most questions.
        mode "keyword" ranks by BM25 over stemmed words; use it for names, identifiers and exact
        terms. mode "semantic" ranks by meaning only (local embeddings, cosine similarity).
        Each hit has id, path, title, type, summary and score, and a short snippet with
        [matched words] when keyword search matched.
        Optional filters: type (concept, pattern, problem, decision, or map for the map notes),
        tag, and project (notes recorded from one project, by its folder name).
        """
        if mode is None:
            mode = DEFAULT_MODE if embedder is not None else "keyword"
        if mode not in MODES:
            raise ToolError(f"mode must be one of: {', '.join(MODES)}")
        if mode != "keyword" and embedder is None:
            raise ToolError(f"{mode} search is not enabled on this server")
        k = max(1, min(k, 25))
        return run(
            lambda c: [
                h.to_dict()
                for h in search(
                    c, query, k=k, type=type, tag=tag, mode=mode, embedder=embedder, project=project
                )
            ],
            embed=mode != "keyword",
        )

    if embedder is not None:

        @server.tool(annotations=READ_ONLY, structured_output=False)
        def suggest_links(note: str, k: int = 3) -> str:
            """Notes similar in meaning to `note` that it isn't linked with yet, most similar
            first, each with its cosine similarity. Use it to link a note into the vault."""
            k = max(1, min(k, 10))

            def go(c: sqlite3.Connection):
                return maintain.suggest_for(c, embedder, lookup.resolve(c, note), k)

            return run(go, embed=True)

    @server.tool(annotations=READ_ONLY, structured_output=False)
    def get_note(note: str) -> str:
        """One note in full: metadata, Markdown body, and the notes it links to and from.

        `note` is an id, a file name or vault path (with or without .md), or a title.
        """
        return run(lambda c: lookup.get_note(c, note))

    @server.tool(annotations=READ_ONLY, structured_output=False)
    def get_related(note: str, k: int = 10) -> str:
        """A note's neighbours with their summaries (no bodies).

        Lists notes it links to, then notes linking to it, then notes sharing its tags,
        up to k in total; `total` says how many there are.
        """
        k = max(1, min(k, 50))
        return run(lambda c: lookup.related(c, note, k=k))

    if allow_writes:

        def refresh(conn: sqlite3.Connection) -> None:
            build_index(conn, vault, excludes)
            if embedder is not None:
                update_embeddings(conn, embedder)

        _add_ingest_tools(server, vault, run, embedder, refresh, history_path(db))
    return server


class NoteDraft(BaseModel):
    """One atomic note for ingest_notes."""

    title: str = Field(description="the idea in a few words")
    type: str = Field(description="concept, pattern, problem or decision")
    summary: str = Field(description="one to three plain sentences that state the idea")
    questions: list[str] = Field(
        description="one to three questions this note answers, asked in your own words"
    )
    sources: list[str] = Field(
        default_factory=list,
        description='where the idea came from, e.g. "conversation:2026-10-26" or a URL '
        "(the capture is added automatically)",
    )
    body: str = Field("", description="Markdown details, no top-level heading")
    tags: list[str] = Field(default_factory=list)
    aliases: list[str] = Field(default_factory=list)
    links: dict[str, str] = Field(
        default_factory=dict, description="existing note name -> why it is related"
    )


class MergedNote(BaseModel):
    """The note that results from merging a held duplicate into an existing note."""

    title: str = Field(description="usually the existing note's title")
    type: str = Field(description="concept, pattern, problem or decision")
    summary: str = Field(
        description="one to three plain sentences that state the idea, including the held "
        "note's distinct point if it has one"
    )
    body: str = Field(
        "",
        description="Markdown details, no top-level heading: the existing note's details plus "
        "anything only the held note says, woven in rather than appended; drop nothing the "
        "existing note says, and keep the whole note under about 400 words, since search by "
        "meaning reads only the first 512 tokens",
    )
    tags: list[str] = Field(default_factory=list)
    aliases: list[str] = Field(default_factory=list)
    links: dict[str, str] = Field(
        default_factory=dict, description="note name -> why it is related, from both notes"
    )


def _add_ingest_tools(
    server: MCPServer,
    vault: Path,
    run,
    embedder: Embedder | None,
    refresh: Callable[[sqlite3.Connection], None],
    history: Path,
) -> None:
    def need_embedder() -> Embedder:
        if embedder is None:
            raise ToolError("duplicate checks need semantic search, which is not enabled")
        return embedder

    @server.tool(annotations=READ_ONLY, structured_output=False)
    def list_inbox() -> str:
        """Captures waiting in the inbox, oldest first: name, source, project, captured, size."""
        return run(lambda c: ingest.list_inbox(vault))

    @server.tool(annotations=READ_ONLY, structured_output=False)
    def read_capture(name: str) -> str:
        """The full text of one inbox capture, by the name list_inbox returned."""
        return run(lambda c: ingest.read_capture(vault, name))

    @server.tool(annotations=READ_ONLY, structured_output=False)
    def check_duplicates(title: str, summary: str, body: str = "") -> str:
        """Existing notes most similar to a would-be note, with cosine similarity and a verdict:
        likely-duplicate, related or new. Use before ingest_notes."""
        e = need_embedder()
        return run(lambda c: ingest.check_duplicates(c, e, title, summary, body), embed=True)

    @server.tool(annotations=WRITES, structured_output=False)
    def ingest_notes(notes: list[NoteDraft], capture: str | None = None) -> str:
        """Write atomic notes into the vault in one run, without review.

        Notes that pass the duplicate check go straight into notes/; a likely duplicate of an
        existing note (or of another note in this call) is held in inbox/held/ instead. Each
        note's questions become test questions for search. The run is one git commit, and the
        capture, if given, moves to inbox/processed/. Returns the notes written, the held ones
        and any warnings (search scores that dropped, notes too long, unlinked or with broken
        links), which you should pass on to the person.
        """
        e = need_embedder()
        drafts = [
            Proposal(
                title=n.title,
                type=n.type,
                summary=n.summary,
                body=n.body,
                tags=list(n.tags),
                aliases=list(n.aliases),
                sources=list(n.sources),
                links=dict(n.links),
                questions=list(n.questions),
            )
            for n in notes
        ]
        return run(
            lambda c: ingest.ingest(
                vault, c, e, drafts, capture=capture, refresh=lambda: refresh(c), history=history
            ),
            embed=True,
        )

    @server.tool(annotations=READ_ONLY, structured_output=False)
    def list_held() -> str:
        """Notes held back as likely duplicates (name, title, type)."""
        return run(lambda c: ingest.list_held(vault))

    @server.tool(annotations=WRITES, structured_output=False)
    def apply_held(name: str, reason: str = "") -> str:
        """Move a held note into notes/ after all and commit it: it is related to the note it
        resembles, not the same idea. Say why in `reason`; it goes in the commit message."""
        return run(lambda c: ingest.apply_held(vault, name, reason))

    @server.tool(annotations=READ_ONLY, structured_output=False)
    def get_held_pair(name: str) -> str:
        """A held note's full text next to the full text of the existing note closest to it,
        with their similarity, plus the next closest notes. Use it to judge whether they are the
        same idea (merge_held) or only related (apply_held)."""
        e = need_embedder()
        return run(lambda c: resolve.held_pair(vault, c, e, name), embed=True)

    @server.tool(annotations=WRITES, structured_output=False)
    def merge_held(name: str, into: str, note: MergedNote, reason: str) -> str:
        """Merge a held note into the existing note it duplicates, as one git commit.

        `into` is the existing note's path. `note` is the merged note you wrote from both; it
        keeps the existing note's id, file name and created date, and gains the held note's
        sources. Links and test questions that named the held note are pointed at the merged
        note, and the held file moves to inbox/merged/. `reason` says in a sentence why the two
        are the same idea. Fails, writing nothing, if a file changed while merging.
        """
        draft = Proposal(
            title=note.title,
            type=note.type,
            summary=note.summary,
            body=note.body,
            tags=list(note.tags),
            aliases=list(note.aliases),
            links=dict(note.links),
        )
        return run(
            lambda c: resolve.merge_held(
                vault, c, name, into, draft, reason, refresh=lambda: refresh(c)
            ),
            embed=True,
        )

    @server.tool(annotations=WRITES, structured_output=False)
    def discard(name: str, kind: str = "held", reason: str = "") -> str:
        """Set aside a held note (kind "held", to inbox/rejected/) or a capture with nothing
        worth keeping (kind "capture", to inbox/processed/). Nothing is deleted. `reason`
        (one sentence) is required for a capture and goes in the vault's commit message."""
        if kind not in ("held", "capture"):
            raise ToolError('kind must be "held" or "capture"')
        if kind == "capture" and not reason.strip():
            raise ToolError("say in one sentence why nothing in this capture is worth keeping")
        return run(lambda c: {"moved_to": ingest.discard(vault, name, kind, reason)})


def run_stdio(server: MCPServer) -> None:
    """Serve one stdio connection in the MCP handshake (`initialize`) protocol only.

    The SDK's default loop lets the client's first request pick the protocol
    era for good. Some Claude Code versions open with a 2026-07-28 request and
    then fall back to `initialize`, which that loop refuses ("the initialize
    handshake is not accepted"), so the server never connects. Serving only the
    handshake era turns the opening request away and accepts the fallback.
    """
    import anyio
    from mcp.server.runner import serve_loop
    from mcp.server.stdio import stdio_server

    low = server._lowlevel_server  # the SDK has no public switch for this

    async def main() -> None:
        async with stdio_server() as (read, write), low.lifespan(low) as state:
            await serve_loop(
                low,
                read,
                write,
                lifespan_state=state,
                init_options=low.create_initialization_options(),
            )

    anyio.run(main)
