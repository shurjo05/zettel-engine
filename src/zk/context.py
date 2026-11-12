"""How much context the zk tools spend to reach an answer, against loading the notes directly.

Read-only on the index, like search. For each labelled question this replays
what an agent does through the MCP server: `search_notes` (k results), then
`get_note` on the first relevant hit, and counts the tokens of exactly the JSON
those tools return. Two baselines show what the tools save:
- whole vault: every note's text in context;
- naive RAG: the top k notes opened in full, as retrieval pipelines that skip
  the summary step do.
When the answer is a section of a project doc (`zk docs`), a third: reading the
whole doc file, which is what an agent without zk does.

Tokens are estimated from characters (`checks.estimate_tokens`), the same
estimate the long-note check uses; it is a ruler for comparing, not a bill.
"""

from __future__ import annotations

import json
import sqlite3
import statistics
from dataclasses import dataclass

from zk.checks import estimate_tokens
from zk.embed import Embedder
from zk.evaluate import _resolver
from zk.lookup import compact, get_note
from zk.search import search


@dataclass
class ContextResult:
    query: str
    found_at: int | None  # rank of the first relevant note in the top k
    search_tokens: int
    note_tokens: int  # get_note of the first relevant hit (0 if none was found)
    naive_tokens: int  # the top k notes opened in full
    file_tokens: int = 0  # the whole doc file the answer is a section of (0: not a doc)

    @property
    def tool_tokens(self) -> int:
        return self.search_tokens + self.note_tokens


def run_context(
    conn: sqlite3.Connection,
    queries: list[dict],
    mode: str,
    embedder: Embedder | None = None,
    k: int = 5,
) -> dict:
    resolve = _resolver(conn)
    vault_tokens = sum(
        estimate_tokens(f"{r['title']}\n{r['body']}")
        for r in conn.execute("SELECT title, body FROM notes")
    )
    notes_cache: dict[str, int] = {}

    def note_tokens(path: str) -> int:
        if path not in notes_cache:
            notes_cache[path] = estimate_tokens(compact(get_note(conn, path)))
        return notes_cache[path]

    def doc_tokens(path: str) -> int:
        """The size of the whole doc a reference section comes from (0 for other notes)."""
        row = conn.execute("SELECT frontmatter FROM notes WHERE path = ?", (path,)).fetchone()
        front = json.loads(row[0] or "{}") if row else {}
        if not front.get("doc"):
            return 0
        contents = conn.execute(
            "SELECT frontmatter FROM notes WHERE type = 'reference' "
            "AND json_extract(frontmatter, '$.contents') "
            "AND json_extract(frontmatter, '$.project') = ? "
            "AND json_extract(frontmatter, '$.doc') = ?",
            (front.get("project"), front["doc"]),
        ).fetchone()
        return int(json.loads(contents[0]).get("doc_tokens") or 0) if contents else 0

    results = []
    for q in queries:
        labelled = [p for p in (resolve(str(x)) for x in q["relevant"]) if p]
        relevant = set(labelled)
        hits = search(conn, q["query"], k=k, mode=mode, embedder=embedder)
        ranked = [h.path for h in hits]
        found = next((i for i, p in enumerate(ranked, 1) if p in relevant), None)
        results.append(
            ContextResult(
                query=q["query"],
                found_at=found,
                search_tokens=estimate_tokens(compact([h.to_dict() for h in hits])),
                note_tokens=note_tokens(ranked[found - 1]) if found else 0,
                naive_tokens=sum(note_tokens(p) for p in ranked),
                file_tokens=doc_tokens(labelled[0]) if labelled else 0,
            )
        )
    return summarise(results, vault_tokens, k)


def _p90(values: list[int]) -> int:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))] if ordered else 0


def summarise(results: list[ContextResult], vault_tokens: int, k: int) -> dict:
    found = [r for r in results if r.found_at]
    tools = [r.tool_tokens for r in found]
    naive = [r.naive_tokens for r in results]
    median_tools = int(statistics.median(tools)) if tools else 0
    files = [r.file_tokens for r in results if r.file_tokens]
    return {
        "queries": len(results),
        "k": k,
        "answered_in_top_k": round(len(found) / len(results), 3) if results else 0.0,
        "tool_tokens_median": median_tools,
        "tool_tokens_p90": _p90(tools),
        "search_tokens_median": int(statistics.median(r.search_tokens for r in results))
        if results
        else 0,
        "naive_rag_tokens_median": int(statistics.median(naive)) if naive else 0,
        "whole_vault_tokens": vault_tokens,
        "doc_questions": len(files),
        "file_tokens_median": int(statistics.median(files)) if files else 0,
        "share_of_vault": round(median_tools / vault_tokens, 4) if vault_tokens else 0.0,
        "results": [
            {
                "query": r.query,
                "found_at": r.found_at,
                "tool_tokens": r.tool_tokens,
                "naive_tokens": r.naive_tokens,
                "file_tokens": r.file_tokens,
            }
            for r in results
        ],
    }


def format_context(report: dict) -> str:
    k = report["k"]
    vault = report["whole_vault_tokens"]
    tools = report["tool_tokens_median"]
    naive = report["naive_rag_tokens_median"]

    def times(n: int) -> str:
        return f"{n / vault:.1%} of the whole vault" if vault else "-"

    lines = [
        f"{report['queries']} questions; the answer note was in the top {k} for "
        f"{report['answered_in_top_k']:.0%}",
        f"  zk tools (search, then open the hit): median {tools} tokens, p90 "
        f"{report['tool_tokens_p90']} ({times(tools)})",
        f"    of which search_notes alone: median {report['search_tokens_median']}",
        f"  naive RAG (open the top {k} in full):  median {naive} tokens ({times(naive)})",
        f"  whole vault in context:              {vault} tokens",
    ]
    if report.get("doc_questions"):
        lines.append(
            f"  reading the doc file ({report['doc_questions']} questions answered by a doc): "
            f"median {report['file_tokens_median']} tokens"
        )
    missed = [r for r in report["results"] if not r["found_at"]]
    if missed:
        lines += ["", f"not in the top {k} (an agent would search again or read more):"]
        lines += [f"  - {r['query']}" for r in missed]
    return "\n".join(lines)
