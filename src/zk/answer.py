"""Answer a question from the notes, and measure how well answers cite them.

Read-only on the index. The pipeline is retrieval-augmented generation in its
plainest form: hybrid search picks the top notes, their text goes into one
prompt, and the model must answer only from them, citing each note it uses as
[[name]], or say the notes don't cover it. Because the citations are names, the
answer can be checked by code:
- citation accuracy: it cites a note labelled relevant for the question;
- grounding: it cites only notes it was given (no invented sources);
- abstention: when no relevant note was retrieved, it says so instead of
  answering from general knowledge.
"""

from __future__ import annotations

import re
import sqlite3
import statistics
from dataclasses import dataclass, field

from zk.agent import Ask
from zk.checks import estimate_tokens
from zk.embed import Embedder
from zk.evaluate import _resolver
from zk.search import DEFAULT_MODE, search

NOT_COVERED = "NOT IN NOTES"
MAX_NOTE_CHARS = 2400  # about 650 tokens: past what search by meaning reads anyway
_CITATION = re.compile(r"\[\[([^\]|#]+)")

PROMPT = """Answer the question using only the notes below. They come from my own Zettelkasten.

Rules:
- Cite every note you use as [[name]], with the name exactly as given after "name:".
- Use nothing from outside these notes. If they don't answer the question, reply with
  exactly "{not_covered}" and one sentence on what is missing.
- Keep the answer short: a few sentences.

Question: {question}

{notes}"""


@dataclass
class Answer:
    question: str
    text: str
    given: list[str]  # note names in the prompt, best first
    context_tokens: int
    cited: list[str] = field(default_factory=list)

    @property
    def abstained(self) -> bool:
        return self.text.lstrip().upper().startswith(NOT_COVERED)

    @property
    def grounded(self) -> bool:
        return all(c in self.given for c in self.cited)


def _name(path: str) -> str:
    return path.rsplit("/", 1)[-1].removesuffix(".md")


def build_prompt(
    conn: sqlite3.Connection,
    question: str,
    embedder: Embedder | None,
    k: int = 5,
    mode: str = DEFAULT_MODE,
) -> tuple[str, list[str]]:
    hits = search(conn, question, k=k, mode=mode, embedder=embedder)
    blocks, names = [], []
    for h in hits:
        row = conn.execute("SELECT title, body FROM notes WHERE path = ?", (h.path,)).fetchone()
        body = row["body"]
        if len(body) > MAX_NOTE_CHARS:
            body = body[:MAX_NOTE_CHARS] + " [...]"
        names.append(_name(h.path))
        blocks.append(f"---\nname: {names[-1]}\ntitle: {row['title']}\n\n{body.strip()}")
    notes = "\n\n".join(blocks) if blocks else "(no notes matched)"
    return PROMPT.format(not_covered=NOT_COVERED, question=question, notes=notes), names


def ask(
    conn: sqlite3.Connection,
    question: str,
    model: Ask,
    embedder: Embedder | None,
    k: int = 5,
    mode: str = DEFAULT_MODE,
) -> Answer:
    prompt, names = build_prompt(conn, question, embedder, k, mode)
    text = model(prompt)
    cited = list(dict.fromkeys(c.strip().removesuffix(".md") for c in _CITATION.findall(text)))
    return Answer(question, text, names, estimate_tokens(prompt), cited)


def run_answers(
    conn: sqlite3.Connection,
    queries: list[dict],
    model: Ask,
    embedder: Embedder | None,
    k: int = 5,
    mode: str = DEFAULT_MODE,
) -> dict:
    resolve = _resolver(conn)
    rows = []
    for q in queries:
        relevant = {_name(p) for p in (resolve(str(x)) for x in q["relevant"]) if p}
        a = ask(conn, q["query"], model, embedder, k, mode)
        retrieved = bool(relevant & set(a.given))
        rows.append(
            {
                "query": q["query"],
                "retrieved": retrieved,
                "cites_relevant": bool(relevant & set(a.cited)),
                "grounded": a.grounded,
                "abstained": a.abstained,
                "cited": a.cited,
                "invented": [c for c in a.cited if c not in a.given],
                "context_tokens": a.context_tokens,
            }
        )
    hit = [r for r in rows if r["retrieved"]]
    miss = [r for r in rows if not r["retrieved"]]

    def share(part: list, whole: list) -> float | None:
        return round(len(part) / len(whole), 3) if whole else None

    return {
        "questions": len(rows),
        "retrieved": len(hit),
        "citation_accuracy": share([r for r in hit if r["cites_relevant"]], hit),
        "grounded": share([r for r in rows if r["grounded"]], rows),
        "abstained_when_missing": share([r for r in miss if r["abstained"]], miss),
        "abstained_when_retrieved": share([r for r in hit if r["abstained"]], hit),
        "context_tokens_median": int(statistics.median(r["context_tokens"] for r in rows))
        if rows
        else 0,
        "results": rows,
    }


def format_answers(report: dict) -> str:
    def pct(value: float | None) -> str:
        return "-" if value is None else f"{value:.0%}"

    lines = [
        f"{report['questions']} questions; a relevant note was retrieved for {report['retrieved']}",
        f"  citation accuracy (cites a relevant note, when one was retrieved): "
        f"{pct(report['citation_accuracy'])}",
        f"  grounded (cites only notes it was given):  {pct(report['grounded'])}",
        f"  says 'not in notes' when none was retrieved: {pct(report['abstained_when_missing'])}",
        f"  says 'not in notes' although one was retrieved: "
        f"{pct(report['abstained_when_retrieved'])}",
        f"  context per question: median {report['context_tokens_median']} tokens",
    ]
    wrong = [
        r
        for r in report["results"]
        if r["invented"] or (r["retrieved"] and not r["cites_relevant"])
    ]
    if wrong:
        lines += ["", "to look at:"]
        for r in wrong:
            why = (
                f"cited notes it wasn't given: {', '.join(r['invented'])}"
                if r["invented"]
                else (
                    "answered 'not in notes'"
                    if r["abstained"]
                    else f"cited {r['cited'] or 'nothing'}"
                )
            )
            lines.append(f"  - {r['query']}: {why}")
    return "\n".join(lines)
