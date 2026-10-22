"""Retrieval evaluation: run labelled queries and measure ranking quality.

A query set is a YAML file:

    queries:
      - query: deduplicate notes
        relevant: [jaccard-similarity-for-note-deduplication]   # note file names or ids
        kind: keyword                                            # optional label

Metrics (averaged over queries):
- Recall@k: share of a query's relevant notes that appear in the top k.
- MRR: mean of 1 / rank of the first relevant note (0 if none in the top `depth`).
- Latency: wall-clock time of each search call.
"""

from __future__ import annotations

import sqlite3
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from zk.search import search

if TYPE_CHECKING:
    from zk.embed import Embedder

CUTOFFS = (1, 3, 5, 10)


@dataclass
class QueryResult:
    query: str
    kind: str | None
    relevant: list[str]  # resolved note paths
    unresolved: list[str]  # labels that match no note
    ranked: list[str]  # top `depth` paths returned
    first_hit_rank: int | None
    recall: dict[int, float]
    latency_ms: float


@dataclass
class EvalReport:
    results: list[QueryResult] = field(default_factory=list)

    def _mean(self, values: list[float]) -> float:
        return round(statistics.fmean(values), 3) if values else 0.0

    def summary(self, kind: str | None = None) -> dict:
        rs = [r for r in self.results if kind is None or r.kind == kind]
        latencies = sorted(r.latency_ms for r in rs)
        p95 = latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))] if latencies else 0.0
        return {
            "queries": len(rs),
            **{f"recall@{k}": self._mean([r.recall[k] for r in rs]) for k in CUTOFFS},
            "mrr": self._mean([1 / r.first_hit_rank if r.first_hit_rank else 0.0 for r in rs]),
            "latency_ms_mean": self._mean([r.latency_ms for r in rs]),
            "latency_ms_p95": round(p95, 3),
        }

    def kinds(self) -> list[str]:
        return sorted({r.kind for r in self.results if r.kind})


def load_queries(path: Path) -> list[dict]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    queries = data.get("queries", [])
    for q in queries:
        if not q.get("query") or not q.get("relevant"):
            raise ValueError(f"each query needs `query` and `relevant`: {q!r}")
    return queries


def _resolver(conn: sqlite3.Connection):
    """Map a label (file name without .md, or frontmatter id) to a note path."""
    by_label: dict[str, str] = {}
    for row in conn.execute("SELECT path, id FROM notes"):
        by_label[row["path"].rsplit("/", 1)[-1][:-3].lower()] = row["path"]
        if row["id"]:
            by_label.setdefault(row["id"].lower(), row["path"])
    return lambda label: by_label.get(str(label).lower().removesuffix(".md"))


def run_eval(
    conn: sqlite3.Connection,
    queries: list[dict],
    depth: int = 10,
    mode: str = "keyword",
    embedder: Embedder | None = None,
) -> EvalReport:
    resolve = _resolver(conn)
    report = EvalReport()
    for q in queries:
        labels = [str(x) for x in q["relevant"]]
        relevant = [p for p in (resolve(x) for x in labels) if p]
        unresolved = [x for x in labels if not resolve(x)]

        start = time.perf_counter()
        hits = search(conn, q["query"], k=depth, mode=mode, embedder=embedder)
        latency_ms = (time.perf_counter() - start) * 1000

        ranked = [h.path for h in hits]
        ranks = [ranked.index(p) + 1 for p in relevant if p in ranked]
        recall = {
            k: (sum(1 for r in ranks if r <= k) / len(relevant)) if relevant else 0.0
            for k in CUTOFFS
        }
        report.results.append(
            QueryResult(
                query=q["query"],
                kind=q.get("kind"),
                relevant=relevant,
                unresolved=unresolved,
                ranked=ranked,
                first_hit_rank=min(ranks) if ranks else None,
                recall=recall,
                latency_ms=round(latency_ms, 3),
            )
        )
    return report


def format_report(report: EvalReport) -> str:
    def row(name: str, s: dict) -> str:
        cells = [f"{s[f'recall@{k}']:.2f}" for k in CUTOFFS]
        return (
            f"{name:<12}{s['queries']:>4}  "
            + "  ".join(f"{c:>6}" for c in cells)
            + f"  {s['mrr']:>5.2f}  {s['latency_ms_mean']:>7.2f}  {s['latency_ms_p95']:>7.2f}"
        )

    header = f"{'':<12}{'n':>4}  " + "  ".join(f"{'R@' + str(k):>6}" for k in CUTOFFS)
    lines = [header + "    MRR  mean ms   p95 ms", row("all", report.summary())]
    lines += [row(kind, report.summary(kind)) for kind in report.kinds()]

    lines += ["", "per query (rank of first relevant note, '-' = not in top 10):"]
    for r in report.results:
        rank = str(r.first_hit_rank) if r.first_hit_rank else "-"
        lines.append(f"  {rank:>2}  [{r.kind or ''}] {r.query}")
        if r.unresolved:
            lines.append(f"      unknown labels: {', '.join(r.unresolved)}")
        if not r.first_hit_rank and r.ranked:
            top = r.ranked[0].rsplit("/", 1)[-1]
            lines.append(f"      top result was {top}")
    return "\n".join(lines)
