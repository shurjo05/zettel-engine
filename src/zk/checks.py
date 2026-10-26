"""Checks that run after every ingest and only warn: search quality and new-note health.

Read-only on the vault and the index. The one file it writes is the score
history, a JSON file next to the index (outside the vault). It keeps each test
question's reciprocal rank from the first time it was scored, so a later run can
tell a real drop from noise.
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
import statistics
import tempfile
import zlib
from datetime import datetime
from pathlib import Path

import numpy as np

from zk.embed import Embedder, note_text
from zk.evaluate import load_queries, run_eval
from zk.proposals import DUPLICATE_AT

QUESTIONS_DIR = "eval/questions"
HOLDOUT_EVERY = 5  # one question in five is held out of any tuning
MAX_TOKENS = 512  # bge-small reads at most this many tokens of a note
# bge-small's tokenizer averaged about 3.7 characters per token on the legacy notes
# (median note 481 tokens); an estimate is enough to flag notes that run long.
CHARS_PER_TOKEN = 3.7
MIN_QUERIES = 5  # fewer paired questions than this can't show a drop beyond noise
MIN_DROP = 0.02  # ignore MRR drops smaller than this even when they are consistent
RESAMPLES = 2000


def is_holdout(question: str) -> bool:
    """A fixed one-in-five sample, chosen by a hash so it never depends on run order."""
    return zlib.crc32(" ".join(question.lower().split()).encode()) % HOLDOUT_EVERY == 0


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def load_vault_queries(vault: Path) -> list[dict]:
    """Every test question saved in the vault's eval/ folder."""
    folder = vault / "eval"
    if not folder.is_dir():
        return []
    files = sorted(p for p in folder.rglob("*") if p.suffix in (".yaml", ".yml"))
    return [q for f in files for q in load_queries(f)]


def history_path(db: Path) -> Path:
    return db.with_name(f"{db.stem}-eval.json")


def _load_history(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    data.setdefault("queries", {})
    data.setdefault("runs", [])
    return data


def _save_history(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".zk-", suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)
    os.replace(tmp, path)


def drop_beyond_noise(baseline: list[float], current: list[float], seed: int = 0) -> float | None:
    """The mean drop in reciprocal rank if it is beyond noise, else None.

    A paired bootstrap: resample the questions with replacement and recompute the
    mean drop many times. The drop counts as real when even its 5th percentile is
    above zero, i.e. it isn't explained by which questions happened to be asked.
    """
    if len(baseline) < MIN_QUERIES:
        return None
    diffs = np.asarray(baseline) - np.asarray(current)
    mean = float(diffs.mean())
    if mean < MIN_DROP:
        return None
    rng = np.random.default_rng(seed)
    means = diffs[rng.integers(0, len(diffs), (RESAMPLES, len(diffs)))].mean(axis=1)
    return mean if float(np.quantile(means, 0.05)) > 0 else None


def check_scores(
    conn: sqlite3.Connection,
    vault: Path,
    history: Path,
    embedder: Embedder | None,
    now: datetime,
) -> list[str]:
    """Re-score every saved question; warn when a group's scores fell beyond noise."""
    queries = load_vault_queries(vault)
    if not queries:
        return []
    mode = "hybrid" if embedder is not None else "keyword"
    report = run_eval(conn, queries, depth=10, mode=mode, embedder=embedder)
    data = _load_history(history)
    groups: dict[str, list[tuple[str, float, float]]] = {}
    for q, result in zip(queries, report.results, strict=True):
        if not result.relevant:
            continue  # its note was removed or is still held; nothing to score
        key = f"{q['query'].strip()} -> {','.join(sorted(result.relevant))}"
        rr = 1 / result.first_hit_rank if result.first_hit_rank else 0.0
        seen = data["queries"].setdefault(key, {"rr": rr, "first_scored": now.isoformat()})
        group = {"holdout": "held-out", "generated": "generated"}.get(q.get("kind"), "hand-written")
        groups.setdefault(group, []).append((q["query"], seen["rr"], rr))

    warnings = []
    for group, rows in sorted(groups.items()):
        base, cur = [r[1] for r in rows], [r[2] for r in rows]
        drop = drop_beyond_noise(base, cur)
        if drop is None:
            continue
        lost = [text for text, b, c in rows if b >= 0.2 and c == 0][:3]
        warnings.append(
            f"Search got worse on the {group} test questions: MRR {statistics.fmean(base):.2f} "
            f"when they were added, {statistics.fmean(cur):.2f} now ({len(rows)} questions), "
            "a drop beyond noise." + (f" No longer found: {'; '.join(lost)}." if lost else "")
        )
    data["runs"] = [
        *data["runs"][-99:],
        {
            "at": now.isoformat(),
            "mode": mode,
            **{g: round(statistics.fmean(r[2] for r in rows), 3) for g, rows in groups.items()},
        },
    ]
    _save_history(history, data)
    return warnings


def check_new_notes(conn: sqlite3.Connection, paths: list[str]) -> list[str]:
    """Health problems in notes that were just added."""
    warnings = []
    for path in paths:
        row = conn.execute("SELECT title, body FROM notes WHERE path = ?", (path,)).fetchone()
        if row is None:
            continue
        tokens = estimate_tokens(note_text(row["title"], row["body"]))
        if tokens > MAX_TOKENS:
            warnings.append(
                f"{path} is about {tokens} tokens, over the {MAX_TOKENS} the embedding model "
                "reads, so its end is invisible to search by meaning."
            )
        for link in conn.execute(
            "SELECT target, status FROM links WHERE src_path = ? AND status != 'resolved' "
            "ORDER BY target",
            (path,),
        ):
            what = "matches several notes" if link["status"] == "ambiguous" else "doesn't exist"
            warnings.append(f"{path} links to [[{link['target']}]], which {what}.")
        linked = conn.execute(
            "SELECT 1 FROM links WHERE status = 'resolved' AND src_path != dst_path "
            "AND (src_path = ? OR dst_path = ?) LIMIT 1",
            (path, path),
        ).fetchone()
        if not linked:
            warnings.append(f"{path} is linked to no other note.")
    return warnings


def long_notes(conn: sqlite3.Connection) -> list[tuple[str, int]]:
    """Every note estimated to be over the embedding model's limit, longest first."""
    rows = conn.execute("SELECT path, title, body FROM notes").fetchall()
    sized = [(r["path"], estimate_tokens(note_text(r["title"], r["body"]))) for r in rows]
    return sorted(((p, n) for p, n in sized if n > MAX_TOKENS), key=lambda x: (-x[1], x[0]))


def near_duplicates(
    conn: sqlite3.Connection, embedder: Embedder, at: float = DUPLICATE_AT, block: int = 512
) -> list[tuple[str, str, float]]:
    """Pairs of notes at cosine similarity `at` or more, most similar first."""
    rows = conn.execute(
        "SELECT n.path, e.vector FROM notes n JOIN embeddings e "
        "ON e.content_hash = n.content_hash AND e.model = ? ORDER BY n.path",
        (embedder.model,),
    ).fetchall()
    if len(rows) < 2:
        return []
    matrix = np.stack([np.frombuffer(r["vector"], dtype=np.float32) for r in rows])
    pairs = []
    for start in range(0, len(rows), block):  # a block at a time keeps memory flat
        scores = matrix[start : start + block] @ matrix.T
        for i, j in zip(*np.nonzero(scores >= at), strict=True):
            a = start + int(i)
            if int(j) > a:
                pairs.append((rows[a]["path"], rows[int(j)]["path"], round(float(scores[i, j]), 3)))
    return sorted(pairs, key=lambda p: (-p[2], p[0], p[1]))


def render_warnings(run: str, warnings: list[str], now: datetime) -> str:
    lines = "\n".join(f"- {w}" for w in warnings)
    return (
        f'---\nkind: "warnings"\nrun: "{run}"\ncreated: "{now.isoformat(timespec="seconds")}"\n'
        f"---\n# zk warnings from ingest run {run}\n\n{lines}\n"
    )
