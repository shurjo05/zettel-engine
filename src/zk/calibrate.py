"""Re-check the similarity thresholds as the vault grows: `zk calibrate`.

The cuts in proposals.py and maintain.py were set by judging pairs by hand on
93 notes. A growing vault fills topics in, so more pairs score high, and a cut
can drift from meaning what it did. This samples note pairs in a narrow band
just below and just above each cut, has a model judge each pair (same idea,
related, or unrelated), and reports precision per band. It warns when the band
above a cut is no longer as precise as it was by hand. It never changes a
threshold: that stays a reviewed change (decisions 17 and 20).

Labels are kept in the vault's eval/calibration/ with both notes' content
hashes. A later run reuses a label while both notes are unchanged, so each run
only pays for new pairs and the labelled set grows.

Reads the index and the vault's git log; writes only label files, through
`zk.vault`, in one commit.
"""

from __future__ import annotations

import json
import re
import sqlite3
import subprocess
import zlib
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from zk import vault
from zk.agent import Ask
from zk.embed import Embedder
from zk.maintain import LINK_AT, PARENT_AT
from zk.proposals import DUPLICATE_AT

FOLDER = "eval/calibration"
BAND = 0.03
LABELS = ("same", "related", "unrelated")
MAX_NOTE_CHARS = 1500

# What each cut separates, which labels count as a hit above it, and the precision
# measured by hand on 2026-11-02/03 (decisions 19 and 20).
CUTS = {
    "duplicate": {"at": DUPLICATE_AT, "hit": {"same"}, "by_hand": 0.6},
    "link": {"at": LINK_AT, "hit": {"same", "related"}, "by_hand": 0.94},
    "parent": {"at": PARENT_AT, "hit": {"same", "related"}, "by_hand": 0.75},
}

PROMPT = """You are checking pairs of notes from a personal Zettelkasten. For each pair, say
whether the two notes state the SAME idea (one could replace the other), are RELATED (a reader of
one would want a link to the other), or are UNRELATED (sharing words or a broad topic is not
enough).

Reply with one JSON object per line and nothing else, like:
{{"pair": 1, "label": "related", "why": "both about RRF, one explains k"}}

{pairs}"""


def _note_text(row) -> str:
    body = row["body"].strip()
    if len(body) > MAX_NOTE_CHARS:
        body = body[:MAX_NOTE_CHARS] + " [...]"
    return f"title: {row['title']}\n{body}"


def _name(path: str) -> str:
    return path.rsplit("/", 1)[-1].removesuffix(".md")


def load_labels(vault_dir: Path) -> dict[tuple[str, str], dict]:
    """Saved labels keyed by the two notes' content hashes (sorted)."""
    out: dict[tuple[str, str], dict] = {}
    folder = vault_dir / FOLDER
    for path in sorted(folder.glob("*.yaml")) if folder.is_dir() else []:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for item in data.get("pairs", []):
            if item.get("label") in LABELS:
                out[tuple(sorted((item["hash_a"], item["hash_b"])))] = item
    return out


def sample_pairs(conn: sqlite3.Connection, embedder: Embedder, per_band: int = 6) -> list[dict]:
    """Up to `per_band` pairs from each band just below and above each cut.

    The choice is a hash of the two names, so a run picks the same pairs as the
    last one while the notes are unchanged, and the saved labels get reused.
    """
    rows = conn.execute(
        "SELECT n.path, n.title, n.body, n.content_hash, e.vector FROM notes n JOIN embeddings e "
        "ON e.content_hash = n.content_hash AND e.model = ? ORDER BY n.path",
        (embedder.model,),
    ).fetchall()
    if len(rows) < 2:
        return []
    matrix = np.stack([np.frombuffer(r["vector"], dtype=np.float32) for r in rows])
    scores = np.triu(matrix @ matrix.T, k=1)
    out = []
    for cut, spec in CUTS.items():
        for side, low, high in (
            ("below", spec["at"] - BAND, spec["at"]),
            ("above", spec["at"], spec["at"] + BAND),
        ):
            idx = np.argwhere((scores >= low) & (scores < high))
            pairs = sorted(
                (zlib.crc32(f"{rows[i]['path']}|{rows[j]['path']}".encode()), int(i), int(j))
                for i, j in idx
            )[:per_band]
            for _, i, j in pairs:
                a, b = rows[i], rows[j]
                out.append(
                    {
                        "cut": cut,
                        "side": side,
                        "similarity": round(float(scores[i, j]), 3),
                        "a": _name(a["path"]),
                        "b": _name(b["path"]),
                        "hash_a": a["content_hash"],
                        "hash_b": b["content_hash"],
                        "text_a": _note_text(a),
                        "text_b": _note_text(b),
                    }
                )
    return out


def judge(pairs: list[dict], model: Ask, batch: int = 8) -> None:
    """Fill in `label` and `why` for each pair, `batch` pairs per model call."""
    for start in range(0, len(pairs), batch):
        chunk = pairs[start : start + batch]
        text = "\n\n".join(
            f"=== Pair {n} ===\n--- note A ---\n{p['text_a']}\n--- note B ---\n{p['text_b']}"
            for n, p in enumerate(chunk, 1)
        )
        reply = model(PROMPT.format(pairs=text))
        for line in reply.splitlines():
            match = re.search(r"\{.*\}", line)
            if not match:
                continue
            try:
                item = json.loads(match.group(0))
                p = chunk[int(item["pair"]) - 1]
            except (ValueError, KeyError, IndexError, TypeError):
                continue
            label = str(item.get("label", "")).lower()
            if label in LABELS:
                p["label"], p["why"] = label, str(item.get("why", ""))[:300]


def precision(pairs: list[dict]) -> dict:
    """Per cut and side: how many pairs were judged, and the share that counts as a hit."""
    out = {}
    for cut, spec in CUTS.items():
        out[cut] = {"at": spec["at"], "by_hand": spec["by_hand"]}
        for side in ("below", "above"):
            judged = [p for p in pairs if p["cut"] == cut and p["side"] == side and p.get("label")]
            hits = [p for p in judged if p["label"] in spec["hit"]]
            out[cut][side] = {
                "judged": len(judged),
                "hits": len(hits),
                "precision": round(len(hits) / len(judged), 3) if judged else None,
            }
    return out


def warnings(report: dict, min_judged: int = 5) -> list[str]:
    out = []
    for cut, r in report.items():
        above, below = r["above"], r["below"]
        if above["judged"] >= min_judged and above["precision"] < r["by_hand"]:
            out.append(
                f"{cut} cut {r['at']}: {above['hits']} of {above['judged']} pairs just above it "
                f"are hits ({above['precision']:.0%}), below the {r['by_hand']:.0%} measured by "
                f"hand. Consider raising it."
            )
        if (
            below["judged"] >= min_judged
            and below["precision"] is not None
            and below["precision"] >= r["by_hand"]
        ):
            out.append(
                f"{cut} cut {r['at']}: {below['hits']} of {below['judged']} pairs just below it "
                f"are hits too ({below['precision']:.0%}). It may be missing good pairs; "
                f"consider lowering it."
            )
    return out


def thin(report: dict, min_judged: int = 5) -> list[str]:
    """Bands that point the way a warning would but hold too few pairs to warn on."""
    out = []
    for cut, r in report.items():
        for side in ("above", "below"):
            s = r[side]
            if not 0 < s["judged"] < min_judged:
                continue
            if (side == "above" and s["precision"] < r["by_hand"]) or (
                side == "below" and s["precision"] >= r["by_hand"]
            ):
                out.append(
                    f"{cut} cut {r['at']}: {s['hits']} of {s['judged']} pairs just {side} it are "
                    f"hits, against {r['by_hand']:.0%} by hand; too few pairs to warn on yet"
                )
    return out


def drift(vault_dir: Path) -> dict:
    """Signals from the vault's git history that need no model."""
    try:
        log = subprocess.run(
            ["git", "-C", str(vault_dir), "log", "--format=%s"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):
        return {}
    ingests = [
        m for m in (re.match(r"zk: ingest .*: (\d+) notes?(?:, (\d+) held)?$", s) for s in log) if m
    ]
    notes = sum(int(m.group(1)) for m in ingests)
    held = sum(int(m.group(2) or 0) for m in ingests)
    merged = sum(1 for s in log if s.startswith("zk: merge held "))
    released = sum(1 for s in log if s.startswith("zk: add held note "))
    return {
        "ingest_runs": len(ingests),
        "notes_written": notes,
        "notes_held": held,
        "hold_rate": round(held / (notes + held), 3) if notes + held else None,
        "resolved_merged": merged,
        "resolved_released": released,
        "false_hold_share": round(released / (merged + released), 3) if merged + released else None,
    }


def save(vault_dir: Path, pairs: list[dict], now: datetime) -> dict:
    """Write the newly judged pairs (texts left out) as one file and commit it."""
    keep = ("cut", "side", "similarity", "a", "b", "hash_a", "hash_b", "label", "why")
    rows = [{k: p[k] for k in keep if k in p} for p in pairs if p.get("label")]
    if not rows:
        return {}
    rel = vault.unique_path(vault_dir, f"{FOLDER}/{now:%Y-%m-%d-%H%M%S}.yaml")
    vault.write_new(
        vault_dir,
        rel,
        "# Note pairs judged by zk calibrate; reused while both notes are unchanged.\n"
        + yaml.safe_dump({"pairs": rows}, sort_keys=False, allow_unicode=True),
    )
    try:
        commit = vault.git_commit(vault_dir, [rel], f"zk: calibrate, {len(rows)} pairs judged")
    except (subprocess.CalledProcessError, OSError):
        commit = None
    return {"file": rel, "commit": commit}


def run(
    vault_dir: Path,
    conn: sqlite3.Connection,
    embedder: Embedder,
    model: Ask | None,
    per_band: int = 6,
    now: datetime | None = None,
) -> dict:
    """Sample, reuse saved labels, judge the rest (unless `model` is None), and report."""
    now = now or datetime.now().astimezone()
    pairs = sample_pairs(conn, embedder, per_band)
    saved = load_labels(vault_dir)
    for p in pairs:
        old = saved.get(tuple(sorted((p["hash_a"], p["hash_b"]))))
        if old:
            p["label"], p["why"], p["reused"] = old["label"], old.get("why", ""), True
    new = [p for p in pairs if not p.get("label")]
    written = {}
    if model is not None and new:
        judge(new, model)
        written = save(vault_dir, new, now)
    report = precision(pairs)
    return {
        "pairs": len(pairs),
        "reused": sum(1 for p in pairs if p.get("reused")),
        "judged_now": sum(1 for p in new if p.get("label")),
        "unjudged": sum(1 for p in pairs if not p.get("label")),
        "cuts": report,
        "warnings": warnings(report),
        "too_few": thin(report),
        "drift": drift(vault_dir),
        **written,
    }


def format_calibration(result: dict) -> str:
    lines = [
        f"{result['pairs']} pairs sampled: {result['reused']} labels reused, "
        f"{result['judged_now']} judged now, {result['unjudged']} unjudged"
    ]

    def cell(s: dict) -> str:
        share = f" ({s['precision']:.0%})" if s["precision"] is not None else ""
        return f"{s['hits']}/{s['judged']}{share}"

    for cut, r in result["cuts"].items():
        lines.append(
            f"  {cut:<9} cut {r['at']:.2f}  below: {cell(r['below']):<12} "
            f"above: {cell(r['above']):<12} by hand: {r['by_hand']:.0%}"
        )
    d = result.get("drift") or {}
    if d:
        rate = "-" if d["hold_rate"] is None else f"{d['hold_rate']:.0%}"
        false = "-" if d["false_hold_share"] is None else f"{d['false_hold_share']:.0%}"
        lines += [
            "",
            f"vault history: {d['ingest_runs']} ingests wrote {d['notes_written']} notes and "
            f"held {d['notes_held']} ({rate}); the resolver merged {d['resolved_merged']} and "
            f"released {d['resolved_released']} (false holds: {false})",
        ]
    if result["warnings"]:
        lines += ["", "warnings:"] + [f"  - {w}" for w in result["warnings"]]
    if result.get("too_few"):
        lines += ["", "worth watching (too few pairs to warn on):"]
        lines += [f"  - {w}" for w in result["too_few"]]
    if result.get("file"):
        lines += ["", f"labels saved to {result['file']}"]
    return "\n".join(lines)
