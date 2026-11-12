"""The weekly vault health review: `zk review VAULT`. No model, nothing sent anywhere.

The daily run keeps the vault growing; this checks, less often, that it stays
healthy as it grows. It gathers what the other reports already measure into one
page:
- the note health check (zk check): errors, broken links, orphans;
- the inbox: captures and held notes still waiting;
- retrieval on the vault's own test questions (zk eval), and how many there are;
- maps that are out of date, and topics that have no map yet (zk maps --dry-run);
- the drift signals from git history (zk calibrate): hold rate and false holds;
- what the session brief costs in tokens;
- which projects `zk learn` has read.

It writes the page to inbox/reports/review-<date>.md, through `zk.vault`, in one
commit, and changes nothing else.
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
from datetime import datetime
from pathlib import Path

from zk import calibrate, checks, learn, maps, process, recall, vault
from zk.embed import Embedder
from zk.evaluate import run_eval
from zk.index import health_report
from zk.ingest import list_held, list_inbox

REPORTS_DIR = "inbox/reports"
FEW_QUESTIONS = 20  # below this, a drop in retrieval can't be told from noise


def gather(vault_dir: Path, conn: sqlite3.Connection, embedder: Embedder) -> dict:
    health = health_report(conn)
    queries = checks.load_vault_queries(vault_dir)
    scores = run_eval(conn, queries, mode="hybrid", embedder=embedder).summary() if queries else {}
    pending = maps.update(vault_dir, conn, embedder, dry_run=True)
    brief = recall.brief(conn)
    projects = [
        json.loads(f or "{}").get("project")
        for (f,) in conn.execute("SELECT frontmatter FROM notes WHERE type = 'map'")
    ]
    largest = max(
        (checks.estimate_tokens(recall.brief(conn, project=p)) for p in projects if p), default=0
    )
    record = learn.load_record()
    return {
        "notes": health.notes - health.by_type.get("map", 0),
        "maps": health.by_type.get("map", 0),
        "errors": health.errors,
        "warnings": health.warnings,
        "broken_links": len(health.broken_links),
        "orphans": len(health.orphans),
        "inbox": len(list_inbox(vault_dir)),
        "held": len(list_held(vault_dir)),
        "questions": len(queries),
        "mrr": scores.get("mrr"),
        "recall@5": scores.get("recall@5"),
        "maps_pending": len(pending["new"]) + len(pending["updated"]) + len(pending["retired"]),
        "maps_edited": len(pending["edited"]),
        "drift": calibrate.drift(vault_dir),
        "brief_tokens": checks.estimate_tokens(brief),
        "brief_tokens_in_project": max(largest, checks.estimate_tokens(brief)),
        "projects_learned": sorted(r["project"] for r in record["repos"].values()),
        "projects_skipped": len(record["skip"]),
        "last_process": process.last_status(),
    }


def advice(r: dict) -> list[str]:
    """What, if anything, needs a person."""
    out = []
    if r["errors"]:
        out.append(f"{r['errors']} note errors: run zk check to see them.")
    if r["broken_links"]:
        out.append(f"{r['broken_links']} broken links: zk check lists them.")
    if r["held"]:
        out.append(f"{r['held']} notes held as likely duplicates are waiting for zk process.")
    last = r.get("last_process") or {}
    if last and last.get("finished") is False and not process.lock_path().exists():
        out.append(
            f"The last zk process run ({last.get('at')}) never finished: it was stopped partway. "
            f"Its log, {last.get('log')}, shows how far it got."
        )
    elif last and last.get("reason") == "usage limit":
        out.append(
            f"The last zk process run ({last.get('at')}) stopped at Claude's usage limit; "
            "the next run picks up the rest."
        )
    elif last and not last.get("ok", True):
        out.append(
            f"The last zk process run ({last.get('at')}) processed none of its captures, so "
            f"they're piling up: its log, {last.get('log')}, says why."
        )
    if r["inbox"]:
        out.append(f"{r['inbox']} captures are waiting in the inbox for zk process.")
    if r["questions"] < FEW_QUESTIONS:
        out.append(
            f"Only {r['questions']} test questions: too few to tell a retrieval drop from noise."
        )
    if r["maps_pending"]:
        out.append(f"{r['maps_pending']} map changes are pending: zk maps brings them up to date.")
    false = (r["drift"] or {}).get("false_hold_share")
    if false is not None and false >= 0.5:
        out.append(
            f"The resolver kept {false:.0%} of held notes as separate notes: the duplicate cut may "
            "be too loose; zk calibrate checks it."
        )
    if r["brief_tokens_in_project"] > 1000:
        out.append(
            f"The session brief reaches {r['brief_tokens_in_project']} tokens in a project's "
            "folder, over the 1,000 budget."
        )
    return out


def render(r: dict, now: datetime) -> str:
    def num(value, fmt="{:.3f}"):
        return "-" if value is None else fmt.format(value)

    d = r["drift"] or {}
    lines = [
        "---",
        "kind: report",
        f"created: {now.isoformat(timespec='seconds')}",
        "---",
        f"# Vault review {now:%Y-%m-%d}",
        "",
        "## Needs attention",
        "",
        *([f"- {a}" for a in advice(r)] or ["- Nothing."]),
        "",
        "## Numbers",
        "",
        f"- Notes: {r['notes']}, maps: {r['maps']}, orphans: {r['orphans']}",
        f"- Errors: {r['errors']}, warnings: {r['warnings']}, broken links: {r['broken_links']}",
        f"- Inbox: {r['inbox']} captures, {r['held']} held",
        f"- Retrieval on {r['questions']} test questions: MRR {num(r['mrr'])}, "
        f"recall@5 {num(r['recall@5'])}",
        f"- Maps: {r['maps_pending']} changes pending, {r['maps_edited']} edited by hand",
        f"- Holds: rate {num(d.get('hold_rate'), '{:.0%}')}, false holds "
        f"{num(d.get('false_hold_share'), '{:.0%}')} "
        f"({d.get('resolved_merged', 0)} merged, {d.get('resolved_released', 0)} released)",
        f"- Session brief: about {r['brief_tokens']} tokens, up to "
        f"{r['brief_tokens_in_project']} in a project's folder",
        "- Last zk process: "
        + (
            f"{r['last_process'].get('at')}, "
            + (
                "never finished"
                if r["last_process"].get("finished") is False
                else "fine"
                if r["last_process"].get("ok", True)
                else "processed nothing"
            )
            if r.get("last_process")
            else "none recorded"
        ),
        f"- Projects read by zk learn: {', '.join(r['projects_learned']) or 'none'}"
        + (f" ({r['projects_skipped']} kept out)" if r["projects_skipped"] else ""),
        "",
    ]
    return "\n".join(lines)


def run(
    vault_dir: Path,
    conn: sqlite3.Connection,
    embedder: Embedder,
    now: datetime | None = None,
    write: bool = True,
) -> dict:
    now = now or datetime.now().astimezone()
    r = gather(vault_dir, conn, embedder)
    r["text"] = render(r, now)
    if write:
        rel = vault.unique_path(vault_dir, f"{REPORTS_DIR}/review-{now:%Y-%m-%d}.md")
        vault.write_new(vault_dir, rel, r["text"])
        r["file"] = rel
        try:
            r["commit"] = vault.git_commit(vault_dir, [rel], f"zk: review {now:%Y-%m-%d}")
        except (subprocess.CalledProcessError, OSError):
            r["commit"] = None
    return r
