"""Map notes: one note per topic and per project, listing its notes. `zk maps VAULT`.

A map note is the Zettelkasten's table of contents for one topic: a line per
note, each a link plus its summary. Here they are generated, so they follow the
vault: topic maps come from the clusters `zk suggest` finds (maintain.clusters),
and project maps from the notes' `project:<name>` sources. Every session is
told the map names (see recall.py); a map is opened only when its topic comes
up, and a note only when it's needed.

Maps are rewritten when their notes change, after an ingest. Each map records
the hash of the body zk wrote, so one edited by hand is left alone from then on.
A topic keeps its map's name while the cluster still shares half its notes with
it, so maps don't churn as notes are added. A map whose topic is gone is moved
to maps/retired/, never deleted. All writes go through `zk.vault`, in one commit.

Patterns and decisions that come from a single project stay in that project's
map; they reach the topic maps once they hold for two or more projects (a merge
gives a note both notes' sources). Concepts and problems are general knowledge
and go in both.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import subprocess
from datetime import datetime
from pathlib import Path

from zk import maintain, vault
from zk.embed import Embedder
from zk.notes import parse_note
from zk.search import project_slug

MAPS_DIR = "maps"
RETIRED_DIR = "maps/retired"
SAME_MAP = 0.5  # a cluster keeps a map's name while they share this much (Jaccard)
PROJECT_MIN = 2  # a project gets a map once it has this many notes
SCOPED_TYPES = ("pattern", "decision")  # these stay in their project until a second one
LINE_CHARS = 160  # each entry: the note's title and the start of its summary
_PROJECT = re.compile(r"^project:([^:]+)", re.I)


def projects_of(sources: list[str]) -> list[str]:
    """The projects a note's sources name, as slugs, in order."""
    found = (_PROJECT.match(s.strip()) for s in sources)
    return list(dict.fromkeys(project_slug(m.group(1)) for m in found if m))


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60] or "topic"


def _rows(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT path, title, type, summary, tags, sources, frontmatter FROM notes ORDER BY path"
    ).fetchall()


def _existing(rows: list[sqlite3.Row]) -> dict[str, dict]:
    """zk's generated maps in the index: name -> kind, notes, project."""
    out = {}
    for r in rows:
        front = json.loads(r["frontmatter"] or "{}")
        if r["type"] == "map" and front.get("generated") == "zk":
            out[r["path"].rsplit("/", 1)[-1].removesuffix(".md")] = {
                "path": r["path"],
                "kind": front.get("map"),
                "project": front.get("project"),
                "notes": set(front.get("members") or []),
            }
    return out


def _entry(row: sqlite3.Row) -> str:
    stem = row["path"].rsplit("/", 1)[-1].removesuffix(".md")
    summary = " ".join((row["summary"] or "").split())
    line = f"{row['title']}: {summary}" if summary else row["title"]
    if len(line) > LINE_CHARS:
        line = line[: LINE_CHARS - 3].rstrip() + "..."
    return f"- [[{stem}]] {line}"


def plan(conn: sqlite3.Connection, embedder: Embedder) -> list[dict]:
    """The maps the vault should have now: name, kind, title, label and member paths."""
    rows = _rows(conn)
    notes = [r for r in rows if r["type"] != "map"]
    existing = _existing(rows)
    projects = {r["path"]: projects_of(json.loads(r["sources"] or "[]")) for r in notes}
    scoped = {
        r["path"] for r in notes if r["type"] in SCOPED_TYPES and len(projects[r["path"]]) == 1
    }
    taken = {r["path"].rsplit("/", 1)[-1].removesuffix(".md") for r in rows}
    out = []

    # Topics, matched to the existing topic map they overlap most (if half or more).
    free = {n: e for n, e in existing.items() if e["kind"] == "topic"}
    for cluster in maintain.clusters(conn, embedder, skip=scoped):
        members = set(cluster["notes"])

        def overlap(e: dict, members: set = members) -> float:
            return len(members & e["notes"]) / len(members | e["notes"])

        best = max(free, key=lambda n: (overlap(free[n]), n), default=None)
        label = ", ".join(cluster["shared_tags"]) or next(
            r["title"] for r in notes if r["path"] == cluster["closest_to_map"]["path"]
        )
        if best is not None and overlap(free[best]) >= SAME_MAP:
            name = best
            del free[best]
        else:
            name = _unique(f"map-{_slug(label)}", taken)
        taken.add(name)
        out.append({"name": name, "kind": "topic", "label": label, "notes": sorted(members)})

    # Projects with enough notes.
    by_project: dict[str, list[str]] = {}
    for path, names in projects.items():
        for p in names:
            by_project.setdefault(p, []).append(path)
    project_maps = {e["project"]: n for n, e in existing.items() if e["kind"] == "project"}
    for slug, members in sorted(by_project.items()):
        if len(members) < PROJECT_MIN:
            continue
        name = project_maps.get(slug) or _unique(f"project-{slug}", taken)
        taken.add(name)
        out.append(
            {
                "name": name,
                "kind": "project",
                "project": slug,
                "label": slug,
                "notes": sorted(members),
            }
        )
    return out


def _unique(name: str, taken: set[str]) -> str:
    candidate, n = name, 1
    while candidate in taken:
        n += 1
        candidate = f"{name}-{n}"
    return candidate


def _body(spec: dict, rows: dict[str, sqlite3.Row]) -> str:
    title = _title(spec)
    count = len(spec["notes"])
    what = (
        f"{count} notes on {spec['label']}."
        if spec["kind"] == "topic"
        else f"{count} notes from the project {spec['label']}."
    )
    entries = sorted(
        (rows[p] for p in spec["notes"] if p in rows), key=lambda r: r["title"].lower()
    )
    lines = "\n".join(_entry(r) for r in entries)
    return (
        f"# {title}\n\n## Summary\n\n{what} Generated by zk; editing this map by hand stops zk "
        f"from rewriting it.\n\n## Notes\n\n{lines}\n"
    )


def _title(spec: dict) -> str:
    return f"Map: {spec['label']}" if spec["kind"] == "topic" else f"Project: {spec['label']}"


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def render(spec: dict, rows: dict[str, sqlite3.Row], created: str) -> str:
    body = _body(spec, rows)
    front = {
        "id": spec["name"],
        "title": _title(spec),
        "type": "map",
        "map": spec["kind"],
        **({"project": spec["project"]} if spec.get("project") else {}),
        "tags": ["map"],
        "sources": ["zk:maps"],
        "created": created,
        "generated": "zk",
        "members": spec["notes"],
        "body_hash": _hash(body),
    }
    lines = "\n".join(f"{k}: {json.dumps(v, ensure_ascii=False)}" for k, v in front.items())
    return f"---\n{lines}\n---\n{body}"


def _edited_by_hand(text: str) -> bool:
    note = parse_note("map.md", text.encode("utf-8"))
    recorded = (note.frontmatter or {}).get("body_hash")
    return recorded != _hash(note.body)


def update(
    vault_dir: Path,
    conn: sqlite3.Connection,
    embedder: Embedder,
    now: datetime | None = None,
    dry_run: bool = False,
) -> dict:
    """Bring maps/ in line with the vault: write new maps, rewrite changed ones, retire old ones.

    Returns what changed. With dry_run, nothing is written.
    """
    now = now or datetime.now().astimezone()
    rows = _rows(conn)
    by_path = {r["path"]: r for r in rows}
    wanted = plan(conn, embedder)
    existing = _existing(rows)
    result: dict = {"new": [], "updated": [], "retired": [], "edited": [], "unchanged": 0}
    touched: list[str] = []
    for spec in wanted:
        rel = f"{MAPS_DIR}/{spec['name']}.md"
        path = vault_dir / rel
        if not path.exists():
            text = render(spec, by_path, now.date().isoformat())
            result["new"].append(rel)
            if not dry_run:
                vault.write_new(vault_dir, rel, text)
                touched.append(rel)
            continue
        old = path.read_text(encoding="utf-8")
        if _edited_by_hand(old):
            result["edited"].append(rel)
            continue
        created = str((parse_note(rel, old.encode("utf-8")).frontmatter or {}).get("created", ""))
        text = render(spec, by_path, created or now.date().isoformat())
        if text == old:
            result["unchanged"] += 1
            continue
        result["updated"].append(rel)
        if not dry_run:
            vault.replace(vault_dir, rel, text, hashlib.sha256(old.encode("utf-8")).hexdigest())
            touched.append(rel)
    keep = {s["name"] for s in wanted}
    for name, e in existing.items():
        if name in keep or not (vault_dir / e["path"]).exists():
            continue
        if _edited_by_hand((vault_dir / e["path"]).read_text(encoding="utf-8")):
            result["edited"].append(e["path"])
            continue
        dst = vault.unique_path(vault_dir, f"{RETIRED_DIR}/{name}.md")
        result["retired"].append(e["path"])
        if not dry_run:
            vault.move_new(vault_dir, e["path"], dst)
            touched += [e["path"], dst]
    if touched:
        message = (
            f"zk: maps, {len(result['new'])} new, {len(result['updated'])} updated, "
            f"{len(result['retired'])} retired"
        )
        try:
            result["commit"] = vault.git_commit(vault_dir, touched, message)
        except (subprocess.CalledProcessError, OSError) as exc:
            result["commit_error"] = f"written but not committed: {exc}"
    return result


def format_update(result: dict) -> str:
    lines = [
        f"maps: {len(result['new'])} new, {len(result['updated'])} updated, "
        f"{len(result['retired'])} retired, {result['unchanged']} unchanged"
    ]
    for key, label in (("new", "new"), ("updated", "updated"), ("retired", "retired")):
        lines += [f"  {label}: {p}" for p in result[key]]
    lines += [f"  left alone (edited by hand): {p}" for p in result["edited"]]
    if result.get("commit_error"):
        lines.append(result["commit_error"])
    return "\n".join(lines)
