"""Import a legacy vault's notes into this vault: convert, dedup, repair links, account for all.

The legacy vault is only ever read. Each legacy note becomes a note in the
current schema with its old id, dates, tags and provenance; links are rewritten
to the new file names and, where they had drifted (a shorter slug, an old id
prefix), repaired. A note that duplicates one imported before it is held in
inbox/held/, like ingest does. Links to it keep its name, so they resolve again
if it is released with apply_held, and are reported as waiting on a held note.
Every file in the legacy folder ends up in the report: imported, held or
skipped, with the reason. The whole import is one git commit.
"""

from __future__ import annotations

import re
import shutil
import sqlite3
import subprocess
import tempfile
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import yaml

from zk import vault
from zk.embed import Embedder, note_text, update_embeddings
from zk.index import build_index, open_db
from zk.notes import NOTE_TYPES, Note, parse_note
from zk.proposals import (
    DUPLICATE_AT,
    HELD_DIR,
    NOTES_DIR,
    InvalidProposal,
    Proposal,
    render,
    slugify,
)
from zk.search import similar_to_vector

TYPE_MAP = {t: t for t in NOTE_TYPES} | {"snippet": "pattern"}
# Sections that become frontmatter or the Related list instead of Details.
_META_SECTIONS = {"summary", "links", "related", "source note", "sources"}
_ID = re.compile(r"^(\d{14})")
_ID_PREFIX = re.compile(r"^\d{14}-")
_WIKILINK = re.compile(r"\[\[([^\[\]|#]*)((?:#[^\[\]|]*)?(?:\|[^\[\]]*)?)\]\]")
REPORTS_DIR = "inbox/reports"
EVAL_FILE = "eval/legacy-queries.yaml"


@dataclass
class Item:
    """One legacy file and what became of it."""

    path: str  # relative to the legacy folder
    status: str  # imported, held or skipped
    reason: str = ""
    target: str = ""  # its path in this vault
    note: Note | None = None
    draft: Proposal | None = None
    name: str = ""  # new file name without .md
    when: datetime | None = None
    vector: np.ndarray | None = None


@dataclass
class ImportReport:
    items: list[Item] = field(default_factory=list)
    links: dict[str, int] = field(default_factory=dict)
    still_broken: list[tuple[str, str]] = field(default_factory=list)  # (new path, target)
    to_held: list[tuple[str, str]] = field(default_factory=list)  # links to a held note
    queries: str | None = None
    commit: str | None = None
    commit_error: str | None = None

    def count(self, status: str) -> int:
        return sum(1 for i in self.items if i.status == status)


def _order(rel: str) -> tuple[int, str]:
    """Accepted notes first, so when two notes match it's the inbox copy that is held."""
    return (0 if "/1_atomic/" in f"/{rel}" else 1, rel)


def _keys(note: Note) -> set[str]:
    """Every way a legacy link might name this note."""
    keys = {note.stem.lower(), _ID_PREFIX.sub("", note.stem).lower(), note.title.lower()}
    keys |= {slugify(note.title), *(a.lower() for a in note.aliases)}
    if note.id:
        keys |= {note.id.lower()}
        if m := _ID.match(note.id):
            keys.add(m.group(1))
    return {k for k in keys if k}


_STOP = {"a", "an", "and", "as", "by", "for", "from", "in", "of", "on", "or", "the", "to", "via"}
_STOP |= {"with", "x"}
FUZZY_AT = 0.75  # share of a drifted link's words that must appear in the note's name


def _words(slug: str) -> set[str]:
    return {w for w in slug.split("-") if w and w not in _STOP}


def _resolver(names: dict[str, str], ids: set[str] = frozenset()) -> Callable[[str], str | None]:
    """Map a link target to a new note name, repairing drift where it is unambiguous.

    Exact names, ids and titles first. Failing that, a drifted slug
    (`claude-code-skill-file-format` for `...-file-format-and-schema`,
    `3-tier-model-routing-task-complexity` for `...-routing-by-task-...`) matches
    the one note whose name holds most of its words, and only if no other note
    does as well. A drifted slug that keeps a note's id (`<id>-batch-model-calls`)
    goes to that note when the id belongs to it alone (`ids`).
    """
    slugs = {k: _words(k) for k in names if re.fullmatch(r"[a-z0-9-]+", k)}

    def resolve(target: str) -> str | None:
        t = target.strip().lower().removesuffix(".md").rsplit("/", 1)[-1]
        for candidate in (t, _ID_PREFIX.sub("", t), slugify(t)):
            if candidate in names:
                return names[candidate]
        if (m := _ID.match(t)) and m.group(1) in ids:
            return names[m.group(1)]
        words = _words(slugify(_ID_PREFIX.sub("", t)))
        if len(words) < 2:
            return None
        scored: dict[str, tuple[float, float]] = {}
        for key, key_words in slugs.items():
            shared = len(words & key_words)
            score = (shared / len(words), shared / max(len(key_words), 1))
            name = names[key]
            scored[name] = max(scored.get(name, (0.0, 0.0)), score)
        ranked = sorted(scored.items(), key=lambda kv: kv[1], reverse=True)
        if not ranked or ranked[0][1][0] < FUZZY_AT:
            return None
        if len(ranked) > 1 and ranked[1][1] == ranked[0][1]:
            return None  # two notes fit equally well: leave the link for a person
        return ranked[0][0]

    return resolve


def _source_note(note: Note) -> str:
    text = note.sections.get("Source Note", "").strip()
    first = text.split("\n", 1)[0].strip() if text else ""
    return re.sub(r"^derived from:\s*", "", first, flags=re.I).strip()


def _details(note: Note) -> str:
    parts = []
    for heading, text in note.sections.items():
        if heading.lower() in _META_SECTIONS or not text.strip():
            continue
        parts.append(text if heading.lower() == "core idea" else f"### {heading}\n\n{text}")
    return "\n\n".join(parts)


def _related(note: Note) -> list[tuple[str, str]]:
    """(target, why) for each link listed under the Links or Related section."""
    out = []
    for heading in ("Links", "Related"):
        for line in note.sections.get(heading, "").split("\n"):
            matches = list(_WIKILINK.finditer(line))
            for m in matches:
                why = line[m.end() :] if m is matches[-1] else ""
                out.append((m.group(1).strip(), why.strip().lstrip("-–—:").strip()))
    return [(t, w) for t, w in out if t]


def plan(legacy: Path) -> list[Item]:
    """Classify every file under the legacy folder (hidden folders aside). Reads only."""
    items = []
    for path in sorted(legacy.rglob("*")):
        rel = path.relative_to(legacy).as_posix()
        if not path.is_file() or any(p.startswith(".") for p in rel.split("/")):
            continue
        if path.suffix.lower() != ".md":
            items.append(Item(rel, "skipped", "not a note (not Markdown)"))
            continue
        note = parse_note(rel, path.read_bytes())
        raw_type = note.type
        if not note.body.strip() and note.frontmatter is None:
            items.append(Item(rel, "skipped", "empty file"))
        elif "/templates/" in f"/{rel}":
            items.append(Item(rel, "skipped", "template"))
        elif raw_type is None:
            items.append(Item(rel, "skipped", "no note type: a system doc, map or other file"))
        elif raw_type not in TYPE_MAP:
            items.append(Item(rel, "skipped", f"unknown note type {raw_type!r}"))
        else:
            items.append(Item(rel, "imported", note=note))
    return sorted(items, key=lambda i: _order(i.path))


def _free(vault_dir: Path, folder: str, name: str, used: set[str]) -> str:
    """folder/<name>.md with a name no note or held note has, on disk or in this run.

    Held and accepted notes share one set of names, so a held note can later be
    released into notes/ under the name its incoming links already use.
    """
    candidate, n = name, 1
    while candidate in used or any(
        (vault_dir / d / f"{candidate}.md").exists() for d in (NOTES_DIR, HELD_DIR)
    ):
        n += 1
        candidate = f"{name}-{n}"
    used.add(candidate)
    return f"{folder}/{candidate}.md"


def _when(note: Note, taken: set[str], conn: sqlite3.Connection, fallback: datetime) -> datetime:
    m = _ID.match(note.id or "")
    when = datetime.strptime(m.group(1), "%Y%m%d%H%M%S") if m else fallback
    while True:
        note_id = when.strftime("%Y%m%d%H%M%S")
        if (
            note_id not in taken
            and not conn.execute("SELECT 1 FROM notes WHERE id = ?", (note_id,)).fetchone()
        ):
            taken.add(note_id)
            return when
        when += timedelta(seconds=1)


def _links(folder: Path, sources: set[str]) -> list[tuple[str, str, bool]]:
    """(source path, target, resolved) for every link out of `sources`, from a throwaway index."""
    with tempfile.TemporaryDirectory() as tmp:
        conn = open_db(Path(tmp) / "i.sqlite")
        try:
            build_index(conn, folder, ("inbox/**", "templates/**"))
            rows = conn.execute("SELECT src_path, target, status FROM links").fetchall()
        finally:
            conn.close()
    return [(r[0], r[1], r[2] == "resolved") for r in rows if r[0] in sources]


def import_legacy(
    legacy: Path,
    vault_dir: Path,
    conn: sqlite3.Connection,
    embedder: Embedder,
    queries: Path | None = None,
    now: datetime | None = None,
) -> ImportReport:
    """Import every legacy note into vault_dir. Nothing is written if a note can't convert."""
    now = now or datetime.now().astimezone()
    legacy, vault_dir = legacy.resolve(), vault_dir.resolve()
    if legacy == vault_dir or vault_dir in legacy.parents or legacy in vault_dir.parents:
        raise ValueError("the legacy folder and the vault must not contain each other")
    report = ImportReport(items=plan(legacy))
    notes = [i for i in report.items if i.status == "imported"]

    # 1. Convert each note's fields (links are rewritten in step 3, once names are final).
    taken: set[str] = set()
    for item in notes:
        n = item.note
        sources = list(n.sources)
        if (extra := _source_note(n)) and extra not in sources:
            sources.append(extra)
        item.draft = Proposal(
            title=n.title.strip(),
            type=TYPE_MAP[n.type],
            summary=" ".join(n.summary.split()) or n.title,
            body=_details(n),
            tags=[t for t in n.tags if t.strip()],
            aliases=list(n.aliases),
            sources=[*sources, f"legacy:{item.path}"],
        )
        item.name = slugify(_ID_PREFIX.sub("", n.stem))
        item.when = _when(n, taken, conn, now)
    vectors = embedder.embed_documents(
        [note_text(i.draft.title, f"{i.draft.summary}\n\n{i.draft.body}") for i in notes]
    )

    # 2. Hold duplicates of an existing note or of a note imported before it.
    names: dict[str, str] = {}
    twin_of: dict[str, str] = {}  # held name -> the note it duplicates
    kept: list[Item] = []
    used: set[str] = set()
    for item, vector in zip(notes, vectors, strict=True):
        item.vector = vector
        match = next(iter(similar_to_vector(conn, embedder, vector, k=1)), None)
        twin = (Path(match.path).stem, match.score) if match else None
        for other in kept:
            score = float(np.dot(vector, other.vector))
            if score >= DUPLICATE_AT and (twin is None or score > twin[1]):
                twin = (other.name, score)
        if twin and twin[1] >= DUPLICATE_AT:
            item.status = "held"
            item.reason = f"{twin[1]:.2f} similar to {twin[0]}"
            item.target = _free(vault_dir, HELD_DIR, item.name, used)
            item.name = Path(item.target).stem
            twin_of[item.name] = twin[0]
        else:
            item.target = _free(vault_dir, NOTES_DIR, item.name, used)
            item.name = Path(item.target).stem
            kept.append(item)
        # Links keep pointing at a held note rather than its twin: a false hold
        # would otherwise move the link's meaning to a merely similar note.
        for key in _keys(item.note):
            names.setdefault(key, item.name)
    stamps = Counter(m.group(1) for i in notes if (m := _ID.match(i.note.id or "")))
    resolve = _resolver(names, {k for k, n in stamps.items() if n == 1})

    # 3. Rewrite links (in Details and the Related list) to the new names.
    legacy_links = _links(legacy, {i.path for i in notes})
    was_broken = {(src, target.strip()) for src, target, ok in legacy_links if not ok}
    counts = {"repaired": 0, "unresolved": 0}

    def fix(src: str, target: str) -> str:
        new = resolve(target)
        if new is None:
            counts["unresolved"] += 1
            return target
        if (src, target.strip()) in was_broken:
            counts["repaired"] += 1
        return new

    rendered = []
    for item in notes:
        body = _WIKILINK.sub(
            lambda m, src=item.path: f"[[{fix(src, m.group(1))}{m.group(2)}]]", item.draft.body
        )
        related: dict[str, str] = {}
        for target, why in _related(item.note):
            new = fix(item.path, target)
            if new != item.name:
                related.setdefault(new, why)
        item.draft.body, item.draft.links = body, related
        try:
            _, text = render(
                item.draft,
                item.when,
                created=item.note.created,
                updated=item.note.updated,
                name=Path(item.target).stem,
            )
        except InvalidProposal as exc:
            raise InvalidProposal(f"{item.path}: {exc}") from exc
        rendered.append((item, text))

    # 4. Write everything, then account for the links and commit once.
    touched = []
    for item, text in rendered:
        vault.write_new(vault_dir, item.target, text)
        touched.append(item.target)
    if queries is not None:
        report.queries = _write_queries(vault_dir, queries, resolve, twin_of)
        touched.append(report.queries)

    imported = {i.target for i in notes if i.status == "imported"}
    after = _links(vault_dir, imported)
    unresolved = sorted((src, target) for src, target, ok in after if not ok)
    report.to_held = [(s, t) for s, t in unresolved if t.strip() in twin_of]
    report.still_broken = [(s, t) for s, t in unresolved if t.strip() not in twin_of]
    report.links = {
        "legacy_links": len(legacy_links),
        "legacy_broken": len(was_broken),
        **counts,
        "links_after": len(after),
        "broken_after": len(report.still_broken),
        "to_held": len(report.to_held),
    }

    run = now.strftime("%Y-%m-%d-%H%M%S")
    report_path = vault.unique_path(vault_dir, f"{REPORTS_DIR}/import-{run}.md")
    vault.write_new(vault_dir, report_path, format_report(report, legacy))
    touched.append(report_path)
    message = f"zk: import {report.count('imported')} legacy notes"
    if report.count("held"):
        message += f" ({report.count('held')} held as duplicates)"
    try:
        report.commit = vault.git_commit(vault_dir, touched, message)
    except (subprocess.CalledProcessError, OSError) as exc:  # written; say so, don't fail
        report.commit_error = f"written but not committed: {getattr(exc, 'stderr', '') or exc}"
    return report


def _write_queries(vault_dir: Path, queries: Path, resolve, twin_of: dict[str, str]) -> str:
    """Copy a legacy query set into the vault with its labels pointing at the new names.

    A label for a held note lists the note it duplicates as well, so the question
    still has an answer in the index; the held name counts again once released.
    """
    data = yaml.safe_load(queries.read_text(encoding="utf-8")) or {}
    for q in data.get("queries", []):
        labels = []
        for label in q.get("relevant", []):
            new = resolve(str(label)) or str(label)
            labels += [new, twin_of[new]] if new in twin_of else [new]
        q["relevant"] = list(dict.fromkeys(labels))  # two old labels can meet at one note
    rel = vault.unique_path(vault_dir, EVAL_FILE)
    vault.write_new(
        vault_dir,
        rel,
        "# Hand-written test questions, carried over from the legacy vault by zk import.\n"
        + yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
    )
    return rel


def dry_run(
    legacy: Path, vault_dir: Path, embedder: Embedder, queries: Path | None = None
) -> ImportReport:
    """The same import into a throwaway copy of the vault: nothing real is written."""
    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / "vault"
        shutil.copytree(vault_dir, copy, ignore=shutil.ignore_patterns(".git"))
        conn = open_db(Path(tmp) / "i.sqlite")
        try:
            build_index(conn, copy)
            update_embeddings(conn, embedder)
            return import_legacy(legacy, copy, conn, embedder, queries)
        finally:
            conn.close()


def summary(report: ImportReport) -> str:
    links = report.links
    return (
        f"{len(report.items)} legacy files: {report.count('imported')} imported, "
        f"{report.count('held')} held as likely duplicates, {report.count('skipped')} skipped.\n"
        f"Links: {links.get('legacy_broken', 0)} of the legacy notes' "
        f"{links.get('legacy_links', 0)} links were broken. {links.get('repaired', 0)} were "
        f"repaired; {links.get('broken_after', 0)} of the imported notes' "
        f"{links.get('links_after', 0)} links are still broken"
        + (f", and {links['to_held']} point at a held note." if links.get("to_held") else ".")
    )


def format_report(report: ImportReport, legacy: Path | None = None) -> str:
    lines = ["# zk import report", ""]
    if legacy:
        lines += [f"From `{legacy}`.", ""]
    lines += summary(report).split("\n")
    for status, title in (
        ("imported", "Imported"),
        ("held", "Held as likely duplicates (in inbox/held/; apply_held releases one)"),
        ("skipped", "Skipped"),
    ):
        rows = [i for i in report.items if i.status == status]
        if rows:
            lines += ["", f"## {title} ({len(rows)})", ""]
            lines += [
                f"- `{i.path}`"
                + (f" -> `{i.target}`" if i.target else "")
                + (f": {i.reason}" if i.reason else "")
                for i in rows
            ]
    if report.to_held:
        lines += ["", f"## Waiting on a held note ({len(report.to_held)})", ""]
        lines += [f"- `{src}` -> [[{target}]]" for src, target in report.to_held]
    if report.still_broken:
        lines += ["", f"## Still broken ({len(report.still_broken)})", ""]
        lines += [f"- `{src}` -> [[{target}]]" for src, target in report.still_broken]
    return "\n".join(lines) + "\n"
