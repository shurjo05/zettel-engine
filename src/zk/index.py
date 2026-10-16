"""The catalog: a SQLite index of every note, its links and its issues.

The vault's Markdown files are the source of truth. Everything in this
database is derived from them, so it can be deleted and rebuilt at any time.
This module only ever reads the vault; it writes nothing but the database.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import sqlite3
from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

from zk.notes import Note, parse_note, validate

SCHEMA_VERSION = "2"

# Raw captures and templates live in the vault but are not atomic notes.
DEFAULT_EXCLUDES = ("inbox/**", "templates/**")

_SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);

CREATE TABLE notes (
    path         TEXT PRIMARY KEY,   -- relative to the vault, forward slashes
    id           TEXT,               -- frontmatter id; stable across renames
    title        TEXT NOT NULL,
    type         TEXT,
    summary      TEXT NOT NULL,
    tags         TEXT NOT NULL,      -- JSON list
    aliases      TEXT NOT NULL,      -- JSON list
    sources      TEXT NOT NULL,      -- JSON list
    created      TEXT,
    updated      TEXT,
    content_hash TEXT NOT NULL,      -- SHA-256 of the file bytes
    frontmatter  TEXT,               -- JSON object, NULL if the note has none
    body         TEXT NOT NULL
);
CREATE INDEX notes_id ON notes(id);

CREATE TABLE links (
    src_path TEXT NOT NULL REFERENCES notes(path) ON DELETE CASCADE,
    target   TEXT NOT NULL,          -- as written inside [[...]]
    heading  TEXT,
    section  TEXT,                   -- the ## section the link sits in
    dst_path TEXT,                   -- resolved note, NULL if broken or ambiguous
    status   TEXT NOT NULL DEFAULT 'unresolved'  -- resolved | broken | ambiguous
);
CREATE INDEX links_src ON links(src_path);
CREATE INDEX links_dst ON links(dst_path);

CREATE TABLE issues (
    path    TEXT NOT NULL REFERENCES notes(path) ON DELETE CASCADE,
    level   TEXT NOT NULL,           -- error | warning
    code    TEXT NOT NULL,
    message TEXT NOT NULL
);
CREATE INDEX issues_path ON issues(path);

-- Full-text index for keyword search (BM25). Porter stemming makes
-- "deduplicate" match "deduplication". path is stored, not searched.
CREATE VIRTUAL TABLE notes_fts USING fts5(
    path UNINDEXED, title, aliases, tags, summary, body,
    tokenize = 'porter unicode61 remove_diacritics 2'
);
"""


@dataclass
class BuildStats:
    added: int = 0
    updated: int = 0
    removed: int = 0
    unchanged: int = 0

    @property
    def total(self) -> int:
        return self.added + self.updated + self.unchanged


def default_db_path(vault: Path) -> Path:
    """Where the index lives: outside the vault, one file per vault.

    ZK_DATA_DIR overrides; otherwise %LOCALAPPDATA%\\zk on Windows and
    ~/.local/share/zk elsewhere.
    """
    base = os.environ.get("ZK_DATA_DIR")
    if base:
        root = Path(base)
    elif os.name == "nt" and os.environ.get("LOCALAPPDATA"):
        root = Path(os.environ["LOCALAPPDATA"]) / "zk"
    else:
        xdg = os.environ.get("XDG_DATA_HOME")
        root = (Path(xdg) if xdg else Path.home() / ".local" / "share") / "zk"
    vault = vault.resolve()
    digest = hashlib.sha1(str(vault).encode()).hexdigest()[:8]
    return root / f"{vault.name or 'vault'}-{digest}.sqlite"


def open_db(db_path: Path | str) -> sqlite3.Connection:
    """Open (and if needed create or reset) the index database."""
    if str(db_path) != ":memory:":
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    row = None
    try:
        row = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
    except sqlite3.OperationalError:
        pass
    if row is None or row["value"] != SCHEMA_VERSION:
        _reset(conn)
    return conn


def _reset(conn: sqlite3.Connection) -> None:
    """Drop everything and recreate the schema. Safe because the index is derived."""
    with conn:
        conn.execute("PRAGMA foreign_keys = OFF")
        # Virtual tables first: dropping one also drops its shadow tables.
        for (name,) in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND sql LIKE 'CREATE VIRTUAL%'"
        ).fetchall():
            conn.execute(f'DROP TABLE IF EXISTS "{name}"')
        for (name,) in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall():
            conn.execute(f'DROP TABLE IF EXISTS "{name}"')
        conn.executescript(_SCHEMA)
        conn.execute("INSERT INTO meta VALUES ('schema_version', ?)", (SCHEMA_VERSION,))
        conn.execute("PRAGMA foreign_keys = ON")


def iter_note_files(vault: Path, excludes: Iterable[str] = DEFAULT_EXCLUDES) -> Iterator[Path]:
    """Every Markdown file in the vault, skipping hidden folders and excluded globs."""
    patterns = list(excludes)
    for path in sorted(vault.rglob("*.md")):
        rel = path.relative_to(vault)
        if any(part.startswith(".") for part in rel.parts):
            continue  # .obsidian, .git, .trash
        rel_posix = rel.as_posix()
        if any(fnmatch.fnmatch(rel_posix, p) for p in patterns):
            continue
        if path.is_file():
            yield path


def build_index(
    conn: sqlite3.Connection, vault: Path, excludes: Iterable[str] = DEFAULT_EXCLUDES
) -> BuildStats:
    """Bring the index in line with the vault, re-parsing only changed files."""
    vault = vault.resolve()
    stats = BuildStats()

    stored_root = conn.execute("SELECT value FROM meta WHERE key = 'vault'").fetchone()
    if stored_root is not None and stored_root["value"] != str(vault):
        _reset(conn)  # a different vault: start over rather than mix them

    known = {
        r["path"]: r["content_hash"] for r in conn.execute("SELECT path, content_hash FROM notes")
    }
    seen: set[str] = set()

    with conn:
        conn.execute("INSERT OR REPLACE INTO meta VALUES ('vault', ?)", (str(vault),))
        for file in iter_note_files(vault, excludes):
            rel = file.relative_to(vault).as_posix()
            seen.add(rel)
            data = file.read_bytes()
            note = parse_note(rel, data)
            if known.get(rel) == note.content_hash:
                stats.unchanged += 1
                continue
            if rel in known:
                stats.updated += 1
            else:
                stats.added += 1
            _store(conn, note)

        for rel in known.keys() - seen:
            _delete(conn, rel)
            stats.removed += 1

        _resolve_links(conn)
    return stats


def _delete(conn: sqlite3.Connection, path: str) -> None:
    # notes_fts shares rowids with notes. Virtual tables don't take part in
    # foreign-key cascades, so its row is removed by hand, by rowid (fast).
    row = conn.execute("SELECT rowid FROM notes WHERE path = ?", (path,)).fetchone()
    if row is not None:
        conn.execute("DELETE FROM notes_fts WHERE rowid = ?", (row[0],))
    conn.execute("DELETE FROM notes WHERE path = ?", (path,))  # cascades to links, issues


def _store(conn: sqlite3.Connection, note: Note) -> None:
    _delete(conn, note.path)
    cursor = conn.execute(
        "INSERT INTO notes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            note.path,
            note.id,
            note.title,
            note.type,
            note.summary,
            json.dumps(note.tags),
            json.dumps(note.aliases),
            json.dumps(note.sources),
            note.created,
            note.updated,
            note.content_hash,
            None if note.frontmatter is None else json.dumps(note.frontmatter),
            note.body,
        ),
    )
    conn.execute(
        "INSERT INTO notes_fts (rowid, path, title, aliases, tags, summary, body) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            cursor.lastrowid,
            note.path,
            note.title,
            " ".join(note.aliases),
            " ".join(note.tags),
            note.summary,
            note.body,
        ),
    )
    conn.executemany(
        "INSERT INTO links (src_path, target, heading, section) VALUES (?, ?, ?, ?)",
        [(note.path, link.target, link.heading, link.section) for link in note.links],
    )
    conn.executemany(
        "INSERT INTO issues VALUES (?, ?, ?, ?)",
        [(note.path, i.level, i.code, i.message) for i in validate(note)],
    )


def _resolve_links(conn: sqlite3.Connection) -> None:
    """Point every link at a note, the way Obsidian does, then fall back to ids.

    Obsidian resolves [[name]] by file name, case-insensitively, anywhere in
    the vault; [[folder/name]] by path. We also accept a note's frontmatter id,
    which is what will keep links stable when a file is renamed.
    """
    by_stem: dict[str, list[str]] = defaultdict(list)
    by_path: dict[str, str] = {}
    by_id: dict[str, list[str]] = defaultdict(list)
    for row in conn.execute("SELECT path, id FROM notes"):
        path = row["path"]
        no_ext = path[:-3].lower()
        by_path[no_ext] = path
        by_stem[no_ext.rsplit("/", 1)[-1]].append(path)
        if row["id"]:
            by_id[row["id"].lower()].append(path)

    updates = []
    for link in conn.execute("SELECT rowid, target FROM links"):
        key = link["target"].strip().replace("\\", "/").lower()
        if key.endswith(".md"):
            key = key[:-3]
        if "/" in key:
            exact = by_path.get(key.lstrip("/"))
            matches = [exact] if exact else [p for k, p in by_path.items() if k.endswith("/" + key)]
        else:
            matches = by_stem.get(key) or by_id.get(key) or []
        if len(matches) == 1:
            updates.append((matches[0], "resolved", link["rowid"]))
        else:
            updates.append((None, "ambiguous" if matches else "broken", link["rowid"]))
    conn.executemany("UPDATE links SET dst_path = ?, status = ? WHERE rowid = ?", updates)


@dataclass
class Report:
    """The vault health check, computed entirely from the index."""

    notes: int = 0
    by_type: dict[str, int] = field(default_factory=dict)
    links: int = 0
    issues: list[dict] = field(default_factory=list)
    duplicate_ids: dict[str, list[str]] = field(default_factory=dict)
    broken_links: list[dict] = field(default_factory=list)
    ambiguous_links: list[dict] = field(default_factory=list)
    orphans: list[str] = field(default_factory=list)
    no_outgoing: list[str] = field(default_factory=list)

    @property
    def errors(self) -> int:
        return sum(1 for i in self.issues if i["level"] == "error") + len(self.duplicate_ids)

    @property
    def warnings(self) -> int:
        return sum(1 for i in self.issues if i["level"] == "warning") + len(self.broken_links)

    def to_dict(self) -> dict:
        return {"errors": self.errors, "warnings": self.warnings, **self.__dict__}


def health_report(conn: sqlite3.Connection) -> Report:
    report = Report()
    report.notes = conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
    report.by_type = {
        (r["type"] or "(none)"): r["n"]
        for r in conn.execute("SELECT type, COUNT(*) AS n FROM notes GROUP BY type ORDER BY n DESC")
    }
    report.links = conn.execute("SELECT COUNT(*) FROM links").fetchone()[0]
    report.issues = [
        dict(r)
        for r in conn.execute("SELECT path, level, code, message FROM issues ORDER BY path, level")
    ]
    report.duplicate_ids = {
        r["id"]: r["paths"].split("\n")
        for r in conn.execute(
            "SELECT id, group_concat(path, char(10)) AS paths FROM notes "
            "WHERE id IS NOT NULL GROUP BY id HAVING COUNT(*) > 1 ORDER BY id"
        )
    }
    for status, bucket in (("broken", report.broken_links), ("ambiguous", report.ambiguous_links)):
        bucket.extend(
            {"path": r["src_path"], "target": r["target"]}
            for r in conn.execute(
                "SELECT src_path, target FROM links WHERE status = ? ORDER BY src_path, target",
                (status,),
            )
        )
    report.orphans = [
        r["path"]
        for r in conn.execute(
            "SELECT path FROM notes n WHERE NOT EXISTS ("
            "  SELECT 1 FROM links l WHERE l.dst_path = n.path AND l.src_path != n.path"
            ") ORDER BY path"
        )
    ]
    report.no_outgoing = [
        r["path"]
        for r in conn.execute(
            "SELECT path FROM notes n WHERE NOT EXISTS ("
            "  SELECT 1 FROM links l WHERE l.src_path = n.path AND l.status = 'resolved'"
            "  AND l.dst_path != n.path"
            ") ORDER BY path"
        )
    ]
    return report
