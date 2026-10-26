"""Turn conversations into notes automatically: check for duplicates, write, commit, check.

The agent reads a capture (or the conversation it is in) and decides which ideas
are worth a note; this module gives every note the vault's shape, writes the
ones that pass the duplicate check straight into notes/, holds the rest in
inbox/held/, saves each note's test questions, commits the whole run once, and
then runs the checks that warn when something looks wrong. All writes go
through zk.vault, which only creates or moves files.
"""

from __future__ import annotations

import sqlite3
import subprocess
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import yaml

from zk import checks, vault
from zk.embed import Embedder, note_text
from zk.notes import parse_note, split_frontmatter
from zk.proposals import (
    DUPLICATE_AT,
    HELD_DIR,
    NOTES_DIR,
    InvalidProposal,
    Proposal,
    render,
    verdict,
)
from zk.search import similar_notes, similar_to_vector

INBOX = "inbox"
PROCESSED = "inbox/processed"
REJECTED = "inbox/rejected"
WARNINGS = "inbox/warnings"


def _front(text: str) -> dict:
    raw, _ = split_frontmatter(text)
    try:
        data = yaml.safe_load(raw) if raw else {}
    except yaml.YAMLError:
        data = {}
    return data if isinstance(data, dict) else {}


def _check_name(name: str) -> str:
    if not name or "/" in name or "\\" in name or name.startswith("."):
        raise ValueError(f"{name!r} is not a file name in the inbox")
    return name if name.endswith(".md") else name + ".md"


def list_inbox(vault_dir: Path) -> list[dict]:
    """Captures waiting to be processed, oldest first."""
    folder = vault_dir / INBOX
    out = []
    for path in sorted(folder.glob("*.md")) if folder.is_dir() else []:
        text = path.read_text(encoding="utf-8", errors="replace")
        front = _front(text)
        out.append(
            {
                "name": path.name,
                "source": front.get("source"),
                "project": front.get("project"),
                "captured": str(front.get("captured") or ""),
                "chars": len(text),
            }
        )
    return out


def list_held(vault_dir: Path) -> list[dict]:
    """Notes held back as likely duplicates."""
    folder = vault_dir / HELD_DIR
    out = []
    for path in sorted(folder.glob("*.md")) if folder.is_dir() else []:
        front = _front(path.read_text(encoding="utf-8", errors="replace"))
        out.append({"name": path.name, "title": front.get("title"), "type": front.get("type")})
    return out


def read_capture(vault_dir: Path, name: str) -> str:
    path = vault_dir / INBOX / _check_name(name)
    if not path.is_file():
        raise FileNotFoundError(f"no capture named {name!r} in the inbox")
    return path.read_text(encoding="utf-8", errors="replace")


def check_duplicates(
    conn: sqlite3.Connection, embedder: Embedder, title: str, summary: str, body: str = "", k=5
) -> list[dict]:
    """Existing notes most similar to a would-be note, each with a verdict."""
    text = note_text(title, f"{summary}\n\n{body}".strip())
    return [
        {
            "path": h.path,
            "title": h.title,
            "summary": h.summary,
            "similarity": h.score,
            "verdict": verdict(h.score),
        }
        for h in similar_notes(conn, embedder, text, k=k)
    ]


MAX_QUESTIONS = 3


def _unique_id(conn: sqlite3.Connection, when: datetime, taken: set[str]) -> datetime:
    """`when`, moved on a second at a time until its id is used by no note."""
    while True:
        note_id = when.strftime("%Y%m%d%H%M%S")
        if (
            note_id not in taken
            and not conn.execute("SELECT 1 FROM notes WHERE id = ?", (note_id,)).fetchone()
        ):
            taken.add(note_id)
            return when
        when += timedelta(seconds=1)


def ingest(
    vault_dir: Path,
    conn: sqlite3.Connection,
    embedder: Embedder,
    drafts: list[Proposal],
    capture: str | None = None,
    refresh: Callable[[], None] | None = None,
    history: Path | None = None,
    now: datetime | None = None,
) -> dict:
    """One ingest run: validate every draft, then write, save questions, check and commit.

    Nothing is written unless every draft is valid. A draft at DUPLICATE_AT or
    more to an existing note, or to an earlier draft in the same run, is held in
    inbox/held/ instead of notes/. `refresh` brings the index up to date with the
    new notes before the checks run; `history` is where scores are remembered.
    """
    now = now or datetime.now().astimezone()
    if not drafts:
        raise InvalidProposal("no notes given; to retire a capture with nothing in it, discard it")
    if capture:
        capture = _check_name(capture)
        if not (vault_dir / INBOX / capture).is_file():
            raise FileNotFoundError(f"no capture named {capture!r} in the inbox")

    rendered, problems, taken = [], [], set()
    for i, draft in enumerate(drafts, 1):
        questions = [" ".join(q.split()) for q in draft.questions if q.strip()]
        if not 1 <= len(questions) <= MAX_QUESTIONS:
            problems.append(f"note {i}: give one to {MAX_QUESTIONS} test questions it answers")
        if capture:
            draft.sources = [*draft.sources, f"capture:{capture}"]
        try:
            name, text = render(draft, _unique_id(conn, now, taken))
        except InvalidProposal as exc:
            problems.append(f"note {i} ({draft.title or 'untitled'}): {exc}")
            continue
        rendered.append((draft, questions, name, text))
    if problems:
        raise InvalidProposal("; ".join(problems))

    bodies = [parse_note(f"{NOTES_DIR}/{n}", t.encode("utf-8")).body for _, _, n, t in rendered]
    vectors = embedder.embed_documents(
        [note_text(d.title, b) for (d, *_), b in zip(rendered, bodies, strict=True)]
    )
    run = now.strftime("%Y-%m-%d-%H%M%S")
    touched, notes, held, questions_out, kept = [], [], [], [], []
    for (_, questions, name, text), vector in zip(rendered, vectors, strict=True):
        match = next(iter(similar_to_vector(conn, embedder, vector, k=1)), None)
        twin = (match.path, match.score) if match else None
        for earlier_path, earlier_vector in kept:  # duplicates within this run
            score = float(np.dot(vector, earlier_vector))
            if score >= DUPLICATE_AT and (twin is None or score > twin[1]):
                twin = (earlier_path, score)
        if twin and twin[1] >= DUPLICATE_AT:
            rel = vault.unique_path(vault_dir, f"{HELD_DIR}/{name}")
            held.append({"path": rel, "duplicate_of": twin[0], "similarity": round(twin[1], 3)})
        else:
            rel = vault.unique_path(vault_dir, f"{NOTES_DIR}/{name}")
            notes.append(rel)
            kept.append((rel, vector))
        vault.write_new(vault_dir, rel, text)
        touched.append(rel)
        label = Path(rel).stem
        questions_out += [
            {
                "query": q,
                "relevant": [label],
                "kind": "holdout" if checks.is_holdout(q) else "generated",
            }
            for q in questions
        ]

    questions_file = vault.unique_path(vault_dir, f"{checks.QUESTIONS_DIR}/{run}.yaml")
    source = (
        f"capture:{capture}" if capture else (drafts[0].sources[0] if drafts[0].sources else "")
    )
    vault.write_new(
        vault_dir,
        questions_file,
        "# Test questions written with the notes of one ingest run.\n"
        + yaml.safe_dump(
            {"source": source, "queries": questions_out}, sort_keys=False, allow_unicode=True
        ),
    )
    touched.append(questions_file)
    if capture:
        done = vault.unique_path(vault_dir, f"{PROCESSED}/{capture}")
        vault.move_new(vault_dir, f"{INBOX}/{capture}", done)
        touched += [f"{INBOX}/{capture}", done]

    warnings = [
        f"{h['path']} was held, not added: it is {h['similarity']:.2f} similar to "
        f"{h['duplicate_of']}. Release it with apply_held or discard it."
        for h in held
    ]
    if refresh is not None:
        refresh()
        warnings += checks.check_new_notes(conn, notes)
        if history is not None:
            warnings += checks.check_scores(conn, vault_dir, history, embedder, now)
    result = {"notes": notes, "held": held, "questions": questions_file, "warnings": warnings}
    if warnings:
        result["warnings_file"] = vault.unique_path(vault_dir, f"{WARNINGS}/{run}.md")
        vault.write_new(
            vault_dir, result["warnings_file"], checks.render_warnings(run, warnings, now)
        )
        touched.append(result["warnings_file"])

    what = f"{len(notes)} note{'s' if len(notes) != 1 else ''}" + (
        f", {len(held)} held" if held else ""
    )
    result.update(_commit(vault_dir, touched, f"zk: ingest {capture or source or run}: {what}"))
    return result


def _commit(vault_dir: Path, paths: list[str], message: str) -> dict:
    try:
        return {"commit": vault.git_commit(vault_dir, paths, message)}
    except (subprocess.CalledProcessError, OSError) as exc:
        detail = getattr(exc, "stderr", "") or str(exc)
        return {"commit": None, "commit_error": f"written but not committed: {detail.strip()}"}


def apply_held(vault_dir: Path, name: str) -> dict:
    """Move a held note into notes/ after all (it was not a duplicate) and commit it."""
    name = _check_name(name)
    src = f"{HELD_DIR}/{name}"
    if not (vault_dir / src).is_file():
        raise FileNotFoundError(f"no held note named {name!r}")
    front = _front((vault_dir / src).read_text(encoding="utf-8"))
    dst = vault.unique_path(vault_dir, f"{NOTES_DIR}/{name}")
    vault.move_new(vault_dir, src, dst)
    result = {"note": dst}
    result.update(_commit(vault_dir, [src, dst], f"zk: add held note {front.get('title') or name}"))
    return result


def discard(vault_dir: Path, name: str, kind: str = "held") -> str:
    """Set a held note (to inbox/rejected/) or a capture (to inbox/processed/) aside.

    Nothing is deleted.
    """
    name = _check_name(name)
    src = f"{HELD_DIR}/{name}" if kind == "held" else f"{INBOX}/{name}"
    folder = REJECTED if kind == "held" else PROCESSED
    if not (vault_dir / src).is_file():
        raise FileNotFoundError(f"no {kind} named {name!r}")
    dst = vault.unique_path(vault_dir, f"{folder}/{name}")
    vault.move_new(vault_dir, src, dst)
    return dst
