"""Resolve held likely-duplicates: a model judges each pair, this module does the writes.

A note held in inbox/held/ either says the same thing as a note in notes/ or is
only related to it. The agent (ideally a strong reasoning model) reads both and
decides. "Related" releases the held note with `ingest.apply_held`. "Same idea"
merges it here: the existing note is rewritten from the agent's merged draft,
keeping its id, file name and created date, and gains the held note's sources.
Links and test questions that named the held note are pointed at the merged
note, the held file is moved to inbox/merged/, and it is all one git commit.
Nothing is deleted, and the old text stays in git history.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from zk import vault
from zk.embed import Embedder, note_text
from zk.ingest import _check_name, _commit
from zk.notes import content_hash, parse_note
from zk.proposals import HELD_DIR, NOTES_DIR, InvalidProposal, Proposal, render
from zk.search import similar_to_vector

MERGED_DIR = "inbox/merged"
EVAL_DIR = "eval"


def _held(vault_dir: Path, name: str):
    name = _check_name(name)
    path = vault_dir / HELD_DIR / name
    if not path.is_file():
        raise FileNotFoundError(f"no held note named {name!r}")
    return name, path.read_bytes()


def held_pair(
    vault_dir: Path, conn: sqlite3.Connection, embedder: Embedder, name: str, k: int = 3
) -> dict:
    """A held note's full text, the full text of its closest note, and the next closest."""
    name, data = _held(vault_dir, name)
    note = parse_note(f"{HELD_DIR}/{name}", data)
    vector = embedder.embed_documents([note_text(note.title, note.body)])[0]
    hits = similar_to_vector(conn, embedder, vector, k)
    if not hits:
        raise LookupError("the vault has no embedded notes to compare with")
    closest = hits[0]
    return {
        "name": name,
        "held": data.decode("utf-8", errors="replace"),
        "closest": {
            "path": closest.path,
            "similarity": round(closest.score, 3),
            "text": (vault_dir / closest.path).read_text(encoding="utf-8", errors="replace"),
        },
        "others": [
            {
                "path": h.path,
                "title": h.title,
                "summary": h.summary,
                "similarity": round(h.score, 3),
            }
            for h in hits[1:]
        ],
    }


def _link_pattern(stem: str) -> re.Pattern:
    # [[stem]], [[stem.md]], [[folder/stem]], [[stem#heading]], [[stem|label]]
    return re.compile(
        r"\[\[(?:[^\[\]|#]*/)?" + re.escape(stem) + r"(?:\.md)?(?=[\]|#])", re.IGNORECASE
    )


def _label_pattern(stem: str) -> re.Pattern:
    return re.compile(r"(?<![\w-])" + re.escape(stem) + r"(?![\w-])")


def _retarget_links(text: str, old: str, new: str) -> str:
    return _link_pattern(old).sub(f"[[{new}", text)


def _retarget_labels(text: str, old: str, new: str) -> str:
    """Rename an answer label in a YAML question file, leaving the questions' own wording alone."""
    pattern = _label_pattern(old)
    return "".join(
        line if re.match(r"\s*-?\s*query\s*:", line) else pattern.sub(new, line)
        for line in text.splitlines(keepends=True)
    )


def merge_held(
    vault_dir: Path,
    conn: sqlite3.Connection,
    name: str,
    into: str,
    draft: Proposal,
    reason: str,
    refresh: Callable[[], None] | None = None,
    now: datetime | None = None,
) -> dict:
    """Merge held note `name` into the note at `into` (a path in notes/), as one commit.

    `draft` is the merged note, written by the agent from both notes. Nothing is
    written unless it renders cleanly and every file still has the content read.
    """
    now = now or datetime.now().astimezone()
    if not reason.strip():
        raise InvalidProposal("say in a sentence why the two notes are the same idea")
    name, held_data = _held(vault_dir, name)
    held = parse_note(f"{HELD_DIR}/{name}", held_data)
    into = into.strip().replace("\\", "/")
    if not into.endswith(".md"):
        into += ".md"
    if "/" not in into:
        into = f"{NOTES_DIR}/{into}"
    target_path = vault_dir / into
    if not into.startswith(f"{NOTES_DIR}/") or not target_path.is_file():
        raise FileNotFoundError(f"no note {into!r} to merge into")
    target_data = target_path.read_bytes()
    target = parse_note(into, target_data)
    old, new = held.stem, target.stem

    def union(*lists):
        return list(dict.fromkeys(x for xs in lists for x in xs if x.strip()))

    draft.sources = union(target.sources, held.sources, draft.sources)
    titles = (draft.title.lower(), target.title.lower())
    draft.aliases = union(
        target.aliases, draft.aliases, [held.title] if held.title.lower() not in titles else []
    )
    draft.tags = union(target.tags, held.tags, draft.tags)
    # Neither note's links are lost: any the draft dropped go back in its Related list.
    links = dict(draft.links)
    mentioned = draft.body.lower()
    for link in [*target.links, *held.links]:
        key = link.target.strip()
        if f"[[{key.lower()}" not in mentioned and key.lower() not in map(str.lower, links):
            links[key] = ""
    draft.links = {
        k: v for k, v in links.items() if k.strip().lower() not in (old.lower(), new.lower())
    }
    try:
        _, text = render(
            draft,
            now,
            created=target.created,
            updated=now.date().isoformat(),
            name=new,
            note_id=target.id,
        )
    except InvalidProposal as exc:
        raise InvalidProposal(f"the merged note: {exc}") from exc

    # Every write is planned and its file read before any is made.
    writes = [(into, _retarget_links(text, old, new), content_hash(target_data))]
    sources = {
        r["src_path"]
        for r in conn.execute("SELECT src_path, target FROM links")
        if _link_pattern(old).search(f"[[{r['target']}]]")
    }
    for src in sorted(sources - {into}):
        data = (vault_dir / src).read_bytes()
        changed = _retarget_links(data.decode("utf-8"), old, new)
        if changed != data.decode("utf-8"):
            writes.append((src, changed, content_hash(data)))
    eval_dir = vault_dir / EVAL_DIR
    for path in sorted(eval_dir.rglob("*.y*ml")) if eval_dir.is_dir() else []:
        data = path.read_bytes()
        changed = _retarget_labels(data.decode("utf-8"), old, new)
        if changed != data.decode("utf-8"):
            writes.append((path.relative_to(vault_dir).as_posix(), changed, content_hash(data)))

    changed_since = [r for r, _, h in writes if content_hash((vault_dir / r).read_bytes()) != h]
    if changed_since:  # checked for all first, so a merge is never left half done
        raise vault.VaultWriteError(f"changed while merging, nothing written: {changed_since}")
    for rel, body, expected in writes:
        vault.replace(vault_dir, rel, body, expected)
    merged_to = vault.unique_path(vault_dir, f"{MERGED_DIR}/{name}")
    vault.move_new(vault_dir, f"{HELD_DIR}/{name}", merged_to)
    if refresh is not None:
        refresh()

    touched = [rel for rel, _, _ in writes] + [f"{HELD_DIR}/{name}", merged_to]
    result = {
        "merged_into": into,
        "held_moved_to": merged_to,
        "links_retargeted": [rel for rel, _, _ in writes[1:] if not rel.startswith(EVAL_DIR)],
        "question_files_updated": [rel for rel, _, _ in writes if rel.startswith(f"{EVAL_DIR}/")],
    }
    message = f"zk: merge held {old} into {new}\n\n{' '.join(reason.split())}"
    result.update(_commit(vault_dir, touched, message))
    return result
