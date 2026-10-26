"""Turn inbox captures into notes: list, propose, check for duplicates, apply.

The agent reads a capture and decides what is worth a note; this module makes
sure every proposal has the vault's shape, is checked against existing notes,
waits in inbox/proposals/ for a person to approve it, and lands with a git
commit. All writes go through zk.vault, which only creates or moves files.
"""

from __future__ import annotations

import sqlite3
import subprocess
from datetime import datetime
from pathlib import Path

import yaml

from zk import vault
from zk.embed import Embedder, note_text
from zk.notes import split_frontmatter
from zk.proposals import NOTES_DIR, PROPOSALS_DIR, Proposal, render, verdict
from zk.search import similar_notes

INBOX = "inbox"
PROCESSED = "inbox/processed"
REJECTED = "inbox/rejected"


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


def list_proposals(vault_dir: Path) -> list[dict]:
    folder = vault_dir / PROPOSALS_DIR
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


def propose(
    vault_dir: Path,
    conn: sqlite3.Connection,
    embedder: Embedder,
    proposal: Proposal,
    capture: str | None = None,
    now: datetime | None = None,
) -> dict:
    """Render, validate and duplicate-check a note, then save it in inbox/proposals/."""
    now = now or datetime.now().astimezone()
    if capture:
        capture = _check_name(capture)
        if not (vault_dir / INBOX / capture).is_file():
            raise FileNotFoundError(f"no capture named {capture!r} in the inbox")
        proposal.sources = [*proposal.sources, f"capture:{capture}"]
    name, text = render(proposal, now)
    duplicates = check_duplicates(conn, embedder, proposal.title, proposal.summary, proposal.body)
    rel = vault.unique_path(vault_dir, f"{PROPOSALS_DIR}/{name}")
    vault.write_new(vault_dir, rel, text)
    return {"proposal": Path(rel).name, "note": text, "duplicates": duplicates}


def apply(vault_dir: Path, proposal_name: str) -> dict:
    """Move an approved proposal into notes/, retire its capture, and commit both."""
    name = _check_name(proposal_name)
    src = f"{PROPOSALS_DIR}/{name}"
    if not (vault_dir / src).is_file():
        raise FileNotFoundError(f"no proposal named {name!r}")
    front = _front((vault_dir / src).read_text(encoding="utf-8"))
    dst = vault.unique_path(vault_dir, f"{NOTES_DIR}/{name}")
    vault.move_new(vault_dir, src, dst)
    touched = [src, dst]

    for source in front.get("sources") or []:
        capture = str(source).removeprefix("capture:")
        if str(source).startswith("capture:") and (vault_dir / INBOX / capture).is_file():
            done = vault.unique_path(vault_dir, f"{PROCESSED}/{capture}")
            vault.move_new(vault_dir, f"{INBOX}/{capture}", done)
            touched += [f"{INBOX}/{capture}", done]

    result = {"note": dst, "commit": None}
    try:
        result["commit"] = vault.git_commit(
            vault_dir, touched, f"zk: add note {front.get('title') or name}"
        )
    except (subprocess.CalledProcessError, OSError) as exc:
        detail = getattr(exc, "stderr", "") or str(exc)
        result["commit_error"] = f"note applied but not committed: {detail.strip()}"
    return result


def discard(vault_dir: Path, name: str, kind: str = "proposal") -> str:
    """Set a proposal (to inbox/rejected/) or a capture (to inbox/processed/) aside.

    Nothing is deleted.
    """
    name = _check_name(name)
    src = f"{PROPOSALS_DIR}/{name}" if kind == "proposal" else f"{INBOX}/{name}"
    folder = REJECTED if kind == "proposal" else PROCESSED
    if not (vault_dir / src).is_file():
        raise FileNotFoundError(f"no {kind} named {name!r}")
    dst = vault.unique_path(vault_dir, f"{folder}/{name}")
    vault.move_new(vault_dir, src, dst)
    return dst
