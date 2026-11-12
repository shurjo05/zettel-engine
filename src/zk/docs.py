"""A project's documentation in the vault, one note per section: `zk docs VAULT FOLDER`.

A project's own docs (specs, design notes, a course's assignment pages) are long,
and an agent working in the project reads whole files to find one part. This
copies each doc into the vault split at its headings, so `search_notes` returns
the one section that answers and `get_note` opens just that.

The text is the doc's own, not rewritten: no model runs, and exact details
(member names, test ids, rubric rules) survive. Sections are cut at headings,
largest first, until each fits MAX_CHARS, which keeps every section inside what
the embedding model reads (512 tokens); a section with no headings left is cut
between paragraphs, and a long table keeps its header row in every part. PDFs
are read with Poppler's `pdftotext` if it is installed, and cut by page.

Each doc gets a contents note listing its sections. The notes go in
refs/<project>/ as type `reference`: they are searchable like any note but are
not atomic notes, so the duplicate check, clustering, topic maps and orphan
counts leave them out. Every note records a hash of its text; running again
rewrites only sections that changed and that nobody edited, and moves sections
that are gone to refs/retired/. Nothing is deleted. One commit per run.

What was imported, and with which options, is remembered in the zk data folder,
so a session starting in the project can re-sync it in the background (hook.py)
and the Read hook can point at the sections (recall.on_read).
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from zk import capture, vault
from zk.index import data_dir
from zk.learn import SECRET_FILES, SKIP_DIRS, _key
from zk.search import project_slug

REFS_DIR = "refs"
RETIRED_DIR = "refs/retired"
INCLUDE = ("*.md", "*.markdown", "*.pdf")
# Files at a project's top that agents load by themselves every session.
AGENT_FILES = ("CLAUDE.md", "AGENTS.md", "CLAUDE.local.md")
MAX_CHARS = 1600  # a section: about 400 tokens, inside the embedder's 512
MIN_CHARS = 200  # an intro shorter than this joins the first subsection
MAX_DOC_BYTES = 2_000_000
CHARS_PER_TOKEN = 4

_FENCE = re.compile(r"^\s*(`{3,}|~{3,})")
_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")


# Splitting.


@dataclass
class Section:
    path: list[str]  # headings from the doc's title down
    text: str  # the section's own Markdown, headings included


@dataclass
class _Node:
    level: int
    title: str
    lines: list[str] = field(default_factory=list)  # own text, after the heading line
    heading: str = ""  # the heading line itself
    children: list[_Node] = field(default_factory=list)

    def own(self) -> str:
        return "\n".join(self.lines).strip()

    def full(self) -> str:
        parts = [self.heading] if self.heading else []
        if self.own():
            parts.append(self.own())
        parts += [c.full() for c in self.children]
        return "\n\n".join(parts).strip()


def _tree(text: str) -> _Node:
    root = _Node(0, "")
    stack = [root]
    in_fence = False
    for line in text.replace("\r\n", "\n").split("\n"):
        if _FENCE.match(line):
            in_fence = not in_fence
        m = None if in_fence else _HEADING.match(line)
        if m:
            level = len(m.group(1))
            while stack[-1].level >= level:
                stack.pop()
            node = _Node(level, m.group(2).strip(), heading=line.strip())
            stack[-1].children.append(node)
            stack.append(node)
        else:
            stack[-1].lines.append(line)
    return root


def _blocks(text: str) -> list[str]:
    """Paragraphs, with each fenced code block kept whole."""
    blocks, current, in_fence = [], [], False
    for line in text.split("\n"):
        if _FENCE.match(line):
            in_fence = not in_fence
        if not line.strip() and not in_fence:
            if current:
                blocks.append("\n".join(current))
                current = []
            continue
        current.append(line)
    if current:
        blocks.append("\n".join(current))
    return blocks


def _cut_block(block: str, limit: int) -> list[str]:
    """A block too long for one section, cut between lines.

    A table repeats its header row in every part, and a code block cut in the middle
    is closed at the end of one part and reopened at the start of the next, so each
    part is valid Markdown on its own.
    """
    lines = block.split("\n")
    header: list[str] = []
    if (
        len(lines) > 2
        and lines[0].lstrip().startswith("|")
        and set(lines[1].strip()) <= set("|-: ")
    ):
        header, lines = lines[:2], lines[2:]
    parts: list[str] = []
    current = list(header)
    fence: str | None = None  # the line that opened the code block we're in
    for line in lines:
        line = line[: limit * 2]  # one absurd line can't stall the cut
        closing = f"\n{_FENCE.match(fence).group(1)}" if fence else ""
        if len("\n".join([*current, line])) + len(closing) > limit and len(current) > len(
            header
        ) + (1 if fence and current and current[-1] == fence else 0):
            parts.append("\n".join(current) + closing)
            current = [*header, fence] if fence else list(header)
        current.append(line)
        if _FENCE.match(line):
            fence = None if fence else line
    if len(current) > len(header):
        parts.append("\n".join(current))
    return parts


def _pack(text: str, limit: int) -> list[str]:
    """Paragraphs packed into parts of at most `limit` characters."""
    parts: list[str] = []
    current = ""
    for block in _blocks(text):
        pieces = [block] if len(block) <= limit else _cut_block(block, limit)
        for piece in pieces:
            # A heading or a line of lead-in stays with what follows it, a little over.
            if current and len(current) + 2 + len(piece) > limit and len(current) >= MIN_CHARS:
                parts.append(current)
                current = ""
            current = f"{current}\n\n{piece}" if current else piece
    if current.strip():
        parts.append(current)
    return parts


def split_markdown(text: str, title: str, limit: int = MAX_CHARS) -> list[Section]:
    """A Markdown doc as sections of at most about `limit` characters, cut at headings."""
    root = _tree(text)
    # A doc with one H1 on top: that H1 is the title, not a section of its own.
    if not root.own() and len(root.children) == 1 and root.children[0].level == 1:
        top = root.children[0]
        title = top.title
        root = _Node(0, "", lines=top.lines, children=top.children)
    out: list[Section] = []

    def walk(node: _Node, path: list[str], carry: str = "") -> None:
        here = [*path, node.title] if node.title else path
        whole = "\n\n".join(p for p in (carry, node.full()) if p)
        if len(whole) <= limit or (not node.children and not carry and len(node.own()) <= limit):
            if whole.strip():
                out.append(Section(here, whole))
            return
        own = "\n\n".join(p for p in (carry, node.heading, node.own()) if p)
        if not node.children:
            for i, part in enumerate(_pack(own, limit), 1):
                out.append(Section([*here, f"part {i}"] if i > 1 else here, part))
            return
        if len(node.own()) < MIN_CHARS:  # a short intro rides with the first subsection
            walk(node.children[0], here, own)
        else:
            for i, part in enumerate(_pack(own, limit), 1):
                out.append(Section([*here, f"part {i}"] if i > 1 else here, part))
            walk(node.children[0], here)
        for child in node.children[1:]:
            walk(child, here)

    walk(root, [title])
    return out


def split_pdf_text(text: str, title: str, limit: int = MAX_CHARS) -> list[Section]:
    """A PDF's text (pages split by form feeds) as sections of whole paragraphs, by page."""
    out: list[Section] = []
    for page_no, page in enumerate(text.split("\f"), 1):
        page = re.sub(r"[ \t]+\n", "\n", page).strip()
        if not page:
            continue
        for i, part in enumerate(_pack(page, limit), 1):
            first = next((ln.strip() for ln in part.split("\n") if ln.strip()), "")
            if len(first) > 60:
                first = first[:60].rsplit(" ", 1)[0] + "..."
            label = f"page {page_no}" + (f" part {i}" if i > 1 else "") + f": {first}"
            out.append(Section([title, label], part))
    return out


# Finding and reading docs.


def _matches(rel: str, patterns) -> bool:
    """A pattern with a slash matches the path (`docs/uml/*.pdf`); one without, the name."""
    name = rel.rsplit("/", 1)[-1].lower()
    return any(
        fnmatch.fnmatch(rel.lower() if "/" in p else name, p.lower().removeprefix("./"))
        for p in patterns
    )


def find_docs(root: Path, include=INCLUDE, exclude=()) -> list[str]:
    """Doc files under `root` (relative paths), skipping build output, dependencies,
    secrets, the agent files at the top, and any PDF with a Markdown file of the same
    name (it was rendered from it)."""
    out: list[str] = []
    for folder, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and not d.startswith("."))
        for name in sorted(files):
            rel = (Path(folder) / name).relative_to(root).as_posix()
            if rel in AGENT_FILES or _matches(name, SECRET_FILES):
                continue
            if _matches(rel, include) and not _matches(rel, exclude):
                out.append(rel)
    stems = {_norm(Path(r).stem) for r in out if not r.lower().endswith(".pdf")}
    return [r for r in out if not (r.lower().endswith(".pdf") and _norm(Path(r).stem) in stems)]


def _norm(stem: str) -> str:
    return re.sub(r"[^a-z0-9]", "", stem.lower())


def pdftotext_path(configured: str | None = None) -> str | None:
    """Poppler's pdftotext: the configured path, ZK_PDFTOTEXT, or one on PATH."""
    for candidate in (configured, os.environ.get("ZK_PDFTOTEXT"), shutil.which("pdftotext")):
        if candidate and Path(candidate).is_file():
            return str(candidate)
    return None


def read_doc(path: Path, pdftotext: str | None) -> str | None:
    """A doc's text, or None if it can't be read (too big, or a PDF without pdftotext)."""
    try:
        if path.stat().st_size > MAX_DOC_BYTES:
            return None
        if path.suffix.lower() != ".pdf":
            return path.read_text(encoding="utf-8", errors="replace").lstrip("﻿")
        if not pdftotext:
            return None
        done = subprocess.run(
            [pdftotext, "-enc", "UTF-8", str(path), "-"],
            capture_output=True,
            timeout=60,
        )
        return done.stdout.decode("utf-8", errors="replace") if done.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


# Rendering notes.


def _slug(text: str, limit: int = 60) -> str:
    """Lower case and dashes, cut at a word boundary to at most `limit` characters."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    if len(slug) > limit:
        cut = slug[: limit + 1]
        slug = cut.rsplit("-", 1)[0] if "-" in cut[limit // 2 :] else slug[:limit]
    return slug.strip("-") or "section"


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _front(fields: dict) -> str:
    return "\n".join(f"{k}: {json.dumps(v, ensure_ascii=False)}" for k, v in fields.items())


@dataclass
class Planned:
    rel: str  # vault path
    text: str


def plan_doc(
    project: str,
    doc_rel: str,
    text: str,
    created: str,
    is_pdf: bool = False,
    doc_slug: str | None = None,
) -> list[Planned]:
    """The vault notes for one doc: its sections and its contents note."""
    title = Path(doc_rel).stem
    sections = split_pdf_text(text, title) if is_pdf else split_markdown(text, title)
    if not sections:
        return []
    doc_title = sections[0].path[0]
    doc_slug = doc_slug or _slug(Path(doc_rel).stem, 50)
    contents_stem = f"{doc_slug}--contents"
    doc_tokens = len(text) // CHARS_PER_TOKEN
    base = {"type": "reference", "project": project, "doc": doc_rel}
    taken: set[str] = {contents_stem}
    planned: list[Planned] = []
    entries: list[str] = []
    for sec in sections:
        label = " / ".join(sec.path[1:]) or "intro"
        stem = f"{doc_slug}--{_slug(label)}"
        n = 2
        while stem in taken:
            stem, n = f"{doc_slug}--{_slug(label)}-{n}", n + 1
        taken.add(stem)
        note_title = f"{doc_title}: {label}" if sec.path[1:] else doc_title
        body = (
            f"# {note_title}\n\n{capture.redact(sec.text).strip()}\n\n"
            f"From `{doc_rel}` in {project}; all sections: [[{contents_stem}]].\n"
        )
        front = {
            "id": f"ref-{project}-{stem}",
            "title": note_title,
            **base,
            "section": label,
            "sources": [f"project:{project}", f"doc:{project}/{doc_rel}"],
            "tags": [],
            "created": created,
            "generated": "zk",
            "body_hash": _hash(body),
        }
        planned.append(
            Planned(f"{REFS_DIR}/{project}/{stem}.md", f"---\n{_front(front)}\n---\n{body}")
        )
        entries.append(f"- [[{stem}]] {label} (~{len(sec.text) // CHARS_PER_TOKEN} tokens)")
    body = (
        f"# {doc_title}: contents\n\n"
        f"`{doc_rel}` from {project}, about {doc_tokens} tokens, in {len(sections)} sections. "
        "Open the section you need rather than the whole file.\n\n## Sections\n\n"
        + "\n".join(entries)
        + "\n"
    )
    front = {
        "id": f"ref-{project}-{contents_stem}",
        "title": f"{doc_title}: contents",
        **base,
        "contents": True,
        "sections": len(sections),
        "doc_tokens": doc_tokens,
        "sources": [f"project:{project}", f"doc:{project}/{doc_rel}"],
        "tags": [],
        "created": created,
        "generated": "zk",
        "body_hash": _hash(body),
    }
    planned.insert(
        0, Planned(f"{REFS_DIR}/{project}/{contents_stem}.md", f"---\n{_front(front)}\n---\n{body}")
    )
    return planned


def _split_front(text: str) -> tuple[dict, str]:
    if not text.startswith("---\n"):
        return {}, text
    head, sep, body = text[4:].partition("\n---\n")
    if not sep:
        return {}, text
    front = {}
    for line in head.splitlines():
        key, _, value = line.partition(": ")
        try:
            front[key] = json.loads(value)
        except ValueError:
            front[key] = value
    return front, body


def edited_by_hand(text: str) -> bool:
    front, body = _split_front(text)
    return front.get("body_hash") != _hash(body)


def _same_but_created(old: str, new: str) -> bool:
    strip = re.compile(r'^created: ".*"$', re.M)
    return strip.sub("", old) == strip.sub("", new)


# The record.


def record_path() -> Path:
    return data_dir() / "docs.json"


def load_record() -> dict:
    try:
        data = json.loads(record_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    data.setdefault("projects", {})
    return data


def save_record(record: dict) -> None:
    path = record_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=1, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def _stamp(path: Path) -> list[float]:
    try:
        st = path.stat()
        return [round(st.st_mtime, 3), st.st_size]
    except OSError:
        return [0, 0]


def project_for(path: str | Path) -> tuple[dict, str] | None:
    """The docs record of the project a file is in, and the file's path inside it."""
    target = Path(path)
    key = _key(target)
    for root_key, entry in load_record()["projects"].items():
        sep = "\\" if os.name == "nt" else "/"
        if key.startswith(root_key.rstrip(sep) + sep):
            rel = Path(key[len(root_key) :].lstrip(sep)).as_posix()
            return entry, rel
    return None


def project_in(folder: str | Path | None) -> str | None:
    """The project name docs were imported under for a folder (or one it is inside)."""
    if not folder:
        return None
    key = _key(Path(folder))
    sep = "\\" if os.name == "nt" else "/"
    for root_key, entry in load_record()["projects"].items():
        if key == root_key or key.startswith(root_key.rstrip(sep) + sep):
            return entry.get("project")
    return None


def stale(root: Path) -> bool:
    """Whether an imported project's docs (or the folders holding them) changed since."""
    entry = load_record()["projects"].get(_key(root))
    if not entry:
        return False
    for rel, stamp in {**entry.get("files", {}), **entry.get("folders", {})}.items():
        if _stamp(root / rel) != stamp:
            return True
    return False


# The import.


def import_docs(
    vault_dir: Path,
    root: Path,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    pdftotext: str | None = None,
    project: str | None = None,
    now: datetime | None = None,
    dry_run: bool = False,
) -> dict:
    """Bring refs/<project>/ in line with the project's docs. Returns what changed.

    Options not given are taken from the last import of this folder, so a re-sync
    keeps them.
    """
    now = now or datetime.now().astimezone()
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"{root} is not a folder")
    if root == vault_dir.resolve() or vault_dir.resolve() in root.parents:
        raise ValueError("that folder is the vault itself")
    record = load_record()
    key = _key(root)
    last = record["projects"].get(key, {})
    include = include or last.get("include") or list(INCLUDE)
    exclude = exclude if exclude is not None else last.get("exclude", [])
    project = project_slug(project or last.get("project") or root.name)
    pdftotext = pdftotext_path(pdftotext or last.get("pdftotext"))
    created = now.date().isoformat()

    result: dict = {
        "project": project, "docs": 0, "sections": 0, "new": [], "updated": [],
        "retired": [], "edited": [], "unchanged": 0, "skipped": [], "duplicates": [],
    }  # fmt: skip
    planned: list[Planned] = []
    seen_text: dict[str, str] = {}
    files: dict[str, list[float]] = {}
    folders: dict[str, list[float]] = {".": _stamp(root)}
    found = find_docs(root, include, exclude)
    stems: dict[str, int] = {}
    for rel in found:
        stems[_slug(Path(rel).stem, 50)] = stems.get(_slug(Path(rel).stem, 50), 0) + 1
    for rel in found:
        path = root / rel
        is_pdf = rel.lower().endswith(".pdf")
        text = read_doc(path, pdftotext)
        files[rel] = _stamp(path)
        parent = Path(rel).parent.as_posix()
        folders[parent] = _stamp(root / parent)
        if text is None or not text.strip():
            result["skipped"].append(rel + (" (no pdftotext)" if is_pdf and not pdftotext else ""))
            continue
        digest = _hash(" ".join(text.split()))
        if digest in seen_text:  # the same doc twice (a copy in two folders)
            result["duplicates"].append(f"{rel} = {seen_text[digest]}")
            continue
        seen_text[digest] = rel
        slug = _slug(Path(rel).stem, 50)
        if stems[slug] > 1:  # two docs share a name (docs/ai/README.md, README.md)
            slug = _slug(Path(rel).with_suffix("").as_posix(), 70)
        notes = plan_doc(project, rel, text, created, is_pdf, slug)
        if notes:
            result["docs"] += 1
            result["sections"] += len(notes) - 1
            planned += notes

    touched: list[str] = []
    wanted = {p.rel for p in planned}
    for p in planned:
        target = vault_dir / p.rel
        if not target.exists():
            result["new"].append(p.rel)
            if not dry_run:
                vault.write_new(vault_dir, p.rel, p.text)
                touched.append(p.rel)
            continue
        old = target.read_text(encoding="utf-8")
        if _same_but_created(old, p.text):
            result["unchanged"] += 1
            continue
        if edited_by_hand(old):
            result["edited"].append(p.rel)
            continue
        old_created = _split_front(old)[0].get("created") or created
        text = p.text.replace(f'created: "{created}"', f'created: "{old_created}"', 1)
        if text == old:
            result["unchanged"] += 1
            continue
        result["updated"].append(p.rel)
        if not dry_run:
            vault.replace(vault_dir, p.rel, text, hashlib.sha256(old.encode("utf-8")).hexdigest())
            touched.append(p.rel)
    folder = vault_dir / REFS_DIR / project
    for path in sorted(folder.glob("*.md")) if folder.is_dir() else []:
        rel = path.relative_to(vault_dir).as_posix()
        if rel in wanted:
            continue
        old = path.read_text(encoding="utf-8")
        front, _ = _split_front(old)
        if front.get("generated") != "zk":
            continue  # someone's own note in refs/: not zk's to move
        if edited_by_hand(old):
            result["edited"].append(rel)
            continue
        dst = vault.unique_path(vault_dir, f"{RETIRED_DIR}/{project}/{path.name}")
        result["retired"].append(rel)
        if not dry_run:
            vault.move_new(vault_dir, rel, dst)
            touched += [rel, dst]
    if dry_run:
        return result
    if touched:
        message = (
            f"zk: docs {project}, {len(result['new'])} new, {len(result['updated'])} updated, "
            f"{len(result['retired'])} retired"
        )
        try:
            result["commit"] = vault.git_commit(vault_dir, touched, message)
        except (subprocess.CalledProcessError, OSError) as exc:
            result["commit_error"] = f"written but not committed: {exc}"
    record = load_record()  # re-read: another run may have saved meanwhile
    record["projects"][key] = {
        "project": project,
        "root": str(root),
        "include": list(include),
        "exclude": list(exclude),
        "pdftotext": pdftotext,
        "files": files,
        "folders": folders,
        "docs": _doc_index(planned),
        "synced": now.isoformat(timespec="seconds"),
    }
    save_record(record)
    return result


def _doc_index(planned: list[Planned]) -> dict[str, dict]:
    """Per doc path: its contents note's id, section count and size, for the Read hook."""
    out: dict[str, dict] = {}
    for p in planned:
        front, _ = _split_front(p.text)
        if front.get("contents"):
            out[front["doc"]] = {
                "contents": front["id"],
                "sections": front["sections"],
                "tokens": front["doc_tokens"],
            }
    return out


def forget(root: Path) -> bool:
    """Stop syncing a project's docs (the notes already in the vault stay)."""
    record = load_record()
    found = record["projects"].pop(_key(root.resolve()), None)
    if found:
        save_record(record)
    return bool(found)


def format_result(result: dict) -> str:
    lines = [
        f"docs for {result['project']}: {result['docs']} docs, {result['sections']} sections; "
        f"{len(result['new'])} new, {len(result['updated'])} updated, "
        f"{len(result['retired'])} retired, {result['unchanged']} unchanged"
    ]
    if result["edited"]:
        lines.append(f"left alone (edited by hand): {', '.join(result['edited'])}")
    if result["skipped"]:
        lines.append(f"not read: {', '.join(result['skipped'])}")
    if result["duplicates"]:
        lines.append(f"copies of another doc: {', '.join(result['duplicates'])}")
    if result.get("commit_error"):
        lines.append(result["commit_error"])
    return "\n".join(lines)


def start(vault_dir: Path, root: Path) -> None:
    """Re-sync a project's docs detached, so the session that is starting doesn't wait."""
    from zk.sessions import INGESTING

    flags = (
        {
            "creationflags": subprocess.DETACHED_PROCESS
            | subprocess.CREATE_NEW_PROCESS_GROUP
            | subprocess.CREATE_NO_WINDOW
        }
        if os.name == "nt"
        else {"start_new_session": True}
    )
    subprocess.Popen(
        [sys.executable, "-m", "zk", "docs", str(vault_dir), str(root), "--quiet"],
        cwd=vault_dir,
        env={**os.environ, INGESTING: "1"},
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        **flags,
    )
