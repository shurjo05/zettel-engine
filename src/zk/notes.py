"""Parse and validate Markdown notes.

This module is pure: it turns text into a `Note` and a `Note` into a list of
issues. It never touches the filesystem, which keeps it trivial to test.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

import yaml

NOTE_TYPES = ("concept", "pattern", "problem", "decision")

_FENCE = re.compile(r"^\s*(```|~~~)")
_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_INLINE_CODE = re.compile(r"`[^`\n]*`")
# [[target]], [[target|label]], [[target#heading]], [[target#heading|label]]
_WIKILINK = re.compile(r"\[\[([^\[\]|#]*)(?:#([^\[\]|]*))?(?:\|([^\[\]]*))?\]\]")


@dataclass(frozen=True)
class Link:
    target: str  # as written, e.g. "bm25-ranking"
    heading: str | None  # "Details" in [[note#Details]]
    section: str | None  # the "## Section" the link appears under


@dataclass(frozen=True)
class Issue:
    level: str  # "error" or "warning"
    code: str  # stable machine-readable name, e.g. "missing-id"
    message: str


@dataclass
class Note:
    path: str  # relative to the vault, forward slashes
    content_hash: str
    frontmatter: dict[str, Any] | None  # None when the note has no frontmatter block
    yaml_error: str | None
    h1: str | None
    sections: dict[str, str]  # "## Heading" -> text under it
    links: list[Link]
    body: str  # everything after the frontmatter

    id: str | None = None
    title: str = ""
    type: str | None = None
    tags: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    created: str | None = None
    updated: str | None = None
    summary: str = ""

    @property
    def stem(self) -> str:
        name = self.path.rsplit("/", 1)[-1]
        return name[:-3] if name.lower().endswith(".md") else name


def content_hash(data: bytes) -> str:
    """SHA-256 of the raw file bytes: changes if and only if the content changes."""
    return hashlib.sha256(data).hexdigest()


def split_frontmatter(text: str) -> tuple[str | None, str]:
    """Return (frontmatter_yaml, body). Frontmatter must open on the first line."""
    lines = text.split("\n")
    if not lines or lines[0].strip() != "---":
        return None, text
    for i in range(1, len(lines)):
        if lines[i].strip() in ("---", "..."):
            return "\n".join(lines[1:i]), "\n".join(lines[i + 1 :])
    return None, text  # an opening --- with no closing line is not frontmatter


def _yaml_value(value: Any) -> Any:
    """YAML turns 2026-10-15 into a date and 20261015 into an int; we want strings."""
    if isinstance(value, dt.datetime | dt.date):
        return value.isoformat()
    if isinstance(value, list):
        return [_yaml_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _yaml_value(v) for k, v in value.items()}
    return value


def _as_str_list(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return [str(v) for v in value if v is not None and str(v) != ""]
    return [str(value)]


def _scan_body(body: str) -> tuple[str | None, dict[str, str], list[Link]]:
    """Walk the body line by line, skipping fenced code blocks.

    Returns the first H1, the text under each H2, and every wikilink with the
    H2 section it sits in. Headings and links inside code are ignored, which is
    one of the bugs the legacy regex parser had.
    """
    h1: str | None = None
    sections: dict[str, list[str]] = {}
    current: str | None = None
    links: list[Link] = []
    fence: str | None = None

    for line in body.split("\n"):
        fence_match = _FENCE.match(line)
        if fence_match:
            marker = fence_match.group(1)
            if fence is None:
                fence = marker
            elif marker == fence:
                fence = None
            if current is not None:
                sections[current].append(line)
            continue
        if fence is not None:
            if current is not None:
                sections[current].append(line)
            continue

        heading = _HEADING.match(line)
        if heading:
            level, text = len(heading.group(1)), heading.group(2).strip()
            if level == 1 and h1 is None:
                h1 = text
                continue
            if level == 2:
                current = text
                sections.setdefault(current, [])
                continue

        if current is not None:
            sections[current].append(line)
        for m in _WIKILINK.finditer(_INLINE_CODE.sub("", line)):
            target = m.group(1).strip()
            if target:
                links.append(Link(target=target, heading=m.group(2), section=current))

    return h1, {k: "\n".join(v).strip() for k, v in sections.items()}, links


def _first_paragraph(body: str) -> str:
    """The first plain paragraph after the H1, used when there is no Summary section."""
    in_fence = False
    para: list[str] = []
    for line in body.split("\n"):
        stripped = line.strip()
        if _FENCE.match(line):
            in_fence = not in_fence
            if para:
                break
            continue
        if in_fence:
            continue
        if stripped.startswith("#"):
            if para:
                break
            continue
        if not stripped:
            if para:
                break
            continue
        para.append(stripped)
    return " ".join(para)


def parse_note(path: str, data: bytes) -> Note:
    """Parse raw file bytes into a Note. Never raises on bad content."""
    text = data.decode("utf-8", errors="replace").lstrip("﻿").replace("\r\n", "\n")
    raw_frontmatter, body = split_frontmatter(text)

    frontmatter: dict[str, Any] | None = None
    yaml_error = None
    if raw_frontmatter is not None:
        try:
            loaded = yaml.safe_load(raw_frontmatter)
        except yaml.YAMLError as exc:
            yaml_error = str(exc).split("\n")[0]
            loaded = None
        if loaded is None and yaml_error is None:
            frontmatter = {}
        elif isinstance(loaded, dict):
            frontmatter = _yaml_value(loaded)
        elif yaml_error is None:
            yaml_error = "frontmatter is not a mapping"

    h1, sections, links = _scan_body(body)
    note = Note(
        path=path,
        content_hash=content_hash(data),
        frontmatter=frontmatter,
        yaml_error=yaml_error,
        h1=h1,
        sections=sections,
        links=links,
        body=body,
    )

    fm = frontmatter or {}
    note.id = str(fm["id"]) if fm.get("id") not in (None, "") else None
    note.title = str(fm.get("title") or h1 or note.stem)
    note.type = str(fm["type"]).strip().lower() if fm.get("type") else None
    note.tags = _as_str_list(fm.get("tags"))
    note.aliases = _as_str_list(fm.get("aliases"))
    # New format uses a `sources` list; the legacy vault used a single `source`.
    note.sources = _as_str_list(fm.get("sources")) or _as_str_list(fm.get("source"))
    note.created = str(fm["created"]) if fm.get("created") else None
    note.updated = str(fm["updated"]) if fm.get("updated") else None
    note.summary = sections.get("Summary") or _first_paragraph(body)
    return note


def validate(note: Note) -> list[Issue]:
    """Check one note against the schema. Cross-note checks live in the index."""
    issues: list[Issue] = []

    def error(code: str, message: str) -> None:
        issues.append(Issue("error", code, message))

    def warn(code: str, message: str) -> None:
        issues.append(Issue("warning", code, message))

    if note.yaml_error:
        error("invalid-yaml", f"frontmatter is not valid YAML: {note.yaml_error}")
        return issues
    if note.frontmatter is None:
        error("missing-frontmatter", "no frontmatter block at the top of the file")
        return issues

    fm = note.frontmatter
    if note.id is None:
        error("missing-id", "frontmatter has no `id`")
    if not fm.get("title"):
        warn("missing-title", f"frontmatter has no `title`; using {note.title!r}")
    if note.type is None:
        error("missing-type", "frontmatter has no `type`")
    elif note.type not in NOTE_TYPES:
        error("invalid-type", f"type {note.type!r} is not one of {', '.join(NOTE_TYPES)}")
    for key in ("tags", "aliases", "sources"):
        if key in fm and fm[key] is not None and not isinstance(fm[key], list):
            error("not-a-list", f"`{key}` should be a list")
    if not note.sources:
        warn("missing-sources", "no `sources` (or legacy `source`) recorded")
    if note.created is None:
        warn("missing-created", "frontmatter has no `created` date")
    if note.h1 is None:
        warn("missing-h1", "no `# Title` heading")
    elif fm.get("title") and note.h1 != str(fm["title"]):
        warn("title-mismatch", f"H1 {note.h1!r} differs from title {fm['title']!r}")
    if not note.summary:
        warn("missing-summary", "no summary paragraph or Summary section")
    return issues
