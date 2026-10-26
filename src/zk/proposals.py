"""Note drafts: render structured fields into a note in the vault's schema.

Pure functions. An agent supplies the fields (it understands the capture); this
module decides the file name, id, frontmatter and layout, and refuses anything
that would not pass `zk check` cleanly, so every note that reaches the vault
has the same shape however it was written.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime

import yaml

from zk.notes import NOTE_TYPES, parse_note, validate

HELD_DIR = "inbox/held"  # drafts that looked like duplicates
NOTES_DIR = "notes"

# Cosine similarity between a proposal and an existing note (bge-small, unit
# vectors). bge-small scores run high: a typical note's nearest neighbour sits
# near 0.80, reworded duplicates scored 0.78-0.94 and same-topic notes 0.62-0.81,
# so the cut favours flagging duplicates; a flagged draft is held, not lost.
DUPLICATE_AT = 0.85
RELATED_AT = 0.80


class InvalidProposal(ValueError):
    pass


@dataclass
class Proposal:
    title: str
    type: str
    summary: str
    body: str = ""  # Markdown for the Details section
    tags: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    links: dict[str, str] = field(default_factory=dict)  # note name -> why it is related
    questions: list[str] = field(default_factory=list)  # test questions the note answers


def slugify(title: str) -> str:
    text = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:80].rstrip("-") or "note"


def render(
    proposal: Proposal,
    now: datetime,
    *,
    created: str | None = None,
    updated: str | None = None,
    name: str | None = None,
) -> tuple[str, str]:
    """(file name, Markdown) for a draft. Raises InvalidProposal if it isn't clean.

    The id comes from `now`; `created`, `updated` (ISO dates) and the file name
    default to today and the title's slug, and an import passes the old ones.
    """
    p = proposal
    problems = []
    if not p.title.strip():
        problems.append("title is empty")
    if p.type not in NOTE_TYPES:
        problems.append(f"type must be one of {', '.join(NOTE_TYPES)}")
    if not p.summary.strip():
        problems.append("summary is empty: state the idea in one to three sentences")
    if not p.sources:
        problems.append("sources is empty: say where the idea came from")
    if "\n#" in "\n" + p.summary:
        problems.append("summary must be plain sentences, not headings")
    if problems:
        raise InvalidProposal("; ".join(problems))

    front = {
        "id": now.strftime("%Y%m%d%H%M%S"),
        "title": p.title.strip(),
        "type": p.type,
        "tags": [t.strip().lstrip("#") for t in p.tags if t.strip()],
        "aliases": [a.strip() for a in p.aliases if a.strip()],
        "sources": [s.strip() for s in p.sources if s.strip()],
        "created": created or now.date().isoformat(),
        "updated": updated or created or now.date().isoformat(),
    }
    if not front["aliases"]:
        del front["aliases"]
    yaml_text = yaml.safe_dump(front, sort_keys=False, allow_unicode=True, default_flow_style=None)
    parts = [
        f"---\n{yaml_text}---\n# {front['title']}",
        " ".join(p.summary.split()),
    ]
    if p.body.strip():
        parts.append("## Details\n\n" + p.body.strip())
    if p.links:
        parts.append(
            "## Related\n\n"
            + "\n".join(
                f"- [[{target.strip()}]]" + (f": {why.strip()}" if why.strip() else "")
                for target, why in p.links.items()
            )
        )
    text = "\n\n".join(parts) + "\n"

    name = f"{name or slugify(p.title)}.md"
    issues = validate(parse_note(f"{NOTES_DIR}/{name}", text.encode("utf-8")))
    if issues:
        raise InvalidProposal("; ".join(i.message for i in issues))
    return name, text


def verdict(similarity: float) -> str:
    if similarity >= DUPLICATE_AT:
        return "likely-duplicate"
    if similarity >= RELATED_AT:
        return "related"
    return "new"
