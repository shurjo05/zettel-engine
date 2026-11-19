"""Links between a project's doc sections and its code, found in the text. No model. Pure.

Two things come out of a project's docs and the outlines of its source files
(outline.py):

- what each doc section refers to: the ids it names (R-08, UC-3, FIX-44, T-54) and
  the code's own class names. Sections that name the same id are linked through it,
  so one lookup follows a requirement to its realization, its tests and its fixes;
- why a section may be out of date: it puts a method in a class (or file) that no
  longer declares it, while another one does. Docs drift as code moves, and a stale
  sentence read as fact is worse than none.

Only names are compared, never behaviour: a section that describes what a method
does, when the method now does something else, is not caught.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath

from zk.outline import _TYPE_NAME, IDS, LANGS, Outline

# Ids that are versions or standards, not a project's own: UTF-8, SHA-256, GPT-4.
NOT_IDS = frozenset({"UTF", "SHA", "ISO", "GPT", "HTTP", "TLS", "MD", "AES", "RSA", "IPV", "CP"})
MAX_STALE = 3  # reasons kept per section

_CODE_SUFFIX = "|".join(s.lstrip(".") for s in sorted(LANGS, key=len, reverse=True))
# `Class.member` with nothing between: "LoginForm. Logout" is two sentences, not a member.
_DOTTED = re.compile(r"\b([A-Z][A-Za-z0-9_]*)(?:\.|::|#)([A-Za-z_][A-Za-z0-9_]*)\b")
_IN_CLASS = re.compile(
    r"\b([A-Za-z_][A-Za-z0-9_]*)(?:\(\))?\s+(?:is\s+)?(?:in|on|inside)\s+"
    r"(?:the\s+)?([A-Z][A-Za-z0-9_]*)\b"
)
_IN_FILE = re.compile(
    rf"\b([A-Za-z_][A-Za-z0-9_]*)(?:\(\))?\s+(?:is\s+)?(?:in|from|inside)\s+"
    rf"(?:the\s+)?([\w./\\-]+\.(?:{_CODE_SUFFIX}))\b"
)
_PATH = re.compile(rf"(?<![\w/\\.-])((?:[\w.-]+[/\\])+[\w.-]+\.(?:{_CODE_SUFFIX}))\b")
_NAME = re.compile(r"(~?[A-Za-z_$][\w$]*)\s*(?:<[^()]*>)?\s*\(")


@dataclass
class CodeFacts:
    """What a project's code declares, by name."""

    members: dict[str, set[str]] = field(default_factory=dict)  # class -> its member names
    bases: dict[str, set[str]] = field(default_factory=dict)  # class -> names it extends
    owners: dict[str, set[str]] = field(default_factory=dict)  # member -> classes with it
    homes: dict[str, set[str]] = field(default_factory=dict)  # class -> files declaring it
    file_names: dict[str, set[str]] = field(default_factory=dict)  # file name -> names in it
    # id -> "file: Class.member", the spots that name fewest other ids first
    ids: dict[str, list[str]] = field(default_factory=dict)
    files: set[str] = field(default_factory=set)  # every source file's name, lower case

    @property
    def classes(self) -> set[str]:
        return set(self.members)


def _id_ok(ref: str) -> bool:
    return ref.split("-", 1)[0] not in NOT_IDS


def ids_in(text: str) -> list[str]:
    """The ids a text names, in order, once each."""
    return list(dict.fromkeys(r for r in IDS.findall(text) if _id_ok(r)))


def member_name(signature: str) -> str | None:
    """`public bool Add(string name)` -> Add; `def name(self) -> str` -> name; a property's
    last word before its accessors."""
    head = signature.split("=>", 1)[0].split("{", 1)[0].split("=", 1)[0]
    found = _NAME.findall(head)
    if found:
        return found[0].lstrip("~")
    words = re.findall(r"[A-Za-z_$][\w$]*", head)
    return words[-1] if words else None


def _bases(signature: str) -> set[str]:
    after = re.split(r":|\bextends\b|\bimplements\b|\(", signature, maxsplit=1)
    if len(after) < 2:
        return set()
    return set(re.findall(r"[A-Za-z_]\w*", after[1])) - {"public", "private", "protected"}


def facts_from(outlines: dict[str, Outline], files: list[str] | None = None) -> CodeFacts:
    """CodeFacts from the outlines of a project's source files, keyed by path. `files`
    lists every source file, generated ones too, so a doc naming one isn't flagged."""
    facts = CodeFacts(files={PurePosixPath(f).name.lower() for f in [*(files or []), *outlines]})
    named: dict[str, int] = {}  # spot -> ids it names: a comment listing ten ids says little
    for rel, o in sorted(outlines.items()):
        name = PurePosixPath(rel).name.lower()
        in_file = facts.file_names.setdefault(name, set())
        scopes: list[str] = []
        for item in o.items:
            if item.kind == "type":
                m = _TYPE_NAME.search(item.signature)
                if not m:
                    continue
                scopes = [*scopes[: item.depth], m[2]]
                facts.members.setdefault(m[2], set())
                facts.bases.setdefault(m[2], set()).update(_bases(item.signature))
                facts.homes.setdefault(m[2], set()).add(rel)
                in_file.add(m[2])
                where = m[2]
            else:
                member = member_name(item.signature)
                if not member:
                    continue
                owner = scopes[item.depth - 1] if 0 < item.depth <= len(scopes) else None
                in_file.add(member)
                if owner:
                    facts.members.setdefault(owner, set()).add(member)
                    facts.owners.setdefault(member, set()).add(owner)
                where = f"{owner}.{member}" if owner else member
            for ref in item.ids:
                if _id_ok(ref):
                    spot = f"{rel}: {where}"
                    named[spot] = max(named.get(spot, 0), len(item.ids))
                    if spot not in facts.ids.setdefault(ref, []):
                        facts.ids[ref].append(spot)
    for spots in facts.ids.values():
        spots.sort(key=lambda s: named[s])
    return facts


def mentions(text: str, facts: CodeFacts | None = None) -> list[str]:
    """What a doc section refers to: its ids, then the code's class names it uses."""
    found = ids_in(text)
    if facts and facts.members:
        words = re.findall(r"[A-Z][A-Za-z0-9_]{2,}", text)
        found += [w for w in dict.fromkeys(words) if w in facts.members]
    return found


def _plain(text: str) -> str:
    return text.replace("`", "").replace("**", "")


def _inherited(facts: CodeFacts, cls: str, member: str) -> bool:
    return any(member in facts.members.get(b, set()) for b in facts.bases.get(cls, set()))


def stale_reasons(text: str, facts: CodeFacts | None) -> list[str]:
    """Why a doc section may no longer match the code: a member it puts in a class or file
    that doesn't declare it while another does, or a source path that is gone."""
    if not facts or not facts.files:
        return []
    text = _plain(text)
    out: list[str] = []

    def misplaced(member: str, cls: str) -> None:
        if cls not in facts.members or member in facts.members[cls] or member in facts.members:
            return  # declared there, or a class itself (its constructor isn't "misplaced")
        others = sorted(facts.owners.get(member, set()) - {cls})
        if others and not _inherited(facts, cls, member):
            out.append(f"{member} is declared in {', '.join(others)}, not {cls}")

    for cls, member in _DOTTED.findall(text):
        misplaced(member, cls)
    for member, cls in _IN_CLASS.findall(text):
        if member in facts.owners:
            misplaced(member, cls)
    for member, path in _IN_FILE.findall(text):
        name = PurePosixPath(path.replace("\\", "/")).name.lower()
        names = facts.file_names.get(name)
        if names is None or member in names:
            continue
        elsewhere = sorted(f for f, n in facts.file_names.items() if member in n)
        if elsewhere:
            out.append(f"{member} is in {', '.join(elsewhere)}, not {name}")
    for path in _PATH.findall(text):
        name = PurePosixPath(path.replace("\\", "/")).name.lower()
        if facts.files and name not in facts.files:
            out.append(f"{path} is not in the code any more")
    return list(dict.fromkeys(out))[:MAX_STALE]
