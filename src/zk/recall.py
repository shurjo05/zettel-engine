"""What reaches Claude without being asked for: a brief at session start, a one-time nudge
before searching files, and notes that match a failed command.

Claude Code runs these through hook.py and adds what they print to the session's
context, so each is built to cost little:
- the brief names the map notes (a capped list) and, in a known project, lists
  that project's notes; it stays under BRIEF_CHARS (about 1,000 tokens);
- the nudge is one line, once per session, the first time Claude greps or globs;
- reading a long doc that `zk docs` put in the vault is turned back once per file per
  session, with the way to its sections (a whole read, or a read of a part, goes ahead);
- after a failed shell command, up to two notes whose summary matches the error,
  and only when they clearly match;
- on each prompt, up to three pointers to notes whose title, tags and summary share
  most of the prompt's words (BM25 first, then that check), capped at PROMPT_CHARS;
  silent when nothing clearly matches.
A note is pointed to at most once per session, by the error or the prompt hook.

Everything here only reads the index (opened read-only) and imports nothing
heavy: hooks run on every session and every failed command. Each hook's small
per-session record lives in the zk data folder, never in the project.
Text is written as facts about the vault, not as instructions.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import time
from pathlib import Path

from zk.index import data_dir, default_db_path
from zk.search import _WORD, is_container, keyword_search, project_slug, same_project

BRIEF_CHARS = 4000
MAX_TOPICS = 15
MAX_PROJECTS = 20
MAX_PROJECT_ENTRIES = 12
ENTRY_CHARS = 100  # a project note in the brief: its link, title and the start of its summary
ERROR_HITS = 2
MAX_ERROR_WORDS = 12
PROMPT_HITS = 3
PROMPT_MATCH = 3  # words a note must share with a prompt (lower: more pointers, more noise)
PROMPT_CHARS = 1200  # about 300 tokens at most per prompt, and usually nothing
MAX_PROMPT_WORDS = 16
MAX_PROJECT_DOCS = 8  # docs listed in the brief for the project a session is in
REDIRECT_TOKENS = 1500  # a doc this long is read through its sections first
STATE_DAYS = 7  # per-session records older than this are cleared at session start
NUDGE = (
    "The zk vault (search_notes) holds notes from past projects that may already "
    "describe this code or problem; a search returns summaries for a few hundred tokens."
)
_ERRORISH = re.compile(
    r"(?i)error|exception|fail|fatal|cannot|can't|not found|no such|undefined|denied|"
    r"traceback|refused|invalid|missing|unresolved|timed out"
)
_STOP = frozenset(
    "the and for with from that this was were are not but you your have has had can cannot "
    "could would should into onto file line exit code error errors failed failure fatal "
    "exception warning found none null true false see more info run".split()
)
_PROMPT_STOP = _STOP | frozenset(
    "what how why when where which who does did doing done make made use using used add want "
    "need needs like just get got also its it's i'm let's lets please thanks thank yes okay "
    "now then than them they their there here about some any all one two new old should "
    "would could will shall may might must been being because what's into over under out "
    "only very really much many each other same different look looks think know tell show "
    "give take try work works working fix help write change update check".split()
)


def open_index(vault_dir: Path, db: Path | None = None) -> sqlite3.Connection | None:
    """The vault's index, read-only, or None if it hasn't been built."""
    path = db or default_db_path(vault_dir)
    if not path.exists():
        return None
    conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def project_root(cwd: str | Path | None) -> Path | None:
    """The git repository a folder is in (its top folder), or None."""
    if not cwd or not Path(cwd).exists():
        return None
    here = Path(cwd)
    for folder in (here, *here.parents):
        if (folder / ".git").exists():
            return folder
    return None


def project_of(cwd: str | Path | None) -> str | None:
    """The project a session is in: its repository's folder name, else the folder's own."""
    if not cwd:
        return None
    root = project_root(cwd) or Path(cwd)
    slug = project_slug(root.name)
    return slug if slug and not is_container(slug) else None


# Per-session records.


def _state_path(session_id: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_-]", "", session_id)[:64] or "unknown"
    return data_dir() / "hook-state" / f"{safe}.json"


def _load_state(session_id: str) -> dict:
    try:
        return json.loads(_state_path(session_id).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save_state(session_id: str, state: dict) -> None:
    path = _state_path(session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state), encoding="utf-8")
    os.replace(tmp, path)


def clear_old_state(now: float | None = None) -> None:
    """Remove per-session records older than STATE_DAYS (zk's own files, not vault content)."""
    now = time.time() if now is None else now
    folder = data_dir() / "hook-state"
    for path in folder.glob("*.json") if folder.is_dir() else []:
        try:
            if now - path.stat().st_mtime > STATE_DAYS * 86400:
                path.unlink()
        except OSError:
            pass


# The brief.


def _notes_section(body: str) -> list[str]:
    """The entry lines under a map's "## Notes" heading."""
    _, _, rest = body.partition("## Notes")
    return [line for line in rest.splitlines() if line.startswith("- [[")]


def _short(entry: str) -> str:
    return entry if len(entry) <= ENTRY_CHARS else entry[: ENTRY_CHARS - 3].rstrip() + "..."


def brief(
    conn: sqlite3.Connection, cwd: str | Path | None = None, project: str | None = None
) -> str:
    """What every session is told about the vault: its maps, and this project's notes.

    The project is the one `cwd` is in, unless `project` names one (the review uses that
    to measure the largest brief).
    """
    count = conn.execute(
        "SELECT count(*) FROM notes WHERE coalesce(type, '') NOT IN ('map', 'reference')"
    ).fetchone()[0]
    docs = [
        json.loads(f or "{}")
        for (f,) in conn.execute(
            "SELECT frontmatter FROM notes WHERE type = 'reference' "
            "AND json_extract(frontmatter, '$.contents') ORDER BY path"
        )
    ]
    if not count and not docs:
        return ""
    maps = conn.execute(
        "SELECT path, body, frontmatter FROM notes WHERE type = 'map' ORDER BY path"
    ).fetchall()
    topics, projects, here = [], [], None
    current = project or project_of(cwd)
    best = 0  # 2: this folder's own name; 1: a name that is the same project
    for m in maps:
        front = json.loads(m["frontmatter"] or "{}")
        name = m["path"].rsplit("/", 1)[-1].removesuffix(".md")
        size = len(front.get("members") or []) or len(_notes_section(m["body"]))
        if front.get("map") == "project":
            projects.append((size, name))
            names = [front.get("project"), *(front.get("aliases") or [])]
            names = [str(n) for n in names if n]
            fit = 2 if current in names else int(any(same_project(current, n) for n in names))
            if current and fit > best:
                best, here = fit, (name, front.get("project") or current, m["body"])
        else:
            label = str(front.get("title", name)).removeprefix("Map: ")
            topics.append((size, f"{name} ({label}; {size})"))
    lines = [
        f"The zk vault holds {count} notes from the user's past work. search_notes finds them "
        "by meaning or keyword and returns summaries; get_note opens one note or map in full."
    ]
    if topics:
        shown = [t for _, t in sorted(topics, key=lambda t: (-t[0], t[1]))[:MAX_TOPICS]]
        more = len(topics) - len(shown)
        lines.append("Topic maps: " + ", ".join(shown) + (f", and {more} more" if more else ""))
    if projects:
        shown = [f"{n} ({s})" for s, n in sorted(projects, key=lambda p: (-p[0], p[1]))]
        more = max(0, len(shown) - MAX_PROJECTS)
        lines.append(
            "Project maps: "
            + ", ".join(shown[:MAX_PROJECTS])
            + (f", and {more} more" if more else "")
        )
    if docs:
        sections = sum(int(d.get("sections") or 0) for d in docs)
        lines.append(
            f"It also holds {sections} sections of {len(docs)} project docs (type reference), "
            "each a doc's own text, so a search returns the part of a doc that answers."
        )
    if docs and cwd:
        from zk import docs as project_docs

        imported_as = project_docs.project_in(cwd)  # `zk docs --project` may name it otherwise
    else:
        imported_as = None
    current_docs = [
        d
        for d in docs
        if d.get("project") == imported_as
        or (current and same_project(current, str(d.get("project"))))
    ]
    if current_docs:
        project = current_docs[0].get("project")
        lines.append(
            f'This project\'s docs are in the vault: search_notes(query, project="{project}") '
            "returns the matching section; get_note on a contents note lists a doc's sections."
        )
        current_docs.sort(key=lambda d: -int(d.get("doc_tokens") or 0))  # biggest first
        for d in current_docs[:MAX_PROJECT_DOCS]:
            stem = str(d.get("id", "")).removeprefix(f"ref-{project}-")
            lines.append(
                f"- {d.get('doc')}: {d.get('sections')} sections, ~{d.get('doc_tokens')} tokens "
                f"(contents: {stem})"
            )
        if len(current_docs) > MAX_PROJECT_DOCS:
            lines.append(f"(and {len(current_docs) - MAX_PROJECT_DOCS} more docs)")
    if here:
        name, project, body = here
        entries = _notes_section(body)
        lines.append(f"This folder is the project {project}; its map {name} lists:")
        lines += [_short(e) for e in entries[:MAX_PROJECT_ENTRIES]]
        if len(entries) > MAX_PROJECT_ENTRIES:
            lines.append(f"(and {len(entries) - MAX_PROJECT_ENTRIES} more in {name})")
    text = "\n".join(lines)
    if len(text) > BRIEF_CHARS:
        text = text[: BRIEF_CHARS - 4].rsplit("\n", 1)[0] + "\n..."
    return text


# The nudge.


def nudge(event: dict) -> str | None:
    """One line, the first time a session greps or globs; nothing after that."""
    if event.get("tool_name") not in ("Grep", "Glob"):
        return None
    sid = str(event.get("session_id") or "")
    state = _load_state(sid)
    if state.get("nudged"):
        return None
    state["nudged"] = True
    _save_state(sid, state)
    return NUDGE


# Reads of a doc that is in the vault.


def on_read(event: dict) -> str | None:
    """Why a Read of a long imported doc should go through its sections, or None to let it run.

    Only a whole-file read of a doc of REDIRECT_TOKENS or more, unchanged since it was
    imported, and only the first time in a session: the second read goes ahead, so the
    agent can still read the whole file when it needs to. ZK_READ_REDIRECT=0 turns it off.
    """
    if event.get("tool_name") != "Read" or os.environ.get("ZK_READ_REDIRECT") == "0":
        return None
    args = event.get("tool_input") or {}
    path = args.get("file_path")
    if not path or args.get("offset") or args.get("limit"):
        return None
    from zk import docs

    found = docs.project_for(path)
    if found is None:
        return None
    entry, rel = found
    doc = (entry.get("docs") or {}).get(rel)
    if not doc or int(doc.get("tokens") or 0) < REDIRECT_TOKENS:
        return None
    if docs._stamp(Path(path)) != (entry.get("files") or {}).get(rel):
        return None  # changed since the import: the vault's copy may be behind
    sid = str(event.get("session_id") or "")
    state = _load_state(sid)
    seen = set(state.get("reads") or [])
    if rel in seen:
        return None
    state["reads"] = sorted(seen | {rel})
    _save_state(sid, state)
    project = entry.get("project")
    stem = str(doc.get("contents", "")).removeprefix(f"ref-{project}-")
    return (
        f"zk: {rel} (about {doc.get('tokens')} tokens) is in the zk vault as "
        f'{doc.get("sections")} sections. search_notes("<what you need>", project="{project}") '
        f'returns the section that answers, and get_note("{stem}") lists them all. '
        "To read the whole file anyway, read it again; this check runs once per file."
    )


# Notes for a failed command.


def error_query(error: str) -> list[str]:
    """The words worth searching for in a failed command's output.

    The error lines (usually at the end), without paths, numbers or common words.
    """
    lines = [ln.strip() for ln in error.splitlines() if ln.strip()]
    if lines and re.match(r"(?i)exit code -?\d+$", lines[0]):
        lines = lines[1:]
    errorish = [ln for ln in lines if _ERRORISH.search(ln)]
    picked = errorish[-3:] or lines[-2:]
    words: list[str] = []
    for line in picked:
        for token in line.split():
            if "/" in token or "\\" in token or "://" in token:
                continue  # paths and URLs are specific to this machine
            for w in _WORD.findall(token.lower()):
                if len(w) >= 3 and not w.isdigit() and w not in _STOP and not _hexlike(w):
                    words.append(w)
    return list(dict.fromkeys(words))[:MAX_ERROR_WORDS]


def _hexlike(word: str) -> bool:
    return len(word) >= 7 and bool(re.fullmatch(r"[0-9a-f]+", word))


def _matched(words: list[str], text: str) -> int:
    """How many query words a note's title, tags and summary contain (by 5-letter stem)."""
    have = {w[:5] for w in _WORD.findall(text.lower())}
    return sum(w[:5] in have for w in words)


def for_error(conn: sqlite3.Connection, error: str, skip: set[str] = frozenset()) -> list[dict]:
    """Notes that clearly match an error: most of their top words appear in it."""
    words = error_query(error)
    if len(words) < 2:
        return []
    out = []
    for hit in keyword_search(conn, " ".join(words), k=5):
        if hit.path in skip:
            continue
        tags = conn.execute("SELECT tags FROM notes WHERE path = ?", (hit.path,)).fetchone()
        text = " ".join([hit.title, hit.summary, " ".join(json.loads(tags[0] or "[]"))])
        matched = _matched(words, text)
        if matched >= 2 and matched >= 0.3 * len(words):
            out.append({"path": hit.path, "title": hit.title, "summary": hit.summary})
        if len(out) == ERROR_HITS:
            break
    return out


def on_failure(conn: sqlite3.Connection, event: dict) -> str | None:
    """Text for a failed shell command, or None when no note clearly matches."""
    if event.get("tool_name") not in ("Bash", "PowerShell") or event.get("is_interrupt"):
        return None
    sid = str(event.get("session_id") or "")
    state = _load_state(sid)
    shown = set(state.get("shown") or [])
    hits = for_error(conn, str(event.get("error") or ""), shown)
    if not hits:
        return None
    state["shown"] = sorted(shown | {h["path"] for h in hits})
    _save_state(sid, state)
    lines = ["Notes in the zk vault match this error (get_note opens one):"]
    for h in hits:
        stem = h["path"].rsplit("/", 1)[-1].removesuffix(".md")
        summary = " ".join(h["summary"].split())
        if len(summary) > 240:
            summary = summary[:237].rstrip() + "..."
        lines.append(f"- {stem}: {h['title']}. {summary}")
    return "\n".join(lines)


# Pointers on each prompt.


def prompt_query(prompt: str) -> list[str]:
    """The words in a prompt worth searching for: no common words, numbers, paths or URLs."""
    words: list[str] = []
    for token in prompt.split():
        if "/" in token or "\\" in token:
            continue
        for w in _WORD.findall(token.lower()):
            if len(w) >= 3 and not w.isdigit() and w not in _PROMPT_STOP and not _hexlike(w):
                words.append(w)
    return list(dict.fromkeys(words))[:MAX_PROMPT_WORDS]


def for_prompt(conn: sqlite3.Connection, prompt: str, skip: set[str] = frozenset()) -> list[dict]:
    """Notes that clearly match a prompt: BM25's best, kept only if they share most of its words.

    BM25 alone always returns something; the word check is the score floor. A note must
    contain three of the prompt's words (both, for a two-word prompt), and a quarter of
    them for long prompts. Measured on the legacy copy (decision 23): 19 of 30 test
    questions got pointers, all with a right note, and 0 of 20 unrelated prompts did.
    """
    words = prompt_query(prompt)
    if len(words) < 2:
        return []
    need = min(len(words), max(PROMPT_MATCH, round(0.25 * len(words))))
    out = []
    for hit in keyword_search(conn, " ".join(words), k=8):
        if hit.path in skip:
            continue
        tags = conn.execute("SELECT tags FROM notes WHERE path = ?", (hit.path,)).fetchone()
        text = " ".join([hit.title, hit.summary, " ".join(json.loads(tags[0] or "[]"))])
        if _matched(words, text) >= need:
            out.append({"path": hit.path, "title": hit.title, "summary": hit.summary})
        if len(out) == PROMPT_HITS:
            break
    return out


def on_prompt(conn: sqlite3.Connection, event: dict) -> str | None:
    """Pointers for a prompt, or None. Slash commands and shell escapes get nothing."""
    prompt = str(event.get("prompt") or "").strip()
    if not prompt or prompt[0] in "/!#":
        return None
    sid = str(event.get("session_id") or "")
    state = _load_state(sid)
    shown = set(state.get("shown") or [])
    hits = for_prompt(conn, prompt, shown)
    if not hits:
        return None
    state["shown"] = sorted(shown | {h["path"] for h in hits})
    _save_state(sid, state)
    lines = ["Notes in the zk vault match this prompt (get_note opens one):"]
    for h in hits:
        stem = h["path"].rsplit("/", 1)[-1].removesuffix(".md")
        line = f"- {stem}: {h['title']}. {' '.join(h['summary'].split())}"
        lines.append(line if len(line) <= 300 else line[:297].rstrip() + "...")
    text = "\n".join(lines)
    return text if len(text) <= PROMPT_CHARS else text[: PROMPT_CHARS - 4].rsplit("\n", 1)[0]
