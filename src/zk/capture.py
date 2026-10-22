"""Turn a Claude Code session transcript (or free text) into an inbox capture.

Pure functions: parsing, redaction and rendering take and return strings. The
CLI reads the transcript and `zk.vault` writes the capture.

A transcript is JSONL, one event per line. Only the conversation is kept: the
person's messages and the assistant's text replies. Tool calls and results,
thinking, and side chains (subagents) are dropped as bulky and mostly noise.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass

MIN_USER_TURNS = 2
MIN_CHARS = 400
MAX_CHARS = 40_000

# Common secret shapes. Captures stay on this machine and in a private repo,
# but a pasted key should not sit in plain text in either.
_SECRETS = [
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S),
    re.compile(r"\bsk-(?:ant-)?[A-Za-z0-9_-]{20,}"),
    re.compile(r"\b(?:ghp|gho|ghs|ghu)_[A-Za-z0-9]{30,}|\bgithub_pat_[A-Za-z0-9_]{30,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}"),
    re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"),
    re.compile(
        r"(?i)\b(api[_-]?key|access[_-]?token|secret|password|passwd)\b(\s*[:=]\s*)(['\"]?)[^\s'\"]{6,}\3"
    ),
]
# Harness blocks that wrap or precede what the person typed; not their words.
_WRAPPERS = re.compile(
    r"<(system-reminder|session-context|command-[a-z-]+|local-command-[a-z-]+)\b[^>]*>.*?</\1>",
    re.S,
)


@dataclass
class Turn:
    role: str  # "user" or "assistant"
    text: str


def redact(text: str) -> str:
    for pattern in _SECRETS:
        if pattern.groups >= 2:
            text = pattern.sub(lambda m: f"{m.group(1)}{m.group(2)}[REDACTED]", text)
        else:
            text = pattern.sub("[REDACTED]", text)
    return text


def _text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n\n".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return ""


def parse_transcript(lines: Iterable[str]) -> list[Turn]:
    """The conversation in a Claude Code transcript, consecutive same-role turns merged."""
    turns: list[Turn] = []
    for line in lines:
        try:
            event = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(event, dict) or event.get("type") not in ("user", "assistant"):
            continue
        if event.get("isSidechain") or event.get("isMeta"):
            continue
        message = event.get("message") or {}
        text = _WRAPPERS.sub("", _text_of(message.get("content"))).strip()
        if not text:
            continue
        role = event["type"]
        if turns and turns[-1].role == role:
            turns[-1].text += "\n\n" + text
        else:
            turns.append(Turn(role, text))
    return turns


def worth_keeping(turns: list[Turn]) -> tuple[bool, str]:
    """Cheap first triage: skip sessions too short to hold anything worth a note."""
    user_turns = sum(1 for t in turns if t.role == "user")
    chars = sum(len(t.text) for t in turns)
    if user_turns < MIN_USER_TURNS:
        return False, f"only {user_turns} message(s) from you"
    if chars < MIN_CHARS:
        return False, f"only {chars} characters of conversation"
    return True, f"{user_turns} messages from you, {chars} characters"


def _clip(body: str, limit: int) -> str:
    if len(body) <= limit:
        return body
    half = limit // 2
    dropped = len(body) - 2 * half
    return f"{body[:half]}\n\n[... {dropped} characters omitted ...]\n\n{body[-half:]}"


def render_capture(
    turns: list[Turn], source: str, captured: str, project: str | None = None
) -> str:
    """The capture file: frontmatter, then the conversation under ## headings."""
    labels = {"user": "## You", "assistant": "## Assistant"}
    body = "\n\n".join(f"{labels[t.role]}\n\n{t.text}" for t in turns)
    meta = {"kind": "capture", "source": source, "captured": captured}
    if project:
        meta["project"] = project
    front = "\n".join(f"{k}: {json.dumps(v, ensure_ascii=False)}" for k, v in meta.items())
    title = f"# Capture: {project or source} {captured[:10]}"
    return f"---\n{front}\n---\n{title}\n\n{_clip(redact(body), MAX_CHARS)}\n"


def render_text_capture(text: str, source: str, captured: str) -> str:
    return render_capture([Turn("user", text.strip())], source, captured)
