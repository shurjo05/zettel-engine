from pathlib import Path

import yaml

from zk.capture import (
    MAX_CHARS,
    Turn,
    parse_transcript,
    redact,
    render_capture,
    worth_keeping,
)
from zk.notes import split_frontmatter

TRANSCRIPT = Path(__file__).parent / "fixtures" / "transcripts" / "session.jsonl"


def turns():
    return parse_transcript(TRANSCRIPT.read_text(encoding="utf-8").splitlines())


def test_only_the_conversation_is_kept():
    result = turns()
    assert [t.role for t in result] == ["user", "assistant", "user", "assistant"]
    text = "\n".join(t.text for t in result)
    assert result[0].text == "How should I merge two ranked result lists?"
    assert (
        "reciprocal rank fusion" in result[1].text and "never have to be compared" in result[1].text
    )
    for dropped in ("harness note", "secret thoughts", "file-list", "subagent chatter", '"ls"'):
        assert dropped not in text


def test_secrets_are_redacted():
    assert "sk-ant" not in redact("key sk-ant-abcdefghijklmnopqrstuvwxyz123456 here")
    assert redact("password = hunter2hunter2") == "password = [REDACTED]"
    assert redact("API_KEY: 'abcdef123456'") == "API_KEY: [REDACTED]"
    assert redact("token AKIAABCDEFGHIJKLMNOP end") == "token [REDACTED] end"
    pem = "-----BEGIN RSA PRIVATE KEY-----\nabc\n-----END RSA PRIVATE KEY-----"
    assert redact(f"x {pem} y") == "x [REDACTED] y"
    assert redact("the password policy is strict") == "the password policy is strict"


def test_short_sessions_are_skipped():
    assert worth_keeping(turns())[0]
    assert not worth_keeping([Turn("user", "hi " * 300), Turn("assistant", "hello")])[0]
    assert not worth_keeping([Turn("user", "a"), Turn("assistant", "b"), Turn("user", "c")])[0]


def test_render_has_parseable_frontmatter_and_redacts():
    text = render_capture(turns(), "claude-code:abc", "2026-10-22T18:00:00+00:00", "zettel-engine")
    raw, body = split_frontmatter(text)
    front = yaml.safe_load(raw)
    assert front["source"] == "claude-code:abc" and front["project"] == "zettel-engine"
    assert body.startswith("# Capture: zettel-engine 2026-10-22")
    assert "## You" in body and "## Assistant" in body
    assert "sk-ant" not in text


def test_long_sessions_keep_start_and_end():
    long = [Turn("user", "start " + "x" * MAX_CHARS), Turn("assistant", "y" * MAX_CHARS + " end")]
    text = render_capture(long, "s", "2026-10-22T00:00:00")
    assert "characters omitted" in text
    assert "start" in text and text.rstrip().endswith("end")
    assert len(text) < MAX_CHARS + 1000
