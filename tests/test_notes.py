from zk.notes import parse_note, split_frontmatter, validate


def note(text: str, path: str = "notes/example.md"):
    return parse_note(path, text.encode())


def codes(n):
    return {i.code for i in validate(n)}


VALID = """---
id: 20261015100000
title: BM25 Ranking
type: concept
tags: [retrieval]
sources: [external:paper]
created: 2026-10-15
---
# BM25 Ranking

First paragraph is the summary.

## Related

- [[reciprocal-rank-fusion]]: fuses rankings
"""


def test_valid_note_parses_all_fields():
    n = note(VALID)
    assert n.id == "20261015100000"  # YAML int becomes a string
    assert n.title == "BM25 Ranking"
    assert n.type == "concept"
    assert n.tags == ["retrieval"]
    assert n.sources == ["external:paper"]
    assert n.created == "2026-10-15"  # YAML date becomes an ISO string
    assert n.summary == "First paragraph is the summary."
    assert [link.target for link in n.links] == ["reciprocal-rank-fusion"]
    assert n.links[0].section == "Related"
    assert validate(n) == []


def test_quoted_scalars_are_accepted():
    # The legacy regex parser rejected `type: "concept"`; real YAML does not.
    n = note(VALID.replace("type: concept", 'type: "concept"'))
    assert n.type == "concept"
    assert validate(n) == []


def test_legacy_single_source_becomes_a_list():
    n = note(VALID.replace("sources: [external:paper]", "source: project:old:2026-03-23"))
    assert n.sources == ["project:old:2026-03-23"]


def test_legacy_summary_section_is_used():
    n = note(VALID.replace("First paragraph is the summary.", "## Summary\nFrom the section."))
    assert n.summary == "From the section."


def test_crlf_and_bom_are_handled():
    n = parse_note("notes/x.md", ("﻿" + VALID).replace("\n", "\r\n").encode())
    assert n.id == "20261015100000"
    assert n.h1 == "BM25 Ranking"


def test_missing_frontmatter_is_an_error():
    assert codes(note("# Title\n\nBody.")) == {"missing-frontmatter"}


def test_invalid_yaml_is_an_error_not_a_crash():
    n = note("---\ntitle: [unclosed\n---\n# T\n")
    assert codes(n) == {"invalid-yaml"}


def test_unclosed_frontmatter_is_treated_as_body():
    assert split_frontmatter("---\nid: 1\n# no closing line") == (
        None,
        "---\nid: 1\n# no closing line",
    )


def test_missing_required_fields_are_reported():
    n = note("---\ntitle: Only a title\n---\n# Only a title\n\nBody.\n")
    assert {"missing-id", "missing-type", "missing-sources", "missing-created"} <= codes(n)


def test_unknown_type_is_an_error():
    assert "invalid-type" in codes(note(VALID.replace("type: concept", "type: snippet")))


def test_title_mismatch_is_a_warning():
    n = note(VALID.replace("# BM25 Ranking", "# Something Else"))
    assert "title-mismatch" in codes(n)


def test_code_blocks_hide_headings_and_links():
    text = VALID + "\n```markdown\n## Fake\n[[fake-link]]\n```\nSee `[[inline]]` too.\n"
    n = note(text)
    assert "Fake" not in n.sections
    assert [link.target for link in n.links] == ["reciprocal-rank-fusion"]


def test_link_with_heading_and_label():
    n = note(VALID + "\n[[Embeddings#Details|vectors]]\n")
    link = n.links[-1]
    assert (link.target, link.heading) == ("Embeddings", "Details")


def test_hash_changes_only_with_content():
    assert note(VALID).content_hash == note(VALID, path="elsewhere.md").content_hash
    assert note(VALID).content_hash != note(VALID + " ").content_hash
