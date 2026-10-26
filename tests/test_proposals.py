from datetime import datetime

import pytest

from zk.notes import parse_note, validate
from zk.proposals import InvalidProposal, Proposal, render, slugify, verdict

NOW = datetime(2026, 10, 22, 19, 5, 1)


def good(**changes) -> Proposal:
    fields = dict(
        title="Reciprocal Rank Fusion",
        type="pattern",
        summary="RRF merges ranked lists by summing 1 / (k + rank).",
        body="It needs only ranks, so scores never have to be comparable.",
        tags=["retrieval", "#hybrid-search"],
        sources=["conversation:2026-10-22"],
        links={"bm25-ranking": "one of the fused lists"},
    )
    fields.update(changes)
    return Proposal(**fields)


def test_rendered_note_has_the_schema_and_passes_check():
    name, text = render(good(), NOW)
    assert name == "reciprocal-rank-fusion.md"
    note = parse_note(f"notes/{name}", text.encode())
    assert validate(note) == []
    assert note.id == "20261022190501" and note.type == "pattern"
    assert note.tags == ["retrieval", "hybrid-search"]
    assert note.summary.startswith("RRF merges")
    assert [link.target for link in note.links] == ["bm25-ranking"]
    assert "## Details" in text and "- [[bm25-ranking]]: one of the fused lists" in text


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"type": "idea"}, "type must be one of"),
        ({"summary": "  "}, "summary is empty"),
        ({"sources": []}, "sources is empty"),
        ({"title": ""}, "title is empty"),
        ({"summary": "# Heading\nText"}, "plain sentences"),
    ],
)
def test_invalid_proposals_are_refused(changes, message):
    with pytest.raises(InvalidProposal, match=message):
        render(good(**changes), NOW)


def test_slugify_and_verdict():
    assert slugify("Héllo, World: RRF & BM25!") == "hello-world-rrf-bm25"
    assert slugify("???") == "note"
    assert (verdict(0.95), verdict(0.8), verdict(0.3)) == ("likely-duplicate", "related", "new")
