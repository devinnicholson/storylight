"""Retained actual local parser rows for nominal absence and nearby scopes."""

import asyncio
import copy
import json
from pathlib import Path

import pytest

from storylight.reviewed_description import PARSER_REVISION, review_description
from storylight.scene_facts import SceneFactsV2
from storylight.voice_language import graph_from_row

ROWS = {row["id"]: row for row in json.loads(
    (Path(__file__).parent / "fixtures/voice-without-object-rows.json").read_bytes()
)["rows"]}


@pytest.mark.parametrize("name,action,absent", [
    ("exact", "walks", "notebooks"), ("bare", "runs", "hats"),
    ("compound", "walks", "paper cups"), ("color", "walks", "red ribbons"),
])
def test_nominal_absence_preserves_action_and_never_creates_positive_object(name, action, absent):
    row = copy.deepcopy(ROWS[name])
    result = graph_from_row(row, "watercolor")
    assert row == ROWS[name]
    assert result["status"] == "draft_ready", result
    assert not result["render_admitted"] and result["requires_fact_review"]
    facts = SceneFactsV2.model_validate(result["facts"])
    assert facts.subjects[0].actions == (action,)
    assert not facts.objects and not facts.relationships
    assert [(n.kind, n.value, n.target) for n in facts.negatives] == [
        ("additional_object", absent, None),
    ]
    assert f"Constraints: no {absent}" in result["renderer_prompt_preview"]
    assert "Object:" not in result["renderer_prompt_preview"]
    facts.validate_source_grounding(source_text=row["text"])


@pytest.mark.parametrize("name", [
    "nested", "verb", "named", "possessive", "partial", "and", "or", "count",
    "qualifier", "multi", "attachment", "negative",
])
def test_uncertain_absence_scope_is_not_silently_projected_as_presence(name):
    result = graph_from_row(ROWS[name], "watercolor")
    assert result["status"] == "needs_review", result
    assert result["facts"] is None
    assert not result["render_admitted"]


def test_with_near_neighbor_keeps_positive_object():
    result = graph_from_row(ROWS["positive"], "watercolor")
    assert result["status"] == "draft_ready"
    assert [o["label"] for o in result["facts"]["objects"]] == ["hats"]
    assert not result["facts"]["negatives"]


def test_exact_demo_source_reaches_reviewed_bridge_with_typed_negative(monkeypatch):
    calls = []

    async def request(payload, socket):
        calls.append(payload)
        assert socket == Path("/tmp/unused-language.sock")
        assert set(payload) == {"text", "visual_style"}  # Full dependency path.
        return json.dumps(graph_from_row(ROWS["exact"], payload["visual_style"]))

    monkeypatch.setattr("storylight.reviewed_description._parser_request", request)
    result = asyncio.run(review_description(
        ROWS["exact"]["text"], "watercolor", 42, parser_socket=Path("/tmp/unused-language.sock"),
    ))
    assert len(calls) == 1
    assert result.result.model_revision == PARSER_REVISION
    assert result.requires_fact_review and result.visual_fact_digest
    assert not result.result.plan.scene_facts.objects
    assert result.result.plan.scene_facts.negatives[0].value == "notebooks"
