"""Actual spaCy predictions for source-preserving copular location continuations."""

import copy
import json
from pathlib import Path

import pytest

from bookforge.scene_facts import SceneFactsV2
from bookforge.voice_dependencies import copular_location_continuation
from bookforge.voice_language import graph_from_row

ROWS = {r["id"]: r for r in json.loads((Path(__file__).parent / "fixtures"
                                      / "voice-copular-location-rows.json").read_bytes())["rows"]}


@pytest.mark.parametrize("name,subject,objects", [
    ("exact", "cow", {"goats", "mountains"}),
    ("horse", "horse", {"sheep", "trees"}),
    ("counts", "cows", {"goats"}),
    ("colors", "cow", {"goats", "mountains"}),
    ("inside", "cat", {"flowers", "trees"}),
    ("nested", "cow", {"goats", "river"}),
])
def test_actual_location_rows_retain_whole_clause_and_explicit_entities(name, subject, objects):
    original = copy.deepcopy(ROWS[name])
    result = graph_from_row(original, "rich watercolor")
    assert original == ROWS[name]
    assert result["status"] == "draft_ready", result
    assert result["render_admitted"] is False and result["requires_fact_review"] is True
    facts = SceneFactsV2.model_validate(result["facts"])
    assert {s.label for s in facts.subjects} == {subject}
    assert {o.label for o in facts.objects} == objects
    continuation = copular_location_continuation(original)
    assert continuation["phrase"] in original["text"]
    assert facts.subjects[0].actions[-1] == continuation["phrase"].lower()
    assert not facts.relationships and not facts.salience  # No guessed location of a conjunct.
    facts.validate_source_grounding(source_text=original["text"])
    assert facts.to_renderer_prompt(
        source_text=original["text"], visual_style="rich watercolor",
    ) == result["renderer_prompt_preview"]
    assert continuation["phrase"].lower() in result["renderer_prompt_preview"]
    assert result["local_omissions"] == []
    if name == "counts":
        assert (facts.subjects[0].count, facts.subjects[0].color) == (2, "brown")
        assert facts.objects[0].count == 3
    if name == "colors":
        assert {o.label: o.color for o in facts.objects} == {"goats": "red", "mountains": "blue"}


@pytest.mark.parametrize("name", [
    "negative", "no_goats", "named", "unrelated", "clipped", "or", "owner", "only",
    "zero", "long", "other_clause", "compound_relation", "time_context",
])
def test_ambiguous_private_negative_or_unbounded_tail_refuses_without_dropping_words(name):
    result = graph_from_row(ROWS[name], "rich watercolor")
    assert result["status"] == "needs_review", result
    assert result["facts"] is None


def test_source_boundary_rejects_changed_actor_count_color_or_literal_context():
    row = ROWS["counts"]
    result = graph_from_row(row, "rich watercolor")
    changes = [("count", 1), ("color", "white"), ("label", "horses"),
               ("actions", ["were in big pasture", "with three wolves in the background"])]
    for field, value in changes:
        altered = copy.deepcopy(result["facts"])
        altered["subjects"][0][field] = value
        with pytest.raises(ValueError, match="source-grounded"):
            SceneFactsV2.model_validate(altered).validate_source_grounding(source_text=row["text"])
    altered = copy.deepcopy(graph_from_row(ROWS["exact"], "rich watercolor")["facts"])
    altered["subjects"][0]["actions"][1] = "with goats and mountains in the foreground"
    with pytest.raises(ValueError, match="source-grounded"):
        SceneFactsV2.model_validate(altered).validate_source_grounding(source_text=ROWS["exact"]["text"])
