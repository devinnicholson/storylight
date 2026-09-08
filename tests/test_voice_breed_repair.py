"""Breed consequences from retained actual spaCy rows; no parser/model/network at test time."""

import asyncio
import copy
import hashlib
import json
from pathlib import Path

import pytest

from bookforge.reviewed_description import review_description
from bookforge.scene_facts import SceneFactsV2
from bookforge.voice_language import graph_from_row

DIRECTORY = Path(__file__).resolve().parents[1] / "benchmarks/voice-retriever-2026-09-08"
RAW = (DIRECTORY / "breed-adversarial-rows.json").read_bytes()
ROWS = {row["id"]: row for row in json.loads(RAW)["rows"]}


def checked(name):
    row = ROWS[name]
    before = copy.deepcopy(row)
    result = graph_from_row(row, "watercolor")
    assert row == before  # Retained parser predictions are evidence, not mutable working state.
    assert result["render_admitted"] is False
    assert result["requires_fact_review"] is True
    if result["facts"] is not None:
        facts = SceneFactsV2.model_validate(result["facts"])
        facts.validate_source_grounding(source_text=row["text"])
        assert result["renderer_prompt_preview"] == facts.to_renderer_prompt(
            source_text=row["text"], visual_style="watercolor",
        )
    return result


def test_actual_jetson_failure_reaches_review_with_literal_breed_and_private_place(monkeypatch):
    assert hashlib.sha256(RAW).hexdigest() == (
        "573e462ef9ed2f1fecdfd5bff547733a0952069681243aa7eba1007a0b458902"
    )
    original = json.loads((DIRECTORY / "parser.json").read_bytes())
    assert {**ROWS["original"], "id": "request"} == original["row"]
    result = checked("original")
    assert result["status"] == "omission_review"
    actor, = result["facts"]["subjects"]
    assert actor["label"] == "english cream golden retriever"
    assert actor["actions"] == ["ran down cobblestone path in city"]
    assert actor["color"] is None  # Golden is part of the breed, not an inferred coat color.
    assert result["facts"]["objects"][0]["label"] == "cobblestone path"
    assert result["facts"]["setting"]["label"] == "city"
    assert [o["local_text"] for o in result["local_omissions"]] == ["Paris"]
    assert "paris" not in result["renderer_prompt_preview"].lower()

    calls = []

    async def parser(payload, socket):
        calls.append(payload)
        assert payload == {"text": ROWS["original"]["text"], "visual_style": "watercolor"}
        assert socket == Path("/tmp/unused-breed-parser.sock")
        return json.dumps(result).encode()

    monkeypatch.setattr("bookforge.reviewed_description._parser_request", parser)
    reviewed = asyncio.run(review_description(
        ROWS["original"]["text"], "watercolor", 42,
        parser_socket=Path("/tmp/unused-breed-parser.sock"),
    ))
    assert len(calls) == 1
    assert not reviewed.is_confirmed(False, reviewed.visual_fact_digest)
    assert not reviewed.is_confirmed(True, "0" * 64)
    assert reviewed.is_confirmed(True, reviewed.visual_fact_digest)
    page = reviewed.result.plan.to_page(
        source_text=ROWS["original"]["text"], visual_style="watercolor", seed=42,
    )
    assert page.scene_spec.master_prompt == result["renderer_prompt_preview"]
    assert "Paris" in page.source_text  # Original stays local while renderer prompt omits it.
    assert graph_from_row(ROWS["original"], "Paris watercolor")["facts"] is None


def test_counts_coat_color_action_particle_and_absence_survive_actual_rows():
    plural = checked("plural")["facts"]
    assert len(plural["subjects"]) == 1
    assert plural["subjects"][0]["count"] == 2
    assert plural["subjects"][0]["label"] == "english cream golden retrievers"
    assert plural["subjects"][0]["actions"] == ["run down cobblestone path"]
    black = checked("black_breed")["facts"]["subjects"][0]
    assert (black["label"], black["color"], black["count"]) == (
        "english cream golden retriever", "black", 1,
    )
    chase = checked("counted_chase")["facts"]
    assert (chase["subjects"][0]["label"], chase["subjects"][0]["count"]) == (
        "cream golden retrievers", 3,
    )
    assert chase["subjects"][0]["actions"] == ["chase two white cats"]
    assert (chase["objects"][0]["label"], chase["objects"][0]["color"],
            chase["objects"][0]["count"]) == ("cats", "white", 2)
    absence = checked("absence")
    assert absence["facts"]["negatives"] == [
        {"kind": "additional_object", "value": "cats", "target": None},
    ]
    assert "Constraints: no cats" in absence["renderer_prompt_preview"]


@pytest.mark.parametrize("name", [
    "coordination", "counted_coordination", "split_nouns", "near_match",
    "negative", "named_actor", "object_near_match", "color",
])
def test_ambiguous_actors_names_negation_and_unparsed_verbs_do_not_become_one_breed(name):
    # These actual predictions are unresolved. A breed repair must not turn
    # neighboring nouns, another actor, or a negative event into one positive dog.
    assert checked(name)["facts"] is None


@pytest.mark.parametrize("name,private_name", [
    ("named_target", "Alice"), ("plural_place", "Paris"),
])
def test_other_names_remain_explicit_local_omissions(name, private_name):
    result = checked(name)
    assert result["status"] == "omission_review"
    assert private_name in [o["local_text"] for o in result["local_omissions"]]
    assert private_name.lower() not in result["renderer_prompt_preview"].lower()
    assert private_name.lower() not in json.dumps(result["facts"]).lower()


@pytest.mark.parametrize("name,collection,field,value", [
    ("original", "subjects", "count", 2),
    ("black_breed", "subjects", "color", "white"),
    ("counted_chase", "subjects", "count", 2),
    ("counted_chase", "objects", "count", 3),
    ("counted_chase", "subjects", "color", "white"),
])
def test_repaired_source_does_not_license_wrong_counts_or_actor_color(
    name, collection, field, value,
):
    facts = checked(name)["facts"]
    facts[collection][0][field] = value
    with pytest.raises(ValueError, match="source-grounded"):
        SceneFactsV2.model_validate(facts).validate_source_grounding(source_text=ROWS[name]["text"])
