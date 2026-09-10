"""Static subjects and refusals from actual retained CPU dependency predictions."""

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from storylight.scene_facts import SceneFactsV2
from storylight.voice_language import extract_graph, graph_from_row

DIRECTORY = Path(__file__).resolve().parent / "fixtures/voice-static-subject"
ROWS = {
    row["id"]: row
    for name in ("actual-parser-rows.json", "additional-parser-rows.json")
    for row in json.loads((DIRECTORY / name).read_bytes())["rows"]
}
PROBES = {row["id"]: row for row in json.loads(
    (DIRECTORY / "standalone-head-rows.json").read_bytes(),
)["probes"]}


def parsed(name):
    row = ROWS[name]
    original = copy.deepcopy(row)
    probe = PROBES[name]["original"]
    assert probe["text"] == row["tokens"][PROBES[name]["root_index"]]["text"]
    result = graph_from_row(row, "watercolor", nominal_head_pos=probe["tokens"][0]["pos"])
    assert row == original
    assert result["render_admitted"] is False
    assert result["requires_fact_review"] is True
    return result


@pytest.mark.parametrize("name,label,color,count", [
    ("balloon", "balloon", "red", 1),
    ("dish_title", "chicken lollipop", None, 1),
    ("dish_lower", "chicken lollipop", None, 1),
    ("balloon_count", "balloons", "red", 2),
    ("dish_count", "chicken lollipops", None, 3),
    ("dish_attribute", "crispy chicken lollipop", None, 1),
    ("animal", "chicken", None, 1),
    ("animal_color", "chicken", "red", 1),
    ("cat", "cat", "black", 1),
])
def test_complete_noun_phrase_preserves_literal_subject_without_inventing_action(
    name, label, color, count,
):
    result = parsed(name)
    assert result["status"] == "draft_ready", result
    facts = SceneFactsV2.model_validate(result["facts"])
    actor, = facts.subjects
    assert (actor.label, actor.color, actor.count, actor.actions) == (label, color, count, ())
    assert not facts.objects and not facts.events and not facts.motions
    assert not facts.relationships and not facts.negatives
    assert facts.setting.label == "unspecified"
    facts.validate_source_grounding(source_text=ROWS[name]["text"])
    assert facts.to_renderer_prompt(source_text=ROWS[name]["text"], visual_style="watercolor") == (
        result["renderer_prompt_preview"]
    )
    assert result["local_omissions"] == []
    assert label in result["renderer_prompt_preview"].lower()
    assert "standing" not in result["renderer_prompt_preview"].lower()
    assert "holding" not in result["renderer_prompt_preview"].lower()


@pytest.mark.parametrize("name", [
    "dish_bare", "coordination", "location", "negative_no", "negative_not",
    "negative_without", "incomplete_chase", "incomplete_copula", "incomplete_negative",
    "name", "named_place", "named_item", "name_modifier", "conflicting_counts", "static_breed",
    "rose", "may", "jordan", "almost", "only", "dangling_and", "dangling_in",
    "colored_coordination", "possessive", "predicate_is",
])
def test_incomplete_predicates_names_negation_and_uncovered_structure_are_not_static_subjects(name):
    assert parsed(name)["facts"] is None


@pytest.mark.parametrize("name,field,value", [
    ("balloon_count", "count", 3), ("balloon", "color", "blue"),
    ("dish_count", "count", 1), ("animal_color", "color", "green"),
    ("dish_title", "actions", ["chasing a mouse"]),
])
def test_static_admission_does_not_license_wrong_counts_color_or_new_actions(name, field, value):
    result = parsed(name)
    changed = copy.deepcopy(result["facts"])
    changed["subjects"][0][field] = value
    with pytest.raises(ValueError, match="source-grounded"):
        SceneFactsV2.model_validate(changed).validate_source_grounding(source_text=ROWS[name]["text"])


def test_service_checks_actual_standalone_head_prediction_without_changing_source():
    class Doc(list):
        def __init__(self, row):
            super().__init__(SimpleNamespace(i=t["i"], text=t["text"], idx=t["offset"],
                                            lemma_=t["lemma"], pos_=t["pos"], dep_=t["dep"])
                             for t in row["tokens"])
            for token, raw in zip(self, row["tokens"], strict=True):
                token.head = self[raw["head"]]
            self.text = row["text"]
            self.ents = [SimpleNamespace(text=e["text"], label_=e["label"],
                                         start=e["start"], end=e["end"]) for e in row["entities"]]

    proof = json.loads((DIRECTORY / "standalone-head-rows.json").read_bytes())
    for name, digest in proof["source_sha256"].items():
        assert hashlib.sha256((DIRECTORY / name).read_bytes()).hexdigest() == digest
    for name in ("dish_title", "balloon", "incomplete_chase"):
        original = ROWS[name]
        head = PROBES[name]["original"]
        calls = []

        def nlp(text, *, original=original, head=head, calls=calls):
            calls.append(text)
            assert text in {original["text"], head["text"]}
            return Doc(original if text == original["text"] else head)

        result = extract_graph(original["text"], "watercolor", nlp=nlp)
        assert calls == [original["text"], head["text"]]
        assert result["facts"] == parsed(name)["facts"]
    assert PROBES["incomplete_chase"]["original"]["tokens"][0]["pos"] == "VERB"
    # A name in the original source cannot be recovered through the style channel.
    assert graph_from_row(ROWS["named_place"], "Paris watercolor", nominal_head_pos="PROPN")[
        "facts"
    ] is None
