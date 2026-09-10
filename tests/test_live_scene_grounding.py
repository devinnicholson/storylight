from types import SimpleNamespace

import pytest

from storylight.live_scene_grounding import LiveSceneGroundingError, validate_wire_grounding


def wire(subject, action, supporting="none", background="neutral background"):
    return SimpleNamespace(
        focus=SimpleNamespace(subject=subject, action=action),
        magic=SimpleNamespace(prompt=supporting),
        background_prompt=background,
    )


def test_source_bound_visual_selections_and_paraphrases():
    source = "A quick brown fox jumps over a lazy dog."
    for subject, action, dog in (
        ("quick brown fox", "jumps over a lazy dog", "lazy dog"),
        ("fast brown fox", "leaping over the sluggish canine", "sluggish canine"),
    ):
        validate_wire_grounding(wire(subject, action, dog), source_text=source)
    validate_wire_grounding(
        wire("pink fox", "jumped over the river stream", "river stream", "river stream"),
        source_text="The pink fox jumped over the river stream.",
    )
    validate_wire_grounding(
        wire("brown fox", "jumps over a dog", "white rabbit carries a red umbrella"),
        source_text="A brown fox jumps over a dog. A white rabbit carries a red umbrella.",
    )


@pytest.mark.parametrize(
    "candidate,source",
    [
        (
            wire("golden retriever puppy", "leaping", "yellow dandelion", "throw lazy dog"),
            "A quick brown box. Don’t throw a little lazy dog.",
        ),
        (
            wire("brown fox", "jumps over a dog", "yellow dandelion"),
            "A brown fox jumps over a dog.",
        ),
        (wire("red fox", "jumps over a dog"), "A brown fox jumps over a dog."),
        (wire("fox", "jumps over a dog"), "A brown fox jumps over a dog."),
        (wire("brown fox", "runs over a dog"), "A brown fox jumps over a dog."),
        (wire("brown fox", "jumps under a dog"), "A brown fox jumps over a dog."),
        (wire("dog", "jumps over a brown fox"), "A brown fox jumps over a dog."),
        (wire("brown fox", "jumps over a dog"), "A brown fox does not jump over a dog."),
        (wire("brown fox", "jumps over a dog"), "A brown fox doesn't jump over a dog."),
        (wire("brown fox", "runs"), "A brown fox jumps. A pink fox runs."),
    ],
)
def test_refuses_unbound_subject_attribute_action_relationship_and_negation(candidate, source):
    with pytest.raises(LiveSceneGroundingError):
        validate_wire_grounding(candidate, source_text=source)


@pytest.mark.parametrize(
    "candidate, source",
    [
        (wire("three brown foxes", "jump"), "Two brown foxes jump. Three pink foxes run."),
        (wire("three brown foxes", "jump"), "Two brown foxes jump. Three brown foxes run."),
        (wire("large brown fox", "jumps"), "A small brown fox jumps. A large pink fox runs."),
        (
            wire("brown fox", "jump over a dog"),
            "A brown fox does not ever under any circumstances jump over a dog.",
        ),
        (wire("brown fox", "jump over a dog"), "Two brown foxes jump over a dog."),
        (wire("brown fox", "jumps over a dog"), "A brown fox jumps over two dogs."),
        (wire("brown fox", "jumps over a white dog"), "A brown fox jumps over a pink dog."),
        (wire("brown fox", ""), "A brown fox jumps over a dog."),
        (wire("brown fox", "jumps"), "A brown fox jumps. A white rabbit carries a red umbrella."),
        (
            wire("three ducks", "stand beside a red boat"),
            "Two foxes and three ducks stand beside a red boat.",
        ),
    ],
)
def test_preserves_bound_counts_attributes_actors_and_negation(candidate, source):
    with pytest.raises(LiveSceneGroundingError):
        validate_wire_grounding(candidate, source_text=source)


def test_explicit_absence_constraint_is_retained_and_not_turned_into_an_object():
    source = "A brown fox stands beside a stream. No dogs."
    validate_wire_grounding(
        wire("brown fox", "stands beside a stream", "no dogs"), source_text=source
    )
    with pytest.raises(LiveSceneGroundingError):
        validate_wire_grounding(wire("brown fox", "stands beside a stream"), source_text=source)
    with pytest.raises(LiveSceneGroundingError):
        validate_wire_grounding(
            wire("brown fox", "stands beside a stream", "no cats"), source_text=source
        )


def test_contradictory_presence_and_absence_cannot_produce_a_scene():
    with pytest.raises(LiveSceneGroundingError):
        validate_wire_grounding(
            wire("brown dog", "stands", "no dogs"), source_text="A brown dog stands. No dogs."
        )
