import pytest

from storylight.privacy_policy import printed_source_payload_candidates, proper_name_candidates
from storylight.scene_facts import (
    SceneFactsPrivacyError,
    SceneFactsV2,
    SceneSettingFact,
    SceneSubjectFact,
)


@pytest.mark.parametrize(
    "name,source",
    [
        ("elena", "In a forest, {name} lifts a lantern."),
        ("elena", "In a forest, a fox watches as {name} lifts a lantern."),
        ("ｅｌｅｎａ", "In a forest, {name} lifts a lantern."),
    ],
)
def test_graph_rejects_unmarked_names_after_local_clause_boundaries(name, source):
    facts = SceneFactsV2(
        setting=SceneSettingFact(label="forest"),
        subjects=(SceneSubjectFact(ref="a", label=name),),
    )
    with pytest.raises(SceneFactsPrivacyError, match="proper-name"):
        facts.to_renderer_prompt(source_text=source.format(name=name))


@pytest.mark.parametrize(
    "printed",
    [
        "a sign reading starlight hangs beside a fox",
        "a sign reading ‘starlight’ hangs beside a fox",
        "a starlight–engraved sign hangs beside a fox",
        "starlight has been written on a sign beside a fox",
    ],
)
def test_graph_rejects_gerund_adjectival_and_perfect_passive_payloads(printed):
    facts = SceneFactsV2(
        setting=SceneSettingFact(label="forest"),
        subjects=(SceneSubjectFact(ref="a", label="starlight"),),
    )
    with pytest.raises(SceneFactsPrivacyError, match="printed source payload"):
        facts.to_renderer_prompt(source_text=f"In a forest, {printed}.")


def test_reading_action_does_not_turn_story_objects_into_printed_payloads():
    assert not printed_source_payload_candidates(
        "In a forest, a fox reading a book sits beside a lantern."
    )


@pytest.mark.parametrize("actor", ["elena the keeper", "a child éléna"])
def test_graph_rejects_lowercase_names_adjacent_to_human_roles(actor):
    facts = SceneFactsV2(
        setting=SceneSettingFact(label="forest"),
        subjects=(SceneSubjectFact(ref="a", label=actor),),
    )
    with pytest.raises(SceneFactsPrivacyError, match="proper-name"):
        facts.to_renderer_prompt(source_text=f"In a forest, {actor} lifts a lantern.")


@pytest.mark.parametrize("source", ["In a forest, a keeper in a cave lifts a lantern."])
def test_role_actions_and_prepositions_are_not_personal_names(source):
    assert not proper_name_candidates(source)


def test_asr_breed_capitalization_keeps_explicit_names_private():
    source = "The white golden retriever and the Merle Aussie are playing in the field."
    assert not proper_name_candidates(source)
    cases = (
        ("A dog named the Merle Aussie runs.", "aussie"),
        ("A dog called the Golden Retriever runs.", "retriever"),
        ("The teacher is named the Border Collie.", "collie"),
        ("A dog known as the Merle Aussie runs.", "merle"),
        ("A dog known  as the Merle Aussie runs.", "aussie"),
        ("A dog named: the Merle Aussie runs.", "merle"),
        ('A dog goes by "the Merle Aussie".', "aussie"),
        ("The Merle Aussie named Alice is playing in the field.", "alice"),
    )
    for source, name in cases:
        assert (name,) in proper_name_candidates(source)
        facts = SceneFactsV2(
            setting=SceneSettingFact(label="unspecified"),
            subjects=(SceneSubjectFact(ref="a", label=name),),
        )
        with pytest.raises(SceneFactsPrivacyError, match="proper-name"):
            facts.to_renderer_prompt(source_text=source)


@pytest.mark.parametrize("payload", ["password hunter2", "credential abracadabra"])
def test_adapter_and_wire_compiler_reject_sensitive_noun_payloads(payload):
    from storylight.live_scene_facts import adapt_live_scene_facts

    source = f"In a forest, a fox lifts a {payload}. A rainbow appears."
    result = adapt_live_scene_facts(
        {"SETTING": "forest", "ACTOR": "fox", "ACTION": f"lifts {payload}", "MAGIC": "rainbow"},
        source_text=source,
    )
    assert result.facts is None
    assert result.refusal.value == "privacy"
    facts = SceneFactsV2(
        setting=SceneSettingFact(label="forest"),
        subjects=(SceneSubjectFact(ref="a", label=payload),),
    )
    restored = SceneFactsV2.from_wire(facts.to_wire())
    with pytest.raises(SceneFactsPrivacyError, match="protected sensitive"):
        restored.to_renderer_prompt(source_text=source)
