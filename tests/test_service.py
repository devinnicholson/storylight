import asyncio

from bookforge.config import Settings
from bookforge.domain import (
    BookPageInput,
    GeneratedPagePlan,
    GeneratedStoryPlan,
    InterventionRequest,
    LayerComposition,
    SceneSpecV2,
    StoryCompileRequest,
    StoryTrigger,
    SupportAction,
    VisualLayer,
)
from bookforge.model_client import FakeModelClient
from bookforge.service import BookforgeService


def make_service() -> BookforgeService:
    settings = Settings(model_backend="fake", model_name="fake")
    return BookforgeService(settings, FakeModelClient())


def test_first_long_pause_uses_instant_grapheme_fast_path() -> None:
    response = asyncio.run(
        make_service().select_intervention(
            InterventionRequest(
                session_id="session-1",
                page_id="page-1",
                expected_word="through",
                expected_grapheme="th",
                pause_ms=1_300,
            )
        )
    )

    assert response.source == "fast_path"
    assert response.decision.action is SupportAction.HIGHLIGHT_GRAPHEME
    assert response.decision.target == "th"
    assert response.metrics is None


def test_second_attempt_uses_structured_model_decision() -> None:
    response = asyncio.run(
        make_service().select_intervention(
            InterventionRequest(
                session_id="session-1",
                page_id="page-1",
                expected_word="through",
                expected_grapheme="th",
                pause_ms=1_300,
                attempt_count=1,
            )
        )
    )

    assert response.source == "model"
    assert response.decision.action is SupportAction.HIGHLIGHT_GRAPHEME
    assert response.decision.target == "th"
    assert response.metrics is not None


def test_disallowed_model_action_fails_closed_to_wait() -> None:
    response = asyncio.run(
        make_service().select_intervention(
            InterventionRequest(
                session_id="session-1",
                page_id="page-1",
                expected_word="through",
                pause_ms=1_300,
                attempt_count=1,
                allowed_actions=[SupportAction.WAIT],
            )
        )
    )

    assert response.decision.action is SupportAction.WAIT
    assert response.decision.confidence == 0


def test_story_compiler_returns_versioned_validated_pack() -> None:
    response = asyncio.run(
        make_service().compile_story(
            StoryCompileRequest(
                story_id="moon-gate",
                title="The Moon Gate",
                pages=[
                    BookPageInput(
                        page_id="page-01",
                        text="The small moth went through the red gate.",
                    )
                ],
            )
        )
    )

    assert response.story_pack.schema_version == "2.0"
    assert response.story_pack.pages[0].scene_spec is not None
    assert response.story_pack.pages[0].page_id == "page-01"
    assert response.story_pack.pages[0].source_text == ("The small moth went through the red gate.")
    assert response.story_pack.pages[0].triggers[0].word == "the"
    assert response.metrics.backend == "fake"


def test_story_compiler_anchors_a_valid_trigger_phrase_to_its_final_word() -> None:
    request = StoryCompileRequest(
        story_id="moon-gate",
        title="The Moon Gate",
        pages=[
            BookPageInput(
                page_id="page-01",
                text="The red gate opened, and the red gate glowed.",
            )
        ],
    )
    plan = GeneratedStoryPlan(
        pages=[
            GeneratedPagePlan(
                page_id="page-01",
                scene_summary="A red gate glows.",
                scene_spec=SceneSpecV2(
                    master_prompt="A glowing red paper gate in a cinematic landscape",
                    composition=[
                        LayerComposition(
                            layer_id="gate",
                            center_x=0.7,
                            center_y=0.5,
                            width=0.3,
                            height=0.7,
                            depth=0.7,
                        )
                    ],
                ),
                layers=[
                    VisualLayer(
                        layer_id="gate",
                        kind="prop",
                        prompt="A glowing red paper gate",
                        z_index=1,
                        motion="Reveal with opacity",
                    )
                ],
                triggers=[
                    StoryTrigger(
                        trigger_id="glow-second-gate",
                        word="red gate",
                        occurrence=2,
                        action="glow",
                        target_layer_id="gate",
                        duration_ms=300,
                    )
                ],
                literacy_support=[],
                comprehension=[],
            )
        ]
    )

    normalized = BookforgeService._validate_story_plan(request, plan)

    trigger = normalized.pages[0].triggers[0]
    assert trigger.word == "gate"
    assert trigger.occurrence == 2
