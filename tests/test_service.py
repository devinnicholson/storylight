import asyncio
from pathlib import Path

import pytest
from pydantic import ValidationError

from bookforge.config import Settings
from bookforge.domain import (
    AssetKind,
    AssetRecord,
    AssetState,
    BookPageInput,
    GeneratedPagePlan,
    GeneratedStoryPlan,
    InterventionRequest,
    LayerComposition,
    SceneSpecV2,
    StoryCompileRequest,
    StoryPack,
    StoryTrigger,
    SupportAction,
    VisualLayer,
)
from bookforge.model_client import FakeModelClient
from bookforge.service import BookforgeService


def make_service() -> BookforgeService:
    settings = Settings(model_backend="fake", model_name="fake")
    return BookforgeService(settings, FakeModelClient())


def test_short_first_pause_uses_zero_model_fast_path() -> None:
    response = asyncio.run(
        make_service().select_intervention(
            InterventionRequest(
                session_id="session-1",
                page_id="page-1",
                expected_word="through",
                expected_grapheme="th",
                pause_ms=420,
            )
        )
    )

    assert response.source == "fast_path"
    assert response.decision.action is SupportAction.WAIT
    assert response.metrics is None


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
    assert response.story_pack.pages[0].source_text == (
        "The small moth went through the red gate."
    )
    assert response.story_pack.pages[0].triggers[0].word == "the"
    assert response.metrics.backend == "fake"


def test_story_authoring_schema_requires_scene_spec_even_with_legacy_playback_support() -> None:
    schema = GeneratedStoryPlan.model_json_schema()

    assert "scene_spec" in schema["$defs"]["GeneratedPagePlan"]["required"]


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


def test_empty_action_allowlist_is_rejected() -> None:
    with pytest.raises(ValueError, match="allowed_actions"):
        InterventionRequest(
            session_id="session-1",
            page_id="page-1",
            expected_word="through",
            allowed_actions=[],
        )


def test_ready_asset_requires_location_and_checksum() -> None:
    with pytest.raises(ValidationError, match="checksum_sha256"):
        AssetRecord(
            asset_id="asset-sky",
            page_id="page-01",
            layer_id="sky",
            kind=AssetKind.PROCEDURAL,
            provider="fixture",
            prompt="Moonlit paper sky",
            seed=7,
            width=1920,
            height=1080,
            state=AssetState.READY,
        )


def test_story_pack_rejects_asset_for_missing_layer() -> None:
    with pytest.raises(ValidationError, match="missing layer"):
        StoryPack(
            story_id="moon-gate",
            title="The Moon Gate",
            reading_level=2,
            visual_style="paper theater",
            compiler_model="fixture",
            pages=[
                GeneratedPagePlan(
                    page_id="page-01",
                    source_text="The moth found the gate.",
                    scene_summary="A moonlit gate.",
                    layers=[
                        VisualLayer(
                            layer_id="sky",
                            kind="background",
                            prompt="Moonlit paper sky",
                            z_index=0,
                            motion="Slow parallax",
                        )
                    ],
                    triggers=[],
                    literacy_support=[],
                    comprehension=[],
                )
            ],
            assets=[
                AssetRecord(
                    asset_id="asset-missing",
                    page_id="page-01",
                    layer_id="not-a-layer",
                    kind=AssetKind.PROCEDURAL,
                    provider="fixture",
                    prompt="Missing layer",
                    seed=7,
                    width=1920,
                    height=1080,
                )
            ],
        )


def test_story_pack_rejects_trigger_for_missing_layer() -> None:
    with pytest.raises(ValidationError, match="Trigger.*missing layer"):
        StoryPack(
            story_id="moon-gate",
            title="The Moon Gate",
            reading_level=2,
            visual_style="paper theater",
            compiler_model="fixture",
            pages=[
                GeneratedPagePlan(
                    page_id="page-01",
                    source_text="The moth found the gate.",
                    scene_summary="A moonlit gate.",
                    layers=[
                        VisualLayer(
                            layer_id="sky",
                            kind="background",
                            prompt="Moonlit paper sky",
                            z_index=0,
                            motion="Slow parallax",
                        )
                    ],
                    triggers=[
                        {
                            "trigger_id": "bad-trigger",
                            "word": "moth",
                            "action": "reveal",
                            "target_layer_id": "missing",
                            "duration_ms": 300,
                        }
                    ],
                    literacy_support=[],
                    comprehension=[],
                )
            ],
        )


def test_projector_fixture_is_a_valid_ready_story_pack() -> None:
    fixture = (
        Path(__file__).parents[1]
        / "src"
        / "bookforge"
        / "static"
        / "moon-gate.story-pack.json"
    )
    pack = StoryPack.model_validate_json(fixture.read_text())

    assert pack.schema_version == "1.1"
    assert len(pack.assets) == 5
    assert all(asset.state is AssetState.READY for asset in pack.assets)
