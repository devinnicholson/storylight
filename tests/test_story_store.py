import asyncio
import json
from pathlib import Path

import pytest

from bookforge.domain import (
    AssetKind,
    AssetRecord,
    AssetRole,
    AssetState,
    GeneratedPagePlan,
    StoryPack,
    VisualLayer,
)
from bookforge.story_store import StoryPackCorruptError, StoryPackStore


def make_pack(story_id: str = "Moon Gate / Demo") -> StoryPack:
    return StoryPack(
        story_id=story_id,
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
    )


def test_story_pack_store_round_trip_and_private_permissions(tmp_path: Path) -> None:
    store = StoryPackStore(tmp_path / "packs")
    pack = make_pack()

    destination = asyncio.run(store.save(pack))
    loaded = asyncio.run(store.latest())

    assert loaded == pack
    assert destination.name.startswith("moon-gate-demo-")
    assert destination.stat().st_mode & 0o777 == 0o600
    assert store.root.stat().st_mode & 0o777 == 0o700
    assert json.loads(destination.read_text())["story_id"] == pack.story_id


def test_story_pack_store_detects_corrupt_payload(tmp_path: Path) -> None:
    store = StoryPackStore(tmp_path / "packs")
    destination = asyncio.run(store.save(make_pack()))
    destination.write_text("{}", encoding="utf-8")

    with pytest.raises(StoryPackCorruptError, match="checksum"):
        asyncio.run(store.latest())


def test_story_pack_store_finds_only_exact_completed_live_scene(tmp_path: Path) -> None:
    store = StoryPackStore(tmp_path / "packs")
    pack = make_pack("reader-1-deadbeefcafe").model_copy(
        update={
            "visual_style": "paper theater",
            "assets": [
                AssetRecord(
                    asset_id=f"scene-{role.value}",
                    page_id="page-01",
                    layer_id="sky",
                    kind=kind,
                    role=role,
                    provider="fake",
                    prompt="cached scene",
                    seed=17,
                    width=896,
                    height=512,
                    checksum_sha256=character * 64,
                    local_uri=f"/v1/assets/{character * 64}/scene-{role.value}.png",
                    state=AssetState.READY,
                )
                for role, kind, character in (
                    (AssetRole.MASTER, AssetKind.IMAGE, "a"),
                    (AssetRole.DEPTH, AssetKind.DEPTH_MAP, "b"),
                )
            ],
        }
    )
    asyncio.run(store.save(pack))
    restarted = StoryPackStore(store.root)
    asyncio.run(restarted.initialize())

    exact = asyncio.run(
        restarted.find_live_scene(
            text="The moth found the gate.",
            visual_style="paper theater",
            seed=17,
            session_id="reader-1",
        )
    )
    wrong_seed = asyncio.run(
        restarted.find_live_scene(
            text="The moth found the gate.",
            visual_style="paper theater",
            seed=18,
            session_id="reader-1",
        )
    )
    other_session = asyncio.run(
        restarted.find_live_scene(
            text="The moth found the gate.",
            visual_style="paper theater",
            seed=17,
            session_id="reader-elsewhere",
        )
    )

    assert exact == pack
    assert other_session == pack
    assert wrong_seed is None
    assert all("The moth found the gate." not in key for key in restarted._live_scene_index)  # noqa: SLF001


def test_story_pack_store_invalidates_a_corrupt_indexed_live_scene(tmp_path: Path) -> None:
    store = StoryPackStore(tmp_path / "packs")
    pack = make_pack("reader-2-feedfacecafe").model_copy(
        update={
            "assets": [
                AssetRecord(
                    asset_id=f"scene-{role.value}",
                    page_id="page-01",
                    layer_id="sky",
                    kind=kind,
                    role=role,
                    provider="fake",
                    prompt="cached scene",
                    seed=29,
                    width=896,
                    height=512,
                    checksum_sha256=character * 64,
                    local_uri=f"/v1/assets/{character * 64}/scene-{role.value}.png",
                    state=AssetState.READY,
                )
                for role, kind, character in (
                    (AssetRole.MASTER, AssetKind.IMAGE, "c"),
                    (AssetRole.DEPTH, AssetKind.DEPTH_MAP, "d"),
                )
            ]
        }
    )
    destination = asyncio.run(store.save(pack))
    destination.write_text("{}", encoding="utf-8")

    restored = asyncio.run(
        store.find_live_scene(
            text="The moth found the gate.",
            visual_style="paper theater",
            seed=29,
            session_id="reader-2",
        )
    )

    assert restored is None
