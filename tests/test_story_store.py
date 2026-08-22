import asyncio
import json
from pathlib import Path

import pytest

from bookforge.domain import GeneratedPagePlan, StoryPack, VisualLayer
from bookforge.story_store import StoryPackCorruptError, StoryPackNotFoundError, StoryPackStore


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


def test_story_pack_store_reports_missing_latest(tmp_path: Path) -> None:
    store = StoryPackStore(tmp_path / "packs")
    asyncio.run(store.initialize())

    with pytest.raises(StoryPackNotFoundError, match="No compiled"):
        asyncio.run(store.latest())


def test_story_pack_store_detects_corrupt_payload(tmp_path: Path) -> None:
    store = StoryPackStore(tmp_path / "packs")
    destination = asyncio.run(store.save(make_pack()))
    destination.write_text("{}", encoding="utf-8")

    with pytest.raises(StoryPackCorruptError, match="checksum"):
        asyncio.run(store.latest())
