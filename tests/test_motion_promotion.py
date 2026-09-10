from pathlib import Path

import pytest

from storylight.domain import AssetRole, StoryPack
from storylight.motion_promotion import MotionPromotionError, promote_motion_asset

ROOT = Path(__file__).parents[1]


def test_promotes_local_video_without_removing_existing_assets(tmp_path: Path) -> None:
    source_pack = StoryPack.model_validate_json(
        (ROOT / "examples/moon-gate.story-pack.sample.json").read_text()
    )
    video = tmp_path / "motion.mp4"
    video.write_bytes(b"fake-mp4-content")

    promoted = promote_motion_asset(
        source_pack,
        video_path=video,
        asset_root=tmp_path,
        page_id="page-01",
        provider="modal:ltx",
        prompt="Locked camera, subtle moth motion",
        seed=42,
        duration_ms=4000,
        generation_ms=1200,
    )

    assert len(promoted.assets) == len(source_pack.assets) + 1
    motion = promoted.assets[-1]
    assert motion.role is AssetRole.MOTION
    assert motion.local_uri == "motion.mp4"
    assert motion.duration_ms == 4000
    StoryPack.model_validate(promoted.model_dump())


def test_replaces_existing_motion_for_same_page(tmp_path: Path) -> None:
    source_pack = StoryPack.model_validate_json(
        (ROOT / "examples/moon-gate.story-pack.sample.json").read_text()
    )
    first = tmp_path / "first.mp4"
    second = tmp_path / "second.mp4"
    first.write_bytes(b"first-video")
    second.write_bytes(b"second-video")
    kwargs = {
        "asset_root": tmp_path,
        "page_id": "page-01",
        "provider": "modal:ltx",
        "prompt": "subtle motion",
        "seed": 42,
        "duration_ms": 4000,
        "generation_ms": 1200,
    }

    promoted = promote_motion_asset(source_pack, video_path=first, **kwargs)
    replaced = promote_motion_asset(promoted, video_path=second, **kwargs)

    assert sum(asset.role is AssetRole.MOTION for asset in replaced.assets) == 1
    assert replaced.assets[-1].local_uri == "second.mp4"


def test_rejects_video_outside_asset_root(tmp_path: Path) -> None:
    source_pack = StoryPack.model_validate_json(
        (ROOT / "examples/moon-gate.story-pack.sample.json").read_text()
    )
    outside = tmp_path.parent / "outside.mp4"
    outside.write_bytes(b"outside")
    try:
        with pytest.raises(MotionPromotionError, match="inside the asset root"):
            promote_motion_asset(
                source_pack,
                video_path=outside,
                asset_root=tmp_path,
                page_id="page-01",
                provider="modal:ltx",
                prompt="motion",
                seed=1,
                duration_ms=4000,
                generation_ms=1,
            )
    finally:
        outside.unlink()
