import asyncio
import hashlib
import json
from pathlib import Path

import pytest

from storylight.config import Settings
from storylight.domain import (
    AssetKind,
    AssetRecord,
    AssetState,
    GeneratedPagePlan,
    StoryPack,
    VisualLayer,
)
from storylight.handoff_bundle import HandoffBundleError, build_handoff_bundle
from storylight.pack_installer import install


def _pack(checksum: str, local_uri: str = "source/sky.png") -> StoryPack:
    return StoryPack(
        story_id="portable-book",
        title="Portable Book",
        reading_level=2,
        visual_style="paper theater",
        compiler_model="fixture",
        pages=[
            GeneratedPagePlan(
                page_id="page-01",
                source_text="The sky glowed.",
                scene_summary="A glowing sky.",
                layers=[
                    VisualLayer(
                        layer_id="sky",
                        kind="background",
                        prompt="Glowing sky",
                        z_index=0,
                        motion="Slow drift",
                    )
                ],
                triggers=[],
                literacy_support=[],
                comprehension=[],
            )
        ],
        assets=[
            AssetRecord(
                asset_id="page-01 master",
                page_id="page-01",
                layer_id="sky",
                kind=AssetKind.IMAGE,
                provider="fixture",
                prompt="Glowing sky",
                seed=1,
                width=16,
                height=9,
                checksum_sha256=checksum,
                local_uri=local_uri,
                state=AssetState.READY,
            )
        ],
    )


def test_builds_portable_bundle_that_reinstalls_offline(tmp_path: Path) -> None:
    content = b"portable-image"
    checksum = hashlib.sha256(content).hexdigest()
    source = tmp_path / "source"
    source.mkdir()
    (source / "sky.png").write_bytes(content)
    pack_path = tmp_path / "source-pack.json"
    pack_path.write_text(_pack(checksum).model_dump_json(), encoding="utf-8")
    output = tmp_path / "handoff"

    result = build_handoff_bundle(pack_path, asset_root=tmp_path, output_dir=output)

    bundled_pack_path = output / "portable-book.story-pack.json"
    bundled = StoryPack.model_validate_json(bundled_pack_path.read_text())
    checksums = json.loads((output / "checksums.json").read_text())
    assert result["assets"] == 1
    assert bundled.assets[0].local_uri == "assets/page-01-master.png"
    assert checksums["assets"][0]["sha256"] == checksum
    assert hashlib.sha256(bundled_pack_path.read_bytes()).hexdigest() == result["story_pack_sha256"]

    settings = Settings(data_dir=tmp_path / "data", cache_dir=tmp_path / "cache")
    installed = asyncio.run(install(bundled_pack_path, output, settings))
    assert installed["ready_assets"] == 1


def test_fails_closed_on_checksum_mismatch_or_existing_output(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "sky.png").write_bytes(b"wrong")
    pack_path = tmp_path / "source-pack.json"
    pack_path.write_text(_pack("0" * 64).model_dump_json(), encoding="utf-8")

    with pytest.raises(HandoffBundleError, match="checksum mismatch"):
        build_handoff_bundle(pack_path, asset_root=tmp_path, output_dir=tmp_path / "handoff")

    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(HandoffBundleError, match="already exists"):
        build_handoff_bundle(pack_path, asset_root=tmp_path, output_dir=output)


def test_rejects_asset_source_outside_root(tmp_path: Path) -> None:
    pack_path = tmp_path / "source-pack.json"
    pack_path.write_text(_pack("0" * 64, "../sky.png").model_dump_json(), encoding="utf-8")

    with pytest.raises(HandoffBundleError, match="escapes"):
        build_handoff_bundle(pack_path, asset_root=tmp_path, output_dir=tmp_path / "handoff")
