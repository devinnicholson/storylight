import asyncio
import hashlib
from pathlib import Path

import pytest

from bookforge.asset_cache import AssetCache, AssetCacheError
from bookforge.config import Settings
from bookforge.domain import (
    AssetKind,
    AssetRecord,
    AssetState,
    GeneratedPagePlan,
    StoryPack,
    VisualLayer,
)
from bookforge.pack_installer import install
from bookforge.story_store import StoryPackStore


def make_pack(checksum: str) -> StoryPack:
    return StoryPack(
        story_id="offline-book",
        title="Offline Book",
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
                        prompt="Glowing paper sky",
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
                asset_id="sky image",
                page_id="page-01",
                layer_id="sky",
                kind=AssetKind.IMAGE,
                provider="fixture",
                prompt="Glowing paper sky",
                seed=1,
                width=16,
                height=9,
                checksum_sha256=checksum,
                local_uri="sky.png",
                state=AssetState.READY,
            )
        ],
    )


def test_offline_pack_installer_verifies_caches_and_rewrites_assets(tmp_path: Path) -> None:
    media = b"verified-image-bytes"
    checksum = hashlib.sha256(media).hexdigest()
    source = tmp_path / "source"
    source.mkdir()
    (source / "sky.png").write_bytes(media)
    story_pack = source / "story-pack.json"
    story_pack.write_text(make_pack(checksum).model_dump_json(), encoding="utf-8")
    settings = Settings(data_dir=tmp_path / "data", cache_dir=tmp_path / "cache")

    result = asyncio.run(install(story_pack, source, settings))
    installed = asyncio.run(StoryPackStore(settings.data_dir / "story-packs").latest())
    local_uri = installed.assets[0].local_uri

    assert result["ready_assets"] == 1
    assert local_uri == f"/v1/assets/{checksum}/sky-image.png"
    cached = AssetCache(settings.cache_dir / "assets").resolve(checksum, "sky-image.png")
    assert cached.read_bytes() == media
    assert cached.stat().st_mode & 0o777 == 0o600


def test_offline_pack_installer_rejects_checksum_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "sky.png").write_bytes(b"wrong")
    cache = AssetCache(tmp_path / "cache")

    with pytest.raises(AssetCacheError, match="checksum"):
        asyncio.run(cache.install_pack(make_pack("0" * 64), source))


def test_asset_cache_rejects_path_traversal(tmp_path: Path) -> None:
    cache = AssetCache(tmp_path)

    with pytest.raises(AssetCacheError, match="Invalid"):
        cache.resolve("0" * 64, "../secret")


def test_generated_depth_jpeg_is_checksum_addressed_and_resolvable(tmp_path: Path) -> None:
    cache = AssetCache(tmp_path / "cache")
    content = b"\xff\xd8depth-map\xff\xd9"

    checksum, uri = asyncio.run(
        cache.store_generated(
            asset_id="scene-depth",
            kind=AssetKind.DEPTH_MAP,
            content=content,
            suffix=".jpg",
        )
    )

    assert uri == f"/v1/assets/{checksum}/scene-depth.jpg"
    assert cache.resolve(checksum, "scene-depth.jpg").read_bytes() == content


@pytest.mark.parametrize("local_uri", ["../secret.png", "file:///etc/passwd"])
def test_offline_pack_installer_rejects_sources_outside_package(
    tmp_path: Path, local_uri: str
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    cache = AssetCache(tmp_path / "cache")
    pack = make_pack("0" * 64)
    unsafe = pack.assets[0].model_copy(update={"local_uri": local_uri})

    with pytest.raises(AssetCacheError, match="escapes its package"):
        asyncio.run(cache.install_pack(pack.model_copy(update={"assets": [unsafe]}), source))


def test_offline_pack_installer_rejects_unsafe_media_format(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    media = b"<script>not media</script>"
    (source / "sky.png").write_bytes(media)
    pack = make_pack(hashlib.sha256(media).hexdigest())
    unsafe = pack.assets[0].model_copy(update={"local_uri": "sky.html"})
    (source / "sky.html").write_bytes(media)
    cache = AssetCache(tmp_path / "cache")

    with pytest.raises(AssetCacheError, match="Unsupported image"):
        asyncio.run(cache.install_pack(pack.model_copy(update={"assets": [unsafe]}), source))


def test_offline_pack_installer_revalidates_cached_asset_uris(tmp_path: Path) -> None:
    checksum = "0" * 64
    cache = AssetCache(tmp_path / "cache")
    pack = make_pack(checksum)
    cached = pack.assets[0].model_copy(update={"local_uri": f"/v1/assets/{checksum}/sky.png"})

    with pytest.raises(AssetCacheError, match="not found"):
        asyncio.run(cache.install_pack(pack.model_copy(update={"assets": [cached]}), tmp_path))
