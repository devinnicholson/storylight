import asyncio
import hashlib

import pytest

from bookforge.asset_cache import AssetCache
from bookforge.asset_generator import (
    FakeAssetGenerator,
    FakeDepthEstimator,
    GeneratedImage,
    ImageGenerationRequest,
    ModalDepthEstimator,
)
from bookforge.config import Settings
from bookforge.domain import AssetRole, BookPageInput, StoryCompileRequest
from bookforge.model_client import FakeModelClient
from bookforge.scene_foundry import SceneFoundry, SceneFoundryError
from bookforge.service import BookforgeService


def test_foundry_generates_checksums_and_caches_master_and_depth(tmp_path) -> None:
    async def run():
        settings = Settings(model_backend="fake", model_name="fake")
        compiled = await BookforgeService(settings, FakeModelClient()).compile_story(
            StoryCompileRequest(
                story_id="silver-fox",
                title="The Silver Fox",
                visual_style="layered watercolor paper theater",
                pages=[
                    BookPageInput(
                        page_id="page-01",
                        text="A silver fox found a lantern under the old tree.",
                    )
                ],
            )
        )
        cache = AssetCache(tmp_path / "assets")
        await cache.initialize()
        foundry = SceneFoundry(
            generator=FakeAssetGenerator(),
            depth_estimator=FakeDepthEstimator(),
            cache=cache,
            width=320,
            height=180,
        )
        return cache, await foundry.build(compiled.story_pack)

    cache, (pack, metrics) = asyncio.run(run())

    assert pack.story_id == "silver-fox"
    assert [asset.role for asset in pack.assets] == [AssetRole.MASTER, AssetRole.DEPTH]
    assert metrics.generated_assets == 2
    assert metrics.generated_bytes > 0
    for asset in pack.assets:
        assert asset.local_uri.startswith(f"/v1/assets/{asset.checksum_sha256}/")
        filename = asset.local_uri.rsplit("/", 1)[-1]
        content = cache.resolve(asset.checksum_sha256, filename).read_bytes()
        assert hashlib.sha256(content).hexdigest() == asset.checksum_sha256


def test_foundry_rejects_legacy_pack_without_scene_spec(tmp_path) -> None:
    from bookforge.domain import GeneratedPagePlan, StoryPack, VisualLayer

    pack = StoryPack(
        story_id="legacy",
        title="Legacy",
        reading_level=2,
        visual_style="paper",
        compiler_model="fixture",
        pages=[
            GeneratedPagePlan(
                page_id="page-01",
                source_text="The fox ran.",
                scene_summary="A fox runs.",
                layers=[
                    VisualLayer(
                        layer_id="background",
                        kind="background",
                        prompt="Forest",
                        z_index=0,
                        motion="Static",
                    )
                ],
                triggers=[],
                literacy_support=[],
                comprehension=[],
            )
        ],
    )
    foundry = SceneFoundry(
        generator=FakeAssetGenerator(),
        depth_estimator=FakeDepthEstimator(),
        cache=AssetCache(tmp_path / "assets"),
        width=320,
        height=180,
    )

    with pytest.raises(SceneFoundryError, match="schema 2.0"):
        asyncio.run(foundry.build(pack))


def test_modal_depth_estimator_uses_embedded_sidecar() -> None:
    async def run():
        source = await FakeAssetGenerator().generate(
            ImageGenerationRequest(
                prompt="forest",
                negative_prompt="text",
                seed=7,
                width=16,
                height=9,
            )
        )
        depth = await FakeDepthEstimator().estimate(source)
        modal_source = GeneratedImage(
            content=source.content,
            mime_type=source.mime_type,
            width=source.width,
            height=source.height,
            provider="modal",
            model="scene-model",
            seed=source.seed,
            generation_ms=10,
            embedded_depth=depth.content,
            embedded_depth_provider="modal",
            embedded_depth_model="depth-anything-v2-small",
            embedded_depth_ms=2,
        )
        return source, depth, await ModalDepthEstimator(Settings()).estimate(modal_source)

    source, depth, result = asyncio.run(run())

    assert result.content == depth.content
    assert result.width == source.width
    assert result.height == source.height
    assert result.model == "depth-anything-v2-small"
    assert result.generation_ms == 2
