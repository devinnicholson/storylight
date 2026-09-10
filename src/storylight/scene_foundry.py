from __future__ import annotations

import hashlib
import time

from storylight.asset_cache import AssetCache
from storylight.asset_generator import AssetGenerator, DepthEstimator, ImageGenerationRequest
from storylight.domain import (
    AssetKind,
    AssetRecord,
    AssetRole,
    AssetState,
    GeneratedPagePlan,
    SceneGenerationMetrics,
    StoryPack,
)


class SceneFoundryError(RuntimeError):
    pass


class SceneFoundry:
    def __init__(
        self,
        *,
        generator: AssetGenerator,
        depth_estimator: DepthEstimator,
        cache: AssetCache,
        width: int,
        height: int,
    ) -> None:
        self.generator = generator
        self.depth_estimator = depth_estimator
        self.cache = cache
        self.width = width
        self.height = height

    async def build(self, pack: StoryPack) -> tuple[StoryPack, SceneGenerationMetrics]:
        if pack.schema_version != "2.0":
            raise SceneFoundryError("Scene generation requires Story Pack schema 2.0")
        started = time.perf_counter()
        retained_assets = [
            asset for asset in pack.assets if asset.role not in {AssetRole.MASTER, AssetRole.DEPTH}
        ]
        generated_assets: list[AssetRecord] = []
        master_ms = 0.0
        depth_ms = 0.0
        generated_bytes = 0
        provider = ""
        model = ""
        for page in pack.pages:
            page_assets, page_master_ms, page_depth_ms, page_bytes = await self._build_page(
                pack, page
            )
            generated_assets.extend(page_assets)
            master_ms += page_master_ms
            depth_ms += page_depth_ms
            generated_bytes += page_bytes
            provider = page_assets[0].provider
            model = page_assets[0].provider.split(":", 1)[-1]
        complete_pack = pack.model_copy(update={"assets": retained_assets + generated_assets})
        return complete_pack, SceneGenerationMetrics(
            provider=provider,
            model=model,
            total_ms=(time.perf_counter() - started) * 1000,
            master_ms=master_ms,
            depth_ms=depth_ms,
            generated_assets=len(generated_assets),
            generated_bytes=generated_bytes,
        )

    async def _build_page(
        self,
        pack: StoryPack,
        page: GeneratedPagePlan,
    ) -> tuple[list[AssetRecord], float, float, int]:
        if page.scene_spec is None:
            raise SceneFoundryError(f"Page {page.page_id} has no SceneSpec v2")
        layer = next(
            (candidate for candidate in page.layers if candidate.kind == "background"),
            page.layers[0],
        )
        seed = _stable_seed(pack.story_id, page.page_id, page.scene_spec.master_prompt)
        prompt = (
            f"{page.scene_spec.master_prompt}\n"
            f"Visual style: {pack.visual_style}. "
            "Full-bleed cinematic 16:9 storybook projection, no words, no letters, no labels, "
            "no interface, no border, no split panels, no watermark."
        )
        negative_prompt = page.scene_spec.negative_prompt or (
            "text, typography, labels, UI, watermark, frame, collage, split screen, low detail"
        )
        master = await self.generator.generate(
            ImageGenerationRequest(
                prompt=prompt,
                negative_prompt=negative_prompt,
                seed=seed,
                width=self.width,
                height=self.height,
            )
        )
        master_id = f"{page.page_id}-master"
        master_checksum, master_uri = await self.cache.store_generated(
            asset_id=master_id,
            kind=AssetKind.IMAGE,
            content=master.content,
            suffix=".png",
        )
        depth = await self.depth_estimator.estimate(master)
        depth_id = f"{page.page_id}-depth"
        depth_checksum, depth_uri = await self.cache.store_generated(
            asset_id=depth_id,
            kind=AssetKind.DEPTH_MAP,
            content=depth.content,
            suffix=".png",
        )
        master_record = AssetRecord(
            asset_id=master_id,
            page_id=page.page_id,
            layer_id=layer.layer_id,
            kind=AssetKind.IMAGE,
            role=AssetRole.MASTER,
            provider=f"{master.provider}:{master.model}",
            prompt=prompt,
            seed=seed,
            width=master.width,
            height=master.height,
            checksum_sha256=master_checksum,
            local_uri=master_uri,
            state=AssetState.READY,
            generation_ms=master.generation_ms,
        )
        depth_record = AssetRecord(
            asset_id=depth_id,
            page_id=page.page_id,
            layer_id=layer.layer_id,
            kind=AssetKind.DEPTH_MAP,
            role=AssetRole.DEPTH,
            provider=f"{depth.provider}:{depth.model}",
            prompt=f"Depth estimate for {master_id}",
            seed=seed,
            width=depth.width,
            height=depth.height,
            checksum_sha256=depth_checksum,
            local_uri=depth_uri,
            state=AssetState.READY,
            generation_ms=depth.generation_ms,
        )
        return (
            [master_record, depth_record],
            master.generation_ms,
            depth.generation_ms,
            len(master.content) + len(depth.content),
        )


def _stable_seed(story_id: str, page_id: str, prompt: str) -> int:
    payload = f"{story_id}\0{page_id}\0{prompt}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")
