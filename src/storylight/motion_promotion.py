from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from storylight.domain import AssetKind, AssetRecord, AssetRole, AssetState, StoryPack


class MotionPromotionError(RuntimeError):
    pass


def promote_motion_asset(
    pack: StoryPack,
    *,
    video_path: Path,
    asset_root: Path,
    page_id: str,
    provider: str,
    prompt: str,
    seed: int,
    duration_ms: int,
    generation_ms: float,
) -> StoryPack:
    root = asset_root.resolve()
    source = video_path.resolve()
    if not source.is_file() or not source.is_relative_to(root):
        raise MotionPromotionError("motion video must be a file inside the asset root")
    if source.suffix.lower() not in {".mp4", ".webm"}:
        raise MotionPromotionError("motion video must be MP4 or WebM")
    page = next((candidate for candidate in pack.pages if candidate.page_id == page_id), None)
    if page is None:
        raise MotionPromotionError(f"Story Pack has no page {page_id!r}")
    background = next(
        (layer for layer in page.layers if layer.kind == "background"),
        page.layers[0],
    )
    content = source.read_bytes()
    if not content:
        raise MotionPromotionError("motion video is empty")
    checksum = hashlib.sha256(content).hexdigest()
    relative_uri = source.relative_to(root).as_posix()
    motion = AssetRecord(
        asset_id=f"{page_id}-motion",
        page_id=page_id,
        layer_id=background.layer_id,
        kind=AssetKind.VIDEO_LOOP,
        role=AssetRole.MOTION,
        provider=provider,
        prompt=prompt,
        seed=seed,
        width=768,
        height=512,
        duration_ms=duration_ms,
        checksum_sha256=checksum,
        local_uri=relative_uri,
        state=AssetState.READY,
        generation_ms=generation_ms,
    )
    retained = [
        asset
        for asset in pack.assets
        if not (asset.page_id == page_id and asset.role is AssetRole.MOTION)
    ]
    return pack.model_copy(update={"assets": [*retained, motion]})


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Attach a verified visual-lab motion loop to a Story Pack"
    )
    parser.add_argument("story_pack", type=Path)
    parser.add_argument("video", type=Path)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--page-id", default="page-01")
    parser.add_argument("--provider", default="modal:Lightricks/LTX-Video")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--duration-ms", default=4000, type=int)
    parser.add_argument("--generation-ms", required=True, type=float)
    arguments = parser.parse_args()
    pack = StoryPack.model_validate_json(arguments.story_pack.read_text())
    promoted = promote_motion_asset(
        pack,
        video_path=arguments.video,
        asset_root=arguments.asset_root,
        page_id=arguments.page_id,
        provider=arguments.provider,
        prompt=arguments.prompt,
        seed=arguments.seed,
        duration_ms=arguments.duration_ms,
        generation_ms=arguments.generation_ms,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(promoted.model_dump_json(indent=2) + "\n")
    print(arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
