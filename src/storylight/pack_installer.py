from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from storylight.asset_cache import AssetCache
from storylight.config import Settings
from storylight.domain import StoryPack
from storylight.story_store import StoryPackStore


async def install(source: Path, source_root: Path, settings: Settings) -> dict[str, object]:
    pack = StoryPack.model_validate_json(source.read_text(encoding="utf-8"))
    cache = AssetCache(settings.cache_dir / "assets")
    store = StoryPackStore(settings.data_dir / "story-packs")
    await cache.initialize()
    await store.initialize()
    installed = await cache.install_pack(pack, source_root)
    destination = await store.save(installed)
    return {
        "story_id": installed.story_id,
        "story_pack": str(destination),
        "assets": len(installed.assets),
        "ready_assets": sum(asset.state.value == "ready" for asset in installed.assets),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify and install an offline Storylight package")
    parser.add_argument("story_pack", type=Path)
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    arguments = parser.parse_args()
    source = arguments.story_pack.resolve()
    source_root = (arguments.asset_root or source.parent).resolve()
    overrides = {
        name: value
        for name, value in (("data_dir", arguments.data_dir), ("cache_dir", arguments.cache_dir))
        if value is not None
    }
    result = asyncio.run(install(source, source_root, Settings(**overrides)))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
