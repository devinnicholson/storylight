import json
from pathlib import Path

import pytest

from storylight.domain import AssetRole, StoryPack
from storylight.literacy_pack import LiteracyPackError, assemble_literacy_pack

ROOT = Path(__file__).parents[1]


def story() -> dict:
    return json.loads((ROOT / "experiments/visual-lab/silver-fox-literacy-story.json").read_text())


def selections(tmp_path: Path) -> dict:
    pages = []
    for index in range(1, 7):
        path = tmp_path / f"page-{index:02d}.mp4"
        path.write_bytes(f"motion-{index}".encode())
        pages.append(
            {
                "page_id": f"page-{index:02d}",
                "motion": {
                    "path": path.name,
                    "provider": "modal:ltx",
                    "prompt": f"motion page {index}",
                    "seed": index,
                    "width": 768,
                    "height": 512,
                    "duration_ms": 4000,
                    "generation_ms": 1000,
                },
            }
        )
    return {"schema_version": "1.0", "pages": pages}


def test_assembles_valid_six_page_pack_with_ready_motion(tmp_path: Path) -> None:
    pack = assemble_literacy_pack(story(), selections(tmp_path), asset_root=tmp_path)

    assert pack.schema_version == "2.0"
    assert len(pack.pages) == 6
    assert len(pack.assets) == 6
    assert all(asset.role is AssetRole.MOTION for asset in pack.assets)
    assert [page.triggers[0].word for page in pack.pages] == [
        "fox",
        "letters",
        "reader",
        "word",
        "library",
        "book",
    ]
    assert all(page.scene_spec is not None for page in pack.pages)
    StoryPack.model_validate(pack.model_dump())


def test_rejects_missing_page_selection(tmp_path: Path) -> None:
    selected = selections(tmp_path)
    selected["pages"].pop()

    with pytest.raises(LiteracyPackError, match="every story page"):
        assemble_literacy_pack(story(), selected, asset_root=tmp_path)


def test_rejects_asset_outside_root(tmp_path: Path) -> None:
    selected = selections(tmp_path)
    outside = tmp_path.parent / "outside-literacy.mp4"
    outside.write_bytes(b"outside")
    selected["pages"][0]["motion"]["path"] = str(outside)
    try:
        with pytest.raises(LiteracyPackError, match="outside its root"):
            assemble_literacy_pack(story(), selected, asset_root=tmp_path)
    finally:
        outside.unlink()
