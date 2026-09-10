from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from storylight.domain import (
    AmbientEffect,
    AmbientMotion,
    AssetKind,
    AssetRecord,
    AssetRole,
    AssetState,
    CameraMotion,
    ComprehensionPrompt,
    GeneratedPagePlan,
    LayerComposition,
    LiteracySupport,
    SceneSpecV2,
    StoryPack,
    StoryTrigger,
    VisualLayer,
)


class LiteracyPackError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _asset_path(raw: str, asset_root: Path, suffixes: set[str]) -> tuple[Path, str]:
    root = asset_root.resolve()
    source = (root / raw).resolve()
    if not source.is_relative_to(root) or not source.is_file():
        raise LiteracyPackError(f"selected asset is missing or outside its root: {raw}")
    if source.suffix.lower() not in suffixes:
        raise LiteracyPackError(f"selected asset has an unsupported format: {raw}")
    return source, source.relative_to(root).as_posix()


def _support(word: str) -> LiteracySupport:
    letters = [character for character in word if character.isalnum()]
    grapheme = "-".join(letters)
    hints = [letters[0], "".join(letters[:2]), "".join(letters)]
    return LiteracySupport(word=word, grapheme=grapheme, hint_ladder=list(dict.fromkeys(hints)))


def _page_plan(page: dict[str, Any], *, style: str, character_bible: str) -> GeneratedPagePlan:
    page_id = str(page["page_id"])
    text = str(page["text"])
    beat = str(page["beat"])
    trigger_word = str(page["trigger_word"])
    if trigger_word.lower() not in {token.strip(".,!?\"'").lower() for token in text.split()}:
        raise LiteracyPackError(f"trigger word {trigger_word!r} is absent from {page_id}")
    focus_id = f"{page_id}-focus"
    layers = [
        VisualLayer(
            layer_id=f"{page_id}-background",
            kind="background",
            prompt=f"Environment for: {beat}",
            z_index=1,
            motion="Subtle depth drift",
        ),
        VisualLayer(
            layer_id=focus_id,
            kind="character",
            prompt=f"Story focus for: {beat}. {character_bible}",
            z_index=2,
            motion="Gentle reveal synchronized to the spoken trigger",
        ),
        VisualLayer(
            layer_id=f"{page_id}-light",
            kind="effect",
            prompt="Warm word-light and restrained paper motes",
            z_index=3,
            motion="Slow breathing glow",
        ),
    ]
    scene_spec = SceneSpecV2(
        master_prompt=(
            f"{beat} Visual style: {style}. Character continuity: {character_bible} "
            "Projection-safe 16:9 illustration without readable text or interface."
        ),
        negative_prompt=(
            "readable text, captions, logos, extra foxes, extra lanterns, fear, violence"
        ),
        camera=CameraMotion(kind="slow_push", start_scale=1.01, end_scale=1.035),
        composition=[
            LayerComposition(
                layer_id=layers[0].layer_id,
                center_x=0.5,
                center_y=0.5,
                width=1,
                height=1,
                depth=1,
                ambient_motion=AmbientMotion(kind="parallax", amplitude_x=0.002),
            ),
            LayerComposition(
                layer_id=focus_id,
                center_x=0.5,
                center_y=0.62,
                width=0.58,
                height=0.66,
                depth=3,
                ambient_motion=AmbientMotion(kind="breathe", scale_delta=0.004),
            ),
            LayerComposition(
                layer_id=layers[2].layer_id,
                center_x=0.56,
                center_y=0.48,
                width=0.7,
                height=0.7,
                depth=4,
                ambient_motion=AmbientMotion(kind="float", amplitude_y=0.006),
            ),
        ],
        ambience=[AmbientEffect(kind="dust", density=0.22, speed=0.35, color="#ffe4a8")],
    )
    return GeneratedPagePlan(
        page_id=page_id,
        source_text=text,
        scene_summary=beat,
        scene_spec=scene_spec,
        layers=layers,
        triggers=[
            StoryTrigger(
                trigger_id=f"{page_id}-{trigger_word}",
                word=trigger_word,
                action=str(page["trigger_action"]),
                target_layer_id=focus_id,
                duration_ms=1200,
            )
        ],
        literacy_support=[_support(trigger_word)],
        comprehension=[
            ComprehensionPrompt(
                question=f"What happened when you read the word {trigger_word}?",
                expected_concepts=[trigger_word, "light", "story"],
            )
        ],
    )


def _asset_record(
    *,
    page: GeneratedPagePlan,
    selection: dict[str, Any],
    role: AssetRole,
    asset_root: Path,
) -> AssetRecord:
    raw = str(selection["path"])
    kind = AssetKind.VIDEO_LOOP if role is AssetRole.MOTION else AssetKind.IMAGE
    suffixes = (
        {".mp4", ".webm"} if kind is AssetKind.VIDEO_LOOP else {".png", ".jpg", ".jpeg", ".webp"}
    )
    source, relative = _asset_path(raw, asset_root, suffixes)
    return AssetRecord(
        asset_id=f"{page.page_id}-{role.value}",
        page_id=page.page_id,
        layer_id=page.layers[0].layer_id,
        kind=kind,
        role=role,
        provider=str(selection["provider"]),
        prompt=str(selection["prompt"]),
        seed=int(selection["seed"]),
        width=int(selection["width"]),
        height=int(selection["height"]),
        duration_ms=int(selection.get("duration_ms", 0)),
        checksum_sha256=_sha256(source),
        local_uri=relative,
        state=AssetState.READY,
        generation_ms=float(selection["generation_ms"]),
    )


def assemble_literacy_pack(
    story: dict[str, Any],
    selections: dict[str, Any],
    *,
    asset_root: Path,
) -> StoryPack:
    raw_pages = story.get("pages")
    selected_pages = selections.get("pages")
    if not isinstance(raw_pages, list) or not raw_pages:
        raise LiteracyPackError("story requires pages")
    if not isinstance(selected_pages, list):
        raise LiteracyPackError("selection manifest requires pages")
    selection_by_page = {str(item["page_id"]): item for item in selected_pages}
    if len(selection_by_page) != len(selected_pages):
        raise LiteracyPackError("selection manifest repeats a page")
    pages = [
        _page_plan(
            raw,
            style=str(story["visual_style"]),
            character_bible=str(story["character_bible"]),
        )
        for raw in raw_pages
    ]
    if {page.page_id for page in pages} != set(selection_by_page):
        raise LiteracyPackError("every story page requires exactly one asset selection")
    assets: list[AssetRecord] = []
    for page in pages:
        selection = selection_by_page[page.page_id]
        if "master" in selection:
            assets.append(
                _asset_record(
                    page=page,
                    selection=selection["master"],
                    role=AssetRole.MASTER,
                    asset_root=asset_root,
                )
            )
        if "motion" not in selection:
            raise LiteracyPackError(f"{page.page_id} requires a selected motion loop")
        assets.append(
            _asset_record(
                page=page,
                selection=selection["motion"],
                role=AssetRole.MOTION,
                asset_root=asset_root,
            )
        )
    return StoryPack(
        schema_version="2.0",
        story_id=str(story["story_id"]),
        title=str(story["title"]),
        reading_level=int(story["reading_level"]),
        visual_style=str(story["visual_style"]),
        compiler_model="storylight-literacy-pack-assembler-v1",
        pages=pages,
        assets=assets,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Assemble the six-page literacy Story Pack")
    parser.add_argument("story", type=Path)
    parser.add_argument("selections", type=Path)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    pack = assemble_literacy_pack(
        json.loads(arguments.story.read_text()),
        json.loads(arguments.selections.read_text()),
        asset_root=arguments.asset_root,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(pack.model_dump_json(indent=2) + "\n")
    print(arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
