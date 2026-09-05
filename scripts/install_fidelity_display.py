#!/usr/bin/env python3
"""Install a completed, bound display batch into a fresh local cache and Story Pack store."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import re
import stat
import sys
from dataclasses import asdict
from pathlib import Path

from bookforge.asset_cache import AssetCache
from bookforge.domain import AssetKind, AssetRecord, AssetRole, AssetState, StoryPack
from bookforge.finite_modal_provider import DEPTH_MODEL, DEPTH_MODEL_REVISION, _jpeg_dimensions
from bookforge.klein_scene_provider import MODEL, REVISION
from bookforge.story_store import StoryPackStore

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import render_fidelity_display as render  # noqa: E402

assembly = render.assembly
METRICS = (
    "remote_seconds",
    "image_seconds",
    "depth_seconds",
    "packaging_seconds",
    "model_load_seconds",
    "startup_seconds",
    "cache_setup_seconds",
    "warm_state",
    "sequence_bucket",
    "token_count",
)


def read_bounded(path: Path, maximum: int) -> bytes:
    if path.is_symlink() or path.resolve() != path.absolute():
        raise ValueError("installation input cannot be a symlink")
    if not path.is_file() or path.stat().st_size > maximum:
        raise ValueError("installation input exceeds its file contract")
    return path.read_bytes()


def read_pack(path: Path, batch, proof_sha256: str):
    assembly.smoke.private_path(path)
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) != 0o600
            or info.st_size > 2_000_000
        ):
            raise ValueError("private Story Pack differs from its file contract")
        data = stream.read()
    if hashlib.sha256(data).hexdigest() != proof_sha256:
        raise ValueError("private Story Pack checksum differs from supplied proof")
    pack = StoryPack.model_validate_json(data)
    manifest, cases = assembly.smoke.load_manifest(assembly.probe.STORY)
    candidates = [row for row in batch.requests if row.variant == "candidate"]
    if (
        pack.schema_version != "2.0"
        or pack.planning_scope != "scene"
        or pack.assets
        or pack.compiler_model != "local-authored-scene-facts-v2"
        or pack.compiler_contract_revision != "authored-display-steps-v1"
        or pack.story_id != manifest["story_id"]
        or pack.visual_style != manifest["visual_style"]
        or [page.page_id for page in pack.pages] != [row.display_step_id for row in candidates]
    ):
        raise ValueError("private Story Pack differs from proved candidate pages")
    for page, row in zip(pack.pages, candidates, strict=True):
        if (
            page.source_text != cases[row.source_page_index - 1][0]
            or page.scene_spec.master_prompt != row.prompt
            or page.scene_spec.negative_prompt != row.negative_prompt
            or page.scene_spec.camera.duration_ms != 8000 + row.seed % 4001
            or sum(layer.kind == "background" for layer in page.layers) != 1
        ):
            raise ValueError("private page differs from its proved request")
    return pack, hashlib.sha256(data).hexdigest()


def completed_bundles(batch, proof: str, directory: Path):
    journal_data = read_bounded(directory / "journal.jsonl", 262144)
    entries = [json.loads(line) for line in journal_data.splitlines()]
    if len(entries) != 24:
        raise ValueError("render journal is not a completed eleven-request run")
    header, complete = entries[0], entries[-1]
    identity = render.expected_identity()
    expected_header = {
        "kind": "header",
        "schema_version": 1,
        "mode": "execute",
        "batch_sha256": proof,
        "harness_sha256": assembly.file_hash(Path(render.__file__)),
        "provider_sha256": assembly.file_hash(
            render.ROOT / "src/bookforge/klein_scene_provider.py"
        ),
        "runtime_sha256": render.EXPECTED_RUNTIME_SHA256,
        "deployment_source_sha256": render.EXPECTED_DEPLOYMENT_SHA256,
        "renderer_receipt_sha256": render.RECEIPT_SHA256,
        "expected_identity_sha256": assembly.digest(identity),
        "cache_id_sha256": assembly.digest(render.CACHE_ID),
        "maximum_requests": 11,
        "call_ceiling_usd": 0.25,
        "batch_ceiling_usd": 2.75,
        "automatic_retries": 0,
        "prewarm_requests": 0,
        "negative_prompt_supported": False,
        "visual_acceptance_measured": False,
    }
    if any(header.get(key) != value for key, value in expected_header.items()):
        raise ValueError("render journal provenance differs")
    if set(header) != {
        *expected_header,
        "plan_sha256",
        "ledger_before_sha256",
        "ledger_path_sha256",
        "attempt_claim_sha256",
        "tokens",
    }:
        raise ValueError("render journal header fields differ")
    tokens = header["tokens"]
    if set(tokens) != {"tokenizer_sha256", "token_counts"} or set(tokens["token_counts"]) != {
        row.id for row in batch.requests
    }:
        raise ValueError("render token evidence differs")
    hashes = [
        header[key]
        for key in (
            "plan_sha256",
            "ledger_before_sha256",
            "ledger_path_sha256",
            "attempt_claim_sha256",
        )
    ]
    hashes += [tokens["tokenizer_sha256"], complete.get("ledger_after_sha256")]
    if any(
        not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value) for value in hashes
    ):
        raise ValueError("render evidence contains invalid digests")
    if complete != {
        "kind": "complete",
        "mode": "execute",
        "ledger_after_sha256": complete["ledger_after_sha256"],
    }:
        raise ValueError("render run did not complete")
    artifacts = {}
    for ordinal, row in enumerate(render.ordered_requests(batch)):
        start, result = entries[1 + ordinal * 2 : 3 + ordinal * 2]
        if start != {
            "kind": "start",
            "ordinal": ordinal,
            "request_sha256": assembly.digest(row.model_dump()),
        }:
            raise ValueError("render request order or proof differs")
        if (
            set(result)
            != {
                "kind",
                "ordinal",
                "status",
                "manifest_sha256",
                "identity_sha256",
                "reservation_sha256",
                "master_sha256",
                "depth_sha256",
                "metrics",
            }
            or result["kind"] != "result"
            or result["ordinal"] != ordinal
            or result["status"] != "ok"
        ):
            raise ValueError("render result is missing or failed")
        image_dir = directory / f"image-{ordinal:02}"
        data = read_bounded(image_dir / "scene.manifest.json", 262144)
        if hashlib.sha256(data).hexdigest() != result["manifest_sha256"]:
            raise ValueError("render manifest checksum differs")
        bundle = json.loads(data)
        if (
            bundle.get("schema_version") != "1.0"
            or bundle.get("provider") != "modal-klein-candidate"
            or bundle.get("scene_id") != row.id
            or bundle.get("request") != asdict(render.fast_request(row))
            or bundle.get("identity") != identity
            or result["identity_sha256"] != assembly.digest(identity)
            or set(bundle.get("artifacts", {})) != {"master", "depth"}
            or set(bundle.get("stages", {})) != {"fast"}
        ):
            raise ValueError("render bundle differs from its request")
        policy = bundle["policy"]
        if (
            assembly.digest(policy["reservation_id"]) != result["reservation_sha256"]
            or policy.get("reserved_compute_usd") != 0.25
            or policy.get("automatic_retries") != 0
            or policy.get("visual_acceptance") != "human-review-required"
        ):
            raise ValueError("render policy receipt differs")
        stage = bundle["stages"]["fast"]
        count = tokens["token_counts"][row.id]
        if (
            type(count) is not int
            or not 0 < count <= 256
            or result["metrics"] != {key: stage[key] for key in METRICS}
            or stage["token_count"] != count
            or stage["sequence_bucket"] != (128 if count <= 128 else 256)
            or stage["warm_state"] not in {"cold", "warm"}
            or stage["model"] != MODEL
            or stage["model_revision"] != REVISION
            or stage["additional_models"]
            != [{"role": "depth", "model": DEPTH_MODEL, "model_revision": DEPTH_MODEL_REVISION}]
            or stage["gpu"] != "L4"
            or stage["steps"] != 4
            or stage["guidance_scale"] != 1.0
            or stage["negative_prompt_supported"] is not False
            or any(
                type(stage[key]) not in {int, float}
                or not math.isfinite(stage[key])
                or stage[key] < 0
                for key in METRICS[:7]
            )
        ):
            raise ValueError("render profile or measurement differs")
        retained = {}
        for role in ("master", "depth"):
            artifact = bundle["artifacts"][role]
            content = read_bounded(image_dir / f"{role}.jpg", 16_000_000)
            checksum = hashlib.sha256(content).hexdigest()
            if (
                artifact
                != {
                    "role": role,
                    "path": f"{role}.jpg",
                    "sha256": checksum,
                    "mime_type": "image/jpeg",
                    "width": 1024,
                    "height": 576,
                    "duration_ms": 0,
                    "frames": 1,
                    "fps": 0,
                }
                or checksum != result[f"{role}_sha256"]
                or _jpeg_dimensions(content) != (1024, 576)
            ):
                raise ValueError("render artifact checksum or dimensions differ")
            if row.variant == "candidate":
                retained[role] = content
        if row.variant == "candidate":
            artifacts[row.display_step_id] = retained
    return artifacts, hashlib.sha256(journal_data).hexdigest()


async def install(pack: StoryPack, artifacts: dict, data_dir: Path, seeds: dict):
    data_dir.mkdir(mode=0o700, exist_ok=False)
    (data_dir / "cache").mkdir(mode=0o700)
    cache = AssetCache(data_dir / "cache/assets")
    records = []
    for page in pack.pages:
        background = next(layer.layer_id for layer in page.layers if layer.kind == "background")
        for role in ("master", "depth"):
            kind = AssetKind.IMAGE if role == "master" else AssetKind.DEPTH_MAP
            asset_id = f"{page.page_id}-{role}"
            checksum, uri = await cache.store_generated(
                asset_id=asset_id, kind=kind, content=artifacts[page.page_id][role], suffix=".jpg"
            )
            records.append(
                AssetRecord(
                    asset_id=asset_id,
                    page_id=page.page_id,
                    layer_id=background,
                    kind=kind,
                    role=AssetRole(role),
                    provider="modal-klein-candidate",
                    prompt=page.scene_spec.master_prompt,
                    seed=seeds[page.page_id],
                    width=1024,
                    height=576,
                    checksum_sha256=checksum,
                    local_uri=uri,
                    state=AssetState.READY,
                )
            )
    installed = StoryPack.model_validate({**pack.model_dump(), "assets": records})
    installed = await cache.install_pack(installed, data_dir)
    store = StoryPackStore(data_dir / "story-packs")
    destination = await store.save(installed)
    replay = await cache.install_pack(await store.latest(), data_dir)
    if replay != installed or len(replay.assets) != 16:
        raise ValueError("cached replay differs from installed pack")
    return installed, assembly.file_hash(destination)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("batch", "render-dir", "private-pack", "data-dir", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--proof-batch-sha256", required=True)
    parser.add_argument("--proof-private-pack-sha256", required=True)
    args = parser.parse_args(argv)
    try:
        if (
            args.data_dir.exists()
            or args.data_dir.is_symlink()
            or args.output.exists()
            or args.output.is_symlink()
        ):
            raise ValueError("installation requires fresh target paths")
        target = args.data_dir.resolve()
        if target.is_relative_to(render.ROOT.resolve()) or any(
            (parent / ".git").exists() or (parent / ".git").is_symlink()
            for parent in target.parents
        ):
            raise ValueError("source-bearing installation requires an external private directory")
        if (
            args.data_dir.parent.resolve() != args.data_dir.parent.absolute()
            or not args.data_dir.parent.is_dir()
        ):
            raise ValueError("installation target parent must already exist without symlinks")
        if not args.output.parent.is_dir() or args.output.resolve().is_relative_to(
            args.data_dir.resolve()
        ):
            raise ValueError("installation receipt requires a separate existing directory")
        batch = render.read_batch(args.batch, args.proof_batch_sha256)
        pack, private_sha = read_pack(args.private_pack, batch, args.proof_private_pack_sha256)
        artifacts, journal_sha = completed_bundles(batch, args.proof_batch_sha256, args.render_dir)
        seeds = {
            row.display_step_id: row.seed for row in batch.requests if row.variant == "candidate"
        }
        installed, stored_sha = asyncio.run(install(pack, artifacts, args.data_dir, seeds))
        receipt = {
            "schema_version": 1,
            "kind": "fidelity-display-installation",
            "batch_sha256": args.proof_batch_sha256,
            "render_journal_sha256": journal_sha,
            "private_pack_sha256": private_sha,
            "stored_pack_sha256": stored_sha,
            "story_id": installed.story_id,
            "page_ids": [page.page_id for page in installed.pages],
            "asset_count": 16,
            "completed_render_requests": 11,
            "cached_replay_checksums_verified": True,
            "visual_acceptance_measured": False,
            "physical_playback_verified": False,
            "assets": [
                {
                    "asset_id": asset.asset_id,
                    "page_id": asset.page_id,
                    "role": asset.role.value,
                    "checksum_sha256": asset.checksum_sha256,
                    "width": asset.width,
                    "height": asset.height,
                }
                for asset in installed.assets
            ],
        }
        descriptor = os.open(
            args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
        )
        with os.fdopen(descriptor, "w") as stream:
            stream.write(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        return 0
    except Exception:
        print("display installation failed; inspect local inputs and bound render evidence")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
