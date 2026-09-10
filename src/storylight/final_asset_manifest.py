from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


class FinalAssetManifestError(RuntimeError):
    pass


def _record_pool(manifests: list[dict[str, Any]], *, stage: str) -> dict[str, dict[str, Any]]:
    pool: dict[str, dict[str, Any]] = {}
    for manifest in manifests:
        records = manifest.get("records")
        if not isinstance(records, list):
            raise FinalAssetManifestError(f"{stage} manifest requires records")
        for record in records:
            candidate_id = str(record["experiment_id"])
            if candidate_id in pool:
                raise FinalAssetManifestError(f"duplicate {stage} record: {candidate_id}")
            pool[candidate_id] = record
    return pool


def _selected_records(
    selection: dict[str, Any],
    pool: dict[str, dict[str, Any]],
    *,
    stage: str,
) -> dict[int, dict[str, Any]]:
    winners = selection.get("winners")
    if not isinstance(winners, list) or not winners:
        raise FinalAssetManifestError(f"{stage} selection requires winners")
    selected: dict[int, dict[str, Any]] = {}
    for winner in winners:
        page_number = int(winner["page_number"])
        candidate_id = str(winner["candidate_id"])
        record = pool.get(candidate_id)
        if record is None or record.get("sha256") != winner.get("sha256"):
            raise FinalAssetManifestError(
                f"{stage} winner is missing or mismatched: {candidate_id}"
            )
        if page_number in selected:
            raise FinalAssetManifestError(f"{stage} selection repeats page {page_number:02d}")
        selected[page_number] = record
    return selected


def _provider(record: dict[str, Any]) -> str:
    return f"modal:{record['model']}@{record['model_revision']}"


def _asset(record: dict[str, Any], *, motion: bool) -> dict[str, Any]:
    payload = {
        "path": str(record["artifact_path"]),
        "provider": _provider(record),
        "prompt": str(record["prompt"]),
        "seed": int(record["seed"]),
        "width": int(record["width"]),
        "height": int(record["height"]),
        "generation_ms": round(float(record["generation_seconds"]) * 1000, 3),
        "checksum_sha256": str(record["sha256"]),
    }
    if motion:
        frames = int(record["frames"])
        fps = int(record["fps"])
        if frames < 1 or fps < 1:
            raise FinalAssetManifestError("motion records require positive frames and fps")
        payload["duration_ms"] = round(frames / fps * 1000)
    return payload


def build_final_asset_manifest(
    master_manifests: list[dict[str, Any]],
    master_selection: dict[str, Any],
    motion_manifest: dict[str, Any],
    motion_selection: dict[str, Any],
) -> dict[str, Any]:
    masters = _selected_records(
        master_selection,
        _record_pool(master_manifests, stage="master"),
        stage="master",
    )
    motions = _selected_records(
        motion_selection,
        _record_pool([motion_manifest], stage="motion"),
        stage="motion",
    )
    if set(masters) != set(motions):
        raise FinalAssetManifestError("master and motion page sets do not match")
    return {
        "schema_version": "1.0",
        "pages": [
            {
                "page_id": f"page-{page_number:02d}",
                "master": _asset(masters[page_number], motion=False),
                "motion": _asset(motions[page_number], motion=True),
            }
            for page_number in sorted(masters)
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Join selected master and motion evidence")
    parser.add_argument("--master-manifest", action="append", required=True, type=Path)
    parser.add_argument("--master-selection", required=True, type=Path)
    parser.add_argument("--motion-manifest", required=True, type=Path)
    parser.add_argument("--motion-selection", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    payload = build_final_asset_manifest(
        [json.loads(path.read_text()) for path in arguments.master_manifest],
        json.loads(arguments.master_selection.read_text()),
        json.loads(arguments.motion_manifest.read_text()),
        json.loads(arguments.motion_selection.read_text()),
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
