from copy import deepcopy

import pytest

from bookforge.final_asset_manifest import (
    FinalAssetManifestError,
    build_final_asset_manifest,
)


def record(candidate_id: str, checksum: str, *, motion: bool) -> dict:
    return {
        "experiment_id": candidate_id,
        "artifact_path": f"artifacts/{candidate_id}.{'mp4' if motion else 'png'}",
        "model": "ltx" if motion else "sana",
        "model_revision": "revision",
        "prompt": "gentle storybook motion" if motion else "watercolor fox",
        "seed": 42,
        "width": 768 if motion else 1024,
        "height": 512 if motion else 576,
        "generation_seconds": 3.5,
        "sha256": checksum,
        "frames": 96 if motion else 1,
        "fps": 24 if motion else 0,
    }


def selection(candidate_id: str, checksum: str) -> dict:
    return {
        "winners": [
            {
                "candidate_id": candidate_id,
                "page_number": 1,
                "sha256": checksum,
            }
        ]
    }


def test_builds_page_manifest_with_provider_and_exact_duration() -> None:
    master = record("lost-words-page-01-a", "a" * 64, motion=False)
    motion = record("lost-words-page-01-a-motion-ambient", "b" * 64, motion=True)

    payload = build_final_asset_manifest(
        [{"records": [master]}],
        selection(master["experiment_id"], master["sha256"]),
        {"records": [motion]},
        selection(motion["experiment_id"], motion["sha256"]),
    )

    assert payload["pages"][0]["page_id"] == "page-01"
    assert payload["pages"][0]["master"]["provider"] == "modal:sana@revision"
    assert payload["pages"][0]["motion"]["duration_ms"] == 4000
    assert payload["pages"][0]["motion"]["checksum_sha256"] == "b" * 64


def test_rejects_selection_checksum_or_page_set_mismatch() -> None:
    master = record("lost-words-page-01-a", "a" * 64, motion=False)
    motion = record("lost-words-page-02-a-motion-ambient", "b" * 64, motion=True)
    broken = selection(master["experiment_id"], master["sha256"])
    broken["winners"][0]["sha256"] = "f" * 64

    with pytest.raises(FinalAssetManifestError, match="missing or mismatched"):
        build_final_asset_manifest(
            [{"records": [master]}],
            broken,
            {"records": [motion]},
            selection(motion["experiment_id"], motion["sha256"]),
        )

    motion_selection = selection(motion["experiment_id"], motion["sha256"])
    motion_selection["winners"][0]["page_number"] = 2
    with pytest.raises(FinalAssetManifestError, match="page sets"):
        build_final_asset_manifest(
            [{"records": [master]}],
            selection(master["experiment_id"], master["sha256"]),
            {"records": [motion]},
            motion_selection,
        )


def test_rejects_duplicate_master_records_across_manifests() -> None:
    master = record("lost-words-page-01-a", "a" * 64, motion=False)

    with pytest.raises(FinalAssetManifestError, match="duplicate master"):
        build_final_asset_manifest(
            [{"records": [master]}, {"records": [deepcopy(master)]}],
            selection(master["experiment_id"], master["sha256"]),
            {"records": [record("motion-page-01", "b" * 64, motion=True)]},
            selection("motion-page-01", "b" * 64),
        )
