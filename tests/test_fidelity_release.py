# ruff: noqa: E402
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.validate_fidelity_release import (
    MAXTEXT_REVISION,
    MODEL_ID,
    MODEL_REVISION,
    ReleaseValidationError,
    canonical_sha256,
    expected_candidate_id,
    sha256_file,
    validate_release,
)


def make_release(tmp_path: Path) -> tuple[Path, Path, dict]:
    source = tmp_path / "merged-hf"
    source.mkdir()
    artifacts = {
        "config.json": b'{"model_type":"gemma4_text"}\n',
        "model-00001-of-00001.safetensors": b"merged-weights",
        "tokenizer.json": b'{"version":"1.0"}\n',
        "tokenizer_config.json": b'{"eos_token_id":[1,106,50]}\n',
    }
    for relative, content in artifacts.items():
        (source / relative).write_bytes(content)
    files = [
        {
            "path": path.relative_to(source).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(source.rglob("*"))
        if path.is_file()
    ]
    document = {
        "schema_version": "1.0",
        "status": "succeeded",
        "release_type": "merged-hf",
        "base_model": {"id": MODEL_ID, "revision": MODEL_REVISION},
        "maxtext_revision": MAXTEXT_REVISION,
        "config_sha256": "a" * 64,
        "dataset_manifest_sha256": "b" * 64,
        "training_run_id": "lora-train-fidelity-001",
        "files_content_sha256": canonical_sha256(files),
        "files": files,
    }
    document["candidate_id"] = expected_candidate_id(document)
    manifest = tmp_path / "release.manifest.json"
    manifest.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    return manifest, source, document


def test_valid_release_binds_every_file_and_lineage(tmp_path: Path) -> None:
    manifest, source, document = make_release(tmp_path)
    validated = validate_release(
        manifest,
        source,
        expected_manifest_sha256=sha256_file(manifest),
        expected_config_sha256="a" * 64,
        expected_dataset_manifest_sha256="b" * 64,
        expected_candidate=document["candidate_id"],
    )

    assert validated.candidate_id == document["candidate_id"]
    assert validated.files_content_sha256 == document["files_content_sha256"]
    assert validated.document["base_model"]["revision"] == MODEL_REVISION


def test_release_rejects_tampered_weight_bytes(tmp_path: Path) -> None:
    manifest, source, _ = make_release(tmp_path)
    (source / "model-00001-of-00001.safetensors").write_bytes(b"changed")

    with pytest.raises(ReleaseValidationError, match="size mismatch"):
        validate_release(
            manifest,
            source,
            expected_manifest_sha256=sha256_file(manifest),
        )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("base_model", {"id": MODEL_ID, "revision": "c" * 40}, "exact Gemma"),
        ("maxtext_revision", "d" * 40, "MaxText"),
        ("config_sha256", "not-a-hash", "config_sha256"),
        ("candidate_id", "fidelity-00000000000000000000", "lineage"),
    ],
)
def test_release_rejects_lineage_or_candidate_drift(
    tmp_path: Path, field: str, value: object, match: str
) -> None:
    manifest, source, document = make_release(tmp_path)
    document[field] = value
    manifest.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ReleaseValidationError, match=match):
        validate_release(
            manifest,
            source,
            expected_manifest_sha256=sha256_file(manifest),
        )


def test_release_rejects_unmanifested_or_symlinked_files(tmp_path: Path) -> None:
    manifest, source, _ = make_release(tmp_path)
    (source / "untracked.bin").write_bytes(b"not declared")

    with pytest.raises(ReleaseValidationError, match="unmanifested"):
        validate_release(
            manifest,
            source,
            expected_manifest_sha256=sha256_file(manifest),
        )


def test_release_manifest_hash_is_an_external_approval_boundary(tmp_path: Path) -> None:
    manifest, source, _ = make_release(tmp_path)

    with pytest.raises(ReleaseValidationError, match="approved value"):
        validate_release(
            manifest,
            source,
            expected_manifest_sha256="f" * 64,
        )
