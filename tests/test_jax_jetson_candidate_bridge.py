from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
DOWNLOADER = ROOT / "deploy/jetson/download-gcs-fidelity-tensorrt-candidate.py"
WRAPPER = ROOT / "deploy/jetson/build-trained-planner-candidate.sh"
BUILDER = ROOT / "deploy/jetson/build-gemma4-tensorrt-edge-engine.sh"
INSTALLER = ROOT / "deploy/jetson/install-trained-planner-candidate.sh"


def _load_downloader():
    spec = importlib.util.spec_from_file_location("fidelity_candidate_download", DOWNLOADER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


downloader = _load_downloader()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _export_bundle(tmp_path: Path, *, prefix: str = "gs://private-bucket/fidelity/run"):
    bundle = tmp_path / "export"
    files = {
        "llm/config.json": b'{"model":"gemma-4-e2b"}\n',
        "llm/model.onnx": b"synthetic-onnx-graph",
        "llm/model.safetensors": b"synthetic-int4-weights",
    }
    entries = []
    for relative, content in sorted(files.items()):
        path = bundle / "onnx" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        entries.append(
            {
                "path": relative,
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    manifest = {
        "schema_version": "1.0",
        "status": "succeeded",
        "candidate_id": "fidelity-0123456789abcdefabcd",
        "base_model": {
            "id": "google/gemma-4-E2B-it",
            "revision": "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
        },
        "source_release_manifest_sha256": "1" * 64,
        "source_files_content_sha256": "2" * 64,
        "config_sha256": "3" * 64,
        "dataset_manifest_sha256": "4" * 64,
        "training_run_id": "fidelity-run-20260901",
        "private_output_prefix": prefix,
        "quantization": "int4_awq",
        "calibration_dataset": "wikitext",
        "calibration_samples": 128,
        "components": ["thinker"],
        "skip_visual": True,
        "skip_audio": True,
        "externalized_weights": ["int4_ffn"],
        "tensorrt_edge_llm_version": "v0.10.0",
        "tensorrt_edge_llm_revision": "71dd1bae032e70771265917ec74d3ff4cad07a10",
        "engine_built_in_cloud": False,
        "file_count": len(entries),
        "total_bytes": sum(item["bytes"] for item in entries),
        "files": entries,
    }
    manifest_path = bundle / "export.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return bundle, _sha256(manifest_path), manifest


def test_export_verification_rejects_manifest_tampering_and_symlinks(tmp_path: Path) -> None:
    source, digest, _ = _export_bundle(tmp_path)
    (source / "export.manifest.json").write_text("{}\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        downloader.verify_export_bundle(source, digest)

    source, digest, _ = _export_bundle(tmp_path / "link")
    weights = source / "onnx/llm/model.safetensors"
    weights.unlink()
    weights.symlink_to(source / "onnx/llm/model.onnx")
    with pytest.raises(ValueError, match="missing or unsafe"):
        downloader.verify_export_bundle(source, digest)


def test_private_gcs_download_requires_external_sha_and_exact_prefix(tmp_path: Path) -> None:
    source, digest, _ = _export_bundle(tmp_path)
    objects = {
        "fidelity/run/export.manifest.json": source / "export.manifest.json",
        **{
            f"fidelity/run/onnx/{path.relative_to(source / 'onnx').as_posix()}": path
            for path in (source / "onnx").rglob("*")
            if path.is_file()
        },
    }

    def fetch(_token: str, bucket: str, object_name: str, destination: Path):
        assert bucket == "private-bucket"
        source_path = objects[object_name]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, destination)
        return _sha256(destination), destination.stat().st_size

    destination = tmp_path / "download"
    result = downloader.download_gcs_bundle(
        "short-lived-token",
        "private-bucket",
        "fidelity/run",
        destination,
        digest,
        fetch=fetch,
    )
    assert result["manifest_sha256"] == digest
    downloader.verify_export_bundle(
        destination,
        digest,
        expected_private_prefix="gs://private-bucket/fidelity/run",
    )

    source, wrong_digest, _ = _export_bundle(
        tmp_path / "wrong-prefix",
        prefix="gs://private-bucket/a-different-run",
    )
    objects.clear()
    objects["fidelity/run/export.manifest.json"] = source / "export.manifest.json"
    with pytest.raises(ValueError, match="prefix differs"):
        downloader.download_gcs_bundle(
            "short-lived-token",
            "private-bucket",
            "fidelity/run",
            tmp_path / "rejected",
            wrong_digest,
            fetch=fetch,
        )


@pytest.mark.parametrize(
    "prefix",
    [
        "gs://private-bucket/fidelity/run",
        (
            "modal-private://bookforge-tensorrt-edge-llm-fidelity/"
            "fidelity-run-20260901/fidelity-0123456789abcdefabcd"
        ),
    ],
)
def test_wrapper_builds_isolated_bundle_accepted_by_candidate_installer(
    tmp_path: Path, prefix: str
) -> None:
    export, export_sha, export_manifest = _export_bundle(tmp_path, prefix=prefix)
    fake_builder = tmp_path / "fake-builder.sh"
    fake_builder.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'test -f "$BOOKFORGE_EDGELLM_MODEL_ROOT/onnx/llm/model.onnx"\n'
        'mkdir -p "$BOOKFORGE_EDGELLM_MODEL_ROOT/engines/llm"\n'
        "printf synthetic-trained-engine > "
        '"$BOOKFORGE_EDGELLM_MODEL_ROOT/engines/llm/llm.engine"\n'
    )
    fake_builder.chmod(0o755)
    fake_timeout = tmp_path / "timeout"
    fake_timeout.write_text('#!/usr/bin/env bash\nshift 3\nexec "$@"\n')
    fake_timeout.chmod(0o755)
    output = tmp_path / "candidate"
    environment = {
        **os.environ,
        "BOOKFORGE_EDGELLM_ROOT": str(tmp_path / "edge"),
        "BOOKFORGE_TRAINED_CANDIDATE_BUILDER": str(fake_builder),
        "BOOKFORGE_TIMEOUT_BIN": str(fake_timeout),
    }
    result = subprocess.run(
        [
            "bash",
            str(WRAPPER),
            "--export-bundle",
            str(export),
            "--export-manifest-sha256",
            export_sha,
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    candidate_manifest = json.loads(
        (output / "candidate.manifest.json").read_text(encoding="utf-8")
    )
    candidate_sha = _sha256(output / "candidate.manifest.json")
    assert f"candidate_manifest_sha256={candidate_sha}" in result.stdout
    assert candidate_manifest["candidate_id"] == export_manifest["candidate_id"]
    assert candidate_manifest["model_revision"] == "sha256:" + _sha256(
        output / "engines/llm/llm.engine"
    )
    assert candidate_manifest["engine_path"] == "engines/llm/llm.engine"
    assert candidate_manifest["source_export_manifest_sha256"] == export_sha
    assert candidate_manifest["source_dataset_manifest_sha256"] == "4" * 64
    assert (output / "onnx/llm/model.safetensors").is_file()

    verified = subprocess.run(
        [
            "bash",
            str(INSTALLER),
            "--bundle",
            str(output),
            "--expected-manifest-sha256",
            candidate_sha,
            "--verify-only",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Candidate verification complete; no files changed." in verified.stdout
