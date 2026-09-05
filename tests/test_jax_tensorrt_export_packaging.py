# ruff: noqa: E402
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deploy.gcp_gemma4_tensorrt_export.export_fidelity_candidate import (
    CALIBRATION_DATASET,
    CALIBRATION_PROVENANCE_SHA256,
    EDGELLM_REVISION,
    build_export_command,
    build_quantize_command,
    export_local_candidate,
)
from scripts.validate_fidelity_release import (
    MAXTEXT_REVISION,
    MODEL_ID,
    MODEL_REVISION,
    canonical_sha256,
    expected_candidate_id,
    sha256_file,
    validate_release,
)

GCP_EXPORT = ROOT / "deploy/gcp_gemma4_tensorrt_export/export_fidelity_candidate.py"
MODAL_EXPORT = ROOT / "deploy/modal_gemma4_tensorrt_edge_fidelity_export.py"


def make_release(tmp_path: Path) -> tuple[Path, Path, dict]:
    source = tmp_path / "merged-hf"
    source.mkdir()
    for relative, content in {
        "config.json": b"{}\n",
        "model.safetensors": b"merged",
        "tokenizer.json": b"{}\n",
        "tokenizer_config.json": b"{}\n",
    }.items():
        (source / relative).write_bytes(content)
    files = [
        {
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(source.iterdir())
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


def fake_tensorrt_command(command, cwd: Path, timeout: int) -> None:
    assert cwd == Path("/opt/tensorrt-edge-llm")
    assert timeout > 0
    if command[0] == "tensorrt-edgellm-quantize":
        output = Path(command[command.index("--output_dir") + 1])
        output.mkdir(parents=True)
        (output / "config.json").write_text("{}\n")
        return
    assert command[0] == "tensorrt-edgellm-export"
    output = Path(command[2]) / "llm"
    output.mkdir(parents=True)
    (output / "config.json").write_text("{}\n")
    (output / "model.onnx").write_bytes(b"onnx")
    (output / "rank0.safetensors").write_bytes(b"int4")


def test_candidate_export_is_content_addressed_and_completion_is_terminal(
    tmp_path: Path,
) -> None:
    manifest, source, document = make_release(tmp_path)
    release = validate_release(
        manifest,
        source,
        expected_manifest_sha256=sha256_file(manifest),
    )
    destination = tmp_path / "exports"
    payload = export_local_candidate(
        release,
        destination,
        private_output_prefix="gs://private-bucket/tensorrt-edge-llm/fidelity",
        run_command=fake_tensorrt_command,
    )
    root = destination / document["candidate_id"] / release.manifest_sha256[:20]

    assert (root / "export.manifest.json").is_file()
    assert payload["candidate_id"] == document["candidate_id"]
    assert payload["source_release_manifest_sha256"] == release.manifest_sha256
    assert payload["calibration_dataset"] == "wikitext"
    assert payload["calibration_decision"] == "preserve-pinned-wikitext"
    assert payload["calibration_provenance_sha256"] == CALIBRATION_PROVENANCE_SHA256
    assert payload["components"] == ["thinker"]
    assert payload["skip_visual"] is True
    assert payload["skip_audio"] is True
    assert payload["base_export_reused"] is False
    assert json.loads((root / "export.manifest.json").read_text()) == payload

    with pytest.raises(RuntimeError, match="cannot be repeated"):
        export_local_candidate(
            release,
            destination,
            private_output_prefix="gs://private-bucket/tensorrt-edge-llm/fidelity",
            run_command=fake_tensorrt_command,
        )


def test_commands_preserve_verified_v010_calibration_and_text_only_export() -> None:
    quantize = build_quantize_command(Path("/merged-hf"), Path("/quantized"))
    export = build_export_command(Path("/quantized"), Path("/onnx"))

    assert "--text_dataset" in quantize
    assert quantize[quantize.index("--text_dataset") + 1] == CALIBRATION_DATASET == "wikitext"
    assert "--num_samples" in quantize
    assert "thinker" in export
    assert "--skip-visual" in export
    assert "--skip-audio" in export
    assert "int4_ffn" in export


def test_gcp_export_reads_private_merged_release_and_uploads_completion_last() -> None:
    source = GCP_EXPORT.read_text()

    assert EDGELLM_REVISION in source
    assert "snapshot_download" not in source
    assert 'SOURCE_MANIFEST_OBJECT = "release.manifest.json"' in source
    assert 'f"tensorrt-edge-llm/fidelity/{release.candidate_id}/' in source
    assert "if_generation_match=0" in source
    assert 'public_access_prevention != "enforced"' in source
    assert "uniform_bucket_level_access_enabled is not True" in source
    assert '"retry_allowed": False' in source
    assert "fidelity-intents" in source
    assert source.index('for item in payload["files"]') < source.index(
        "completion.upload_from_string("
    )
    assert '"base_export_reused": False' in source
    assert '"engine_built_in_cloud": False' in source


def test_modal_export_is_one_l40s_finite_current_month_call() -> None:
    source = MODAL_EXPORT.read_text()

    assert 'GPU = "L40S"' in source
    assert "MAX_CONTAINERS = 1" in source
    assert "RETRIES = 0" in source
    assert "timeout=REMOTE_TIMEOUT_SECONDS" in source
    assert "retries=RETRIES" in source
    assert "max_containers=MAX_CONTAINERS" in source
    assert 'BUDGET_MONTH = "2026-09"' in source
    assert '["modal", "billing", "report", "--for", "this month", "--json"]' in source
    assert "FULL_COMMAND_CEILING_USD" in source
    assert "FULL_COMMAND_CEILING_PROVIDER_ENFORCED = False" in source
    assert "WORKSPACE_HARD_STOP_USD" in source
    assert ".remote(" in source
    assert "@modal.fastapi_endpoint" not in source
    assert "@modal.web_endpoint" not in source


def test_modal_request_approval_is_bound_to_all_lineage_hashes() -> None:
    source = MODAL_EXPORT.read_text()

    assert "APPROVE_MODAL_FIDELITY_EXPORT:" in source
    assert 'request.get("release_manifest_sha256")' in source
    assert 'request.get("config_sha256")' in source
    assert 'request.get("dataset_manifest_sha256")' in source
    assert "candidate export state already exists" in source
    assert '"export-intent-recorded"' in source
    assert "export_volume.commit()" in source
    assert '"export_manifest_sha256"' in source
    assert '"export_volume_prefix"' in source
    assert "BOOKFORGE_NVIDIA_PYTORCH_IMAGE" in source
    assert "@sha256:" in source
