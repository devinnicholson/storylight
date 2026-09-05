from pathlib import Path

EXPORT_ROOT = Path("deploy/gcp_gemma4_tensorrt_export")
CLOUD_BUILD = Path("infra/gcp/cloud-run/cloudbuild-gemma4-tensorrt-export.yaml")


def test_gcp_export_is_bounded_and_completion_manifest_is_last() -> None:
    exporter = (EXPORT_ROOT / "export.py").read_text()
    cloud_build = CLOUD_BUILD.read_text()

    assert "JOB_TIMEOUT_SECONDS = 1_200" in exporter
    assert "if_generation_match=0" in exporter
    assert "completion.upload_from_string(" in exporter
    assert '"bookforge_tensorrt_export_progress"' in exporter
    assert '_progress("python_entrypoint")' in exporter
    assert '"cuda_ready"' in exporter
    assert '"artifact_upload_complete"' in exporter
    assert exporter.index('for item in manifest["files"]') < exporter.index(
        "completion.upload_from_string("
    )
    assert "timeout: 1800s" in cloud_build
    assert "machineType: E2_HIGHCPU_8" in cloud_build
    assert "          .\n" in cloud_build
