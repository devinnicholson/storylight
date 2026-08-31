from pathlib import Path

EXPORT_ROOT = Path("deploy/gcp_gemma4_tensorrt_export")
CLOUD_BUILD = Path("infra/gcp/cloud-run/cloudbuild-gemma4-tensorrt-export.yaml")


def test_gcp_export_is_pinned_and_target_only() -> None:
    dockerfile = (EXPORT_ROOT / "Dockerfile").read_text()
    exporter = (EXPORT_ROOT / "export.py").read_text()

    assert "nvcr.io/nvidia/pytorch:25.12-py3" in dockerfile
    assert "71dd1bae032e70771265917ec74d3ff4cad07a10" in dockerfile
    assert 'MODEL_ID = "google/gemma-4-E2B-it"' in exporter
    assert 'MODEL_REVISION = "3e22461f65e89153144f8adb70e3b8c2cc9845a7"' in exporter
    assert '"--components",\n            "thinker"' in exporter
    assert '"--skip-visual"' in exporter
    assert '"--skip-audio"' in exporter
    assert '"int4_ffn"' in exporter
    assert '"mtp_included": False' in exporter
    assert '"engine_built_in_cloud": False' in exporter


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


def test_gcp_export_rejects_wrong_gpu_and_incomplete_checkpoint() -> None:
    exporter = (EXPORT_ROOT / "export.py").read_text()

    assert '"RTX PRO 6000" not in gpu_name.upper()' in exporter
    assert 'onnx_dir / "llm" / "config.json"' in exporter
    assert 'onnx_dir / "llm" / "model.onnx"' in exporter
    assert 'glob("*.safetensors")' in exporter
