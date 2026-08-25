from pathlib import Path

WORKER_ROOT = Path("deploy/gcp_live_scene_worker")
DEPLOY_SCRIPT = Path("infra/gcp/cloud-run/deploy-live-scene.sh")
CLOUD_BUILD = Path("infra/gcp/cloud-run/cloudbuild-live-scene.yaml")


def test_rtx_worker_uses_blackwell_compatible_pytorch_and_cuda() -> None:
    dockerfile = (WORKER_ROOT / "Dockerfile").read_text()
    requirements = (WORKER_ROOT / "requirements.txt").read_text()
    app = (WORKER_ROOT / "app.py").read_text()

    assert "nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04" in dockerfile
    assert "https://download.pytorch.org/whl/cu128" in requirements
    assert "torch==2.8.0+cu128" in requirements
    assert "torchvision==0.23.0+cu128" in requirements
    assert "torch.cuda.get_device_capability(0)" in app
    assert "torch.cuda.get_arch_list()" in app
    deploy_script = DEPLOY_SCRIPT.read_text()
    assert "HF_HUB_OFFLINE=1" in deploy_script
    assert "TRANSFORMERS_OFFLINE=1" in deploy_script
    assert "snapshot_download(" in app
    assert "local_files_only=True" in app


def test_cloud_build_reuses_the_latest_immutable_worker_layers() -> None:
    dockerfile = (WORKER_ROOT / "Dockerfile").read_text()
    deploy_script = DEPLOY_SCRIPT.read_text()
    cloud_build = CLOUD_BUILD.read_text()

    assert "chown -R bookforge:bookforge /app\n" in dockerfile
    assert "chown -R bookforge:bookforge /app /models" not in dockerfile
    assert "COPY --chown=65532:65532 app.py ./" in dockerfile
    assert "spec.template.spec.containers[0].image" in deploy_script
    assert "cloudbuild-live-scene.yaml" in deploy_script
    assert 'docker pull "${_CACHE_IMAGE}" || true' in cloud_build
    assert 'docker build --cache-from "${_CACHE_IMAGE}"' in cloud_build


def test_cloud_run_worker_avoids_reserved_healthz_route() -> None:
    app = (WORKER_ROOT / "app.py").read_text()

    assert '@app.get("/health")' in app
    assert '@app.get("/healthz")' not in app
    assert "steps: Annotated[int, Field(ge=2, le=2)]" in app
