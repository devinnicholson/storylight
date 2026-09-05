import asyncio
import hashlib
from pathlib import Path

from bookforge.finite_modal_provider import (
    FastSceneRequest,
    FiniteSceneBundle,
    SceneArtifact,
    WarmPrewarmReport,
)
from bookforge.gcp_scene_benchmark import BenchmarkConfig, collect_benchmark


class FakeProvider:
    def __init__(self, tmp_path: Path, *, ready: bool = True) -> None:
        self.tmp_path = tmp_path
        self.ready = ready
        self.closed = False
        self.requests: list[FastSceneRequest] = []

    async def probe(self) -> tuple[bool, str]:
        return self.ready, "fixture"

    async def prewarm(
        self,
        *,
        prewarm_id: str,
        include_motion: bool,
        scaledown_window_seconds: int,
    ) -> WarmPrewarmReport:
        return WarmPrewarmReport(
            prewarm_id=prewarm_id,
            reservation_id="fixture",
            include_motion=include_motion,
            fast_remote_seconds=0.2,
            motion_remote_seconds=0,
            full_session_ceiling_usd=0.5,
            fast_model_load_seconds=0.1,
            motion_model_load_seconds=0,
            expires_in_seconds=scaledown_window_seconds,
            scaledown_window_seconds=scaledown_window_seconds,
        )

    async def generate_fast(
        self, request: FastSceneRequest, *, output_dir: Path
    ) -> FiniteSceneBundle:
        self.requests.append(request)
        output_dir.mkdir(parents=True)
        master_path = output_dir / "master.jpg"
        depth_path = output_dir / "depth.jpg"
        master_path.write_bytes(b"master")
        depth_path.write_bytes(b"depth")
        manifest = {
            "stages": {
                "fast": {
                    "remote_seconds": 0.7,
                    "inference_seconds": 0.5,
                    "image_seconds": 0.4,
                    "depth_seconds": 0.1,
                    "image_gpu_ms": 350.0,
                    "depth_gpu_ms": 80.0,
                    "packaging_seconds": 0.02,
                    "model_load_seconds": 0.0,
                    "container_age_seconds": 30.0,
                    "warm_state": "warm",
                    "estimated_gpu_usd": 0.001,
                }
            },
            "artifacts": {
                "master": {"sha256": hashlib.sha256(b"master").hexdigest(), "bytes": 6},
                "depth": {"sha256": hashlib.sha256(b"depth").hexdigest(), "bytes": 5},
            },
        }
        return FiniteSceneBundle(
            manifest_path=output_dir / "scene.manifest.json",
            scene_id=request.scene_id,
            artifacts={
                "master": SceneArtifact(
                    role="master",
                    path=master_path,
                    sha256=manifest["artifacts"]["master"]["sha256"],
                    mime_type="image/jpeg",
                    width=1024,
                    height=576,
                ),
                "depth": SceneArtifact(
                    role="depth",
                    path=depth_path,
                    sha256=manifest["artifacts"]["depth"]["sha256"],
                    mime_type="image/jpeg",
                    width=1024,
                    height=576,
                ),
            },
            manifest=manifest,
        )

    async def aclose(self) -> None:
        self.closed = True


def test_prepared_benchmark_is_bounded_and_omits_prompt_text(tmp_path: Path) -> None:
    provider = FakeProvider(tmp_path)
    report = asyncio.run(
        collect_benchmark(
            provider,
            BenchmarkConfig(
                mode="prepared",
                samples=3,
                output_root=tmp_path / "output",
                gpu="RTX_PRO_6000",
            ),
        )
    )

    assert report["result"] == "passed"
    assert report["automatic_retries"] == 0
    assert report["aggregate"]["completed_samples"] == 3
    assert report["aggregate"]["remote_ms"]["p95"] == 700
    assert report["aggregate"]["estimated_gpu_usd"] == 0.003
    assert report["privacy"]["prompt_text_recorded"] is False
    assert report["cost_policy"]["zero_cost_claim"] is False
    assert all(request.prompt not in str(report) for request in provider.requests)
    assert provider.closed is True


def test_failed_probe_stops_before_prewarm_or_generation(tmp_path: Path) -> None:
    provider = FakeProvider(tmp_path, ready=False)
    report = asyncio.run(
        collect_benchmark(
            provider,
            BenchmarkConfig(
                mode="prepared",
                samples=1,
                output_root=tmp_path / "output",
                gpu="RTX_PRO_6000",
            ),
        )
    )

    assert report["result"] == "failed"
    assert report["failure"]["type"] == "RuntimeError"
    assert report["cost_policy"]["billing_reconciliation_required"] is True
    assert provider.requests == []
    assert provider.closed is True
