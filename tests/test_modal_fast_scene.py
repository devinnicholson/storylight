from pathlib import Path

ROOT = Path(__file__).parents[1]
SOURCE = (ROOT / "deploy/modal_fast_scene.py").read_text()


def test_fast_scene_models_and_revisions_are_pinned() -> None:
    assert (
        'FAST_MODEL_REVISION = "19683c58b7ea290e55cedd8950ae1d86ada7ef96"' in SOURCE
    )
    assert (
        'DEPTH_MODEL_REVISION = "b4769fd619394250528294b658587285526fab1c"' in SOURCE
    )
    assert 'DEPTH_DTYPE = "float16"' in SOURCE
    assert (
        'MOTION_MODEL_REVISION = "a6d59ee37c13c58261aa79027d3e41cd41960925"' in SOURCE
    )
    assert 'revision=FAST_MODEL_REVISION' in SOURCE
    assert 'revision=DEPTH_MODEL_REVISION' in SOURCE
    assert 'revision=MOTION_MODEL_REVISION' in SOURCE


def test_modal_scene_has_no_public_endpoint_and_keeps_finite_cli() -> None:
    assert "@app.local_entrypoint()" in SOURCE
    assert "@modal.method()" in SOURCE
    assert "@modal.web_endpoint" not in SOURCE
    assert "@modal.asgi_app" not in SOURCE
    assert ".deploy(" not in SOURCE
    assert '"persistent_endpoint": False' in SOURCE


def test_deployed_classes_scale_to_zero_and_require_explicit_prewarm() -> None:
    assert SOURCE.count("min_containers=0") == 2
    assert SOURCE.count("max_containers=1") == 2
    assert "SCALEDOWN_WINDOW_SECONDS = 90" in SOURCE
    assert SOURCE.count("scaledown_window=SCALEDOWN_WINDOW_SECONDS") == 2
    assert SOURCE.count("def prewarm(self)") == 2
    assert "modal.Cls.from_name" not in SOURCE


def test_fast_scene_uses_low_latency_projection_quality_packaging() -> None:
    fast = SOURCE.split("class FastSceneStudio:", 1)[1].split(
        "class MotionUpgradeStudio:", 1
    )[0]
    packaging = SOURCE.split("def _encode_scene_assets", 1)[1].split("@app.cls", 1)[0]

    assert 'format="JPEG"' in packaging
    assert "MASTER_JPEG_QUALITY = 95" in SOURCE
    assert "DEPTH_JPEG_QUALITY = 85" in SOURCE
    assert "quality=MASTER_JPEG_QUALITY" in packaging
    assert "quality=DEPTH_JPEG_QUALITY" in packaging
    assert "subsampling=0" in packaging
    assert "ThreadPoolExecutor(max_workers=2" in packaging
    assert '"packaging_seconds": packaging_seconds' in fast
    assert '"master_media_type": "image/jpeg"' in fast
    assert '"depth_media_type": "image/jpeg"' in fast
    assert '"depth_dtype": DEPTH_DTYPE' in fast
    assert '"negative_prompt_supported": False' in fast
    assert 'result.get("master_media_type") != "image/jpeg"' in SOURCE
    assert 'result.get("depth_media_type") != "image/jpeg"' in SOURCE
    assert 'depth.convert("L").save(' in packaging
    assert "optimize=True" not in packaging
    assert "torch.cuda.empty_cache()" not in fast
    assert "set_progress_bar_config(disable=True)" in fast


def test_sana_sprint_is_the_pinned_production_fast_renderer() -> None:
    fast = SOURCE.split("class FastSceneStudio:", 1)[1].split(
        "class MotionUpgradeStudio:", 1
    )[0]

    assert 'FAST_MODEL = "Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers"' in SOURCE
    assert "diffusers.SanaSprintPipeline.from_pretrained(" in fast
    assert "if not 1 <= steps <= 4" in fast
    assert "num_inference_steps=steps" in fast
    assert 'sprint_timing = {"intermediate_timesteps": None} if steps != 2 else {}' in fast
    assert "**sprint_timing" in fast
    assert "class SprintSceneStudio" not in SOURCE
    assert "sprint_scene_experiment_cli" not in SOURCE


def test_fast_prewarm_executes_shape_matched_cuda_and_depth_work() -> None:
    fast = SOURCE.split("class FastSceneStudio:", 1)[1].split(
        "class MotionUpgradeStudio:", 1
    )[0]

    assert "FAST_PREWARM_WIDTH = 896" in SOURCE
    assert "FAST_PREWARM_HEIGHT = 512" in SOURCE
    assert "if not self.inference_warmed:" in fast
    assert "num_inference_steps=2" in fast
    assert "self.depth_pipe(warmup_master)" in fast
    assert "dtype=torch.float16" in fast
    assert '"inference_warmup_seconds"' in fast


def test_motion_is_projection_native_and_exactly_seamless() -> None:
    motion = SOURCE.split("class MotionUpgradeStudio:", 1)[1].split(
        "def _budget_types", 1
    )[0]

    assert "result + list(reversed(result[:-1]))" in motion
    assert 'width: int = 800' in SOURCE
    assert 'height: int = 448' in SOURCE
    assert 'generated_frames: int = 41' in SOURCE
    assert 'fps: int = 24' in SOURCE


def test_each_stage_reserves_budget_before_gpu_call_and_fails_closed() -> None:
    fast = SOURCE.split("def fast_scene_cli(", 1)[1].split("def motion_upgrade_cli(", 1)[0]
    motion = SOURCE.split("def motion_upgrade_cli(", 1)[1]

    assert fast.index("_guard_and_reserve(") < fast.index("FastSceneStudio().generate.remote(")
    assert motion.index("_guard_and_reserve(") < motion.index(
        "MotionUpgradeStudio().generate.remote("
    )
    assert "Do not release the ledger reservation" in fast
    assert 'FAST_TIMEOUT_SECONDS = 180' in SOURCE
    assert 'MOTION_TIMEOUT_SECONDS = 300' in SOURCE
    assert "expected_experiment_id=experiment_id" in SOURCE


def test_manifest_captures_model_cost_prompt_seed_and_checksum_provenance() -> None:
    for required in (
        '"model_revision"',
        '"estimated_gpu_usd"',
        '"prompt"',
        '"negative_prompt"',
        '"seed"',
        '"sha256"',
        '"source_master_sha256"',
        '"loop_strategy": "exact-ping-pong"',
    ):
        assert required in SOURCE
