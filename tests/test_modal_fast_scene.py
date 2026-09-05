from pathlib import Path

ROOT = Path(__file__).parents[1]
SOURCE = (ROOT / "deploy/modal_fast_scene.py").read_text()


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
    assert SOURCE.count("def prewarm(self") == 2
    assert "modal.Cls.from_name" not in SOURCE


def test_provisional_preview_is_explicitly_low_resolution_and_depth_free() -> None:
    studio = SOURCE.split("def generate_preview(", 1)[1].split("def _generate_master(", 1)[0]
    cli = SOURCE.split("def preview_scene_cli(", 1)[1].split("def motion_upgrade_cli(", 1)[0]

    assert "minimum=256" in studio
    assert "preview dimensions cannot exceed 640x384" in studio
    assert "steps=1" in studio
    assert "self.depth_pipe" not in studio
    assert '"provisional_preview_only": True' in cli
    assert '"source_text_allowed": False' in cli
    assert "prewarm_receipt = studio.prewarm.remote()" in cli
    assert "billable_remote_seconds = prewarm_remote_seconds + remote_seconds" in cli
    assert cli.index("_guard_and_reserve(") < cli.index("studio.generate_preview.remote(")


def test_each_stage_reserves_budget_before_gpu_call_and_fails_closed() -> None:
    fast = SOURCE.split("def fast_scene_cli(", 1)[1].split("def motion_upgrade_cli(", 1)[0]
    motion = SOURCE.split("def motion_upgrade_cli(", 1)[1]

    assert fast.index("_guard_and_reserve(") < fast.index("FastSceneStudio().generate.remote(")
    assert motion.index("_guard_and_reserve(") < motion.index(
        "MotionUpgradeStudio().generate.remote("
    )
    assert "Do not release the ledger reservation" in fast
    assert "FAST_TIMEOUT_SECONDS = 180" in SOURCE
    assert "MOTION_TIMEOUT_SECONDS = 300" in SOURCE
    assert "expected_experiment_id=experiment_id" in SOURCE
