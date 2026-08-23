from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_system_service_has_persistent_private_paths_and_preflight() -> None:
    unit = (ROOT / "deploy/jetson/systemd/bookforge@.service").read_text()

    assert unit.count("[Service]") == 1
    assert "ExecStartPre=/opt/bookforge/.venv/bin/python -m bookforge.edge_preflight" in unit
    assert "--host 127.0.0.1" in unit
    assert "StateDirectory=bookforge" in unit
    assert "CacheDirectory=bookforge" in unit
    assert "UMask=0077" in unit
    assert "EnvironmentFile=/etc/bookforge/bookforge.env" in unit


def test_kiosk_preserves_chromium_sandbox_and_waits_for_readiness() -> None:
    unit = (ROOT / "deploy/jetson/systemd/bookforge-kiosk.service").read_text()
    launcher = (ROOT / "deploy/jetson/launch-kiosk.sh").read_text()

    assert unit.count("[Service]") == 1
    assert "Restart=always" in unit
    assert "/readyz" in launcher
    assert "--no-sandbox" not in launcher


def test_projector_only_loads_assets_from_the_loopback_cache() -> None:
    projector = (ROOT / "src/bookforge/static/projector.js").read_text()

    assert 'uri.startsWith("/v1/assets/")' in projector
    assert "asset?.storage_uri" not in projector
    assert 'localFetch("/workbench-assets/moon-gate.story-pack.json"' in projector


def test_projector_uses_depth_webgl_and_enforces_offline_replay_boundary() -> None:
    projector = (ROOT / "src/bookforge/static/projector.js").read_text()
    stylesheet = (ROOT / "src/bookforge/static/projector.css").read_text()

    assert 'asset.role === "master"' in projector
    assert 'asset.role === "depth"' in projector
    assert 'canvas.getContext("webgl2"' in projector
    assert "texture(u_depth, uv).r" in projector
    assert "appendSceneHotspots" in projector
    assert 'query.get("offline") === "1"' in projector
    assert "url.origin !== window.location.origin" in projector
    assert ".depth-scene-canvas.ready" in stylesheet
    assert ".scene-hotspot.action-glow::after" in stylesheet


def test_projector_has_a_generated_motion_hero_instead_of_debug_layers() -> None:
    projector = (ROOT / "src/bookforge/static/projector.js").read_text()
    css = (ROOT / "src/bookforge/static/projector.css").read_text()
    hero = ROOT / "src/bookforge/static/assets/moon-gate-hero-v1.png"
    motion = ROOT / "src/bookforge/static/assets/moon-gate-loop-v1.mp4"
    silver_fox_motion = ROOT / "src/bookforge/static/assets/silver-fox-loop-v1.mp4"

    assert hero.is_file()
    assert hero.stat().st_size > 1_000_000
    assert motion.is_file()
    assert motion.stat().st_size > 50_000
    assert silver_fox_motion.is_file()
    assert silver_fox_motion.stat().st_size > 100_000
    assert "/workbench-assets/assets/moon-gate-loop-v1.mp4" in projector
    assert "/workbench-assets/assets/silver-fox-loop-v1.mp4" in projector
    assert 'poster: "/workbench-assets/assets/moon-gate-hero-v1.png"' in projector
    assert "video.autoplay = true" in projector
    assert "video.loop = true" in projector
    assert "elements.generatedScene.querySelector(selector)" in projector
    assert "hero-trigger-layer" in projector
    assert ".bundled-hero-scene" in css


def test_generated_pack_placeholders_use_separate_storyboard_positions() -> None:
    projector = (ROOT / "src/bookforge/static/projector.js").read_text()
    stylesheet = (ROOT / "src/bookforge/static/projector.css").read_text()

    assert "placeholderLayout(layer.kind, index)" in projector
    assert 'node.classList.add("development-layer")' in projector
    assert "--placeholder-x" in stylesheet
    assert ".generic-layer.development-layer::before" in stylesheet


def test_projector_shortcuts_do_not_fire_while_typing() -> None:
    projector = (ROOT / "src/bookforge/static/projector.js").read_text()

    assert "target instanceof HTMLInputElement" in projector
    assert "target instanceof HTMLTextAreaElement" in projector


def test_projector_validates_and_navigates_every_story_page() -> None:
    projector = (ROOT / "src/bookforge/static/projector.js").read_text()
    markup = (ROOT / "src/bookforge/static/projector.html").read_text()

    assert "for (const page of pack.pages)" in projector
    assert "async function activatePage(nextIndex)" in projector
    assert "await renderPackLayers(state.pack, state.page);" in projector
    assert "await configureReaderSession();" in projector
    assert 'currentUrl.searchParams.set("page", String(nextIndex + 1));' in projector
    assert 'event.key === "["' in projector
    assert 'event.key === "]"' in projector
    assert 'id="previousPageButton"' in markup
    assert 'id="nextPageButton"' in markup
    assert 'id="pageLabel"' in markup


def test_workbench_starts_each_recording_with_a_fresh_reader_session() -> None:
    workbench = (ROOT / "src/bookforge/static/workbench.js").read_text()

    assert "await configureReaderSession(pageText);" in workbench
    assert "await resetReaderSession()).generation" in workbench
    assert "if (!partialBusy) partialInFlight" in workbench


def test_projector_recovers_queue_gaps_without_rewinding() -> None:
    projector = (ROOT / "src/bookforge/static/projector.js").read_text()

    assert "message.dropped_before_sequence" in projector
    assert "requestReaderSynchronization();" in projector
    assert "if (forceAfterCurrent) state.resyncAfterCurrent = true;" in projector
    assert "status.generation === state.generation && nextCursor <= state.cursor" in projector
    assert "state.pendingReaderEvents.push(message);" in projector


def test_projector_rejects_partial_transcripts_from_other_readings() -> None:
    projector = (ROOT / "src/bookforge/static/projector.js").read_text()

    partial_handler = projector.split('if (message.type === "transcript.partial")', 1)[1]
    assert "payload.page_id !== state.page?.page_id" in partial_handler
    assert "payload.generation !== state.generation" in partial_handler


def test_workbench_guards_async_start_and_keeps_page_text_immutable() -> None:
    workbench = (ROOT / "src/bookforge/static/workbench.js").read_text()

    assert "if (starting || listening) return;" in workbench
    assert 'elements.micButtonText.textContent = "Starting…";' in workbench
    assert "activePageText = pageText;" in workbench
    publish_body = workbench.split("async function publishReaderTranscript", 1)[1].split(
        "async function configureReaderSession", 1
    )[0]
    assert "elements.story.value" not in publish_body
    assert "configureReaderSession" not in publish_body


def test_workbench_explains_the_three_step_reader_flow() -> None:
    workbench = (ROOT / "src/bookforge/static/workbench.html").read_text()

    assert "Add a page" in workbench
    assert "Create its scene" in workbench
    assert "Read it aloud" in workbench
    assert "it does not create the page text" in workbench
    assert "Open projection view" in workbench
    assert "Exact projector output" in workbench
    assert 'id="projectorFrame"' in workbench

    controller = (ROOT / "src/bookforge/static/workbench.js").read_text()
    assert "reloadProjectionPreview();" in controller
    assert (
        "elements.projectorFrame.src = `/projector?pack=latest&session=${readerSessionId}"
        in controller
    )
    assert "&present=1" in controller

    projector = (ROOT / "src/bookforge/static/projector.js").read_text()
    assert 'query.get("debug") !== "1"' in projector
    assert 'document.body.classList.add("hud-hidden")' in projector


def test_hardware_evidence_and_privacy_scripts_are_executable() -> None:
    for name in ("collect-evidence.sh", "check-privacy.sh", "check-device.sh", "warm-asr.sh"):
        path = ROOT / "deploy/jetson" / name
        assert path.stat().st_mode & 0o111


def test_device_check_reads_nvcc_release_line_instead_of_build_footer() -> None:
    checker = (ROOT / "deploy/jetson/check-device.sh").read_text()

    assert 'NVCC_OUTPUT="$(nvcc --version' in checker
    assert "grep -m 1 -E 'release [0-9]+\\.[0-9]+'" in checker


def test_hardware_evidence_wrapper_requires_and_passes_service_pid() -> None:
    wrapper = (ROOT / "deploy/jetson/collect-evidence.sh").read_text()

    assert "systemctl show --property MainPID --value" in wrapper
    assert "Bookforge service is not running" in wrapper
    assert '--service-pid "$SERVICE_PID"' in wrapper


def test_bootstrap_keeps_jetpack_python_packages_visible() -> None:
    bootstrap = (ROOT / "deploy/jetson/bootstrap.sh").read_text()

    assert "python3 -m venv --system-site-packages" in bootstrap
    assert 'pip install --upgrade "$REPO_ROOT"' in bootstrap
