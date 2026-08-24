import json
import os
import subprocess
from pathlib import Path

from bookforge.live_scene_planner import LiveSceneWirePlan

ROOT = Path(__file__).parents[1]
KIOSK_LAUNCHER = ROOT / "deploy/jetson/launch-kiosk.sh"
KIOSK_PREFLIGHT = ROOT / "deploy/jetson/check-kiosk-session.sh"


def _write_executable(path: Path, body: str) -> None:
    path.write_text(f"#!/bin/sh\n{body}")
    path.chmod(0o755)


def _write_kiosk_system_fakes(
    fake_bin: Path,
    *,
    locked_hint: str,
    idle_hint: str,
    monitor_state: str,
) -> None:
    loginctl = f"""
property=
for argument in "$@"; do
  case "$argument" in
    --property=*) property="${{argument#--property=}}" ;;
  esac
done
case "$1" in
  list-sessions) printf '2 {os.getuid()} tester seat0 tty2 active yes 1h\\n' ;;
  show-session)
    case "$property" in
      User) printf '{os.getuid()}\\n' ;;
      Type) printf 'x11\\n' ;;
      Remote) printf 'no\\n' ;;
      Active) printf 'yes\\n' ;;
      State) printf 'active\\n' ;;
      LockedHint) printf '{locked_hint}\\n' ;;
      IdleHint) printf '{idle_hint}\\n' ;;
      *) exit 1 ;;
    esac
    ;;
  *) exit 1 ;;
esac
"""
    _write_executable(fake_bin / "loginctl", loginctl)
    _write_executable(
        fake_bin / "xset",
        f"printf 'DPMS is Enabled\\n  Monitor is {monitor_state}\\n'\n",
    )
    _write_executable(
        fake_bin / "systemd-inhibit",
        """printf 'inhibitor=sleep\n'
while [ "$#" -gt 0 ]; do
  case "$1" in
    --what=*|--who=*|--why=*|--mode=*) shift ;;
    *) exec "$@" ;;
  esac
done
exit 1
""",
    )


def _run_fake_kiosk(
    tmp_path: Path,
    *browser_names: str,
    overrides: dict[str, str] | None = None,
    locked_hint: str = "no",
    idle_hint: str = "no",
    monitor_state: str = "On",
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(parents=True)
    _write_executable(fake_bin / "curl", "exit 0\n")
    _write_kiosk_system_fakes(
        fake_bin,
        locked_hint=locked_hint,
        idle_hint=idle_hint,
        monitor_state=monitor_state,
    )
    for browser_name in browser_names:
        _write_executable(
            fake_bin / browser_name,
            f"printf 'browser={browser_name}\\n'\nprintf 'arg=%s\\n' \"$@\"\n",
        )

    environment = os.environ.copy()
    environment.pop("BOOKFORGE_BROWSER_BIN", None)
    environment.pop("BOOKFORGE_CHROMIUM_BIN", None)
    environment.update(
        {
            "PATH": f"{fake_bin}:/usr/bin:/bin",
            "HOME": str(tmp_path / "home"),
            "XDG_STATE_HOME": str(tmp_path / "state"),
            "XDG_SESSION_ID": "2",
            "DISPLAY": ":1",
            "XAUTHORITY": str(tmp_path / "Xauthority"),
            "BOOKFORGE_KIOSK_URL": "http://127.0.0.1:18081/projector?live=1",
            "BOOKFORGE_READY_URL": "http://127.0.0.1:18081/readyz",
            "BOOKFORGE_KIOSK_STARTUP_TIMEOUT": "1",
        }
    )
    (tmp_path / "Xauthority").write_text("test authority")
    environment.update(overrides or {})
    return subprocess.run(
        [str(KIOSK_LAUNCHER)],
        check=check,
        capture_output=True,
        text=True,
        env=environment,
    )


def test_system_service_has_persistent_private_paths_and_preflight() -> None:
    unit = (ROOT / "deploy/jetson/systemd/bookforge@.service").read_text()

    assert unit.count("[Service]") == 1
    assert "ExecStartPre=/opt/bookforge/.venv/bin/python -m bookforge.edge_preflight" in unit
    assert "--host 127.0.0.1" in unit
    assert "StateDirectory=bookforge" in unit
    assert "CacheDirectory=bookforge" in unit
    assert "UMask=0077" in unit
    assert "EnvironmentFile=/etc/bookforge/bookforge.env" in unit


def test_projector_adapts_dark_scenes_without_an_extra_generation_pass() -> None:
    projector = (ROOT / "src/bookforge/static/projector.js").read_text()
    stylesheet = (ROOT / "src/bookforge/static/projector.css").read_text()

    assert "sample.width = 32;" in projector
    assert "sample.height = 18;" in projector
    assert "Math.log(targetLuma) / Math.log(boundedLuma)" in projector
    assert "fallbackExposure: Math.min(1.45" in projector
    assert "uniform float u_gamma;" in projector
    assert "color = pow(max(color, vec3(0.0)), vec3(u_gamma));" in projector
    assert "gl.uniform1f(gammaLocation, projectionTone.gamma);" in projector
    assert "projectionGamma: projectionTone.gamma" in projector
    assert "scene.dataset.projectionGamma = renderer.projectionGamma.toFixed(3);" in projector
    assert "timings.projectionMeanLuma = renderer.projectionMeanLuma;" in projector
    assert "projectionExposureForImage(masterImage).toFixed(3)" in projector
    assert "filter: brightness(var(--projection-exposure, 1));" in stylesheet


def test_projector_renders_depth_at_source_resolution_before_display_upscale() -> None:
    projector = (ROOT / "src/bookforge/static/projector.js").read_text()

    assert "function sizeDepthCanvasToSource(canvas, image)" in projector
    assert "canvas.width = Math.min(LOGICAL_WIDTH, sourceWidth);" in projector
    assert "canvas.height = Math.min(LOGICAL_HEIGHT, sourceHeight);" in projector
    assert "const renderSize = sizeDepthCanvasToSource(canvas, fallback);" in projector
    assert "timings.renderPixels = renderSize.pixels;" in projector
    assert "canvas.width = LOGICAL_WIDTH;" not in projector
    assert "canvas.height = LOGICAL_HEIGHT;" not in projector


def test_projector_caps_only_the_continuous_depth_draw_pass_at_30_fps() -> None:
    projector = (ROOT / "src/bookforge/static/projector.js").read_text()

    assert "const DEPTH_RENDER_TARGET_FPS = 30;" in projector
    assert "const frameIntervalMs = 1000 / DEPTH_RENDER_TARGET_FPS;" in projector
    assert "timestamp - lastRenderedAt >= frameIntervalMs - 1" in projector
    assert "canvas.dataset.depthRenderedFrames = String(renderedFrames);" in projector
    assert "canvas.dataset.depthSkippedFrames = String(skippedFrames);" in projector
    assert "targetFps: DEPTH_RENDER_TARGET_FPS" in projector


def test_projector_promotes_provisional_preview_without_calling_it_final() -> None:
    projector = (ROOT / "src/bookforge/static/projector.js").read_text()
    stylesheet = (ROOT / "src/bookforge/static/projector.css").read_text()
    workbench = (ROOT / "src/bookforge/static/workbench.js").read_text()

    assert 'asset.role === "preview" && asset.kind === "image"' in projector
    assert 'commitSceneVersion(version, "preview-composed"' in projector
    assert 'preview_ready: "Generated visual sketch live"' in projector
    assert 'stage === "preview_ready"' in projector
    assert ".preview-scene-image" in stylesheet
    assert 'preview_ready: "Generated visual sketch is live"' in workbench
    assert "Gemma is directing the final artwork" in workbench


def test_kiosk_preserves_chromium_sandbox_and_waits_for_readiness() -> None:
    unit = (ROOT / "deploy/jetson/systemd/bookforge-kiosk.service").read_text()
    launcher = KIOSK_LAUNCHER.read_text()

    assert unit.count("[Service]") == 1
    assert "Restart=always" in unit
    assert "RestartPreventExitStatus=78" in unit
    assert "/readyz" in launcher
    assert "--no-sandbox" not in launcher


def test_kiosk_launcher_shell_is_valid_and_browser_precedence_is_explicit() -> None:
    subprocess.run(["bash", "-n", str(KIOSK_LAUNCHER)], check=True)
    subprocess.run(["bash", "-n", str(KIOSK_PREFLIGHT)], check=True)
    launcher = KIOSK_LAUNCHER.read_text()

    assert launcher.index("${BOOKFORGE_BROWSER_BIN:-}") < launcher.index(
        "${BOOKFORGE_CHROMIUM_BIN:-}"
    )
    assert launcher.index("command -v chromium") < launcher.index("command -v firefox")
    assert '--private-window "$KIOSK_URL"' in launcher
    assert '"${SCRIPT_DIR}/check-kiosk-session.sh" || exit 78' in launcher
    assert "--what=sleep" in launcher
    assert "--what=idle:sleep" not in launcher


def test_kiosk_preflight_refuses_locked_idle_and_dark_sessions(tmp_path: Path) -> None:
    cases = (
        ("locked", {"locked_hint": "yes"}, "is locked"),
        ("idle", {"idle_hint": "yes"}, "is idle"),
        ("display-off", {"monitor_state": "Off"}, "display is off"),
    )
    for name, state, expected_error in cases:
        result = _run_fake_kiosk(
            tmp_path / name,
            "firefox",
            check=False,
            **state,
        )

        assert result.returncode == 78
        assert expected_error in result.stderr
        assert "browser=firefox" not in result.stdout


def test_preferred_firefox_override_uses_only_supported_kiosk_flags(tmp_path: Path) -> None:
    firefox = tmp_path / "bin" / "firefox"
    legacy = tmp_path / "bin" / "legacy-chromium-wrapper"
    result = _run_fake_kiosk(
        tmp_path,
        "firefox",
        "legacy-chromium-wrapper",
        overrides={
            "BOOKFORGE_BROWSER_BIN": str(firefox),
            "BOOKFORGE_CHROMIUM_BIN": str(legacy),
        },
    )

    assert "browser=firefox" in result.stdout
    assert "inhibitor=sleep" in result.stdout
    assert "arg=--kiosk" in result.stdout
    assert "arg=--private-window" in result.stdout
    assert "arg=http://127.0.0.1:18081/projector?live=1" in result.stdout
    assert "--app=" not in result.stdout
    assert "--disable-background-networking" not in result.stdout
    assert "--user-data-dir" not in result.stdout


def test_legacy_chromium_override_retains_hardened_arguments(tmp_path: Path) -> None:
    legacy = tmp_path / "bin" / "legacy-chromium-wrapper"
    result = _run_fake_kiosk(
        tmp_path,
        "legacy-chromium-wrapper",
        overrides={"BOOKFORGE_CHROMIUM_BIN": str(legacy)},
    )

    assert "browser=legacy-chromium-wrapper" in result.stdout
    assert "inhibitor=sleep" in result.stdout
    assert "arg=--kiosk" in result.stdout
    assert "arg=--app=http://127.0.0.1:18081/projector?live=1" in result.stdout
    assert "arg=--disable-background-networking" in result.stdout
    assert "arg=--disable-component-update" in result.stdout
    assert "arg=--disable-sync" in result.stdout
    assert "arg=--user-data-dir=" in result.stdout


def test_browser_autodetection_prefers_chromium_and_accepts_firefox(tmp_path: Path) -> None:
    preferred = _run_fake_kiosk(tmp_path / "preferred", "chromium", "firefox")
    fallback = _run_fake_kiosk(tmp_path / "fallback", "firefox")

    assert "browser=chromium" in preferred.stdout
    assert "browser=firefox" not in preferred.stdout
    assert "browser=firefox" in fallback.stdout
    assert "arg=--private-window" in fallback.stdout


def test_projector_only_loads_assets_from_the_loopback_cache() -> None:
    projector = (ROOT / "src/bookforge/static/projector.js").read_text()

    assert 'uri.startsWith("/v1/assets/")' in projector
    assert "asset?.storage_uri" not in projector
    assert 'localFetch("/workbench-assets/moon-gate.story-pack.json"' in projector


def test_projector_wake_lock_is_visibility_scoped_and_never_bypasses_login() -> None:
    projector = (ROOT / "src/bookforge/static/projector.js").read_text()

    assert 'navigator.wakeLock.request("screen")' in projector
    assert 'document.visibilityState !== "visible"' in projector
    assert 'document.addEventListener("visibilitychange"' in projector
    assert "state.screenWakeLock?.release()" in projector
    assert 'document.body.dataset.projectorWakeLock = "active"' in projector
    assert "setupProjectorWakeLock();" in projector


def test_projector_does_not_count_background_tab_time_as_dropped_frames() -> None:
    projector = (ROOT / "src/bookforge/static/projector.js").read_text()

    assert "function resetFrameSampling({resetDropped = false} = {})" in projector
    assert "monitorFrames.previous = null" in projector
    assert "monitorFrames.startedAt = null" in projector
    assert 'document.addEventListener("visibilitychange", resetFrameSampling)' in projector
    assert "resetFrameSampling({resetDropped: true})" in projector
    assert "state.droppedFrames = 0" in projector
    assert "monitorFrames.baselineMs = calibration" in projector
    assert "delta > baseline * 1.5" in projector
    assert "Math.round(delta / baseline) - 1" in projector


def test_projector_reports_asset_renderer_paint_and_reader_activation_timings() -> None:
    projector = (ROOT / "src/bookforge/static/projector.js").read_text()

    assert "timings.mediaReadyMs = performance.now() - mediaStartedAt" in projector
    assert "timings.rendererSetupMs = performance.now() - rendererStartedAt" in projector
    assert "await committedPaint.promise" in projector
    assert "timings.firstPaintMs = Math.max(" in projector
    assert "timings.readerSyncMs = performance.now() - readerSyncStartedAt" in projector
    assert "void nextVersion.offsetWidth" in projector
    assert "requestAnimationFrame(() => {" in projector
    assert "activationBreakdown: state.liveActivationBreakdown" in projector
    assert "dataset.mediaReadyMs" in projector
    assert "dataset.rendererSetupMs" in projector
    assert "dataset.firstPaintMs" in projector
    assert "dataset.readerSyncMs" in projector
    assert "const SCENE_CROSSFADE_MS = 320" in projector
    assert "crossfadeMs: SCENE_CROSSFADE_MS" in projector


def test_projector_uses_fast_nonblanking_live_scene_crossfade() -> None:
    projector = (ROOT / "src/bookforge/static/projector.js").read_text()
    stylesheet = (ROOT / "src/bookforge/static/projector.css").read_text()

    assert "const SCENE_RETIRE_GRACE_MS = 360" in projector
    assert "rapidReplacement ? 50 : SCENE_RETIRE_GRACE_MS" in projector
    assert "opacity 280ms ease" in stylesheet
    assert "opacity 320ms ease" in stylesheet
    assert "opacity 180ms ease" in stylesheet
    assert 'version.classList.contains("incoming")' in projector
    assert 'version.classList.contains("retiring")' in projector
    assert 'nextVersion.classList.add("rapid-replacement")' in projector
    assert ".scene-version.current.rapid-replacement { transition: none; }" in stylesheet
    assert "opacity 680ms" not in stylesheet
    assert "opacity 720ms" not in stylesheet
    assert "opacity 700ms" not in stylesheet


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
    assert "passageDraftTheme(page)" in projector
    assert "sceneCompositionLayout(page.scene_spec, layer, index)" in projector
    assert "version.dataset.passageTheme = draftTheme.name" in projector
    assert "version.dataset.draftFocus = draftTheme.focusMotif" in projector
    assert "version.dataset.draftEffect = draftTheme.effectMotif" in projector
    for motif in ("reader", "fox", "whale", "turtle"):
        assert f'data-draft-focus="{motif}"' in stylesheet
    for motif in ("flock", "jellyfish", "school", "swarm", "bloom", "constellation"):
        assert f'data-draft-effect="{motif}"' in stylesheet
    for theme in ("space", "ocean", "forest", "storm", "literacy"):
        assert f'data-passage-theme="{theme}"' in stylesheet
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
    assert "async function activatePage(nextIndex, renderToken = null)" in projector
    assert "await renderPackLayers(state.pack, state.page, renderToken, timings);" in projector
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


def test_live_scene_workbench_uses_progressive_job_contract() -> None:
    markup = (ROOT / "src/bookforge/static/workbench.html").read_text()
    controller = (ROOT / "src/bookforge/static/workbench.js").read_text()

    assert "Generate moving scene" in markup
    assert 'id="generationProgress"' in markup
    assert 'data-stage="draft_ready"' in markup
    assert 'data-stage="master_ready"' in markup
    assert 'data-stage="motion_ready"' in markup
    assert "Later: Read it aloud" in markup
    assert 'fetch("/v1/live-scenes"' in controller
    assert "function acceptedLiveScenePointer(response, snapshot)" in controller
    assert 'response.headers.get("X-Bookforge-Server-Instance-Id")' in controller
    assert 'response.headers.get("X-Bookforge-Session-Revision")' in controller
    assert (
        "acceptedLiveScenePointer(response, snapshot) || await fetchLiveSceneSession()"
        in controller
    )
    assert "/v1/live-scene-sessions/${encodeURIComponent(readerSessionId)}/events" in controller
    assert 'source.addEventListener("scene.session", receive)' in controller
    assert (
        "new EventSource(`/v1/live-scenes/${encodeURIComponent(jobId)}/events`)"
        not in controller
    )
    assert "response.status !== 202" in controller
    assert "snapshot.story_pack" in controller
    assert "broadcastLiveSnapshot" not in controller
    assert "BroadcastChannel" not in controller
    assert "bookforge.liveSceneSnapshot.v1" not in controller
    assert "contentWindow?.postMessage" not in controller
    assert "if (revision < lastLiveRevision) return;" in controller
    assert 'elements.compileButton.textContent = "Try generation again"' in controller
    assert "const shouldPersist = !liveSnapshot || isTerminalSnapshot(liveSnapshot);" in controller
    assert "if (shouldPersist)" in controller
    assert (
        "const semanticsChanged = semanticFingerprint !== lastRenderedSemanticFingerprint;"
        in controller
    )
    assert "if (semanticsChanged)" in controller
    assert "the session SSE remains authoritative" in controller
    assert "dataset.storyPackPersistCount" in controller
    assert "dataset.semanticRenderCount" in controller


def test_projector_hot_swaps_generated_stages_without_reloading() -> None:
    markup = (ROOT / "src/bookforge/static/projector.html").read_text()
    controller = (ROOT / "src/bookforge/static/projector.js").read_text()
    stylesheet = (ROOT / "src/bookforge/static/projector.css").read_text()

    assert 'id="liveGenerationBadge"' in markup
    assert 'query.get("live") === "1"' in controller
    assert "BroadcastChannel" not in controller
    assert "bookforge.liveSceneSnapshot.v1" not in controller
    assert 'window.addEventListener("storage"' not in controller
    assert 'window.addEventListener("message"' not in controller
    assert (
        "async function renderPackLayers(pack, page, renderToken = null, timings = null)"
        in controller
    )
    assert 'commitSceneVersion(version, "motion-composed", null, renderToken)' in controller
    assert "previousVersions.forEach" in controller
    assert "previousRenderer?.destroy();" in controller
    assert "await activatePage(nextIndex, renderToken);" in controller
    assert "liveRenderTokenIsCurrent(renderToken)" in controller
    assert "state.liveRenderAbortController?.abort();" in controller
    assert "new AbortController()" in controller
    assert "Scene asset timed out" in controller
    assert "livePageAssetFingerprint" in controller
    assert "liveCommittedRevision" in controller
    assert "liveRenderPending" in controller
    assert "liveModeSatisfiesStage" in controller
    assert "renderLiveGenerationBadge(snapshot, {activated: false})" in controller
    assert "state.liveCommittedRevision === revision" in controller
    assert 'setEvent("scene.retrying"' in controller
    assert "window.location.reload" not in controller
    assert ".scene-version.retiring" in stylesheet
    assert "@keyframes draft-camera" in stylesheet
    assert ".generated-scene.draft-composed" in stylesheet
    assert "dataset.terminal = String(state.liveTerminal)" in controller
    assert '.live-generation-badge[data-terminal="true"]' in stylesheet


def test_server_rendezvous_synchronizes_separate_workbench_and_kiosk_browsers() -> None:
    workbench = (ROOT / "src/bookforge/static/workbench.js").read_text()
    projector = (ROOT / "src/bookforge/static/projector.js").read_text()

    assert 'workbenchQuery.get("session") || "bookforge-live"' in workbench
    assert 'query.get("session") || "bookforge-live"' in projector
    assert "/v1/live-scene-sessions/${encodeURIComponent(readerSessionId)}" in workbench
    assert "/v1/live-scene-sessions/${encodeURIComponent(readerSessionId)}/events" in workbench
    assert 'source.addEventListener("scene.session", receive)' in workbench
    assert "async function recoverLiveSceneSession()" in workbench
    assert "restoreInitialScene();" in workbench
    assert "/v1/live-scene-sessions/${encodeURIComponent(SESSION_ID)}" in projector
    assert "/v1/live-scene-sessions/${encodeURIComponent(SESSION_ID)}/events" in projector
    assert "function connectLiveSceneSessionEvents()" in projector
    assert 'source.addEventListener("scene.session", receive)' in projector
    assert "async function rendezvousLiveScene()" in projector
    assert "connectLiveJobEvents" not in projector
    assert "new EventSource(`/v1/live-scenes/${encodeURIComponent(jobId)}/events`)" not in projector
    assert "sessionRevision < state.liveSessionRevision" in projector
    assert "state.liveSessionJobId !== jobId" in projector
    assert "server_instance_id" in workbench
    assert "serverChanged || revisionAdvanced" in workbench
    assert "!serverInstanceId || !Number.isInteger(sessionRevision)" in projector
    assert "scheduleLiveSceneRendezvous(250);" in projector
    # The authoritative session stream is the only live-stage delivery path.
    assert "BroadcastChannel" not in projector
    assert "bookforge.liveSceneSnapshot.v1" not in projector


def test_projector_uses_session_polling_only_while_session_sse_is_unhealthy() -> None:
    projector = (ROOT / "src/bookforge/static/projector.js").read_text()

    assert "liveSessionStreamHealthy: false" in projector
    assert "function liveSessionStreamIsHealthy()" in projector
    assert "source.readyState === EventSource.OPEN" in projector
    assert "state.liveSessionStreamHealthy = true;" in projector
    assert "stopLiveSceneRendezvous();" in projector
    assert "state.liveSessionStreamHealthy = false;" in projector
    assert "scheduleLiveSceneRendezvous(0);" in projector
    assert (
        "if (!LIVE_MODE || state.liveRendezvousInFlight || liveSessionStreamIsHealthy()) return;"
    ) in projector
    assert "scheduleLiveSceneRendezvous();" in projector


def test_workbench_uses_session_polling_only_while_session_sse_is_unhealthy() -> None:
    workbench = (ROOT / "src/bookforge/static/workbench.js").read_text()

    assert "let liveSessionStreamHealthy = false" in workbench
    assert "function liveSessionStreamIsHealthy()" in workbench
    assert "if (!liveSessionStreamIsHealthy()) pollLiveScene" in workbench
    assert "if (liveSessionStreamIsHealthy()) return" in workbench
    assert "liveSessionStreamHealthy = true" in workbench
    assert "liveSessionStreamHealthy = false" in workbench
    assert "function startLivePollingFallback()" in workbench
    assert workbench.count("startLivePollingFallback();") == 2
    assert "pollLiveScene(activeLiveJobId, liveRequestEpoch)" in workbench


def test_projector_restores_session_or_fallback_before_subscribing() -> None:
    projector = (ROOT / "src/bookforge/static/projector.js").read_text()

    startup = projector.split("async function startProjector()", 1)[1].split(
        'elements.previous.addEventListener("click"', 1
    )[0]
    assert "restoreAuthoritativeLiveSession()" in startup
    assert startup.index("restoreAuthoritativeLiveSession()") < startup.index(
        "setupLiveSceneTransport();"
    )
    assert startup.index("loadStoryPack();") < startup.index("setupLiveSceneTransport();")
    assert startup.index("setupLiveSceneTransport();") < startup.index(
        "if (READER_MODE && state.page) connectReaderSession();"
    )
    restore = projector.split("async function restoreAuthoritativeLiveSession()", 1)[1].split(
        "async function startProjector()", 1
    )[0]
    assert "/v1/live-scene-sessions/${encodeURIComponent(SESSION_ID)}" in restore
    assert "if (!pointer.job?.story_pack) return false;" in restore
    assert "acceptLiveSceneSessionPointer(pointer);" in restore
    assert "await state.liveTransition;" in restore
    assert "state.liveCommittedJobId === pointer.job.job_id" in restore
    assert "stream replays its current pointer" in startup
    assert "if (!state.liveServerInstanceId) scheduleLiveSceneRendezvous(250);" in projector
    assert projector.count("setupLiveSceneTransport();") == 1
    assert "void startProjector();" in projector


def test_embedded_visual_preview_defers_reader_transport() -> None:
    workbench = (ROOT / "src/bookforge/static/workbench.js").read_text()
    projector = (ROOT / "src/bookforge/static/projector.js").read_text()

    assert "&reader=0" in workbench
    assert 'const READER_MODE = query.get("reader") !== "0";' in projector
    assert "if (!READER_MODE || readerSessionReusable)" in projector
    assert "if (READER_MODE && state.page) connectReaderSession();" in projector
    assert "readerEnabled: READER_MODE" in projector
    assert "reader=0" not in workbench.split("elements.projectorLink.href", 1)[1].split(";", 1)[0]


def test_projector_elapsed_clock_stops_after_terminal_scene() -> None:
    projector = (ROOT / "src/bookforge/static/projector.js").read_text()

    assert "liveClockTimer: null" in projector
    assert "function synchronizeLiveGenerationClock()" in projector
    assert "if (state.liveJobId && !state.liveTerminal)" in projector
    assert "dataset.clockRunning" in projector
    assert "synchronizeLiveGenerationClock();" in projector
    assert "window.clearInterval(state.liveClockTimer);" in projector
    setup = projector.split("function setupLiveSceneTransport()", 1)[1].split(
        "function defaultCorners()", 1
    )[0]
    assert "setInterval(updateLiveGenerationClock" not in setup


def test_projector_reuses_reader_session_for_visual_only_scene_upgrades() -> None:
    projector = (ROOT / "src/bookforge/static/projector.js").read_text()

    assert "const readerSessionReusable = READER_MODE && Boolean(" in projector
    assert "state.readerConfiguredPageId === nextPage.page_id" in projector
    assert "state.readerConfiguredPageText === nextPage.source_text" in projector
    assert "state.readerConfiguredPageId = state.page.page_id" in projector
    assert "state.readerConfiguredPageText = state.page.source_text" in projector
    assert "readerConfiguredPageId: null" in projector
    assert "readerConfiguredPageText: null" in projector
    assert "state.readerConfiguredPageId = null" in projector
    assert "state.readerConfiguredPageText = null" in projector
    assert "readerSessionReused: readerSessionReusable" in projector
    assert "if (!READER_MODE || readerSessionReusable) {\n    rebuildScene();" in projector
    assert "if (READER_MODE && !readerSessionReusable) goToWord(readerCursor);" in projector
    assert "dataset.readerSessionReused" in projector


def test_projector_never_reveals_an_undrawn_or_lost_webgl_canvas() -> None:
    projector = (ROOT / "src/bookforge/static/projector.js").read_text()

    assert "1.0 - smoothstep(0.26, 0.88" in projector
    assert "smoothstep(0.88, 0.26" not in projector
    first_draw = projector.index("draw(performance.now());")
    reveal_canvas = projector.index('canvas.classList.add("ready");', first_draw)
    assert first_draw < reveal_canvas
    assert "const firstDrawError = gl.getError();" in projector
    assert "gl.isContextLost() || firstDrawError !== gl.NO_ERROR" in projector
    assert 'canvas.addEventListener("webglcontextlost", revealFallback);' in projector
    assert 'canvas.classList.remove("ready");' in projector
    assert "provider artwork remains visible" in projector
    assert "loadSceneImage(masterAsset.local_uri" in projector
    decoded_call = "startDepthRenderer(\n        canvas,\n        fallback,\n        depthImage,"
    assert decoded_call in projector
    assert "The decoded master image" in projector
    assert "client activate ${Math.round(state.liveActivationMs)} ms" in projector
    assert 'publish("scene.activated"' in projector
    assert "activationMs: state.liveActivationMs" in projector


def test_packaged_kiosk_defaults_join_the_canonical_live_session() -> None:
    example = (ROOT / "deploy/jetson/kiosk.env.example").read_text()
    launcher = (ROOT / "deploy/jetson/launch-kiosk.sh").read_text()

    expected_query = "pack=latest&session=bookforge-live&live=1"
    assert expected_query in example
    assert expected_query in launcher


def test_live_ui_displays_backend_metrics_without_a_saved_local_fallback() -> None:
    markup = (ROOT / "src/bookforge/static/workbench.html").read_text()
    controller = (ROOT / "src/bookforge/static/workbench.js").read_text()
    stylesheet = (ROOT / "src/bookforge/static/workbench.css").read_text()
    assert "packaging ${formatBackendMs(metrics.packaging_ms)}" in controller
    assert "renderer prep ${formatBackendMs(metrics.preparation_ms)}" in controller

    assert 'id="generationMetrics"' in markup
    assert 'id="planningPrivacy"' in markup
    assert "backend wall" in controller
    assert "provider remote" in controller
    assert "inference" in controller
    assert "cache promotion" in controller
    assert "estimated_gpu_usd" in controller
    assert "metrics.models" in controller
    assert "Local Gemma plan" in controller
    assert "Cached local Gemma plan" in controller
    assert 'metrics.planning_cache_hit ? "local cache"' in controller
    assert "renderer received only visual direction" in controller
    assert "Saved locally" not in controller
    assert "dataset.terminal = String(isTerminalSnapshot(snapshot))" in controller
    assert '.generation-progress:not([data-terminal="true"])' in stylesheet
    assert '.generation-progress:not([data-terminal="true"]) .stage-rail li.current i' in stylesheet


def test_workbench_rehearsal_prepares_edge_plan_and_text_free_renderer_concurrently() -> None:
    markup = (ROOT / "src/bookforge/static/workbench.html").read_text()
    controller = (ROOT / "src/bookforge/static/workbench.js").read_text()

    assert 'id="prewarmButton"' in markup
    assert 'id="rendererReadiness"' in markup
    assert 'workbenchQuery.get("rehearsal") === "1"' in controller
    assert 'post("/v1/live-scene-provider/prewarm"' in controller
    assert 'post("/v1/live-scene-planner/prepare"' in controller
    assert "Promise.allSettled" in controller
    assert "scaledown_window_seconds: 600" in controller
    assert "if (rehearsalMode) prewarmRenderer();" in controller
    prewarm_source = controller[
        controller.index("async function prewarmRenderer()") : controller.index(
            "function setSceneReady"
        )
    ]
    assert "const text = elements.story.value.trim();" in prewarm_source
    assert "const visualStyle = elements.style.value.trim()" in prewarm_source
    renderer_body = prewarm_source[
        prewarm_source.index('post("/v1/live-scene-provider/prewarm"') : prewarm_source.index(
            'post("/v1/live-scene-planner/prepare"'
        )
    ]
    assert "text" not in renderer_body
    assert "visual_style" not in renderer_body
    assert "include_motion: false" in prewarm_source
    assert "preparedPlanKey" in prewarm_source
    assert "const preparedKey = text;" in prewarm_source
    plan_key_source = controller[
        controller.index("function currentPlanKey()") : controller.index(
            "function markRendererReady"
        )
    ]
    assert "elements.story.value.trim()" in plan_key_source
    assert "elements.style" not in plan_key_source
    assert "visual-style auditions reuse the private semantic plan" in controller


def test_workbench_starts_only_text_free_edge_warmup_in_background() -> None:
    controller = (ROOT / "src/bookforge/static/workbench.js").read_text()

    assert 'fetch("/v1/live-scene-planner/warmup"' in controller
    assert "void warmEdgePlanner();" in controller
    warmup = controller.split("async function warmEdgePlanner()", 1)[1].split(
        "async function", 1
    )[0]
    assert "elements.story" not in warmup
    assert "JSON.stringify" not in warmup


def test_hardware_evidence_and_privacy_scripts_are_executable() -> None:
    for name in (
        "collect-evidence.sh",
        "check-privacy.sh",
        "check-device.sh",
        "check-kiosk-session.sh",
        "warm-asr.sh",
    ):
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


def test_api_launchers_bound_shutdown_with_long_lived_scene_streams() -> None:
    makefile = (ROOT / "Makefile").read_text()
    service = (ROOT / "deploy/jetson/systemd/bookforge@.service").read_text()

    assert "--timeout-graceful-shutdown 3" in makefile
    assert "--timeout-graceful-shutdown 3" in service


def test_optimized_jetson_gemma_fixture_matches_the_production_wire_contract() -> None:
    fixture = json.loads(
        (ROOT / "benchmarks/jetson-gemma3-optimized-schema-request.json").read_text()
    )

    assert fixture["format"] == LiveSceneWirePlan.model_json_schema()
    assert fixture["options"] == {
        "temperature": 0,
        "num_ctx": 4096,
        "num_predict": 180,
    }
    assert fixture["model"] == "gemma3:1b-it-q4_K_M"
