(function () {
  "use strict";
  const PROFILE_KEY = "bookforge.hand-surface.v1";
  const TARGET = [{x: .1, y: .1}, {x: .9, y: .1}, {x: .9, y: .9}, {x: .1, y: .9}];

  function validCorners(points) {
    if (!Array.isArray(points) || points.length !== 4) return false;
    if (points.some(p => !Number.isFinite(p.x) || !Number.isFinite(p.y) || p.x < 0 || p.x > 1 || p.y < 0 || p.y > 1)) return false;
    const crosses = points.map((a, i) => {
      const b = points[(i + 1) % 4], c = points[(i + 2) % 4];
      return (b.x - a.x) * (c.y - b.y) - (b.y - a.y) * (c.x - b.x);
    });
    const area = Math.abs(points.reduce((sum, a, i) => {
      const b = points[(i + 1) % 4];
      return sum + a.x * b.y - b.x * a.y;
    }, 0)) / 2;
    return area > .02 && (crosses.every(v => v > .001) || crosses.every(v => v < -.001));
  }

  function project(matrix, point) {
    const d = matrix[6] * point.x + matrix[7] * point.y + 1;
    if (!Number.isFinite(d) || Math.abs(d) < 1e-8) return null;
    const x = (matrix[0] * point.x + matrix[1] * point.y + matrix[2]) / d;
    const y = (matrix[3] * point.x + matrix[4] * point.y + matrix[5]) / d;
    return Number.isFinite(x) && Number.isFinite(y) ? {x, y} : null;
  }

  function init({stage, solve, screenToStage, blocked, projectionIdentity, rendererTelemetry = () => null}) {
    const byId = id => document.getElementById(id);
    const panel = byId("handPanel"), status = byId("handStatus"), video = byId("handVideo");
    const canvas = byId("handFireflies"), ctx = canvas.getContext("2d", {alpha: true});
    const metrics = {mode: "off", frames: 0, inferenceMs: 0, roundTripMs: 0, drawMs: 0, reason: ""};
    let worker = null, stream = null, epoch = 0, busy = false, sentAt = 0, frameId = 0;
    let timer = null, watchdog = null, raf = null, lastDraw = 0, lastPoint = 0, point = null;
    let mapping = null, calibration = null, identity = null, points = [], capturing = false;
    let baselineFps = 0, sampleStart = 0, sampleFrames = 0, slowWindows = 0, slowInference = 0;
    let suspended = false;
    let previousRenderCount = null;
    let lastTimingUpdate = 0;
    const particles = Array.from({length: 28}, (_, i) => ({x: .5, y: .5, phase: i * 2.399, radius: .018 + (i % 7) * .009}));
    const sprite = document.createElement("canvas");
    sprite.width = sprite.height = 40;
    const glow = sprite.getContext("2d"), gradient = glow.createRadialGradient(20, 20, 0, 20, 20, 20);
    gradient.addColorStop(0, "#fff9d0"); gradient.addColorStop(.12, "#ffeaa0");
    gradient.addColorStop(.4, "#edc65a66"); gradient.addColorStop(1, "#edc65a00");
    glow.fillStyle = gradient; glow.fillRect(0, 0, 40, 40);

    function report(message) { status.textContent = message; }
    function clearTargets() {
      stage.querySelectorAll(".hand-target").forEach(node => node.remove());
      document.body.classList.remove("hand-calibrating");
    }
    function stop(reason = "Off. Camera released.") {
      epoch += 1;
      clearTimeout(timer); clearTimeout(watchdog); cancelAnimationFrame(raf);
      worker?.terminate(); worker = null;
      stream?.getTracks().forEach(track => track.stop()); stream = null;
      video.srcObject = null; video.hidden = true;
      busy = false; point = null; capturing = false; mapping = null; suspended = false;
      clearTargets(); ctx.clearRect(0, 0, canvas.width, canvas.height);
      canvas.hidden = true; metrics.mode = "off"; metrics.reason = reason;
      byId("handTiming").textContent = "";
      report(reason);
    }
    function fail(message) { stop(message); }
    function loadMapping() {
      mapping = null;
      try { calibration = JSON.parse(localStorage.getItem(PROFILE_KEY)); } catch (_) { calibration = null; }
      if (calibration?.identity === identity && calibration.projection === projectionIdentity() && validCorners(calibration.points)) {
        mapping = solve(calibration.points, TARGET);
      }
    }
    function frame(now) {
      if (metrics.mode === "off") return;
      const isBlocked = blocked();
      if (isBlocked !== suspended) {
        suspended = isBlocked; point = null;
        sampleStart = now; sampleFrames = 0; slowWindows = 0;
        previousRenderCount = null;
        if (isBlocked) ctx.clearRect(0, 0, canvas.width, canvas.height);
      }
      if (!isBlocked) {
        sampleFrames += 1;
        if (now - sampleStart >= 5000) {
          const fps = sampleFrames * 1000 / (now - sampleStart);
          const renderer = rendererTelemetry();
          const renderFps = renderer && previousRenderCount !== null
            ? (renderer.renderedFrames - previousRenderCount) * 1000 / (now - sampleStart) : null;
          previousRenderCount = renderer?.renderedFrames ?? null;
          if ((baselineFps && fps < baselineFps * .8) || (renderFps !== null && renderFps < renderer.targetFps * .8)) slowWindows += 1;
          else slowWindows = 0;
          sampleStart = now; sampleFrames = 0;
          if (slowWindows >= 2) { fail("Hand effect stopped: playback cadence fell by more than 20%."); return; }
        }
        if (now - lastDraw >= 32) {
          const dt = Math.min(50, now - lastDraw || 33), started = performance.now();
          lastDraw = now;
          ctx.clearRect(0, 0, canvas.width, canvas.height);
          if (metrics.mode === "pointer-demo" && point) lastPoint = now;
          const alpha = point ? Math.max(0, 1 - (now - lastPoint) / 350) : 0;
          if (alpha > 0) {
            ctx.globalCompositeOperation = "lighter";
            for (const p of particles) {
              const angle = p.phase + now * .00045;
              const follow = 1 - Math.exp(-dt / 160);
              p.x += (point.x + Math.cos(angle) * p.radius - p.x) * follow;
              p.y += (point.y + Math.sin(angle) * p.radius * 1.65 - p.y) * follow;
              ctx.globalAlpha = alpha * (.55 + .4 * Math.sin(p.phase + now * .002) ** 2);
              const size = 12 + (p.phase % 4) * 2;
              ctx.drawImage(sprite, p.x * canvas.width - size / 2, p.y * canvas.height - size / 2, size, size);
            }
            ctx.globalAlpha = 1;
          }
          metrics.drawMs = performance.now() - started;
        }
      }
      raf = requestAnimationFrame(frame);
    }
    function acceptPoint(p, now) {
      if (!p || p.x < 0 || p.x > 1 || p.y < 0 || p.y > 1) return;
      if (!point || now - lastPoint > 350) particles.forEach(particle => { particle.x = p.x; particle.y = p.y; });
      point = p; lastPoint = now;
    }
    async function capture(activeEpoch) {
      if (activeEpoch !== epoch || metrics.mode !== "camera") return;
      if (projectionIdentity() !== calibration?.projection && mapping) {
        mapping = null; report("Projection alignment changed. Calibrate the camera again.");
      }
      if (!busy && !blocked() && video.readyState >= 2) {
        busy = true; sentAt = performance.now();
        try {
          const bitmap = await createImageBitmap(video, {resizeWidth: 320, resizeHeight: Math.round(320 * video.videoHeight / video.videoWidth)});
          if (activeEpoch !== epoch) { bitmap.close(); return; }
          const id = ++frameId;
          watchdog = setTimeout(() => fail("Hand tracking stopped: worker did not respond."), 2500);
          worker.postMessage({type: "frame", id, bitmap, timestamp: performance.now()}, [bitmap]);
        } catch (error) { if (activeEpoch === epoch) fail(`Camera frame unavailable: ${error.message}`); }
      }
      if (activeEpoch === epoch) timer = setTimeout(() => capture(activeEpoch), 100);
    }
    async function measureBaseline(activeEpoch) {
      let count = 0, start = performance.now();
      return new Promise(resolve => {
        function tick(now) {
          if (epoch !== activeEpoch) { resolve(0); return; }
          count += 1;
          if (now - start >= 1000) resolve(count * 1000 / (now - start));
          else requestAnimationFrame(tick);
        }
        requestAnimationFrame(tick);
      });
    }
    async function start() {
      stop();
      const activeEpoch = epoch;
      if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
        fail("Camera requires localhost or HTTPS on the device with the webcam."); return;
      }
      if (!window.Worker || !window.OffscreenCanvas || !window.createImageBitmap) {
        fail("This browser cannot run isolated hand tracking. Projection is unchanged."); return;
      }
      metrics.mode = "starting"; report("Checking local model assets…");
      try {
        const response = await fetch("/workbench-assets/mediapipe/manifest.json", {cache: "no-store"});
        if (!response.ok) throw Error("Local model assets missing. Run scripts/install_hand_tracking_assets.py on this server.");
        if ((await response.json()).version !== "0.10.32") throw Error("Unsupported local MediaPipe assets.");
        if (epoch !== activeEpoch) return;
        report("Measuring playback baseline…");
        baselineFps = await measureBaseline(activeEpoch);
        if (epoch !== activeEpoch) return;
        const camera = await navigator.mediaDevices.getUserMedia({audio: false, video: {width: {ideal: 640}, height: {ideal: 480}, frameRate: {ideal: 15, max: 15}}});
        if (epoch !== activeEpoch) { camera.getTracks().forEach(track => track.stop()); return; }
        stream = camera; video.srcObject = camera; video.hidden = false;
        camera.getVideoTracks()[0].addEventListener("ended", () => { if (epoch === activeEpoch) fail("Camera disconnected. Projection is unchanged."); });
        await video.play();
        if (epoch !== activeEpoch) return;
        const settings = camera.getVideoTracks()[0].getSettings();
        identity = JSON.stringify([settings.deviceId, settings.width, settings.height]);
        loadMapping();
        worker = new Worker("/workbench-assets/hand-tracking-worker.js?v=1");
        watchdog = setTimeout(() => fail("Local hand model did not become ready within 30 seconds."), 30000);
        worker.onerror = () => { if (epoch === activeEpoch) fail("Hand worker failed. Check local model assets and browser support."); };
        worker.onmessage = ({data}) => {
          if (epoch !== activeEpoch) return;
          if (data.type === "error") { fail(`Hand tracking stopped: ${data.message}`); return; }
          if (data.type === "ready") {
            clearTimeout(watchdog); metrics.mode = "camera"; canvas.hidden = false;
            sampleStart = performance.now(); sampleFrames = slowWindows = slowInference = 0; previousRenderCount = null;
            report(mapping ? "Local tracking ready. Move one hand over the calibrated surface." : "Camera ready. Calibrate its view before interacting.");
            raf = requestAnimationFrame(frame); void capture(activeEpoch);
          } else if (data.type === "result" && data.id === frameId) {
            clearTimeout(watchdog); busy = false; metrics.frames += 1;
            metrics.inferenceMs = data.inferenceMs; metrics.roundTripMs = performance.now() - sentAt;
            if (!panel.hidden && performance.now() - lastTimingUpdate >= 1000) {
              byId("handTiming").textContent = `Local inference ${metrics.inferenceMs.toFixed(1)} ms · frame round trip ${metrics.roundTripMs.toFixed(1)} ms`;
              lastTimingUpdate = performance.now();
            }
            if (data.inferenceMs > 150) slowInference += 1; else slowInference = 0;
            if (slowInference >= 5) { fail("Hand tracking stopped: sustained inference exceeds the 150 ms budget."); return; }
            if (mapping && data.point && !blocked()) acceptPoint(project(mapping, data.point), performance.now());
          }
        };
        worker.postMessage({type: "init"});
      } catch (error) { if (epoch === activeEpoch) fail(error.message); }
    }
    byId("handOpen").addEventListener("click", () => { panel.hidden = !panel.hidden; });
    byId("handClose").addEventListener("click", () => { panel.hidden = true; });
    byId("handStart").addEventListener("click", start);
    byId("handStop").addEventListener("click", () => stop());
    byId("handDemo").addEventListener("click", () => {
      stop(); metrics.mode = "pointer-demo"; canvas.hidden = false;
      baselineFps = 0; sampleStart = performance.now();
      report("Pointer preview only — no camera or AI tracking. Move the pointer over the scene.");
      raf = requestAnimationFrame(frame);
    });
    window.addEventListener("pointermove", event => {
      if (metrics.mode === "pointer-demo" && !blocked()) acceptPoint(screenToStage(event.clientX, event.clientY), performance.now());
    });
    document.addEventListener("pointerleave", () => { if (metrics.mode === "pointer-demo") point = null; });
    byId("handCalibrate").addEventListener("click", () => {
      if (metrics.mode !== "camera") { report("Start the camera first."); return; }
      mapping = null; point = null; points = []; capturing = true; clearTargets();
      document.body.classList.add("hand-calibrating");
      TARGET.forEach((p, i) => {
        const marker = document.createElement("span"); marker.className = "hand-target";
        marker.textContent = String(i + 1); marker.style.left = `${p.x * 100}%`; marker.style.top = `${p.y * 100}%`;
        stage.appendChild(marker);
      });
      report("In the camera preview, click projected targets 1, 2, 3, 4 in order. Keep the camera fixed.");
    });
    video.addEventListener("click", event => {
      if (!capturing) return;
      const rect = video.getBoundingClientRect();
      points.push({x: (event.clientX - rect.left) / rect.width, y: (event.clientY - rect.top) / rect.height});
      if (points.length < 4) { report(`Target ${points.length} captured. Click target ${points.length + 1}.`); return; }
      capturing = false; clearTargets();
      if (!validCorners(points)) { report("Invalid surface corners. Recalibrate; targets must form a clear quadrilateral."); return; }
      mapping = solve(points, TARGET);
      calibration = {identity, points, projection: projectionIdentity()};
      try { localStorage.setItem(PROFILE_KEY, JSON.stringify(calibration)); } catch (_) { report("Calibration is active for this session; local storage is unavailable."); return; }
      report("Surface calibrated locally. Move one hand near the surface to gather fireflies.");
    });
    document.addEventListener("visibilitychange", () => { if (document.hidden) stop("Paused: tab hidden. Start camera again when ready."); });
    window.addEventListener("pagehide", () => stop());
    return {stop, metrics};
  }
  window.BookforgeHands = {init, validCorners, project};
})();
