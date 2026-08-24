const elements = {
  style: document.querySelector("#styleInput"),
  story: document.querySelector("#storyInput"),
  micButton: document.querySelector("#micButton"),
  micButtonText: document.querySelector("#micButtonText"),
  compileButton: document.querySelector("#compileButton"),
  prewarmButton: document.querySelector("#prewarmButton"),
  rendererPreflight: document.querySelector("#rendererPreflight"),
  rendererReadiness: document.querySelector("#rendererReadiness"),
  rendererReadinessDetail: document.querySelector("#rendererReadinessDetail"),
  projectorLink: document.querySelector("#projectorLink"),
  projectionPreview: document.querySelector("#projectionPreview"),
  projectorFrame: document.querySelector("#projectorFrame"),
  interim: document.querySelector("#interimText"),
  status: document.querySelector("#status"),
  empty: document.querySelector("#emptyState"),
  results: document.querySelector("#results"),
  error: document.querySelector("#errorBox"),
  model: document.querySelector("#modelMetric"),
  time: document.querySelector("#timeMetric"),
  tokens: document.querySelector("#tokenMetric"),
  summary: document.querySelector("#sceneSummary"),
  layers: document.querySelector("#layerList"),
  layerCount: document.querySelector("#layerCount"),
  triggers: document.querySelector("#triggerList"),
  triggerCount: document.querySelector("#triggerCount"),
  supports: document.querySelector("#supportList"),
  questions: document.querySelector("#questionList"),
  raw: document.querySelector("#rawOutput"),
  micLevel: document.querySelector("#micLevel"),
  browserNote: document.querySelector("#browserNote"),
  generationProgress: document.querySelector("#generationProgress"),
  generationStage: document.querySelector("#generationStage"),
  generationElapsed: document.querySelector("#generationElapsed"),
  generationBar: document.querySelector("#generationBar"),
  stageRail: document.querySelector("#stageRail"),
  artifactStrip: document.querySelector("#artifactStrip"),
  generationProvider: document.querySelector("#generationProvider"),
  generationJob: document.querySelector("#generationJob"),
  generationRevision: document.querySelector("#generationRevision"),
  generationMetrics: document.querySelector("#generationMetrics"),
  planningPrivacy: document.querySelector("#planningPrivacy"),
  generationWarning: document.querySelector("#generationWarning"),
};

let listening = false;
let audioContext = null;
let analyser = null;
let stream = null;
let mediaRecorder = null;
let audioChunks = [];
let partialTimer = null;
let partialBusy = false;
let partialInFlight = Promise.resolve();
let partialBytes = 0;
let recordingEpoch = 0;
let readerGeneration = null;
let activePageText = null;
let starting = false;
let sceneReady = false;
const workbenchQuery = new URLSearchParams(window.location.search);
const readerSessionId = workbenchQuery.get("session") || "bookforge-live";
const rehearsalMode = workbenchQuery.get("rehearsal") === "1";
const LIVE_SCENE_STORAGE_KEY = "bookforge.liveSceneSnapshot.v1";
const LIVE_SCENE_CHANNEL = "bookforge.live-scenes";
const liveSceneChannel = "BroadcastChannel" in window
  ? new BroadcastChannel(LIVE_SCENE_CHANNEL)
  : null;
let liveEventSource = null;
let liveSessionEventSource = null;
let livePollTimer = null;
let liveElapsedTimer = null;
let liveStartedAt = 0;
let liveElapsedBaseMs = 0;
let liveElapsedBaseAt = 0;
let liveIsTerminal = false;
let liveRequestEpoch = 0;
let activeLiveJobId = null;
let latestLiveSnapshot = null;
let lastLiveRevision = -1;
let liveSessionRevision = 0;
let liveServerInstanceId = null;
let rendererPrewarming = false;
let rendererWarmExpiryTimer = null;

const LIVE_STAGES = ["queued", "planning", "draft_ready", "master_ready", "motion_ready"];
const STAGE_LABELS = {
  queued: "Generation job queued",
  planning: "Planning the visual world",
  draft_ready: "Animated draft is live",
  master_ready: "Artwork and depth are live",
  motion_ready: "Cinematic motion loop is live",
  failed: "Generation stopped",
};

function setStatus(state, text) {
  elements.status.dataset.state = state;
  elements.status.lastChild.textContent = ` ${text}`;
}

function setRendererReadiness(state, title, detail, {buttonDisabled = false} = {}) {
  elements.rendererPreflight.dataset.state = state;
  elements.rendererReadiness.textContent = title;
  elements.rendererReadinessDetail.textContent = detail;
  elements.prewarmButton.disabled = buttonDisabled;
}

function markRendererReady(expiresInSeconds) {
  window.clearTimeout(rendererWarmExpiryTimer);
  const boundedSeconds = Math.max(0, Number(expiresInSeconds) || 0);
  const minutes = Math.max(1, Math.ceil(boundedSeconds / 60));
  elements.prewarmButton.textContent = "Renderer ready";
  setRendererReadiness(
    "ready",
    `Cloud renderer ready for about ${minutes} min`,
    "Submit a story now to avoid cold-start delay.",
    {buttonDisabled: true},
  );
  rendererWarmExpiryTimer = window.setTimeout(() => {
    elements.prewarmButton.textContent = "Prepare renderer";
    setRendererReadiness(
      "idle",
      "Renderer may be asleep",
      "Prepare it before a judged run; no story text is sent.",
    );
  }, boundedSeconds * 1000);
}

async function inspectRendererReadiness() {
  try {
    const response = await fetch("/v1/live-scene-provider/warm-status", {cache: "no-store"});
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.detail || `Readiness failed (${response.status})`);
    if (payload.state === "prewarmed") {
      markRendererReady(payload.expires_in_seconds);
      return;
    }
    setRendererReadiness(
      "idle",
      "Renderer sleeps between scenes",
      "Prepare it before a judged run; no story text is sent.",
    );
  } catch (error) {
    setRendererReadiness(
      "error",
      "Renderer prewarm unavailable",
      error.message,
      {buttonDisabled: true},
    );
  }
}

async function prewarmRenderer() {
  if (rendererPrewarming) return;
  rendererPrewarming = true;
  elements.prewarmButton.textContent = "Preparing…";
  setRendererReadiness(
    "warming",
    "Preparing the cloud renderer…",
    "This can take about 30–50 seconds from sleep; no story text is sent.",
    {buttonDisabled: true},
  );
  try {
    const response = await fetch("/v1/live-scene-provider/prewarm", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        prewarm_id: `rehearsal-${Date.now().toString(36)}`,
        include_motion: false,
        scaledown_window_seconds: 600,
      }),
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.detail || `Prewarm failed (${response.status})`);
    markRendererReady(payload.expires_in_seconds);
  } catch (error) {
    elements.prewarmButton.textContent = "Retry preparation";
    setRendererReadiness(
      "error",
      "Renderer preparation failed",
      error.message,
    );
  } finally {
    rendererPrewarming = false;
  }
}

function setSceneReady(ready) {
  sceneReady = ready;
  elements.projectorLink.classList.toggle("disabled", !ready);
  elements.projectorLink.setAttribute("aria-disabled", String(!ready));
  elements.projectorLink.tabIndex = ready ? 0 : -1;
  elements.micButton.disabled = !ready || !canRecordAudio;
  elements.projectionPreview.classList.toggle("hidden", !ready);
  if (ready && !listening && !starting) {
    elements.interim.textContent = "Open the projection view, then press Start reading.";
  }
}

function ensureProjectionPreview() {
  if (!elements.projectorFrame.src) {
    elements.projectorFrame.src = `/projector?pack=latest&session=${readerSessionId}&present=1&live=1`;
  }
}

elements.projectorLink.href = `/projector?pack=latest&session=${encodeURIComponent(readerSessionId)}&present=1&live=1`;

// Kept as the public preview hook; it initializes once and never reloads during stage upgrades.
function reloadProjectionPreview() {
  ensureProjectionPreview();
}

function broadcastLiveSnapshot(snapshot) {
  const envelope = {
    type: "bookforge.live-scene",
    sessionId: readerSessionId,
    serverInstanceId: liveServerInstanceId,
    sessionRevision: liveSessionRevision,
    sentAt: Date.now(),
    snapshot,
  };
  try {
    localStorage.setItem(LIVE_SCENE_STORAGE_KEY, JSON.stringify(envelope));
  } catch (_) {
    // Direct messaging and BroadcastChannel still provide the live path if storage is full.
  }
  liveSceneChannel?.postMessage(envelope);
  elements.projectorFrame.contentWindow?.postMessage(envelope, window.location.origin);
}

function safeText(value) {
  const node = document.createElement("span");
  node.textContent = value == null ? "" : String(value);
  return node.innerHTML;
}

async function startAudioMeter() {
  if (stream) return;
  stream = await navigator.mediaDevices.getUserMedia({audio: true, video: false});
  const Context = window.AudioContext || window.webkitAudioContext;
  audioContext = new Context();
  analyser = audioContext.createAnalyser();
  analyser.fftSize = 512;
  audioContext.createMediaStreamSource(stream).connect(analyser);
  const values = new Uint8Array(analyser.fftSize);

  function meter() {
    if (!analyser) return;
    analyser.getByteTimeDomainData(values);
    let energy = 0;
    values.forEach((value) => {
      const normalized = (value - 128) / 128;
      energy += normalized * normalized;
    });
    const rms = Math.sqrt(energy / values.length);
    elements.micLevel.style.width = `${Math.max(6, Math.min(100, rms * 620))}%`;
    requestAnimationFrame(meter);
  }
  requestAnimationFrame(meter);
}

function releaseMicrophone() {
  analyser = null;
  if (stream) stream.getTracks().forEach((track) => track.stop());
  stream = null;
  if (audioContext) audioContext.close();
  audioContext = null;
  elements.micLevel.style.width = "6%";
}

async function startSpeaking() {
  if (starting || listening) return;
  if (!sceneReady) {
    elements.interim.textContent = "Create the scene before starting the reader.";
    return;
  }
  if (!window.MediaRecorder || !navigator.mediaDevices?.getUserMedia) {
    elements.interim.textContent = "Audio recording is unavailable in this browser.";
    return;
  }
  starting = true;
  elements.micButton.disabled = true;
  elements.micButtonText.textContent = "Starting…";
  elements.compileButton.disabled = true;
  try {
    const pageText = elements.story.value.trim();
    if (!pageText) throw new Error("Enter the trusted page text before starting the reader.");
    activePageText = pageText;
    await configureReaderSession(pageText);
    readerGeneration = (await resetReaderSession()).generation;
    await startAudioMeter();
    audioChunks = [];
    partialBytes = 0;
    partialInFlight = Promise.resolve();
    recordingEpoch += 1;
    const epoch = recordingEpoch;
    const preferredType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
      ? "audio/webm;codecs=opus"
      : "";
    mediaRecorder = preferredType
      ? new MediaRecorder(stream, {mimeType: preferredType})
      : new MediaRecorder(stream);
    mediaRecorder.addEventListener("dataavailable", (event) => {
      if (event.data.size > 0) audioChunks.push(event.data);
    });
    mediaRecorder.addEventListener("stop", transcribeRecording, {once: true});
    mediaRecorder.start(500);
    listening = true;
    elements.micButton.disabled = false;
    elements.micButton.classList.add("listening");
    elements.micButtonText.textContent = "Stop reading";
    elements.compileButton.disabled = true;
    elements.interim.textContent = "Listening locally—read the exact page text above.";
    partialTimer = window.setInterval(() => {
      if (!partialBusy) partialInFlight = transcribePartialRecording(epoch);
    }, 2000);
  } catch (error) {
    elements.interim.textContent = `Microphone unavailable: ${error.message}`;
    if (mediaRecorder?.state === "recording") {
      mediaRecorder.removeEventListener("stop", transcribeRecording);
      mediaRecorder.stop();
    }
    mediaRecorder = null;
    audioChunks = [];
    recordingEpoch += 1;
  } finally {
    starting = false;
    if (!listening) resetMicControls();
  }
}

function stopSpeaking() {
  listening = false;
  elements.micButton.classList.remove("listening");
  elements.micButton.disabled = true;
  elements.micButtonText.textContent = "Finishing…";
  elements.interim.textContent = "Finishing the local transcript…";
  window.clearInterval(partialTimer);
  partialTimer = null;
  if (mediaRecorder?.state === "recording") mediaRecorder.stop();
  else resetMicControls();
}

function resetMicControls() {
  elements.micButton.classList.remove("listening");
  elements.micButton.disabled = !sceneReady || !canRecordAudio;
  elements.micButtonText.textContent = "Start reading";
  elements.compileButton.disabled = false;
  window.clearInterval(partialTimer);
  partialTimer = null;
  activePageText = null;
  readerGeneration = null;
  releaseMicrophone();
}

async function transcribeBlob(recording, mimeType) {
  const response = await fetch("/v1/audio:transcribe", {
      method: "POST",
      headers: {"Content-Type": mimeType},
      body: recording,
  });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.detail || `Transcription failed (${response.status})`);
  return payload;
}

async function publishReaderTranscript(text, isFinal, generation = readerGeneration) {
  if (!activePageText || generation === null) {
    throw new Error("The reader session changed; stop and start this reading again.");
  }
  const response = await fetch(`/v1/reader-sessions/${readerSessionId}/transcripts:simulate`, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({
      source: "asr",
      text,
      page_id: "page-01",
      language: "en",
      is_final: isFinal,
      generation,
    }),
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.detail || `Reader event failed (${response.status})`);
  }
}

async function configureReaderSession(pageText) {
  const configureResponse = await fetch(`/v1/reader-sessions/${readerSessionId}`, {
    method: "PUT",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({page_id: "page-01", page_text: pageText}),
  });
  if (!configureResponse.ok) {
    const payload = await configureResponse.json().catch(() => ({}));
    throw new Error(payload.detail || `Reader setup failed (${configureResponse.status})`);
  }
  return configureResponse.json();
}

async function resetReaderSession() {
  const response = await fetch(`/v1/reader-sessions/${readerSessionId}:reset`, {
    method: "POST",
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.detail || `Reader reset failed (${response.status})`);
  }
  return response.json();
}

async function transcribePartialRecording(epoch) {
  if (partialBusy || !listening || epoch !== recordingEpoch || audioChunks.length === 0) return;
  const generation = readerGeneration;
  const mimeType = mediaRecorder?.mimeType || "audio/webm";
  const recording = new Blob(audioChunks, {type: mimeType});
  if (recording.size < 1000 || recording.size === partialBytes) return;
  partialBusy = true;
  partialBytes = recording.size;
  try {
    const payload = await transcribeBlob(recording, mimeType);
    if (!listening || epoch !== recordingEpoch) return;
    elements.interim.textContent = `Whisper hears: ${payload.text}`;
    await publishReaderTranscript(payload.text, false, generation);
  } catch (error) {
    if (listening && epoch === recordingEpoch) {
      elements.interim.textContent = `Still listening; transcript retrying: ${error.message}`;
    }
  } finally {
    partialBusy = false;
  }
}

async function transcribeRecording() {
  const mimeType = mediaRecorder?.mimeType || "audio/webm";
  const recording = new Blob(audioChunks, {type: mimeType});
  const generation = readerGeneration;
  try {
    await partialInFlight;
    if (recording.size < 1000) throw new Error("Recording was too short. Try speaking for a little longer.");
    const payload = await transcribeBlob(recording, mimeType);
    await publishReaderTranscript(payload.text, true, generation);
    elements.interim.textContent = `Finished in ${(payload.total_ms / 1000).toFixed(1)} s. Whisper heard: ${payload.text}`;
  } catch (error) {
    elements.interim.textContent = error.message;
  } finally {
    mediaRecorder = null;
    audioChunks = [];
    partialInFlight = Promise.resolve();
    resetMicControls();
  }
}

function liveRevision(snapshot) {
  const revision = Number(snapshot?.revision);
  return Number.isFinite(revision) ? revision : 0;
}

function providerLabel(provider) {
  if (!provider) return "Provider pending";
  if (typeof provider === "string") return provider;
  const name = provider.name || provider.provider || provider.backend || "generation provider";
  const model = provider.model || provider.model_id || provider.variant;
  const location = provider.region || provider.location;
  return [name, model, location].filter(Boolean).join(" · ");
}

function artifactIsReady(snapshot, role) {
  if (role === "draft") return Boolean(snapshot?.story_pack);
  const explicit = snapshot?.artifacts;
  if (Array.isArray(explicit)) {
    return explicit.some((asset) => (
      (asset.role === role || asset.kind === role) && asset.state !== "failed"
    ));
  }
  if (explicit && typeof explicit === "object") {
    const artifact = explicit[role];
    if (artifact && typeof artifact === "object") return artifact.state !== "failed";
    if (artifact) return true;
  }
  const assets = snapshot?.story_pack?.assets || [];
  return assets.some((asset) => asset.role === role && asset.state === "ready");
}

function snapshotProgress(snapshot) {
  if (snapshot?.complete === true) return 1;
  if (Number.isFinite(snapshot?.progress)) {
    return Math.max(0, Math.min(1, snapshot.progress > 1 ? snapshot.progress / 100 : snapshot.progress));
  }
  return {
    queued: 0.04,
    planning: 0.16,
    draft_ready: 0.38,
    master_ready: 0.74,
    motion_ready: 1,
    failed: 1,
  }[snapshot?.stage] || 0;
}

function stageIsReached(stage, currentStage) {
  if (currentStage === "failed") return false;
  const stageIndex = LIVE_STAGES.indexOf(stage);
  const currentIndex = LIVE_STAGES.indexOf(currentStage);
  return stageIndex >= 0 && currentIndex >= stageIndex;
}

function currentElapsedMs(snapshot = latestLiveSnapshot) {
  if (liveElapsedBaseAt) {
    return liveElapsedBaseMs + (liveIsTerminal ? 0 : performance.now() - liveElapsedBaseAt);
  }
  return liveStartedAt ? performance.now() - liveStartedAt : 0;
}

function updateElapsedClock() {
  elements.generationElapsed.textContent = `${(currentElapsedMs() / 1000).toFixed(1)} s`;
}

function formatBackendMs(value) {
  return Number.isFinite(value) ? `${Math.round(value)} ms` : "pending";
}

function renderPlanningPrivacy(metrics) {
  const scenePlan = (metrics?.models || []).find((model) => model.role === "scene_plan");
  const planningStatus = metrics?.planning_status || "pending";
  const localGemma = planningStatus === "model"
    && /gemma/i.test(scenePlan?.model || "")
    && /^ollama(?:-|$)/i.test(scenePlan?.revision || "");
  if (localGemma) {
    const source = metrics?.planning_cache_hit ? "Cached local Gemma plan" : "Local Gemma plan";
    elements.planningPrivacy.textContent = `${source} · Gemma planned this scene locally; the renderer received only visual direction.`;
  } else if (planningStatus === "model") {
    elements.planningPrivacy.textContent = "Model scene plan · The renderer received only visual direction, not the source passage.";
  } else if (planningStatus === "fallback") {
    elements.planningPrivacy.textContent = "Local fallback plan · The renderer received only visual direction, not the source passage.";
  } else if (planningStatus === "deterministic") {
    elements.planningPrivacy.textContent = "Local deterministic plan · The renderer received only visual direction, not the source passage.";
  } else {
    elements.planningPrivacy.textContent = "Private visual handoff pending.";
  }
}

function renderBackendMetrics(snapshot) {
  const metrics = snapshot?.metrics;
  if (!metrics) {
    elements.generationMetrics.textContent = "Backend metrics pending";
    renderPlanningPrivacy(null);
    return;
  }
  const modelEvidence = (metrics.models || [])
    .map((model) => `${model.role}:${model.model}@${model.revision}`)
    .join(", ") || "models pending";
  const gpu = metrics.gpu || "GPU not reported";
  const warmState = metrics.warm_state || "unknown";
  const estimatedCost = Number(metrics.estimated_gpu_usd || 0).toFixed(4);
  const planningEvidence = metrics.planning_cache_hit ? "local cache" : (metrics.planning_status || "pending");
  elements.generationMetrics.textContent = [
    `backend wall ${formatBackendMs(metrics.elapsed_ms)}`,
    `edge plan ${formatBackendMs(metrics.planning_ms)} (${planningEvidence})`,
    `renderer prep ${formatBackendMs(metrics.preparation_ms)}`,
    `provider remote ${formatBackendMs(metrics.provider_ms)}`,
    `inference ${formatBackendMs(metrics.inference_ms)}`,
    `packaging ${formatBackendMs(metrics.packaging_ms)}`,
    `cache promotion ${formatBackendMs(metrics.cache_ms)}`,
    `provider overhead ${formatBackendMs(metrics.overhead_ms)}`,
    `${warmState} · ${gpu}`,
    `est. GPU $${estimatedCost} (${metrics.cost_source || "unavailable"})`,
    modelEvidence,
  ].join(" · ");
  renderPlanningPrivacy(metrics);
}

function renderGenerationProgress(snapshot) {
  const stage = snapshot.stage || "queued";
  const progress = snapshotProgress(snapshot);
  elements.generationProgress.classList.remove("hidden");
  elements.generationProgress.dataset.stage = stage;
  elements.generationProgress.dataset.terminal = String(isTerminalSnapshot(snapshot));
  elements.generationStage.textContent = STAGE_LABELS[stage] || stage.replaceAll("_", " ");
  elements.generationBar.style.width = `${progress * 100}%`;
  elements.generationProvider.textContent = providerLabel(snapshot.provider);
  elements.generationJob.textContent = `job ${snapshot.job_id || activeLiveJobId || "—"}`;
  elements.generationRevision.textContent = `revision ${liveRevision(snapshot)}`;
  renderBackendMetrics(snapshot);
  elements.stageRail.querySelectorAll("[data-stage]").forEach((item) => {
    const itemStage = item.dataset.stage;
    item.classList.toggle("reached", stageIsReached(itemStage, stage));
    item.classList.toggle("current", itemStage === stage);
  });
  elements.artifactStrip.querySelectorAll("[data-artifact]").forEach((item) => {
    item.classList.toggle("ready", artifactIsReady(snapshot, item.dataset.artifact));
  });
  const warning = snapshot.warning?.message || snapshot.warning || "";
  elements.generationWarning.textContent = warning ? `Motion optional: ${warning}` : "";
  elements.generationWarning.classList.toggle("hidden", !warning);
  updateElapsedClock();
  setStatus(stage === "failed" ? "error" : "working", STAGE_LABELS[stage] || stage);
}

function setGenerateButtonForStage(stage) {
  const labels = {
    queued: "Waiting for generation provider…",
    planning: "Building animated draft…",
    draft_ready: "Preparing artwork + depth…",
    master_ready: "Preparing motion loop…",
    motion_ready: "Generate another moving scene",
    failed: "Try generation again",
  };
  elements.compileButton.textContent = labels[stage] || "Generating moving scene…";
}

function isTerminalSnapshot(snapshot) {
  return snapshot.stage === "motion_ready"
    || snapshot.stage === "failed"
    || snapshot.complete === true
    || snapshot.terminal === true;
}

function stopLiveJobTransport() {
  liveEventSource?.close();
  liveEventSource = null;
  window.clearTimeout(livePollTimer);
  livePollTimer = null;
  window.clearInterval(liveElapsedTimer);
  liveElapsedTimer = null;
}

function setSceneInputsDisabled(disabled) {
  elements.story.disabled = disabled;
  elements.style.disabled = disabled;
}

function finishLiveJob(snapshot) {
  stopLiveJobTransport();
  activeLiveJobId = null;
  setSceneInputsDisabled(false);
  elements.compileButton.disabled = false;
  setGenerateButtonForStage(snapshot.stage);
  if (snapshot.stage === "failed") {
    const detail = snapshot.error?.message || snapshot.error || snapshot.detail || "The generator did not finish.";
    elements.error.textContent = detail;
    elements.error.classList.remove("hidden");
    elements.interim.textContent = "Generation failed before the scene could finish.";
    setStatus("error", "Generation failed");
    return;
  }
  elements.compileButton.textContent = "Generate another moving scene";
  elements.interim.textContent = snapshot.stage === "motion_ready"
    ? "The final moving scene is live. No projector reload occurred."
    : "The best available scene is live; this provider returned no additional motion stage.";
  setStatus("idle", snapshot.stage === "motion_ready" ? "Moving scene ready" : "Scene ready");
}

function renderLiveSnapshot(snapshot, epoch = liveRequestEpoch) {
  if (!snapshot || epoch !== liveRequestEpoch) return;
  const jobId = snapshot.job_id || activeLiveJobId;
  if (activeLiveJobId && jobId && jobId !== activeLiveJobId) return;
  const revision = liveRevision(snapshot);
  if (revision < lastLiveRevision) return;
  if (revision === lastLiveRevision && latestLiveSnapshot?.stage === snapshot.stage) return;
  lastLiveRevision = revision;
  latestLiveSnapshot = snapshot;
  const reportedElapsed = Number.isFinite(snapshot.metrics?.elapsed_ms)
    ? snapshot.metrics.elapsed_ms
    : Number.isFinite(snapshot.elapsed_ms)
      ? snapshot.elapsed_ms
    : Number.isFinite(snapshot.latency_ms)
      ? snapshot.latency_ms
      : Number.isFinite(snapshot.timings?.total_ms) ? snapshot.timings.total_ms : null;
  liveElapsedBaseMs = reportedElapsed ?? (performance.now() - liveStartedAt);
  liveElapsedBaseAt = performance.now();
  liveIsTerminal = isTerminalSnapshot(snapshot);
  renderGenerationProgress(snapshot);
  setGenerateButtonForStage(snapshot.stage);
  if (snapshot.story_pack) {
    renderPack({story_pack: snapshot.story_pack, live_snapshot: snapshot}, {hotSwap: true});
    const stageCopy = {
      draft_ready: "Animated draft live—the generation provider is preparing the master.",
      master_ready: snapshot.complete
        ? "Artwork and depth are live with local WebGL motion."
        : "Artwork and depth are live—optional video motion is preparing next.",
      motion_ready: "Cinematic motion loop live.",
    }[snapshot.stage];
    if (stageCopy) elements.interim.textContent = stageCopy;
  }
  if (isTerminalSnapshot(snapshot)) finishLiveJob(snapshot);
}

async function fetchLiveSceneSession() {
  const response = await fetch(
    `/v1/live-scene-sessions/${encodeURIComponent(readerSessionId)}`,
    {cache: "no-store"},
  );
  if (response.status === 404) return null;
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.detail || `Session rendezvous failed (${response.status})`);
  if (payload.session_id !== readerSessionId || !payload.job?.job_id) {
    throw new Error("Generation session returned an invalid job pointer.");
  }
  return payload;
}

function trackLiveSceneSession(pointer, {restoreInputs = false} = {}) {
  const serverInstanceId = pointer?.server_instance_id;
  const sessionRevision = Number(pointer?.session_revision);
  const snapshot = pointer?.job;
  if (
    !snapshot?.job_id
    || typeof serverInstanceId !== "string"
    || !serverInstanceId
    || !Number.isInteger(sessionRevision)
    || sessionRevision < 1
  ) return false;
  if (liveServerInstanceId && liveServerInstanceId !== serverInstanceId) {
    liveSessionRevision = 0;
    activeLiveJobId = null;
    latestLiveSnapshot = null;
    lastLiveRevision = -1;
  }
  if (sessionRevision < liveSessionRevision) return false;
  const knownJobId = latestLiveSnapshot?.job_id || activeLiveJobId;
  if (sessionRevision === liveSessionRevision && knownJobId && knownJobId !== snapshot.job_id) {
    return false;
  }

  stopLiveJobTransport();
  liveServerInstanceId = serverInstanceId;
  liveSessionRevision = sessionRevision;
  liveRequestEpoch += 1;
  const epoch = liveRequestEpoch;
  activeLiveJobId = snapshot.job_id;
  latestLiveSnapshot = null;
  lastLiveRevision = -1;
  const createdAt = Date.parse(snapshot.created_at || "");
  const ageMs = Number.isFinite(createdAt) ? Math.max(0, Date.now() - createdAt) : 0;
  liveStartedAt = performance.now() - ageMs;
  liveElapsedBaseMs = ageMs;
  liveElapsedBaseAt = performance.now();
  liveIsTerminal = false;
  if (restoreInputs) {
    elements.story.value = snapshot.request?.text || elements.story.value;
    elements.style.value = snapshot.request?.visual_style || elements.style.value;
  }
  elements.compileButton.disabled = true;
  setSceneInputsDisabled(true);
  elements.error.classList.add("hidden");
  elements.generationProgress.classList.remove("hidden");
  ensureProjectionPreview();
  renderLiveSnapshot(snapshot, epoch);
  if (!isTerminalSnapshot(snapshot)) {
    liveElapsedTimer = window.setInterval(updateElapsedClock, 100);
    connectLiveSceneEvents(snapshot.job_id, epoch);
    pollLiveScene(snapshot.job_id, epoch);
  }
  return true;
}

async function pollLiveScene(jobId, epoch) {
  if (epoch !== liveRequestEpoch || !activeLiveJobId) return;
  try {
    const pointer = await fetchLiveSceneSession();
    if (pointer && handleLiveSceneSessionPointer(pointer, {restoreInputs: true})) return;
    if (pointer?.job?.job_id === jobId) renderLiveSnapshot(pointer.job, epoch);
  } catch (error) {
    if (epoch === liveRequestEpoch) elements.interim.textContent = `Scene is still rendering; status retrying: ${error.message}`;
  }
  if (epoch === liveRequestEpoch && activeLiveJobId) {
    livePollTimer = window.setTimeout(() => pollLiveScene(jobId, epoch), 2500);
  }
}

function handleLiveSceneSessionPointer(pointer, {restoreInputs = false} = {}) {
  if (!pointer || pointer.session_id !== readerSessionId) return false;
  const serverInstanceId = pointer.server_instance_id;
  if (typeof serverInstanceId !== "string" || !serverInstanceId) return false;
  if (!pointer.job) {
    if (liveServerInstanceId && liveServerInstanceId !== serverInstanceId) {
      stopLiveJobTransport();
      liveSessionRevision = 0;
      activeLiveJobId = null;
      latestLiveSnapshot = null;
      lastLiveRevision = -1;
    }
    liveServerInstanceId = serverInstanceId;
    return false;
  }
  const sessionRevision = Number(pointer.session_revision);
  const serverChanged = serverInstanceId !== liveServerInstanceId;
  const revisionAdvanced = sessionRevision > liveSessionRevision;
  const jobChanged = pointer.job.job_id !== (activeLiveJobId || latestLiveSnapshot?.job_id);
  if (serverChanged || revisionAdvanced || jobChanged) {
    return trackLiveSceneSession(pointer, {restoreInputs});
  }
  renderLiveSnapshot(pointer.job, liveRequestEpoch);
  return false;
}

function connectLiveSceneSessionEvents() {
  liveSessionEventSource?.close();
  const source = new EventSource(
    `/v1/live-scene-sessions/${encodeURIComponent(readerSessionId)}/events`,
  );
  liveSessionEventSource = source;
  const receive = (event) => {
    try {
      handleLiveSceneSessionPointer(JSON.parse(event.data), {restoreInputs: true});
    } catch (_) {
      elements.interim.textContent = "Ignored an invalid session update; polling remains active.";
    }
  };
  source.addEventListener("scene.session", receive);
  source.addEventListener("message", receive);
  source.addEventListener("error", () => {
    elements.interim.textContent = "Session updates reconnecting; status polling remains available.";
  });
}

function connectLiveSceneEvents(jobId, epoch) {
  liveEventSource?.close();
  const source = new EventSource(`/v1/live-scenes/${encodeURIComponent(jobId)}/events`);
  liveEventSource = source;
  const receive = (event) => {
    if (epoch !== liveRequestEpoch) return;
    try {
      const snapshot = JSON.parse(event.data);
      if (snapshot.revision === undefined && event.lastEventId) snapshot.revision = Number(event.lastEventId);
      renderLiveSnapshot(snapshot, epoch);
    } catch (_) {
      elements.interim.textContent = "Ignored an invalid generation update; polling remains active.";
    }
  };
  source.addEventListener("scene.job", receive);
  source.addEventListener("message", receive);
  source.addEventListener("error", () => {
    if (epoch === liveRequestEpoch && activeLiveJobId) {
      elements.interim.textContent = "Live updates reconnecting; the scene status is also being polled.";
    }
  });
}

function renderPack(payload, {hotSwap = false} = {}) {
  const pack = payload.story_pack;
  const metrics = payload.compile_metrics || payload.metrics;
  const generation = payload.generation_metrics;
  const liveMetrics = payload.live_snapshot?.metrics;
  localStorage.setItem("bookforge.latestStoryPack", JSON.stringify(pack));
  const page = pack.pages[0];
  elements.model.textContent = liveMetrics?.models?.length
    ? liveMetrics.models.map((model) => `${model.model}@${model.revision}`).join(" + ")
    : metrics?.model || pack.compiler_model;
  const totalMs = (metrics?.total_ms || 0) + (generation?.total_ms || 0);
  elements.time.textContent = Number.isFinite(liveMetrics?.elapsed_ms)
    ? `${(liveMetrics.elapsed_ms / 1000).toFixed(1)} s backend wall`
    : totalMs ? `${(totalMs / 1000).toFixed(1)} s` : "Not measured";
  elements.tokens.textContent = metrics ? `${metrics.output_tokens} tokens` : `${page.layers.length + page.triggers.length} parts`;
  elements.summary.textContent = page.scene_summary;
  elements.layerCount.textContent = `${page.layers.length} layers`;
  elements.triggerCount.textContent = `${page.triggers.length} triggers`;
  elements.layers.innerHTML = page.layers.map((layer) => `
    <article class="layer">
      <div class="layer-top"><b>${safeText(layer.kind)}</b><small>z ${layer.z_index}</small></div>
      <p>${safeText(layer.prompt)}</p>
      <em>${safeText(layer.motion)}</em>
    </article>
  `).join("");
  elements.triggers.innerHTML = page.triggers.length ? page.triggers.map((trigger) => `
    <div class="trigger">
      <b>“${safeText(trigger.word)}”</b>
      <span>${safeText(trigger.action)} → ${safeText(trigger.target_layer_id)}</span>
      <small>${trigger.duration_ms} ms</small>
    </div>
  `).join("") : '<div class="trigger"><span>No word triggers generated.</span></div>';
  elements.supports.innerHTML = page.literacy_support.length ? page.literacy_support.map((support) => `
    <div class="support"><b>${safeText(support.word)}</b><span>${support.hint_ladder.map(safeText).join(" → ")}</span></div>
  `).join("") : '<div class="support">No scaffolds generated.</div>';
  elements.questions.innerHTML = page.comprehension.length ? page.comprehension.map((item) => `
    <div class="question">${safeText(item.question)}<span>${item.expected_concepts.map(safeText).join(", ")}</span></div>
  `).join("") : '<div class="question">No questions generated.</div>';
  elements.raw.textContent = JSON.stringify(pack, null, 2);
  elements.empty.classList.add("hidden");
  elements.error.classList.add("hidden");
  elements.results.classList.remove("hidden");
  setSceneReady(true);
  reloadProjectionPreview();
  if (hotSwap) broadcastLiveSnapshot(payload.live_snapshot || {story_pack: pack, stage: "master_ready", revision: 0});
}

async function compileStory() {
  const text = elements.story.value.trim();
  if (!text) {
    elements.interim.textContent = "Add the exact words from one book page first.";
    return;
  }
  if (listening) stopSpeaking();
  stopLiveJobTransport();
  liveRequestEpoch += 1;
  activeLiveJobId = null;
  latestLiveSnapshot = null;
  lastLiveRevision = -1;
  liveStartedAt = performance.now();
  liveElapsedBaseMs = 0;
  liveElapsedBaseAt = liveStartedAt;
  liveIsTerminal = false;
  elements.compileButton.disabled = true;
  setSceneInputsDisabled(true);
  elements.compileButton.textContent = "Starting live generation…";
  elements.error.classList.add("hidden");
  renderGenerationProgress({stage: "queued", revision: 0});
  liveElapsedTimer = window.setInterval(updateElapsedClock, 100);
  ensureProjectionPreview();

  try {
    const response = await fetch("/v1/live-scenes", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        text,
        visual_style: elements.style.value.trim() || "luminous paper theater",
        session_id: readerSessionId,
      }),
    });
    const snapshot = await response.json();
    if (response.status !== 202) throw new Error(snapshot.detail || `Request failed (${response.status})`);
    if (!snapshot.job_id) throw new Error("Generation service returned no job ID.");
    const pointer = await fetchLiveSceneSession();
    if (!pointer || !trackLiveSceneSession(pointer)) {
      throw new Error("Generation session did not retain the accepted job.");
    }
    elements.interim.textContent = "Generation job accepted. The projector will upgrade itself as each stage arrives.";
  } catch (error) {
    stopLiveJobTransport();
    activeLiveJobId = null;
    setSceneInputsDisabled(false);
    elements.error.textContent = error.message;
    elements.error.classList.remove("hidden");
    setStatus("error", "Could not create scene");
    elements.compileButton.disabled = false;
    elements.compileButton.textContent = "Try generation again";
  }
}

async function loadLatestScene() {
  try {
    const response = await fetch("/v1/story-packs/latest", {cache: "no-store"});
    if (!response.ok) return;
    const pack = await response.json();
    const page = pack.pages[0];
    elements.style.value = pack.visual_style;
    elements.story.value = page.source_text;
    renderPack({story_pack: pack});
    setStatus("idle", "Last scene restored");
  } catch (_) {
    // A saved scene is optional; the empty state already explains the first action.
  }
}

async function recoverLiveSceneSession() {
  try {
    const pointer = await fetchLiveSceneSession();
    if (!pointer) return false;
    return handleLiveSceneSessionPointer(pointer, {restoreInputs: true})
      || Boolean(pointer.job?.story_pack);
  } catch (error) {
    elements.interim.textContent = `Could not restore the live session: ${error.message}`;
    return false;
  }
}

async function restoreInitialScene() {
  if (await recoverLiveSceneSession()) return;
  await loadLatestScene();
}

function invalidateScene() {
  if (!sceneReady || listening || starting || activeLiveJobId) return;
  setSceneReady(false);
  setStatus("stale", "Page changed—create it again");
  elements.interim.textContent = "The page changed. Create the scene again before reading it.";
}

elements.micButton.addEventListener("click", () => {
  if (starting) return;
  if (listening) stopSpeaking();
  else startSpeaking();
});
elements.compileButton.addEventListener("click", compileStory);
elements.prewarmButton.addEventListener("click", prewarmRenderer);
elements.projectorLink.addEventListener("click", (event) => {
  if (!sceneReady) event.preventDefault();
});
elements.projectorFrame.addEventListener("load", () => {
  if (latestLiveSnapshot?.story_pack) broadcastLiveSnapshot(latestLiveSnapshot);
});
[elements.style, elements.story].forEach((element) => {
  element.addEventListener("input", invalidateScene);
});

const canRecordAudio = Boolean(window.MediaRecorder && navigator.mediaDevices?.getUserMedia);
document.body.dataset.audioSupport = canRecordAudio ? "available" : "unavailable";
if (!canRecordAudio) {
  elements.micButton.disabled = true;
  elements.interim.textContent = "This browser cannot use the microphone. Open this local page in Chrome to try reading aloud.";
  elements.browserNote.textContent = "The scene creator still works here. Chrome on localhost supports the private local microphone flow.";
}

connectLiveSceneSessionEvents();
restoreInitialScene();
if (rehearsalMode) prewarmRenderer();
else inspectRendererReadiness();
window.addEventListener("beforeunload", () => {
  window.clearTimeout(rendererWarmExpiryTimer);
  liveSessionEventSource?.close();
  stopLiveJobTransport();
});
