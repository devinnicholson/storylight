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
  voiceReview: document.querySelector("#voiceReview"),
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
let finalizing = false;
let generationSubmitting = false;
let pendingSubmission = null;
let generationReconciling = false;
let sceneReady = false;
let voiceGeneration = {
  latest: null, active: null, completed: null, timer: null,
  publishing: false, presentationInFlight: null, attempted: new Set(),
};
let projectorPreviewUrl = null;
const workbenchQuery = new URLSearchParams(window.location.search);
const voiceMode = workbenchQuery.get("voice") === "1";
const demoMode = !voiceMode && workbenchQuery.get("demo") === "1";
document.body.dataset.demo = String(demoMode);
document.body.dataset.voice = String(voiceMode);
const readerSessionId = workbenchQuery.get("session") || (voiceMode ? "voice-demo" : "bookforge-live");
const restoreLatestScene = !workbenchQuery.has("session")
  || workbenchQuery.get("restore") === "latest";
let liveSessionEventSource = null;
let liveSessionStreamHealthy = false;
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
let lastRenderedSemanticFingerprint = null;
let storyPackPersistCount = 0;
let semanticRenderCount = 0;
let liveSessionRevision = 0;
let liveServerInstanceId = null;
let rendererPrewarming = false;
let rendererWarmExpiryTimer = null;
let rendererWarmUntil = 0;
let preparedPlanKey = null;
let edgePlanPreparationTimer = null;
let edgePlanPreparingKey = null;
let edgePlannerWarmUntil = 0;
const EDGE_PLANNER_KEEP_WARM_MS = 8 * 60 * 1000;

const LIVE_STAGES = [
  "queued",
  "planning",
  "draft_ready",
  "preview_ready",
  "master_ready",
  "motion_ready",
];
const STAGE_LABELS = {
  queued: "Generation job queued",
  planning: "Planning the visual world",
  draft_ready: "Animated draft is live",
  preview_ready: "Generated visual sketch is live",
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

function currentPlanKey() {
  return elements.story.value.trim();
}

function rendererExpiry(expiresInSeconds) {
  return Date.now() + Math.max(0, Number(expiresInSeconds) || 0) * 1000;
}

function expireRendererReadiness() {
  rendererWarmUntil = 0;
  elements.prewarmButton.textContent = "Prepare full path";
  setRendererReadiness(
    "idle",
    preparedPlanKey === currentPlanKey() ? "Edge plan cached; renderer may be asleep" : "Renderer may be asleep",
    "Prepare again before a judged run. Story text stays on the local edge planner.",
  );
}

function markRendererReady(expiresAt, planner = null, preparedKey = null) {
  window.clearTimeout(rendererWarmExpiryTimer);
  const boundedSeconds = Math.max(0, (expiresAt - Date.now()) / 1000);
  const minutes = Math.max(1, Math.ceil(boundedSeconds / 60));
  const planIsCurrent = planner && preparedKey === currentPlanKey();
  if (planIsCurrent) preparedPlanKey = preparedKey;
  if (!boundedSeconds) {
    expireRendererReadiness();
    return false;
  }
  rendererWarmUntil = expiresAt;
  elements.prewarmButton.textContent = planIsCurrent ? "Full path ready" : "Prepare edge plan";
  setRendererReadiness(
    "ready",
    planIsCurrent ? `Gemma + renderer ready for about ${minutes} min` : `Cloud renderer ready for about ${minutes} min`,
    planIsCurrent
      ? `Private edge plan cached in ${(planner.planning_ms / 1000).toFixed(1)} s; Generate can skip that wait.`
      : "Submit a story now to avoid cold-start delay.",
    {buttonDisabled: Boolean(planIsCurrent)},
  );
  rendererWarmExpiryTimer = window.setTimeout(expireRendererReadiness, boundedSeconds * 1000);
  return true;
}

async function inspectRendererReadiness() {
  try {
    const response = await fetch("/v1/live-scene-provider/warm-status", {cache: "no-store"});
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.detail || `Readiness failed (${response.status})`);
    if (payload.state === "prewarmed") {
      return markRendererReady(rendererExpiry(payload.expires_in_seconds)) ? "ready" : "idle";
    }
    setRendererReadiness(
      "idle",
      "Renderer sleeps between scenes",
      "Prepare the full path before a judged run.",
    );
    return "idle";
  } catch (error) {
    setRendererReadiness(
      "error",
      "Renderer prewarm unavailable",
      error.message,
      {buttonDisabled: true},
    );
    return "unavailable";
  }
}

async function warmEdgePlanner() {
  try {
    // Fixed synthetic input only. This can load the private Jetson model while
    // the reader types, but never transmits or stores the story textarea.
    const response = await fetch("/v1/live-scene-planner/warmup", {
      method: "POST",
      cache: "no-store",
    });
    const payload = await response.json().catch(() => ({}));
    if (response.ok && payload.ready === true) {
      edgePlannerWarmUntil = Date.now() + EDGE_PLANNER_KEEP_WARM_MS;
      return true;
    }
  } catch (_) {
    // The generation path will fail closed before cloud rendering if the
    // configured local planner is unavailable.
  }
  return false;
}

function scheduleEdgePlanPreparation(event) {
  if (voiceMode) return;
  window.clearTimeout(edgePlanPreparationTimer);
  edgePlanPreparationTimer = null;
  const text = currentPlanKey();
  if (text.length < 3 || text === preparedPlanKey || text === edgePlanPreparingKey) return;
  // Paste commonly supplies the complete passage in one operation, so begin
  // immediately. Normal typing keeps a short quiet period to avoid spending
  // the Jetson's single model slot on incomplete phrases.
  const delayMs = event?.inputType === "insertFromPaste" ? 0 : 450;
  edgePlanPreparationTimer = window.setTimeout(() => {
    edgePlanPreparationTimer = null;
    void prepareEdgePlan(text);
  }, delayMs);
}

async function prepareEdgePlan(text) {
  if (text !== currentPlanKey() || text === preparedPlanKey || text === edgePlanPreparingKey) return;
  edgePlanPreparingKey = text;
  try {
    const response = await fetch("/v1/live-scene-planner/prepare", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        text,
        visual_style: elements.style.value.trim() || "luminous paper theater",
        session_id: readerSessionId,
        ...(voiceMode ? {reviewed_description: true} : {}),
      }),
    });
    const planner = await response.json().catch(() => ({}));
    if (!response.ok || text !== currentPlanKey()) return;
    preparedPlanKey = text;
    const rendererReady = Date.now() < rendererWarmUntil;
    elements.prewarmButton.textContent = rendererReady ? "Full path ready" : "Prepare renderer";
    setRendererReadiness(
      rendererReady ? "ready" : "idle",
      rendererReady ? "Gemma + renderer ready" : "Gemma plan ready; renderer may be asleep",
      `Private edge plan cached in ${(planner.planning_ms / 1000).toFixed(1)} s; Generate can skip that wait.`,
      {buttonDisabled: rendererReady},
    );
  } catch (_) {
    // Automatic planning is latency hiding only. Manual preparation and
    // Generate preserve their explicit error/fallback behavior.
  } finally {
    if (edgePlanPreparingKey === text) edgePlanPreparingKey = null;
  }
}

async function prewarmRenderer() {
  if (rendererPrewarming) return;
  const text = elements.story.value.trim();
  const visualStyle = elements.style.value.trim() || "luminous paper theater";
  if (text.length < 3) {
    setRendererReadiness("error", "Add a story moment first", "The edge planner needs at least three characters.");
    return;
  }
  const preparedKey = text;
  rendererPrewarming = true;
  elements.prewarmButton.textContent = "Preparing…";
  setRendererReadiness(
    "warming",
    "Preparing Gemma and the cloud renderer…",
    "They run concurrently. Story text goes only to the private local planner.",
    {buttonDisabled: true},
  );
  try {
    const post = async (url, body) => {
      const response = await fetch(url, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(body),
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.detail || `Preparation failed (${response.status})`);
      return payload;
    };
    const rendererPreparation = rendererWarmUntil > Date.now()
      ? Promise.resolve({expiresAt: rendererWarmUntil})
      : post("/v1/live-scene-provider/prewarm", {
          prewarm_id: `rehearsal-${Date.now().toString(36)}`,
          include_motion: false,
          scaledown_window_seconds: 90,
        }).then((renderer) => ({expiresAt: rendererExpiry(renderer.expires_in_seconds)}));
    const [rendererResult, plannerResult] = await Promise.allSettled([
      rendererPreparation,
      post("/v1/live-scene-planner/prepare", {
        text,
        visual_style: visualStyle,
        session_id: readerSessionId,
        ...(voiceMode ? {reviewed_description: true} : {}),
      }),
    ]);
    const renderer = rendererResult.status === "fulfilled" ? rendererResult.value : null;
    const planner = plannerResult.status === "fulfilled" ? plannerResult.value : null;
    const rendererReady = renderer && markRendererReady(renderer.expiresAt, planner, preparedKey);
    if (planner && preparedKey === currentPlanKey()) preparedPlanKey = preparedKey;
    if (renderer && !planner) {
      elements.prewarmButton.textContent = rendererReady ? "Retry edge plan" : "Prepare full path";
      setRendererReadiness(
        rendererReady ? "ready" : "idle",
        rendererReady ? "Renderer ready; edge plan unavailable" : "Renderer may be asleep; edge plan unavailable",
        plannerResult.reason.message,
      );
    } else if (!renderer && planner) {
      elements.prewarmButton.textContent = "Retry renderer";
      setRendererReadiness("error", "Edge plan cached; renderer preparation failed", rendererResult.reason.message);
    } else if (!renderer && !planner) {
      throw new Error(`${plannerResult.reason.message}; ${rendererResult.reason.message}`);
    }
  } catch (error) {
    elements.prewarmButton.textContent = "Retry preparation";
    setRendererReadiness("error", "Full-path preparation failed", error.message);
  } finally {
    rendererPrewarming = false;
  }
}

function invalidatePreparation() {
  if (!preparedPlanKey || preparedPlanKey === currentPlanKey()) return;
  preparedPlanKey = null;
  elements.prewarmButton.textContent = "Prepare full path";
  setRendererReadiness(
    Date.now() < rendererWarmUntil ? "ready" : "idle",
    Date.now() < rendererWarmUntil ? "Renderer ready; edge plan changed" : "Scene preparation changed",
    "Prepare again only after the passage changes; visual-style auditions reuse the private semantic plan.",
  );
}

function updateMicAvailability() {
  elements.micButton.disabled = starting || finalizing
    || (!voiceMode && generationSubmitting)
    || (voiceMode && voiceGeneration.publishing && !listening) || !canRecordAudio
    || (!voiceMode && !listening && !sceneReady);
}

function setSceneReady(ready) {
  sceneReady = ready;
  elements.projectorLink.classList.toggle("disabled", !ready);
  elements.projectorLink.setAttribute("aria-disabled", String(!ready));
  elements.projectorLink.tabIndex = ready ? 0 : -1;
  updateMicAvailability();
  elements.projectionPreview.classList.toggle("hidden", !ready);
  if (ready && !listening && !starting && !finalizing) {
    elements.interim.textContent = voiceMode
      ? "Describe a scene. Generation starts as you speak; only completed artwork appears."
      : "Press Start reading and read the page aloud.";
  }
}

function ensureProjectionPreview() {
  const url = projectorUrl();
  elements.projectorLink.href = url;
  if (projectorPreviewUrl !== url) {
    elements.projectorFrame.src = url;
    projectorPreviewUrl = url;
  }
}

function projectorUrl() {
  const live = !demoMode;
  return `/projector?pack=latest&session=${encodeURIComponent(readerSessionId)}&present=1&reader=${voiceMode ? "0" : "1"}${live ? "&live=1" : ""}${voiceMode ? "&complete_only=1" : ""}`;
}

elements.projectorLink.href = projectorUrl();

// Kept as the public preview hook; it initializes once and never reloads during stage upgrades.
function reloadProjectionPreview() {
  ensureProjectionPreview();
}

function safeText(value) {
  const node = document.createElement("span");
  node.textContent = value == null ? "" : String(value);
  return node.innerHTML;
}

async function startAudioMeter() {
  if (stream) return;
  stream = await new Promise((resolve, reject) => {
    let expired = false;
    const timer = window.setTimeout(() => {
      expired = true;
      reject(new Error("Microphone permission timed out. Allow microphone access, then try again."));
    }, 20000);
    navigator.mediaDevices.getUserMedia({audio: true, video: false}).then((capture) => {
      window.clearTimeout(timer);
      if (expired) capture.getTracks().forEach((track) => track.stop());
      else resolve(capture);
    }, (error) => {
      window.clearTimeout(timer);
      reject(error);
    });
  });
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
  if (starting || listening || finalizing || (!voiceMode && generationSubmitting)
    || (voiceMode && voiceGeneration.publishing)) return;
  if (!voiceMode && !sceneReady) {
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
  setSceneInputsDisabled(true);
  try {
    const pageText = elements.story.value.trim();
    if (!voiceMode && !pageText) throw new Error("Enter the trusted page text before starting the reader.");
    const runtime = await readerRequest("/v1/runtime:status", {cache: "no-store"}, 10000);
    if (runtime.asr?.ready !== true) throw new Error("Local transcription is disabled or unavailable. Enable the local ASR backend before reading.");
    if (!voiceMode) {
      activePageText = pageText;
      await configureReaderSession(pageText);
      readerGeneration = (await resetReaderSession()).generation;
    }
    await startAudioMeter();
    audioChunks = [];
    partialBytes = 0;
    partialInFlight = Promise.resolve();
    recordingEpoch += 1;
    if (voiceMode) {
      voiceGeneration.latest = null;
      voiceGeneration.attempted.clear();
      window.clearTimeout(voiceGeneration.timer);
      voiceGeneration.timer = null;
    }
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
    elements.micButtonText.textContent = voiceMode ? "Finish recording" : "Stop reading";
    elements.compileButton.disabled = true;
    elements.interim.textContent = voiceMode
      ? "Listening locally. Generation starts as your description becomes clear; only completed scenes will appear."
      : "Listening locally—read the exact page text above.";
    partialTimer = window.setInterval(() => {
      if (!partialBusy && (!voiceMode || !voiceGeneration.publishing)) {
        partialInFlight = transcribePartialRecording(epoch);
      }
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
  if (finalizing) return;
  finalizing = true;
  listening = false;
  elements.micButton.classList.remove("listening");
  elements.micButton.disabled = true;
  elements.micButtonText.textContent = "Finishing…";
  elements.interim.textContent = "Finishing the local transcript…";
  window.clearInterval(partialTimer);
  partialTimer = null;
  if (mediaRecorder?.state === "recording") mediaRecorder.stop();
  else resetMicControls();
  releaseMicrophone();
}

function resetMicControls() {
  listening = false;
  finalizing = false;
  elements.micButton.classList.remove("listening");
  updateMicAvailability();
  elements.micButtonText.textContent = voiceMode ? "Describe scene" : "Start reading";
  elements.compileButton.disabled = demoMode || generationSubmitting || Boolean(activeLiveJobId);
  setSceneInputsDisabled(Boolean(activeLiveJobId));
  window.clearInterval(partialTimer);
  partialTimer = null;
  activePageText = null;
  readerGeneration = null;
  releaseMicrophone();
}

async function readerRequest(url, options, timeoutMs = 45000) {
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(url, {...options, signal: controller.signal});
    const payload = await response.json();
    if (!response.ok) throw new Error(typeof payload.detail === "string"
      ? payload.detail : `Local reader request failed (${response.status}).`);
    return payload;
  } catch (error) {
    if (controller.signal.aborted) throw new Error("The local reader timed out. Start reading again to retry.");
    throw error;
  } finally {
    window.clearTimeout(timer);
  }
}

async function transcribeBlob(recording, mimeType) {
  return readerRequest("/v1/audio:transcribe", {
      method: "POST",
      headers: {"Content-Type": mimeType},
      body: recording,
  });
}

async function publishReaderTranscript(text, isFinal, generation = readerGeneration) {
  if (!activePageText || generation === null) {
    throw new Error("The reader session changed; stop and start this reading again.");
  }
  await readerRequest(`/v1/reader-sessions/${readerSessionId}/transcripts:simulate`, {
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
}

async function configureReaderSession(pageText) {
  return readerRequest(`/v1/reader-sessions/${readerSessionId}`, {
    method: "PUT",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({page_id: "page-01", page_text: pageText}),
  });
}

async function resetReaderSession() {
  return readerRequest(`/v1/reader-sessions/${readerSessionId}:reset`, {
    method: "POST",
  });
}

function voiceIntentKey(text, style) {
  return JSON.stringify({text, style});
}

function voiceSnapshotKey(snapshot) {
  const request = snapshot?.request;
  return request?.text && request?.visual_style
    ? voiceIntentKey(request.text.trim(), request.visual_style.trim()) : null;
}

function voiceSnapshotMatchesIntent(snapshot, intent) {
  if (!intent) return false;
  if (voiceSnapshotKey(snapshot) === intent.key) return true;
  const completed = voiceGeneration.completed;
  return Boolean(completed?.semanticKey && completed.key === intent.key
    && completed.snapshot.job_id === snapshot?.job_id
    && completed.sourceKey === voiceSnapshotKey(snapshot));
}

function offerVoiceTranscript(text, {final = false, epoch = recordingEpoch} = {}) {
  if (!voiceMode || epoch !== recordingEpoch || typeof text !== "string") return;
  text = text.trim();
  if (!text) return;
  const style = elements.style.value.trim() || "luminous paper theater";
  const key = voiceIntentKey(text, style);
  const previous = voiceGeneration.latest;
  voiceGeneration.latest = {
    key, text, style, epoch,
    observations: previous?.key === key ? previous.observations + 1 : 1,
    final: final || (previous?.key === key && previous.final),
  };
  elements.story.value = text;
  delete elements.compileButton.dataset.visualVariation;
  invalidatePreparation();
  window.clearTimeout(voiceGeneration.timer);
  voiceGeneration.timer = null;
  if (final) {
    if (!finalizing) void pumpVoiceGeneration();
  } else {
    voiceGeneration.timer = window.setTimeout(() => {
      voiceGeneration.timer = null;
      void pumpVoiceGeneration();
    }, 350);
  }
  void tryPresentVoiceGeneration();
}

async function pumpVoiceGeneration() {
  if (!voiceMode || finalizing || starting || generationSubmitting || pendingSubmission
    || activeLiveJobId || voiceGeneration.publishing) return;
  const intent = voiceGeneration.latest;
  if (!intent || intent.epoch !== recordingEpoch) return;
  if (voiceGeneration.completed?.key === intent.key) {
    await tryPresentVoiceGeneration();
    return;
  }
  if (voiceGeneration.attempted.has(intent.key)) return;
  voiceGeneration.attempted.add(intent.key);
  await compileStory({voiceIntent: intent});
}

async function tryPresentVoiceGeneration() {
  const completed = voiceGeneration.completed;
  const intent = voiceGeneration.latest;
  const current = () => voiceGeneration.completed === completed
    && voiceGeneration.latest?.key === intent?.key
    && voiceGeneration.latest?.epoch === recordingEpoch
    && voiceIntentKey(elements.story.value.trim(),
      elements.style.value.trim() || "luminous paper theater") === intent?.key
    && !partialBusy && !finalizing && !starting && !activeLiveJobId
    && !generationSubmitting && !pendingSubmission;
  if (!voiceMode || !completed || !intent || completed.key !== intent.key
    || completed.presentationAttempted || voiceGeneration.publishing || !current()
    || (!intent.final && intent.observations < 2)) return;
  voiceGeneration.publishing = true;
  let finishPresentation;
  voiceGeneration.presentationInFlight = new Promise((resolve) => { finishPresentation = resolve; });
  setSceneInputsDisabled(true);
  updateMicAvailability();
  try {
    const pointer = await fetchLiveSceneSession();
    if (!current() || pointer?.job?.job_id !== completed.snapshot.job_id
      || !voiceSnapshotMatchesIntent(pointer.job, intent) || !pointer.job.complete
      || pointer.job.stage === "failed") return;
    completed.presentationAttempted = true;
    const shown = await readerRequest(`/v1/live-scenes/${encodeURIComponent(pointer.job.job_id)}/present`, {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({server_instance_id: pointer.server_instance_id,
        session_revision: pointer.session_revision}),
    }, 10000);
    if (!current() || shown.job_id !== pointer.job.job_id || shown.presentation_ready !== true
      || shown.complete !== true || !voiceSnapshotMatchesIntent(shown, intent)) return;
    renderLiveSnapshot(shown);
    elements.interim.textContent = listening
      ? "Your completed scene is showing. Keep describing to change it."
      : "Your completed scene is showing.";
  } catch (error) {
    completed.presentationAttempted = true;
    elements.interim.textContent = `The scene finished, but display confirmation is pending: ${error.message}`;
  } finally {
    voiceGeneration.publishing = false;
    finishPresentation();
    setSceneInputsDisabled(Boolean(activeLiveJobId));
    updateMicAvailability();
    if (voiceGeneration.latest?.key !== intent.key) void pumpVoiceGeneration();
  }
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
    if (voiceMode) offerVoiceTranscript(payload.text, {epoch});
    else await publishReaderTranscript(payload.text, false, generation);
  } catch (error) {
    if (listening && epoch === recordingEpoch) {
      elements.interim.textContent = `Still listening; transcript retrying: ${error.message}`;
    }
  } finally {
    partialBusy = false;
    if (voiceMode) void tryPresentVoiceGeneration();
  }
}

async function transcribeRecording() {
  listening = false;
  finalizing = true;
  releaseMicrophone();
  const mimeType = mediaRecorder?.mimeType || "audio/webm";
  const recording = new Blob(audioChunks, {type: mimeType});
  const generation = readerGeneration;
  try {
    await partialInFlight;
    if (voiceMode) await voiceGeneration.presentationInFlight;
    if (recording.size < 1000) throw new Error("Recording was too short. Try speaking for a little longer.");
    const payload = await transcribeBlob(recording, mimeType);
    if (voiceMode) {
      const text = typeof payload.text === "string" ? payload.text.trim() : "";
      if (!text) throw new Error("No speech was recognized. Describe the scene again, or type it below.");
      offerVoiceTranscript(text, {final: true});
      elements.interim.textContent = "Finishing the scene for your latest description…";
    } else {
      await publishReaderTranscript(payload.text, true, generation);
      elements.interim.textContent = `Finished in ${(payload.total_ms / 1000).toFixed(1)} s. Whisper heard: ${payload.text}`;
    }
  } catch (error) {
    if (voiceMode) {
      voiceGeneration.latest = null;
      window.clearTimeout(voiceGeneration.timer);
      voiceGeneration.timer = null;
    }
    elements.interim.textContent = error.message;
  } finally {
    mediaRecorder = null;
    audioChunks = [];
    partialInFlight = Promise.resolve();
    resetMicControls();
    if (voiceMode) void pumpVoiceGeneration();
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
    preview_ready: 0.58,
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
  if (metrics?.scene_cache_hit) {
    elements.planningPrivacy.textContent = "Verified local replay · This exact completed scene was restored without a new cloud image request.";
    return;
  }
  const scenePlan = (metrics?.models || []).find((model) => model.role === "scene_plan");
  const planningStatus = metrics?.planning_status || "pending";
  const localGemma = planningStatus === "model"
    && /gemma/i.test(scenePlan?.model || "")
    && /^ollama(?:-|$)/i.test(scenePlan?.revision || "");
  if (scenePlan?.model?.startsWith("bounded-description-")) {
    elements.planningPrivacy.textContent = "Reviewed scene description · Local rules verified the visual facts; the renderer received only visual direction.";
  } else if (scenePlan?.model?.startsWith("reviewed-language-")) {
    elements.planningPrivacy.textContent = "Reviewed local language plan · Local language checks verified the visual direction sent to the renderer.";
  } else if (localGemma) {
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
  const sceneEvidence = metrics.scene_cache_hit ? "verified completed-scene cache" : "new scene";
  elements.generationMetrics.textContent = [
    `backend wall ${formatBackendMs(metrics.elapsed_ms)}`,
    sceneEvidence,
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
  if (pendingSubmission) {
    elements.compileButton.disabled = generationReconciling;
    elements.compileButton.textContent = "Check generation status";
    return;
  }
  const labels = {
    queued: "Waiting for generation provider…",
    planning: "Building animated draft…",
    draft_ready: "Preparing artwork + depth…",
    preview_ready: "Gemma is directing the final artwork…",
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
  window.clearTimeout(livePollTimer);
  livePollTimer = null;
  window.clearInterval(liveElapsedTimer);
  liveElapsedTimer = null;
}

function liveSessionStreamIsHealthy() {
  return Boolean(
    liveSessionStreamHealthy
    && liveSessionEventSource
    && liveSessionEventSource.readyState === EventSource.OPEN
  );
}

function startLivePollingFallback() {
  if (!activeLiveJobId || livePollTimer !== null) return;
  livePollTimer = window.setTimeout(() => {
    livePollTimer = null;
    pollLiveScene(activeLiveJobId, liveRequestEpoch);
  }, 0);
}

function setSceneInputsDisabled(disabled) {
  const locked = disabled || demoMode || starting || listening || finalizing || generationSubmitting
    || (voiceMode && voiceGeneration.publishing);
  elements.story.disabled = locked;
  elements.style.disabled = locked;
}

function finishLiveJob(snapshot) {
  stopLiveJobTransport();
  activeLiveJobId = null;
  if (voiceMode) {
    const key = voiceSnapshotKey(snapshot) || voiceGeneration.active?.key;
    const semanticKey = voiceGeneration.active?.key === key
      ? voiceGeneration.active.semanticKey : null;
    voiceGeneration.active = null;
    if (snapshot.complete && snapshot.stage !== "failed" && key) {
      if (voiceGeneration.completed?.snapshot.job_id === snapshot.job_id) {
        voiceGeneration.completed.snapshot = snapshot;
      } else {
        voiceGeneration.completed = {key, sourceKey: key, semanticKey, snapshot, presentationAttempted: false};
      }
    }
    updateMicAvailability();
    setSceneInputsDisabled(false);
    elements.compileButton.disabled = listening || finalizing || generationSubmitting;
    elements.compileButton.textContent = snapshot.stage === "failed"
      ? "Try generation again" : "Generate scene";
    elements.interim.textContent = snapshot.stage === "failed"
      ? "Generation did not finish. Edit the description or retry when ready."
      : (snapshot.presentation_ready
        ? "The completed scene is showing."
        : "The scene finished; waiting for the latest stable description before showing it.");
    setStatus(snapshot.stage === "failed" ? "error" : "idle",
      snapshot.stage === "failed" ? "Generation failed" : "Scene finished");
    void pumpVoiceGeneration();
    return;
  }
  updateMicAvailability();
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
  elements.compileButton.dataset.visualVariation = "true";
  elements.compileButton.textContent = pendingSubmission
    ? "Check generation status" : "Generate a new visual variation";
  elements.interim.textContent = snapshot.stage === "motion_ready"
    ? (snapshot.metrics?.scene_cache_hit
      ? "The exact moving scene was restored locally. No Gemma or cloud renderer call was needed."
      : "The final moving scene is live. No projector reload occurred.")
    : (snapshot.metrics?.scene_cache_hit
      ? "The exact artwork and depth scene were restored locally with zero new GPU cost."
      : "The best available scene is live; this provider returned no additional motion stage.");
  setStatus("idle", snapshot.stage === "motion_ready" ? "Moving scene ready" : "Scene ready");
}

function renderLiveSnapshot(snapshot, epoch = liveRequestEpoch) {
  if (!snapshot || epoch !== liveRequestEpoch) return;
  const jobId = snapshot.job_id || activeLiveJobId;
  if (activeLiveJobId && jobId && jobId !== activeLiveJobId) return;
  const revision = liveRevision(snapshot);
  if (revision < lastLiveRevision) return;
  if (revision === lastLiveRevision && latestLiveSnapshot?.stage === snapshot.stage
    && latestLiveSnapshot?.presentation_ready === snapshot.presentation_ready) return;
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
  const voiceCanDisplay = !voiceMode || (snapshot.complete === true
    && snapshot.stage !== "failed"
    && (!snapshot.request?.defer_presentation || snapshot.presentation_ready === true)
    && (!voiceGeneration.latest || voiceSnapshotMatchesIntent(snapshot, voiceGeneration.latest)));
  if (snapshot.story_pack && voiceCanDisplay) {
    renderPack({story_pack: snapshot.story_pack, live_snapshot: snapshot});
    const stageCopy = {
      draft_ready: "Animated draft live—the generation provider is preparing the master.",
      preview_ready: "Generated visual sketch live—Gemma is directing the final artwork.",
      master_ready: snapshot.complete
        ? "Artwork and depth are live with local WebGL motion."
        : "Artwork and depth are live—optional video motion is preparing next.",
      motion_ready: "Cinematic motion loop live.",
    }[snapshot.stage];
    if (stageCopy) elements.interim.textContent = stageCopy;
  }
  if (isTerminalSnapshot(snapshot)) finishLiveJob(snapshot);
}

async function fetchLiveSceneSession(emptySession = null) {
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), 10000);
  try {
    const response = await fetch(
      `/v1/live-scene-sessions/${encodeURIComponent(readerSessionId)}`,
      {cache: "no-store", signal: controller.signal},
    );
    if (response.status === 404) {
      const server = response.headers.get("X-Bookforge-Server-Instance-Id");
      if (!/^server_[a-f0-9]{32}$/.test(server || "")) {
        throw new Error("Generation service did not provide a current server ID. Update or reconnect before generating.");
      }
      if (emptySession) emptySession.serverInstanceId = server;
      return null;
    }
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || `Session rendezvous failed (${response.status})`);
    if (payload.session_id !== readerSessionId || !payload.job?.job_id) {
      throw new Error("Generation session returned an invalid job pointer.");
    }
    return payload;
  } finally {
    window.clearTimeout(timer);
  }
}

function acceptedLiveScenePointer(response, snapshot) {
  const serverInstanceId = response.headers.get("X-Bookforge-Server-Instance-Id");
  const sessionRevision = Number(response.headers.get("X-Bookforge-Session-Revision"));
  if (
    !serverInstanceId
    || !Number.isInteger(sessionRevision)
    || sessionRevision < 1
    || !snapshot?.job_id
  ) return null;
  return {
    session_id: readerSessionId,
    server_instance_id: serverInstanceId,
    session_revision: sessionRevision,
    job: snapshot,
  };
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
  if (restoreInputs && !voiceMode) {
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
    if (!liveSessionStreamIsHealthy()) pollLiveScene(snapshot.job_id, epoch);
  }
  return true;
}

async function pollLiveScene(jobId, epoch) {
  livePollTimer = null;
  if (epoch !== liveRequestEpoch || !activeLiveJobId) return;
  if (liveSessionStreamIsHealthy()) return;
  try {
    const pointer = await fetchLiveSceneSession();
    if (pointer && handleLiveSceneSessionPointer(pointer, {restoreInputs: true})) return;
    if (pointer?.job?.job_id === jobId) renderLiveSnapshot(pointer.job, epoch);
  } catch (error) {
    if (epoch === liveRequestEpoch) elements.interim.textContent = `Scene is still rendering; status retrying: ${error.message}`;
  }
  if (
    epoch === liveRequestEpoch
    && activeLiveJobId
    && !liveSessionStreamIsHealthy()
  ) {
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
  liveSessionStreamHealthy = false;
  const source = new EventSource(
    `/v1/live-scene-sessions/${encodeURIComponent(readerSessionId)}/events`,
  );
  liveSessionEventSource = source;
  const receive = (event) => {
    try {
      handleLiveSceneSessionPointer(JSON.parse(event.data), {restoreInputs: true});
      if (voiceMode && pendingSubmission && !generationReconciling) void reconcileGeneration();
      if (source === liveSessionEventSource) {
        liveSessionStreamHealthy = true;
        window.clearTimeout(livePollTimer);
        livePollTimer = null;
      }
    } catch (_) {
      liveSessionStreamHealthy = false;
      elements.interim.textContent = "Ignored an invalid session update; polling remains active.";
      startLivePollingFallback();
    }
  };
  source.addEventListener("scene.session", receive);
  source.addEventListener("message", receive);
  source.addEventListener("error", () => {
    if (source !== liveSessionEventSource) return;
    liveSessionStreamHealthy = false;
    elements.interim.textContent = "Session updates reconnecting; status polling remains available.";
    startLivePollingFallback();
  });
}

function renderPack(payload) {
  const pack = payload.story_pack;
  const metrics = payload.compile_metrics || payload.metrics;
  const generation = payload.generation_metrics;
  const liveSnapshot = payload.live_snapshot;
  const liveMetrics = liveSnapshot?.metrics;
  const page = pack.pages[0];
  const semanticFingerprint = JSON.stringify([
    pack.compiler_model,
    page.page_id,
    page.scene_summary,
    page.scene_spec,
    page.layers,
    page.triggers,
    page.literacy_support,
    page.comprehension,
  ]);
  const semanticsChanged = semanticFingerprint !== lastRenderedSemanticFingerprint;
  const shouldPersist = !liveSnapshot || isTerminalSnapshot(liveSnapshot);
  if (shouldPersist) {
    try {
      localStorage.setItem("bookforge.latestStoryPack", JSON.stringify(pack));
      storyPackPersistCount += 1;
      document.body.dataset.storyPackPersistCount = String(storyPackPersistCount);
    } catch (_) {
      // Browser storage is a convenience fallback; the session SSE remains authoritative.
    }
    elements.raw.textContent = JSON.stringify(pack, null, 2);
  }
  elements.model.textContent = liveMetrics?.models?.length
    ? liveMetrics.models.map((model) => `${model.model}@${model.revision}`).join(" + ")
    : metrics?.model || pack.compiler_model;
  const totalMs = (metrics?.total_ms || 0) + (generation?.total_ms || 0);
  elements.time.textContent = Number.isFinite(liveMetrics?.elapsed_ms)
    ? `${(liveMetrics.elapsed_ms / 1000).toFixed(1)} s backend wall`
    : totalMs ? `${(totalMs / 1000).toFixed(1)} s` : "Not measured";
  elements.tokens.textContent = metrics ? `${metrics.output_tokens} tokens` : `${page.layers.length + page.triggers.length} parts`;
  if (semanticsChanged) {
    semanticRenderCount += 1;
    document.body.dataset.semanticRenderCount = String(semanticRenderCount);
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
    lastRenderedSemanticFingerprint = semanticFingerprint;
  }
  elements.empty.classList.add("hidden");
  elements.error.classList.add("hidden");
  elements.results.classList.remove("hidden");
  setSceneReady(true);
  reloadProjectionPreview();
}

async function reconcileGeneration() {
  if (!pendingSubmission || generationReconciling) return;
  generationReconciling = true;
  elements.compileButton.disabled = true;
  try {
    const pointer = await fetchLiveSceneSession();
    const request = pointer?.job?.request;
    const expected = pendingSubmission.request;
    if (pointer?.job?.job_id !== pendingSubmission.previousJobId
      && (!pendingSubmission.previousServer || (pointer.server_instance_id === pendingSubmission.previousServer
        && pointer.session_revision > pendingSubmission.previousRevision))
      && request?.text === expected.text && request?.visual_style === expected.visual_style
      && typeof expected.submission_id === "string"
      && request.submission_id === expected.submission_id
      && Boolean(request.reviewed_description) === Boolean(expected.reviewed_description)
      && Boolean(request.display_when_complete) === Boolean(expected.display_when_complete)
      && Boolean(request.defer_presentation) === Boolean(expected.defer_presentation)
      && Boolean(request.confirm_visual_facts) === Boolean(expected.confirm_visual_facts)
      && (request.visual_fact_digest ?? null) === (expected.visual_fact_digest ?? null)
      && (request.seed ?? null) === (expected.seed ?? null)) {
      handleLiveSceneSessionPointer(pointer);
      if ((activeLiveJobId || latestLiveSnapshot?.job_id) === pointer.job.job_id) {
        pendingSubmission = null;
        generationSubmitting = false;
        setGenerateButtonForStage(pointer.job.stage);
        elements.compileButton.disabled = Boolean(activeLiveJobId);
        elements.error.classList.add("hidden");
        setSceneInputsDisabled(Boolean(activeLiveJobId));
        updateMicAvailability();
        if (voiceMode) void pumpVoiceGeneration();
        return;
      }
    }
  } catch (_) {
    // A lost response cannot establish that the server rejected the paid request.
  } finally {
    generationReconciling = false;
  }
  elements.interim.textContent = "Generation acceptance is uncertain. Check status to reconnect; this will not submit another scene.";
  elements.compileButton.disabled = false;
  elements.compileButton.textContent = "Check generation status";
}

async function checkVoiceDescription(request) {
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), 10000);
  try {
    const response = await fetch("/v1/live-scene-planner/prepare", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({text: request.text, visual_style: request.visual_style,
        seed: request.seed ?? 0, session_id: request.session_id,
        reviewed_description: request.reviewed_description === true}),
      signal: controller.signal,
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail?.message
      || (typeof result.detail === "string" ? result.detail : "The local scene check is unavailable. Try again."));
    if (!result.visual_facts) throw new Error("The local scene checker needs updating. No image was requested.");
    if (result.requires_fact_review && (typeof result.visual_fact_digest !== "string"
      || !/^[a-f0-9]{64}$/.test(result.visual_fact_digest))) {
      throw new Error("The local scene review is incomplete. No image was requested.");
    }
    const facts = result.visual_facts;
    const describe = (entity) => [entity.count, entity.color, ...entity.attributes, entity.label]
      .filter(Boolean).join(" ");
    const labels = Object.fromEntries([...facts.subjects, ...facts.objects].map((entity) => [entity.ref, describe(entity)]));
    const descriptions = facts.subjects.map((subject) => `${describe(subject)}: ${subject.actions.join("; ")}`);
    if (facts.setting.label !== "unspecified") descriptions.push(`Setting: ${[
      ...facts.setting.attributes, facts.setting.label,
    ].join(" ")}`);
    if (facts.objects.length) descriptions.push(`Also visible: ${facts.objects.map(describe).join(", ")}`);
    for (const relation of facts.relationships) descriptions.push([
      labels[relation.source], relation.relation.replaceAll("_", " "), labels[relation.target],
      relation.secondary_target ? `and ${labels[relation.secondary_target]}` : "",
    ].filter(Boolean).join(" "));
    for (const negative of facts.negatives) descriptions.push(negative.target
      ? `${labels[negative.target]}: not ${negative.value}` : `Exclude: ${negative.value}`);
    const omissions = result.local_omissions || [];
    if (omissions.length) descriptions.push(`Kept on this device: ${omissions.map((item) => item.local_text).join(", ")}. These details will not appear in the image request.`);
    elements.voiceReview.textContent = `Scene to generate\n${descriptions.join("\n")}`;
    return result;
  } catch (error) {
    throw controller.signal.aborted
      ? new Error("The local scene check timed out. Your previous scene is unchanged; try again.")
      : error;
  } finally {
    window.clearTimeout(timer);
  }
}

async function compileStory(options = {}) {
  const voiceIntent = options.voiceIntent || null;
  const automatic = Boolean(voiceMode && voiceIntent);
  if (pendingSubmission) return automatic ? undefined : reconcileGeneration();
  if ((!automatic && (starting || listening || finalizing))
    || generationSubmitting || activeLiveJobId) return;
  const text = voiceIntent?.text || elements.story.value.trim();
  if (voiceMode && !automatic && voiceGeneration.completed?.key === voiceIntentKey(text,
    elements.style.value.trim() || "luminous paper theater")
    && !voiceGeneration.completed.snapshot.presentation_ready) {
    const style = elements.style.value.trim() || "luminous paper theater";
    voiceGeneration.latest = {key: voiceIntentKey(text, style), text, style,
      epoch: recordingEpoch, observations: 1, final: true};
    voiceGeneration.completed.presentationAttempted = false;
    await tryPresentVoiceGeneration();
    return;
  }
  if (!text) {
    elements.interim.textContent = voiceMode ? "Describe the scene first." : "Add the exact words from one book page first.";
    return;
  }
  generationSubmitting = true;
  updateMicAvailability();
  const visualVariation = !automatic && elements.compileButton.dataset.visualVariation === "true";
  const variationSeed = visualVariation
    ? window.crypto.getRandomValues(new Uint32Array(1))[0]
    : null;
  const submission = {
    previousJobId: latestLiveSnapshot?.job_id || null,
    request: {
      text,
      visual_style: voiceIntent?.style || elements.style.value.trim() || "luminous paper theater",
      session_id: readerSessionId,
      ...(voiceMode ? {reviewed_description: true, display_when_complete: true, defer_presentation: true} : {}),
      ...(variationSeed === null ? {} : {seed: variationSeed}),
    },
  };
  let semanticKey = null;
  if (voiceMode) {
    elements.compileButton.disabled = true;
    setSceneInputsDisabled(true);
    elements.compileButton.textContent = "Checking description…";
    elements.voiceReview.textContent = "Checking the scene locally before generation.";
    try {
      const checked = await checkVoiceDescription(submission.request);
      if (elements.story.value.trim() !== submission.request.text
        || (elements.style.value.trim() || "luminous paper theater") !== submission.request.visual_style) {
        throw new Error("The description changed during the check. Review it and try again.");
      }
      if (automatic && voiceGeneration.latest?.key !== voiceIntent.key) {
        throw new Error("A newer description is ready; checking that instead.");
      }
      semanticKey = JSON.stringify({facts: checked.visual_facts,
        style: submission.request.visual_style, seed: submission.request.seed ?? 0,
        revision: checked.revision});
      if (automatic && voiceGeneration.completed?.semanticKey === semanticKey) {
        voiceGeneration.completed.key = voiceIntent.key;
        generationSubmitting = false;
        setSceneInputsDisabled(false);
        elements.compileButton.disabled = listening || finalizing;
        elements.compileButton.textContent = "Generate scene";
        elements.interim.textContent = "The scene details are unchanged; reusing the completed artwork.";
        updateMicAvailability();
        void tryPresentVoiceGeneration();
        return;
      }
      if (checked.requires_fact_review) {
        submission.request.confirm_visual_facts = true;
        submission.request.visual_fact_digest = checked.visual_fact_digest;
      }
    } catch (error) {
      elements.voiceReview.textContent = error.message;
      elements.interim.textContent = automatic && listening
        ? "Still listening for a clear scene description. No image was requested for this partial transcript."
        : "The description could not be verified. Your previous scene is unchanged; edit it or describe the scene again.";
      generationSubmitting = false;
      setSceneInputsDisabled(false);
      elements.compileButton.disabled = listening || finalizing;
      elements.compileButton.textContent = "Check description again";
      updateMicAvailability();
      if (automatic && voiceGeneration.latest?.key !== voiceIntent.key) {
        voiceGeneration.attempted.delete(voiceIntent.key);
        void pumpVoiceGeneration();
      }
      return;
    }
  }
  if (voiceMode) {
    const key = voiceIntentKey(submission.request.text, submission.request.visual_style);
    if (!automatic) {
      voiceGeneration.latest = {key, text, style: submission.request.visual_style,
        epoch: recordingEpoch, observations: 1, final: true};
    }
    voiceGeneration.active = {key, semanticKey};
  }
  delete elements.compileButton.dataset.visualVariation;
  // If the user clicks before the typing-pause timer fires, let the accepted
  // live job start the planner directly. If preparation is already in flight,
  // the server coalesces both waiters onto that one local Gemma call.
  window.clearTimeout(edgePlanPreparationTimer);
  edgePlanPreparationTimer = null;
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

  let rejected = true;
  try {
    const emptySession = {};
    const prior = await fetchLiveSceneSession(emptySession);
    if (prior && (!/^server_[a-f0-9]{32}$/.test(prior.server_instance_id || "")
      || !Number.isSafeInteger(prior.session_revision) || prior.session_revision < 1)) {
      throw new Error("Generation session returned an invalid revision.");
    }
    submission.previousJobId = prior?.job?.job_id || null;
    submission.previousServer = prior?.server_instance_id || emptySession.serverInstanceId;
    submission.previousRevision = prior?.session_revision || 0;
    if (prior && !isTerminalSnapshot(prior.job)) {
      if (automatic && voiceSnapshotKey(prior.job) !== voiceIntent.key) {
        voiceGeneration.attempted.delete(voiceIntent.key);
      }
      handleLiveSceneSessionPointer(prior);
      elements.interim.textContent = "The current scene is still generating. Wait for it to finish before submitting another.";
      return;
    }
    if (automatic && voiceGeneration.latest?.key !== voiceIntent.key) {
      voiceGeneration.attempted.delete(voiceIntent.key);
      return;
    }
    if (!submission.previousServer) {
      throw new Error("Generation service did not provide a current server ID. Reconnect before generating.");
    }
    submission.request.submission_id = window.crypto.randomUUID();
    submission.request.expected_server_instance_id = submission.previousServer;
    submission.request.expected_session_revision = submission.previousRevision;
    const controller = new AbortController();
    const timer = window.setTimeout(() => controller.abort(), 15000);
    let response;
    let snapshot;
    try {
      rejected = false;
      response = await fetch("/v1/live-scenes", {
        method: "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify(submission.request), signal: controller.signal,
      });
      rejected = response.status >= 400 && response.status < 500;
      snapshot = await response.json();
    } finally {
      window.clearTimeout(timer);
    }
    if (response.status === 409) {
      if (voiceMode) voiceGeneration.attempted.add(voiceIntentKey(text, submission.request.visual_style));
      const current = await fetchLiveSceneSession();
      if (current) handleLiveSceneSessionPointer(current);
      elements.interim.textContent = "Another scene request changed this session. Showing its status; no additional image was requested.";
      return;
    }
    if (response.status !== 202) throw new Error(snapshot.detail?.message || snapshot.detail || `Request failed (${response.status})`);
    if (!snapshot.job_id) throw new Error("Generation service returned no job ID.");
    if (snapshot.request?.submission_id !== submission.request.submission_id) {
      throw new Error("Generation service returned a different submission.");
    }
    const pointer = acceptedLiveScenePointer(response, snapshot) || await fetchLiveSceneSession();
    if (!pointer) throw new Error("Generation session did not retain the accepted job.");
    handleLiveSceneSessionPointer(pointer);
    const retainedJobId = activeLiveJobId || latestLiveSnapshot?.job_id;
    if (retainedJobId !== snapshot.job_id) {
      throw new Error("Generation session did not retain the accepted job.");
    }
    elements.interim.textContent = voiceMode
      ? "Generating in the background. Keep speaking; only the completed matching scene will appear."
      : "Generation job accepted. The projector will upgrade itself as each stage arrives.";
  } catch (error) {
    if (!rejected) {
      pendingSubmission = submission;
      await reconcileGeneration();
      return;
    }
    stopLiveJobTransport();
    activeLiveJobId = null;
    setSceneInputsDisabled(false);
    elements.error.textContent = error.message;
    elements.error.classList.remove("hidden");
    setStatus("error", "Could not create scene");
    elements.compileButton.disabled = voiceMode && (listening || finalizing);
    elements.compileButton.textContent = "Try generation again";
  } finally {
    generationSubmitting = Boolean(pendingSubmission);
    setSceneInputsDisabled(Boolean(activeLiveJobId));
    updateMicAvailability();
    if (voiceMode) void pumpVoiceGeneration();
  }
}

async function loadLatestScene() {
  try {
    const response = await fetch("/v1/story-packs/latest", {cache: "no-store"});
    if (!response.ok) return;
    const pack = await response.json();
    const page = pack.pages[0];
    if (!voiceMode) {
      elements.style.value = pack.visual_style;
      elements.story.value = page.source_text;
    }
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
  if (demoMode) return loadLatestScene();
  if (await recoverLiveSceneSession()) return;
  if (restoreLatestScene || voiceMode) await loadLatestScene();
}

function invalidateScene() {
  if (voiceMode) return;
  if (!sceneReady || listening || starting || activeLiveJobId) return;
  setSceneReady(false);
  setStatus("stale", "Page changed—create it again");
  elements.interim.textContent = "The page changed. Create the scene again before reading it.";
}

elements.micButton.addEventListener("click", () => {
  if (starting || finalizing) return;
  if (listening) stopSpeaking();
  else startSpeaking();
});
elements.compileButton.addEventListener("click", compileStory);
elements.prewarmButton.addEventListener("click", prewarmRenderer);
elements.projectorLink.addEventListener("click", (event) => {
  if (!sceneReady) event.preventDefault();
});
[elements.style, elements.story].forEach((element) => {
  element.addEventListener("input", () => {
    delete elements.compileButton.dataset.visualVariation;
    invalidatePreparation();
    invalidateScene();
    if (voiceMode) {
      voiceGeneration.latest = null;
      window.clearTimeout(voiceGeneration.timer);
      voiceGeneration.timer = null;
      elements.voiceReview.textContent = "";
    }
  });
});
elements.story.addEventListener("input", scheduleEdgePlanPreparation);

const canRecordAudio = Boolean(window.MediaRecorder && navigator.mediaDevices?.getUserMedia);
document.body.dataset.audioSupport = canRecordAudio ? "available" : "unavailable";
if (!canRecordAudio) {
  elements.micButton.disabled = true;
  elements.interim.textContent = "This browser cannot use the microphone. Open this local page in Chrome to try reading aloud.";
  elements.browserNote.textContent = "The scene creator still works here. Chrome on localhost supports the private local microphone flow.";
}

let edgePlannerKeepWarmTimer = null;
if (!demoMode) {
  window.BookforgeAnticipation.init({
    sessionId: readerSessionId,
    visualStyle: () => elements.style.value.trim(),
    currentProjection: () => ({
      server_instance_id: liveServerInstanceId,
      session_revision: liveSessionRevision,
    }),
    onShow: (pointer) => {
      handleLiveSceneSessionPointer(pointer, {restoreInputs: true});
      ensureProjectionPreview();
    },
  });
  connectLiveSceneSessionEvents();
  void warmEdgePlanner();
  edgePlannerKeepWarmTimer = window.setInterval(() => {
    if (document.visibilityState === "visible") void warmEdgePlanner();
  }, EDGE_PLANNER_KEEP_WARM_MS);
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "visible" && Date.now() >= edgePlannerWarmUntil) {
      void warmEdgePlanner();
    }
  });
  void inspectRendererReadiness();
}
setSceneInputsDisabled(false);
if (voiceMode) {
  elements.story.value = "";
  elements.style.value = "rich luminous watercolor storybook illustration, layered depth, detailed natural scenery, full-bleed 16:9";
  elements.micButtonText.textContent = "Describe scene";
  elements.compileButton.textContent = "Generate scene";
  elements.interim.textContent = "Describe what you want to see. Generation starts as you speak; completed artwork appears automatically.";
  elements.story.placeholder = "Your spoken description appears here. Edit it to refine or retry the scene.";
  updateMicAvailability();
}
if (demoMode) {
  elements.compileButton.disabled = true;
  elements.prewarmButton.disabled = true;
}
restoreInitialScene();
window.addEventListener("beforeunload", () => {
  window.clearInterval(edgePlannerKeepWarmTimer);
  window.clearTimeout(rendererWarmExpiryTimer);
  window.clearTimeout(edgePlanPreparationTimer);
  liveSessionEventSource?.close();
  stopLiveJobTransport();
  releaseMicrophone();
});
