const LOGICAL_WIDTH = 1920;
const LOGICAL_HEIGHT = 1080;
const PROFILE_KEY = "bookforge.projectionProfile.yaber-t1-pro";
const FIRED_CLASSES = ["action-reveal", "action-move", "action-transform", "action-open", "action-glow", "action-fade"];
const query = new URLSearchParams(window.location.search);
const SESSION_ID = query.get("session") || "moon-gate-demo";
const PACK_SOURCE = query.get("pack") || "fixture";

const elements = {
  stage: document.querySelector("#projectionStage"),
  scene: document.querySelector("#scene"),
  fixtureScene: document.querySelector("#fixtureScene"),
  generatedScene: document.querySelector("#generatedScene"),
  readerLine: document.querySelector("#readerLine"),
  packLabel: document.querySelector("#packLabel"),
  eventType: document.querySelector("#eventType"),
  eventDetail: document.querySelector("#eventDetail"),
  readerConnection: document.querySelector("#readerConnection"),
  readerConnectionText: document.querySelector("#readerConnectionText"),
  transcriptSimulator: document.querySelector("#transcriptSimulator"),
  transcriptInput: document.querySelector("#transcriptInput"),
  sessionLabel: document.querySelector("#sessionLabel"),
  triggerDelay: document.querySelector("#triggerDelay"),
  livePathDelay: document.querySelector("#livePathDelay"),
  averageDelay: document.querySelector("#averageDelay"),
  frameRate: document.querySelector("#frameRate"),
  droppedFrames: document.querySelector("#droppedFrames"),
  previous: document.querySelector("#previousButton"),
  next: document.querySelector("#nextButton"),
  reset: document.querySelector("#resetButton"),
  fullscreen: document.querySelector("#fullscreenButton"),
  calibrate: document.querySelector("#calibrateButton"),
  blackout: document.querySelector("#blackoutButton"),
  saveCalibration: document.querySelector("#saveCalibrationButton"),
  resetCalibration: document.querySelector("#resetCalibrationButton"),
  handles: [...document.querySelectorAll("#cornerHandles button")],
  startupError: document.querySelector("#startupError"),
};

const bus = new EventTarget();
const state = {
  pack: null,
  page: null,
  tokens: [],
  cursor: -1,
  triggerIndices: new Map(),
  delays: [],
  droppedFrames: 0,
  frameSamples: [],
  corners: [],
  socket: null,
  reconnectTimer: null,
  reconnectAttempt: 0,
  generation: null,
  lastReaderSequence: null,
  readerSync: null,
  resyncAfterCurrent: false,
  pendingReaderEvents: [],
};

function publish(type, detail = {}) {
  bus.dispatchEvent(new CustomEvent(type, {detail: {...detail, emittedAt: performance.now()}}));
}

function setEvent(type, detail) {
  elements.eventType.textContent = type;
  elements.eventDetail.textContent = detail;
}

function setReaderConnection(status, text) {
  elements.readerConnection.dataset.state = status;
  elements.readerConnectionText.textContent = text;
}

function normalizeWord(word) {
  return word.toLocaleLowerCase().replace(/^[^\p{L}\p{N}]+|[^\p{L}\p{N}]+$/gu, "");
}

function tokenize(text) {
  return [...text.matchAll(/[\p{L}\p{N}]+(?:['’\-][\p{L}\p{N}]+)*/gu)].map((match, index) => ({
    index,
    text: match[0],
    normalized: normalizeWord(match[0]),
  }));
}

function indexTriggers(page) {
  const indices = new Map();
  for (const trigger of page.triggers) {
    let seen = 0;
    const index = state.tokens.findIndex((token) => {
      if (token.normalized !== normalizeWord(trigger.word)) return false;
      seen += 1;
      return seen === trigger.occurrence;
    });
    if (index >= 0) {
      const existing = indices.get(index) || [];
      existing.push(trigger);
      indices.set(index, existing);
    }
  }
  return indices;
}

function assertStoryPack(pack) {
  if (!pack || pack.schema_version !== "1.1" || !Array.isArray(pack.pages) || !pack.pages.length) {
    throw new Error("Story Pack must use schema 1.1 and contain at least one page");
  }
  const page = pack.pages[0];
  if (!page.page_id || !page.source_text?.trim() || !Array.isArray(page.layers) || !Array.isArray(page.triggers)) {
    throw new Error("Story Pack page is missing source text, layers, or triggers");
  }
  const layerIds = new Set(page.layers.map((layer) => layer.layer_id));
  for (const trigger of page.triggers) {
    if (!layerIds.has(trigger.target_layer_id)) {
      throw new Error(`Trigger ${trigger.trigger_id} targets a missing layer`);
    }
  }
  return pack;
}

function layerPalette(index) {
  const palettes = [
    ["#111940", "#163438", "#82e6bd88"],
    ["#2b1231", "#421d26", "#ed654f88"],
    ["#102f31", "#1f4b46", "#fff1c988"],
    ["#20183d", "#102b3c", "#7cbff688"],
  ];
  return palettes[index % palettes.length];
}

function renderPackLayers(pack, page) {
  const fixture = pack.story_id === "moon-gate-projector-fixture";
  elements.fixtureScene.hidden = !fixture;
  elements.generatedScene.innerHTML = "";
  if (fixture) return;

  const assets = new Map(
    (pack.assets || [])
      .filter((asset) => asset.page_id === page.page_id && asset.state === "ready")
      .map((asset) => [asset.layer_id, asset]),
  );
  [...page.layers].sort((left, right) => left.z_index - right.z_index).forEach((layer, index) => {
    const node = document.createElement("div");
    const [start, end, accent] = layerPalette(index);
    node.className = `visual-layer generic-layer layer-kind-${layer.kind}`;
    node.dataset.layerId = layer.layer_id;
    node.style.zIndex = String(layer.z_index);
    node.style.setProperty("--layer-start", start);
    node.style.setProperty("--layer-end", end);
    node.style.setProperty("--layer-accent", accent);
    node.style.setProperty("--layer-x", `${25 + ((index * 23) % 55)}%`);
    node.style.setProperty("--layer-y", `${25 + ((index * 17) % 50)}%`);
    node.style.setProperty("--layer-angle", `${120 + index * 19}deg`);
    const asset = assets.get(layer.layer_id);
    const uri = asset?.local_uri || "";
    if (asset && uri.startsWith("/v1/assets/")) {
      const media = document.createElement(asset.kind === "video_loop" ? "video" : "img");
      media.src = uri;
      if (media instanceof HTMLVideoElement) {
        media.muted = true;
        media.loop = true;
        media.autoplay = true;
        media.playsInline = true;
      }
      media.alt = layer.prompt;
      node.append(media);
    } else {
      const label = document.createElement("span");
      label.className = "layer-development-label";
      label.textContent = `${layer.kind} · ${layer.prompt}`;
      node.append(label);
    }
    elements.generatedScene.append(node);
  });
}

function renderTimeline() {
  elements.readerLine.innerHTML = "";
  for (const token of state.tokens) {
    const span = document.createElement("span");
    span.textContent = token.text;
    span.dataset.index = String(token.index);
    elements.readerLine.append(span);
  }
  updateTimeline();
}

function updateTimeline() {
  for (const node of elements.readerLine.children) {
    const index = Number(node.dataset.index);
    node.classList.toggle("reached", index <= state.cursor);
    node.classList.toggle("current", index === state.cursor);
  }
}

function clearLayerState() {
  for (const layer of elements.scene.querySelectorAll("[data-layer-id]")) {
    layer.className = layer.className
      .split(" ")
      .filter((name) => !name.startsWith("trigger-") && !FIRED_CLASSES.includes(name))
      .join(" ");
    layer.style.removeProperty("--trigger-duration");
  }
}

function applyTrigger(trigger, emittedAt, measure = true, publishedAt = null) {
  const target = elements.scene.querySelector(`[data-layer-id="${CSS.escape(trigger.target_layer_id)}"]`);
  if (!target) return;
  target.style.setProperty("--trigger-duration", `${trigger.duration_ms}ms`);
  target.classList.add(`action-${trigger.action}`, `trigger-${trigger.trigger_id.replace(/[^a-z0-9_-]/gi, "-")}`);
  publish("trigger.fired", {trigger, targetLayerId: trigger.target_layer_id});
  if (!measure) return;
  requestAnimationFrame(() => {
    const delay = performance.now() - emittedAt;
    state.delays.push(delay);
    const average = state.delays.reduce((sum, value) => sum + value, 0) / state.delays.length;
    elements.triggerDelay.textContent = `${delay.toFixed(1)} ms`;
    elements.averageDelay.textContent = `${average.toFixed(1)} ms`;
    if (Number.isFinite(publishedAt)) {
      elements.livePathDelay.textContent = `${Math.max(0, Date.now() - publishedAt).toFixed(1)} ms`;
    }
    setEvent("trigger.fired", `${trigger.word} → ${trigger.action} ${trigger.target_layer_id}`);
  });
}

function rebuildScene() {
  elements.scene.classList.add("no-motion");
  clearLayerState();
  for (const [index, triggers] of state.triggerIndices) {
    if (index > state.cursor) continue;
    for (const trigger of triggers) applyTrigger(trigger, performance.now(), false);
  }
  requestAnimationFrame(() => elements.scene.classList.remove("no-motion"));
}

function goToWord(nextCursor, publishedAt = null) {
  if (!state.page) return;
  const clamped = Math.max(-1, Math.min(state.tokens.length - 1, nextCursor));
  const movingForwardOne = clamped === state.cursor + 1;
  state.cursor = clamped;
  updateTimeline();
  if (clamped < 0) {
    clearLayerState();
    state.delays = [];
    elements.triggerDelay.textContent = "—";
    elements.livePathDelay.textContent = "—";
    elements.averageDelay.textContent = "—";
    setEvent("session.reset", "Ready for the first word");
    return;
  }
  const token = state.tokens[clamped];
  const emittedAt = performance.now();
  publish("word.reached", {word: token.text, index: clamped});
  setEvent("word.reached", `${clamped + 1}/${state.tokens.length} · ${token.text}`);
  if (!movingForwardOne) {
    rebuildScene();
    return;
  }
  const triggers = state.triggerIndices.get(clamped) || [];
  for (const trigger of triggers) applyTrigger(trigger, emittedAt, true, publishedAt);
}

function handleReaderEvent(message) {
  if (!message || typeof message.type !== "string") return;
  const payload = message.payload && typeof message.payload === "object" ? message.payload : message;
  if (message.type === "word.reached" && Number.isInteger(payload.index)) {
    const token = state.tokens[payload.index];
    if (
      payload.page_id !== state.page?.page_id
      || payload.generation !== state.generation
      || !token
      || normalizeWord(payload.word || "") !== token.normalized
    ) return;
    const publishedAt = Date.parse(message.published_at || "");
    if (payload.index > state.cursor) goToWord(payload.index, publishedAt);
    return;
  }
  if (message.type === "session.reset") {
    if (payload.page_id !== state.page?.page_id || payload.page_text !== state.page?.source_text) {
      setEvent("reader.mismatch", "The live reader is using a different Story Pack");
      return;
    }
    if (!Number.isInteger(payload.generation)) return;
    if (state.generation !== null && payload.generation <= state.generation) return;
    state.generation = payload.generation;
    goToWord(-1);
    return;
  }
  if (message.type === "transcript.partial") {
    if (payload.page_id !== state.page?.page_id || payload.generation !== state.generation) {
      setEvent("reader.mismatch", "Ignored transcript from a different reading");
      return;
    }
    const transcript = payload.transcript || payload.text || "";
    setEvent("transcript.partial", transcript || "Listening…");
  }
}

function applyReaderStatus(status) {
  if (status.page_id !== state.page?.page_id || status.page_text !== state.page?.source_text) {
    setEvent("reader.mismatch", "The live reader is using a different Story Pack");
    return false;
  }
  const nextCursor = status.last_reached_index ?? -1;
  if (state.generation !== null && status.generation < state.generation) return true;
  if (status.generation === state.generation && nextCursor <= state.cursor) return true;
  state.generation = status.generation;
  goToWord(nextCursor);
  return true;
}

async function synchronizeReaderSession() {
  if (!state.page) return;
  const response = await fetch(`/v1/reader-sessions/${encodeURIComponent(SESSION_ID)}`, {
    cache: "no-store",
  });
  if (response.status === 404) return false;
  if (!response.ok) throw new Error(`Reader synchronization failed (${response.status})`);
  const status = await response.json();
  return applyReaderStatus(status);
}

function requestReaderSynchronization(forceAfterCurrent = false) {
  if (state.readerSync) {
    if (forceAfterCurrent) state.resyncAfterCurrent = true;
    return state.readerSync;
  }
  const synchronization = (async () => {
    try {
      await synchronizeReaderSession();
      if (!state.resyncAfterCurrent) {
        const pending = state.pendingReaderEvents.splice(0);
        pending.forEach(handleReaderEvent);
      }
    } catch (error) {
      state.pendingReaderEvents = [];
      state.resyncAfterCurrent = false;
      setEvent("reader.error", error.message);
      state.socket?.close();
    } finally {
      if (state.readerSync === synchronization) {
        state.readerSync = null;
        if (state.resyncAfterCurrent) {
          state.resyncAfterCurrent = false;
          requestReaderSynchronization();
        }
      }
    }
  })();
  state.readerSync = synchronization;
  return synchronization;
}

function receiveReaderEvent(message) {
  if (!message || !Number.isInteger(message.sequence)) return;
  if (state.lastReaderSequence !== null && message.sequence <= state.lastReaderSequence) return;
  const sequenceGap = (
    message.dropped_before_sequence !== null
    && message.dropped_before_sequence !== undefined
  ) || (
    state.lastReaderSequence !== null
    && message.sequence > state.lastReaderSequence + 1
  );
  state.lastReaderSequence = message.sequence;
  if (state.readerSync || sequenceGap) {
    state.pendingReaderEvents.push(message);
    if (sequenceGap) requestReaderSynchronization(true);
    return;
  }
  handleReaderEvent(message);
}

function connectReaderSession() {
  if (state.socket?.readyState === WebSocket.OPEN || state.socket?.readyState === WebSocket.CONNECTING) return;
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  setReaderConnection("connecting", `Connecting to ${SESSION_ID}`);
  const socket = new WebSocket(`${protocol}//${window.location.host}/v1/reader-sessions/${encodeURIComponent(SESSION_ID)}/events`);
  state.socket = socket;
  socket.addEventListener("open", () => {
    state.reconnectAttempt = 0;
    setReaderConnection("connected", "Connected · waiting for transcript");
    requestReaderSynchronization(true);
  });
  socket.addEventListener("message", (event) => {
    try {
      receiveReaderEvent(JSON.parse(event.data));
    } catch (_) {
      setEvent("reader.error", "Ignored an invalid live event");
    }
  });
  socket.addEventListener("close", () => {
    if (state.socket !== socket) return;
    state.socket = null;
    state.lastReaderSequence = null;
    state.pendingReaderEvents = [];
    state.reconnectAttempt += 1;
    const delay = Math.min(10_000, 500 * 2 ** Math.min(state.reconnectAttempt, 5));
    setReaderConnection("disconnected", `Disconnected · retrying in ${(delay / 1000).toFixed(1)}s`);
    clearTimeout(state.reconnectTimer);
    state.reconnectTimer = setTimeout(connectReaderSession, delay);
  });
  socket.addEventListener("error", () => socket.close());
}

async function simulateTranscript(text) {
  const response = await fetch(`/v1/reader-sessions/${encodeURIComponent(SESSION_ID)}/transcripts:simulate`, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({
      source: "typed",
      text,
      page_id: state.page?.page_id || "page-01",
      generation: state.generation,
      language: "en",
      is_final: false,
    }),
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.detail || `Transcript simulation failed (${response.status})`);
  }
}

async function configureReaderSession() {
  const response = await fetch(`/v1/reader-sessions/${encodeURIComponent(SESSION_ID)}`, {
    method: "PUT",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({page_id: state.page.page_id, page_text: state.page.source_text}),
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.detail || `Reader session setup failed (${response.status})`);
  }
  const status = await response.json();
  state.generation = status.generation;
  return status;
}

async function resetReaderSession() {
  const response = await fetch(`/v1/reader-sessions/${encodeURIComponent(SESSION_ID)}:reset`, {
    method: "POST",
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.detail || `Reader session reset failed (${response.status})`);
  }
  const status = await response.json();
  state.generation = status.generation;
  goToWord(-1);
  return status;
}

function defaultCorners() {
  const viewportAspect = window.innerWidth / window.innerHeight;
  const stageAspect = LOGICAL_WIDTH / LOGICAL_HEIGHT;
  if (viewportAspect > stageAspect) {
    const width = stageAspect / viewportAspect;
    const inset = (1 - width) / 2;
    return [{x: inset, y: 0}, {x: 1 - inset, y: 0}, {x: 1 - inset, y: 1}, {x: inset, y: 1}];
  }
  const height = viewportAspect / stageAspect;
  const inset = (1 - height) / 2;
  return [{x: 0, y: inset}, {x: 1, y: inset}, {x: 1, y: 1 - inset}, {x: 0, y: 1 - inset}];
}

function solveLinearSystem(matrix, values) {
  const size = values.length;
  const augmented = matrix.map((row, index) => [...row, values[index]]);
  for (let column = 0; column < size; column += 1) {
    let pivot = column;
    for (let row = column + 1; row < size; row += 1) {
      if (Math.abs(augmented[row][column]) > Math.abs(augmented[pivot][column])) pivot = row;
    }
    [augmented[column], augmented[pivot]] = [augmented[pivot], augmented[column]];
    const divisor = augmented[column][column];
    if (Math.abs(divisor) < 1e-10) throw new Error("Calibration corners cannot form a valid surface.");
    for (let item = column; item <= size; item += 1) augmented[column][item] /= divisor;
    for (let row = 0; row < size; row += 1) {
      if (row === column) continue;
      const factor = augmented[row][column];
      for (let item = column; item <= size; item += 1) augmented[row][item] -= factor * augmented[column][item];
    }
  }
  return augmented.map((row) => row[size]);
}

function homography(source, target) {
  const matrix = [];
  const values = [];
  for (let index = 0; index < 4; index += 1) {
    const {x, y} = source[index];
    const {x: u, y: v} = target[index];
    matrix.push([x, y, 1, 0, 0, 0, -u * x, -u * y]);
    values.push(u);
    matrix.push([0, 0, 0, x, y, 1, -v * x, -v * y]);
    values.push(v);
  }
  return solveLinearSystem(matrix, values);
}

function updateProjection() {
  const source = [
    {x: 0, y: 0}, {x: LOGICAL_WIDTH, y: 0},
    {x: LOGICAL_WIDTH, y: LOGICAL_HEIGHT}, {x: 0, y: LOGICAL_HEIGHT},
  ];
  const target = state.corners.map((corner) => ({x: corner.x * window.innerWidth, y: corner.y * window.innerHeight}));
  try {
    const [h11, h12, h13, h21, h22, h23, h31, h32] = homography(source, target);
    elements.stage.style.transform = `matrix3d(${h11},${h21},0,${h31},${h12},${h22},0,${h32},0,0,1,0,${h13},${h23},0,1)`;
    elements.startupError.classList.add("hidden");
  } catch (error) {
    elements.startupError.textContent = error.message;
    elements.startupError.classList.remove("hidden");
  }
  state.corners.forEach((corner, index) => {
    elements.handles[index].style.left = `${corner.x * window.innerWidth}px`;
    elements.handles[index].style.top = `${corner.y * window.innerHeight}px`;
  });
}

function loadCalibration() {
  try {
    const saved = JSON.parse(localStorage.getItem(PROFILE_KEY));
    if (saved?.profile === "yaber-t1-pro" && saved.corners?.length === 4) {
      state.corners = saved.corners;
      return;
    }
  } catch (_) {
    localStorage.removeItem(PROFILE_KEY);
  }
  state.corners = defaultCorners();
}

function saveCalibration() {
  localStorage.setItem(PROFILE_KEY, JSON.stringify({
    profile: "yaber-t1-pro",
    logicalCanvas: {width: LOGICAL_WIDTH, height: LOGICAL_HEIGHT},
    corners: state.corners,
    savedAt: new Date().toISOString(),
  }));
  setEvent("calibration.saved", "Yaber T1 Pro profile stored locally");
}

function toggleCalibration() {
  document.body.classList.toggle("calibrating");
  elements.calibrate.textContent = document.body.classList.contains("calibrating") ? "Finish calibration" : "Calibrate";
  updateProjection();
}

function bindCalibrationHandles() {
  elements.handles.forEach((handle, index) => {
    handle.addEventListener("pointerdown", (event) => {
      handle.setPointerCapture(event.pointerId);
    });
    handle.addEventListener("pointermove", (event) => {
      if (!handle.hasPointerCapture(event.pointerId)) return;
      state.corners[index] = {
        x: Math.max(0, Math.min(1, event.clientX / window.innerWidth)),
        y: Math.max(0, Math.min(1, event.clientY / window.innerHeight)),
      };
      updateProjection();
    });
  });
}

function monitorFrames(timestamp) {
  if (!monitorFrames.startedAt) monitorFrames.startedAt = timestamp;
  const previous = monitorFrames.previous || timestamp;
  const delta = timestamp - previous;
  monitorFrames.previous = timestamp;
  const sampling = timestamp - monitorFrames.startedAt > 1500 && document.visibilityState === "visible";
  if (sampling && delta > 25) {
    state.droppedFrames += Math.max(1, Math.round(delta / 16.67) - 1);
  }
  state.frameSamples.push(delta);
  if (state.frameSamples.length > 60) state.frameSamples.shift();
  const average = state.frameSamples.reduce((sum, value) => sum + value, 0) / state.frameSamples.length;
  elements.frameRate.textContent = average > 0 ? `${Math.min(240, 1000 / average).toFixed(0)} fps` : "—";
  elements.droppedFrames.textContent = String(state.droppedFrames);
  requestAnimationFrame(monitorFrames);
}

async function loadStoryPack() {
  try {
    if (PACK_SOURCE === "latest") {
      const response = await fetch("/v1/story-packs/latest", {cache: "no-store"});
      if (response.ok) {
        state.pack = assertStoryPack(await response.json());
      } else {
        const saved = localStorage.getItem("bookforge.latestStoryPack");
        if (saved) {
          state.pack = assertStoryPack(JSON.parse(saved));
        } else {
          const fixture = await fetch("/workbench-assets/moon-gate.story-pack.json", {cache: "no-store"});
          if (!fixture.ok) throw new Error(`Fallback Story Pack failed to load (${fixture.status})`);
          state.pack = assertStoryPack(await fixture.json());
        }
      }
    } else {
      const response = await fetch("/workbench-assets/moon-gate.story-pack.json", {cache: "no-store"});
      if (!response.ok) throw new Error(`Story Pack failed to load (${response.status})`);
      state.pack = assertStoryPack(await response.json());
    }
    state.page = state.pack.pages[0];
    renderPackLayers(state.pack, state.page);
    state.tokens = tokenize(state.page.source_text);
    state.triggerIndices = indexTriggers(state.page);
    elements.packLabel.textContent = `${state.pack.title} · ${state.pack.schema_version}`;
    renderTimeline();
    const session = await configureReaderSession();
    goToWord(session.last_reached_index ?? -1);
    publish("page.loaded", {pageId: state.page.page_id, assetCount: state.pack.assets.length});
    setEvent("page.loaded", `${state.tokens.length} words · ${state.pack.assets.length} cached layers`);
  } catch (error) {
    elements.startupError.textContent = `Bookforge could not start: ${error.message}`;
    elements.startupError.classList.remove("hidden");
  }
}

elements.previous.addEventListener("click", () => goToWord(state.cursor - 1));
elements.next.addEventListener("click", () => goToWord(state.cursor + 1));
elements.reset.addEventListener("click", async () => {
  try {
    await resetReaderSession();
  } catch (error) {
    setEvent("reader.error", error.message);
  }
});
elements.calibrate.addEventListener("click", toggleCalibration);
elements.blackout.addEventListener("click", () => document.body.classList.toggle("blackout"));
elements.fullscreen.addEventListener("click", async () => {
  if (document.fullscreenElement) await document.exitFullscreen();
  else await document.documentElement.requestFullscreen();
});
elements.saveCalibration.addEventListener("click", saveCalibration);
elements.resetCalibration.addEventListener("click", () => {
  state.corners = defaultCorners();
  updateProjection();
  setEvent("calibration.reset", "Corners returned to a centered 16:9 canvas");
});

document.addEventListener("keydown", (event) => {
  if (event.repeat) return;
  const target = event.target;
  if (target instanceof HTMLInputElement || target instanceof HTMLTextAreaElement || target?.isContentEditable) return;
  const key = event.key.toLowerCase();
  if (event.key === " " || event.key === "ArrowRight") {
    event.preventDefault();
    goToWord(state.cursor + 1);
  } else if (event.key === "ArrowLeft") {
    event.preventDefault();
    goToWord(state.cursor - 1);
  } else if (key === "r") elements.reset.click();
  else if (key === "f") elements.fullscreen.click();
  else if (key === "c") toggleCalibration();
  else if (key === "b") document.body.classList.toggle("blackout");
  else if (key === "h") document.body.classList.toggle("hud-hidden");
});

window.addEventListener("resize", updateProjection);
window.addEventListener("beforeunload", () => {
  clearTimeout(state.reconnectTimer);
  state.socket?.close();
});
elements.transcriptSimulator.addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = elements.transcriptInput.value.trim();
  if (!text) return;
  try {
    await simulateTranscript(text);
  } catch (error) {
    setEvent("reader.error", error.message);
  }
});
elements.sessionLabel.textContent = SESSION_ID;
loadCalibration();
bindCalibrationHandles();
updateProjection();
loadStoryPack().then(() => {
  if (state.page) connectReaderSession();
});
requestAnimationFrame(monitorFrames);
