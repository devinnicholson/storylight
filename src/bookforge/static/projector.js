const LOGICAL_WIDTH = 1920;
const LOGICAL_HEIGHT = 1080;
const PROFILE_KEY = "bookforge.projectionProfile.yaber-t1-pro";
const FIRED_CLASSES = ["action-reveal", "action-move", "action-transform", "action-open", "action-glow", "action-fade"];

const elements = {
  stage: document.querySelector("#projectionStage"),
  scene: document.querySelector("#scene"),
  readerLine: document.querySelector("#readerLine"),
  packLabel: document.querySelector("#packLabel"),
  eventType: document.querySelector("#eventType"),
  eventDetail: document.querySelector("#eventDetail"),
  triggerDelay: document.querySelector("#triggerDelay"),
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
};

function publish(type, detail = {}) {
  bus.dispatchEvent(new CustomEvent(type, {detail: {...detail, emittedAt: performance.now()}}));
}

function setEvent(type, detail) {
  elements.eventType.textContent = type;
  elements.eventDetail.textContent = detail;
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

function applyTrigger(trigger, emittedAt, measure = true) {
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

function goToWord(nextCursor) {
  if (!state.page) return;
  const clamped = Math.max(-1, Math.min(state.tokens.length - 1, nextCursor));
  const movingForwardOne = clamped === state.cursor + 1;
  state.cursor = clamped;
  updateTimeline();
  if (clamped < 0) {
    clearLayerState();
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
  for (const trigger of triggers) applyTrigger(trigger, emittedAt);
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
    const response = await fetch("/workbench-assets/moon-gate.story-pack.json", {cache: "no-store"});
    if (!response.ok) throw new Error(`Story Pack failed to load (${response.status})`);
    state.pack = await response.json();
    state.page = state.pack.pages[0];
    state.tokens = tokenize(state.page.source_text);
    state.triggerIndices = indexTriggers(state.page);
    elements.packLabel.textContent = `${state.pack.title} · ${state.pack.schema_version}`;
    renderTimeline();
    publish("page.loaded", {pageId: state.page.page_id, assetCount: state.pack.assets.length});
    setEvent("page.loaded", `${state.tokens.length} words · ${state.pack.assets.length} cached layers`);
  } catch (error) {
    elements.startupError.textContent = `Bookforge could not start: ${error.message}`;
    elements.startupError.classList.remove("hidden");
  }
}

elements.previous.addEventListener("click", () => goToWord(state.cursor - 1));
elements.next.addEventListener("click", () => goToWord(state.cursor + 1));
elements.reset.addEventListener("click", () => goToWord(-1));
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
  const key = event.key.toLowerCase();
  if (event.key === " " || event.key === "ArrowRight") {
    event.preventDefault();
    goToWord(state.cursor + 1);
  } else if (event.key === "ArrowLeft") {
    event.preventDefault();
    goToWord(state.cursor - 1);
  } else if (key === "r") goToWord(-1);
  else if (key === "f") elements.fullscreen.click();
  else if (key === "c") toggleCalibration();
  else if (key === "b") document.body.classList.toggle("blackout");
  else if (key === "h") document.body.classList.toggle("hud-hidden");
});

window.addEventListener("resize", updateProjection);
loadCalibration();
bindCalibrationHandles();
updateProjection();
loadStoryPack();
requestAnimationFrame(monitorFrames);
