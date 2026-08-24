const LOGICAL_WIDTH = 1920;
const LOGICAL_HEIGHT = 1080;
const PROFILE_KEY = "bookforge.projectionProfile.yaber-t1-pro";
const FIRED_CLASSES = ["action-reveal", "action-move", "action-transform", "action-open", "action-glow", "action-fade"];
const query = new URLSearchParams(window.location.search);
const SESSION_ID = query.get("session") || "bookforge-live";
const PACK_SOURCE = query.get("pack") || "fixture";
const PRESENTATION_MODE = query.get("debug") !== "1";
const OFFLINE_REPLAY = query.get("offline") === "1";
const LIVE_MODE = query.get("live") === "1";
const SCENE_CROSSFADE_MS = 320;
const SCENE_RETIRE_GRACE_MS = 360;
const DEPTH_RENDER_TARGET_FPS = 30;

if (PRESENTATION_MODE) document.body.classList.add("hud-hidden");
if (OFFLINE_REPLAY) document.body.dataset.replayBoundary = "loopback-only";

function localFetch(input, init) {
  const rawUrl = typeof input === "string" ? input : input.url;
  const url = new URL(rawUrl, window.location.href);
  if (OFFLINE_REPLAY && url.origin !== window.location.origin) {
    throw new Error(`Offline replay blocked a non-local request to ${url.origin}`);
  }
  return window.fetch(input, init);
}

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
  previousPage: document.querySelector("#previousPageButton"),
  nextPage: document.querySelector("#nextPageButton"),
  pageLabel: document.querySelector("#pageLabel"),
  reset: document.querySelector("#resetButton"),
  fullscreen: document.querySelector("#fullscreenButton"),
  calibrate: document.querySelector("#calibrateButton"),
  blackout: document.querySelector("#blackoutButton"),
  saveCalibration: document.querySelector("#saveCalibrationButton"),
  resetCalibration: document.querySelector("#resetCalibrationButton"),
  handles: [...document.querySelectorAll("#cornerHandles button")],
  startupError: document.querySelector("#startupError"),
  liveGenerationBadge: document.querySelector("#liveGenerationBadge"),
  liveGenerationStage: document.querySelector("#liveGenerationStage"),
  liveGenerationDetail: document.querySelector("#liveGenerationDetail"),
  liveGenerationElapsed: document.querySelector("#liveGenerationElapsed"),
};

const bus = new EventTarget();
const state = {
  pack: null,
  page: null,
  pageIndex: 0,
  pageTransition: null,
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
  readerConfiguredPageId: null,
  readerConfiguredPageText: null,
  lastReaderSequence: null,
  readerSync: null,
  resyncAfterCurrent: false,
  pendingReaderEvents: [],
  depthRenderer: null,
  liveSessionEventSource: null,
  liveSessionStreamHealthy: false,
  liveRendezvousTimer: null,
  liveRendezvousInFlight: false,
  liveServerInstanceId: null,
  liveSessionRevision: 0,
  liveSessionJobId: null,
  liveAcceptedJobId: null,
  liveAcceptedRevision: -1,
  liveCommittedJobId: null,
  liveCommittedRevision: -1,
  liveRenderPending: false,
  liveRenderEpoch: 0,
  liveRenderAbortController: null,
  liveAssetFingerprint: null,
  liveRevision: -1,
  liveJobId: null,
  liveStage: null,
  liveStartedAt: 0,
  liveElapsedMs: 0,
  liveActivationMs: null,
  liveActivationBreakdown: null,
  liveUpdatedAt: 0,
  liveTerminal: false,
  liveLastEnvelopeAt: 0,
  liveTransition: Promise.resolve(),
  lastSceneCommitPaint: null,
  screenWakeLock: null,
  screenWakeLockRequest: null,
};

function publish(type, detail = {}) {
  bus.dispatchEvent(new CustomEvent(type, {detail: {...detail, emittedAt: performance.now()}}));
}

function setEvent(type, detail) {
  elements.eventType.textContent = type;
  elements.eventDetail.textContent = detail;
}

async function requestProjectorWakeLock() {
  if (!PRESENTATION_MODE || document.visibilityState !== "visible") return;
  if (!("wakeLock" in navigator)) {
    document.body.dataset.projectorWakeLock = "unsupported";
    return;
  }
  if (state.screenWakeLock || state.screenWakeLockRequest) return;

  document.body.dataset.projectorWakeLock = "requesting";
  state.screenWakeLockRequest = navigator.wakeLock.request("screen");
  try {
    const sentinel = await state.screenWakeLockRequest;
    state.screenWakeLock = sentinel;
    document.body.dataset.projectorWakeLock = "active";
    sentinel.addEventListener("release", () => {
      if (state.screenWakeLock === sentinel) state.screenWakeLock = null;
      document.body.dataset.projectorWakeLock = "released";
    });
  } catch (_) {
    document.body.dataset.projectorWakeLock = "unavailable";
  } finally {
    state.screenWakeLockRequest = null;
  }
}

function setupProjectorWakeLock() {
  if (!PRESENTATION_MODE) return;
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "visible") void requestProjectorWakeLock();
    else void state.screenWakeLock?.release();
  });
  void requestProjectorWakeLock();
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
  const supportedSchema = pack && ["1.1", "2.0"].includes(pack.schema_version);
  if (!supportedSchema || !Array.isArray(pack.pages) || !pack.pages.length) {
    throw new Error("Story Pack must use schema 1.1 or 2.0 and contain at least one page");
  }
  const pageIds = new Set();
  for (const page of pack.pages) {
    if (!page.page_id || !page.source_text?.trim() || !Array.isArray(page.layers) || !Array.isArray(page.triggers)) {
      throw new Error("Story Pack page is missing source text, layers, or triggers");
    }
    if (pageIds.has(page.page_id)) throw new Error(`Story Pack repeats page ${page.page_id}`);
    pageIds.add(page.page_id);
    const layerIds = new Set(page.layers.map((layer) => layer.layer_id));
    for (const trigger of page.triggers) {
      if (!layerIds.has(trigger.target_layer_id)) {
        throw new Error(`Trigger ${trigger.trigger_id} targets a missing layer`);
      }
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

function stablePassageHash(text) {
  let hash = 2166136261;
  for (const character of text) {
    hash ^= character.codePointAt(0);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

function passageDraftTheme(page) {
  const text = page.source_text.toLocaleLowerCase();
  const ambientKinds = new Set((page.scene_spec?.ambience || []).map((effect) => effect.kind));
  const themes = [
    {
      name: "space",
      matches: ambientKinds.has("stars") || /\b(star|moon|planet|space|galaxy|rocket)\b/u.test(text),
      palettes: [
        ["#080d2b", "#22245a", "#a9c9ff99"],
        ["#241344", "#0d2848", "#e5dcff99"],
        ["#071c39", "#1b1745", "#86bfff99"],
      ],
    },
    {
      name: "ocean",
      matches: /\b(ocean|sea|wave|whale|fish|river|boat|water)\b/u.test(text),
      palettes: [
        ["#052f45", "#087b7c", "#83f3e499"],
        ["#07364e", "#155f6d", "#b9fff499"],
        ["#0a2945", "#0b8086", "#65d7d599"],
      ],
    },
    {
      name: "forest",
      matches: ambientKinds.has("fireflies") || /\b(forest|tree|fox|deer|mushroom|garden|leaf|wood)\b/u.test(text),
      palettes: [
        ["#0b281d", "#315232", "#ffe58a99"],
        ["#132b22", "#536331", "#a9e59099"],
        ["#081f1a", "#3e4e2a", "#f5d06f99"],
      ],
    },
    {
      name: "storm",
      matches: /\b(storm|rain|thunder|lightning|cloud|wind|snow)\b/u.test(text),
      palettes: [
        ["#111a2b", "#3e4a60", "#e7f2ff99"],
        ["#172033", "#5a6071", "#b9c8dc99"],
        ["#0b1528", "#3d4862", "#d6e9ff99"],
      ],
    },
    {
      name: "literacy",
      matches: /\b(book|letter|word|read|library|story|page|school)\b/u.test(text),
      palettes: [
        ["#21182f", "#5b3c33", "#ffe2a199"],
        ["#15233a", "#614b35", "#fff0c799"],
        ["#2a1830", "#69412d", "#f4d27d99"],
      ],
    },
  ];
  const hash = stablePassageHash(page.source_text);
  const selected = themes.find((theme) => theme.matches) || {
    name: "imaginative",
    palettes: [
      ["#111940", "#163438", "#82e6bd88"],
      ["#2b1231", "#421d26", "#ed654f88"],
      ["#20183d", "#102b3c", "#7cbff688"],
    ],
  };
  const focusMotif = [
    ["whale", /\bwhale\b/u],
    ["fox", /\bfox\b/u],
    ["turtle", /\bturtle\b/u],
    ["reader", /\b(child|student|reader|person|teacher)\b/u],
  ].find(([, pattern]) => pattern.test(text))?.[0] || "story-subject";
  const effectMotif = [
    ["flock", /\b(origami|paper bird|birds?)\b/u],
    ["jellyfish", /\bjellyfish\b/u],
    ["school", /\b(fish|school of fish)\b/u],
    ["swarm", /\b(moths?|butterfl(?:y|ies)|fireflies)\b/u],
    ["bloom", /\b(flowers?|garden|blooms?)\b/u],
    ["constellation", /\b(constellations?|stars?|galaxy)\b/u],
  ].find(([, pattern]) => pattern.test(text))?.[0] || "story-magic";
  return {
    ...selected,
    paletteOffset: hash % selected.palettes.length,
    focusMotif,
    effectMotif,
  };
}

function placeholderLayout(kind, index) {
  const layouts = {
    background: {x: 60, y: 18, width: 48, height: 15},
    character: {x: 50, y: 66, width: 25, height: 36},
    prop: {x: 76, y: 62, width: 25, height: 34},
    effect: {x: 76, y: 43, width: 27, height: 29},
    typography: {x: 58, y: 31, width: 42, height: 15},
  };
  const layout = layouts[kind] || layouts.prop;
  const offset = (index % 3) * 3;
  return {
    ...layout,
    x: Math.min(84, layout.x + offset),
    y: Math.min(78, layout.y + offset),
  };
}

function sceneCompositionLayout(sceneSpec, layer, index) {
  const composition = (sceneSpec?.composition || []).find(
    (item) => item.layer_id === layer.layer_id,
  );
  if (!composition || layer.kind === "background") return placeholderLayout(layer.kind, index);
  return {
    x: composition.center_x * 100,
    y: composition.center_y * 100,
    width: composition.width * 100,
    height: composition.height * 100,
  };
}

function bundledHeroForPage(page) {
  if (LIVE_MODE) return null;
  const source = page.source_text.trim().toLocaleLowerCase();
  if (source === "the small moth went through the red gate.") {
    return {
      video: "/workbench-assets/assets/moon-gate-loop-v1.mp4",
      poster: "/workbench-assets/assets/moon-gate-hero-v1.png",
      label: "Moonlit cut-paper valley with a moth near a glowing red gate",
    };
  }
  if (source === "at dusk, a silver fox carried a golden lantern beneath the enormous cedar trees.") {
    return {
      video: "/workbench-assets/assets/silver-fox-loop-v1.mp4",
      poster: "",
      label: "Silver fox carrying a golden lantern beneath enormous watercolor cedar trees",
    };
  }
  return null;
}

function loadSceneImage(uri, {signal = null, timeoutMs = 5000} = {}) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.decoding = "async";
    let settled = false;
    const finish = (callback, value) => {
      if (settled) return;
      settled = true;
      window.clearTimeout(timeout);
      signal?.removeEventListener("abort", onAbort);
      callback(value);
    };
    const onAbort = () => {
      image.src = "";
      finish(reject, new Error("Scene image load superseded"));
    };
    const timeout = window.setTimeout(() => {
      image.src = "";
      finish(reject, new Error(`Scene asset timed out: ${uri}`));
    }, timeoutMs);
    image.addEventListener("load", () => finish(resolve, image), {once: true});
    image.addEventListener(
      "error",
      () => finish(reject, new Error(`Scene asset failed to load: ${uri}`)),
      {once: true},
    );
    if (signal?.aborted) {
      onAbort();
      return;
    }
    signal?.addEventListener("abort", onAbort, {once: true});
    image.src = uri;
  });
}

function projectionToneForImage(image) {
  const sample = document.createElement("canvas");
  sample.width = 32;
  sample.height = 18;
  const context = sample.getContext("2d", {alpha: false, willReadFrequently: true});
  if (!context) return {fallbackExposure: 1, gamma: 1, meanLuma: null};
  try {
    context.drawImage(image, 0, 0, sample.width, sample.height);
    const pixels = context.getImageData(0, 0, sample.width, sample.height).data;
    let luma = 0;
    for (let offset = 0; offset < pixels.length; offset += 4) {
      luma += (0.2126 * pixels[offset] + 0.7152 * pixels[offset + 1] + 0.0722 * pixels[offset + 2]) / 255;
    }
    const meanLuma = luma / (pixels.length / 4);
    // Gamma lifts shadow detail without clipping highlights like a large linear
    // multiplier would. Bright artwork remains neutral. The still-image fallback
    // cannot apply gamma, so it uses a separately capped brightness correction.
    const targetLuma = 0.32;
    const boundedLuma = Math.min(0.99, Math.max(0.01, meanLuma));
    const gamma = meanLuma < targetLuma
      ? Math.max(0.72, Math.min(1, Math.log(targetLuma) / Math.log(boundedLuma)))
      : 1;
    return {
      fallbackExposure: Math.min(1.45, Math.max(1, targetLuma / boundedLuma)),
      gamma,
      meanLuma,
    };
  } catch (_error) {
    return {fallbackExposure: 1, gamma: 1, meanLuma: null};
  }
}

function projectionExposureForImage(image) {
  return projectionToneForImage(image).fallbackExposure;
}

function sizeDepthCanvasToSource(canvas, image) {
  const sourceWidth = Number(image.naturalWidth) || LOGICAL_WIDTH;
  const sourceHeight = Number(image.naturalHeight) || LOGICAL_HEIGHT;
  canvas.width = Math.min(LOGICAL_WIDTH, sourceWidth);
  canvas.height = Math.min(LOGICAL_HEIGHT, sourceHeight);
  canvas.dataset.renderWidth = String(canvas.width);
  canvas.dataset.renderHeight = String(canvas.height);
  return {
    width: canvas.width,
    height: canvas.height,
    pixels: canvas.width * canvas.height,
    logicalPixels: LOGICAL_WIDTH * LOGICAL_HEIGHT,
  };
}

function compileShader(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const detail = gl.getShaderInfoLog(shader);
    gl.deleteShader(shader);
    throw new Error(`Depth shader failed: ${detail}`);
  }
  return shader;
}

function createTexture(gl, image, textureUnit) {
  const texture = gl.createTexture();
  gl.activeTexture(textureUnit);
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image);
  return texture;
}

async function startDepthRenderer(canvas, masterImage, depthImage, sceneSpec) {
  const gl = canvas.getContext("webgl2", {
    alpha: false,
    antialias: false,
    depth: false,
    powerPreference: "high-performance",
  });
  if (!gl) throw new Error("WebGL 2 is unavailable; using the still-image fallback");
  const vertexSource = `#version 300 es
    in vec2 a_position;
    out vec2 v_uv;
    void main() {
      v_uv = a_position * 0.5 + 0.5;
      gl_Position = vec4(a_position, 0.0, 1.0);
    }
  `;
  const fragmentSource = `#version 300 es
    precision highp float;
    uniform sampler2D u_master;
    uniform sampler2D u_depth;
    uniform float u_time;
    uniform float u_strength;
    uniform float u_camera_scale;
    uniform float u_gamma;
    uniform vec2 u_camera_travel;
    in vec2 v_uv;
    out vec4 out_color;
    void main() {
      float phase = u_time * 0.00018;
      float cameraWave = 0.5 - 0.5 * cos(phase * 6.2831853);
      float scale = 1.0 + u_camera_scale * cameraWave;
      vec2 uv = (v_uv - 0.5) / scale + 0.5 - u_camera_travel * cameraWave;
      float depth = texture(u_depth, uv).r;
      vec2 drift = vec2(sin(phase * 6.2831853), cos(phase * 4.7123890));
      vec2 parallax = drift * (depth - 0.42) * u_strength;
      vec3 color = texture(u_master, clamp(uv + parallax, 0.002, 0.998)).rgb;
      float vignette = 1.0 - smoothstep(0.26, 0.88, length(v_uv - 0.5));
      float lanternBreath = 1.0 + 0.018 * sin(u_time * 0.0017);
      color = pow(max(color, vec3(0.0)), vec3(u_gamma));
      color *= mix(0.92, lanternBreath, vignette);
      out_color = vec4(color, 1.0);
    }
  `;
  const program = gl.createProgram();
  const vertex = compileShader(gl, gl.VERTEX_SHADER, vertexSource);
  const fragment = compileShader(gl, gl.FRAGMENT_SHADER, fragmentSource);
  gl.attachShader(program, vertex);
  gl.attachShader(program, fragment);
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    throw new Error(`Depth shader link failed: ${gl.getProgramInfoLog(program)}`);
  }
  gl.useProgram(program);
  const buffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]), gl.STATIC_DRAW);
  const position = gl.getAttribLocation(program, "a_position");
  gl.enableVertexAttribArray(position);
  gl.vertexAttribPointer(position, 2, gl.FLOAT, false, 0, 0);
  const masterTexture = createTexture(gl, masterImage, gl.TEXTURE0);
  const depthTexture = createTexture(gl, depthImage, gl.TEXTURE1);
  gl.uniform1i(gl.getUniformLocation(program, "u_master"), 0);
  gl.uniform1i(gl.getUniformLocation(program, "u_depth"), 1);
  const timeLocation = gl.getUniformLocation(program, "u_time");
  const strengthLocation = gl.getUniformLocation(program, "u_strength");
  const scaleLocation = gl.getUniformLocation(program, "u_camera_scale");
  const gammaLocation = gl.getUniformLocation(program, "u_gamma");
  const travelLocation = gl.getUniformLocation(program, "u_camera_travel");
  const camera = sceneSpec?.camera || {};
  const scaleDelta = Math.max(0, (camera.end_scale || 1.04) - (camera.start_scale || 1.01));
  gl.uniform1f(strengthLocation, 0.010);
  gl.uniform1f(scaleLocation, Math.min(0.08, scaleDelta));
  const projectionTone = projectionToneForImage(masterImage);
  gl.uniform1f(gammaLocation, projectionTone.gamma);
  gl.uniform2f(travelLocation, camera.travel_x || 0, -(camera.travel_y || 0));
  gl.viewport(0, 0, canvas.width, canvas.height);
  let animationFrame = null;
  let stopped = false;
  let renderedFrames = 0;
  let skippedFrames = 0;
  let lastRenderedAt = null;
  let lastTelemetryAt = 0;
  const frameIntervalMs = 1000 / DEPTH_RENDER_TARGET_FPS;
  canvas.dataset.depthTargetFps = String(DEPTH_RENDER_TARGET_FPS);
  const updateRenderTelemetry = (timestamp) => {
    if (timestamp - lastTelemetryAt < 500 && renderedFrames > 1) return;
    lastTelemetryAt = timestamp;
    canvas.dataset.depthRenderedFrames = String(renderedFrames);
    canvas.dataset.depthSkippedFrames = String(skippedFrames);
  };
  const release = () => {
    gl.deleteTexture(masterTexture);
    gl.deleteTexture(depthTexture);
    gl.deleteBuffer(buffer);
    gl.deleteProgram(program);
    gl.deleteShader(vertex);
    gl.deleteShader(fragment);
  };
  const revealFallback = () => {
    stopped = true;
    if (animationFrame !== null) cancelAnimationFrame(animationFrame);
    canvas.classList.remove("ready");
    setEvent("renderer.fallback", "WebGL context was lost; provider artwork remains visible");
  };
  canvas.addEventListener("webglcontextlost", revealFallback);
  const draw = (timestamp) => {
    if (stopped) return;
    gl.uniform1f(timeLocation, timestamp);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
    renderedFrames += 1;
    lastRenderedAt = timestamp;
    updateRenderTelemetry(timestamp);
  };
  const render = (timestamp) => {
    if (stopped) return;
    if (lastRenderedAt === null || timestamp - lastRenderedAt >= frameIntervalMs - 1) {
      draw(timestamp);
    } else {
      skippedFrames += 1;
      updateRenderTelemetry(timestamp);
    }
    animationFrame = requestAnimationFrame(render);
  };
  // Paint once before the opaque canvas is revealed. If Firefox accepts WebGL but
  // cannot execute the first draw on Jetson, the caller keeps the master image fallback.
  draw(performance.now());
  animationFrame = requestAnimationFrame(render);
  const firstDrawError = gl.getError();
  if (gl.isContextLost() || firstDrawError !== gl.NO_ERROR) {
    stopped = true;
    if (animationFrame !== null) cancelAnimationFrame(animationFrame);
    canvas.removeEventListener("webglcontextlost", revealFallback);
    release();
    throw new Error(`WebGL first frame failed (${firstDrawError}); using the still-image fallback`);
  }
  canvas.classList.add("ready");
  return {
    projectionExposure: projectionTone.fallbackExposure,
    projectionGamma: projectionTone.gamma,
    projectionMeanLuma: projectionTone.meanLuma,
    targetFps: DEPTH_RENDER_TARGET_FPS,
    destroy() {
      stopped = true;
      if (animationFrame !== null) cancelAnimationFrame(animationFrame);
      canvas.removeEventListener("webglcontextlost", revealFallback);
      release();
    },
  };
}

function appendAmbientEffects(container, sceneSpec) {
  const effects = (sceneSpec?.ambience || []).filter((effect) => effect.kind !== "none");
  if (!effects.length) return;
  const field = document.createElement("div");
  field.className = "ambient-field";
  effects.forEach((effect, effectIndex) => {
    const count = Math.max(4, Math.round(8 + effect.density * 28));
    for (let index = 0; index < count; index += 1) {
      const mote = document.createElement("i");
      mote.className = `ambient-particle ambient-${effect.kind}`;
      mote.style.setProperty("--ambient-color", effect.color);
      mote.style.setProperty("--ambient-x", `${(index * 37 + effectIndex * 19) % 101}%`);
      mote.style.setProperty("--ambient-y", `${(index * 61 + effectIndex * 23) % 101}%`);
      mote.style.setProperty("--ambient-delay", `${-((index * 0.71) % 8)}s`);
      mote.style.setProperty("--ambient-duration", `${Math.max(3, 12 - effect.speed * 4 + (index % 4))}s`);
      field.append(mote);
    }
  });
  container.append(field);
}

function appendSceneHotspots(container, page) {
  const compositionByLayer = new Map(
    (page.scene_spec?.composition || []).map((item) => [item.layer_id, item]),
  );
  [...page.layers].sort((left, right) => left.z_index - right.z_index).forEach((layer, index) => {
    const region = compositionByLayer.get(layer.layer_id);
    if (!region) return;
    const node = document.createElement("div");
    node.className = `visual-layer scene-hotspot layer-kind-${layer.kind}`;
    node.dataset.layerId = layer.layer_id;
    node.style.zIndex = String(10 + layer.z_index);
    node.style.setProperty("--hotspot-x", `${region.center_x * 100}%`);
    node.style.setProperty("--hotspot-y", `${region.center_y * 100}%`);
    node.style.setProperty("--hotspot-width", `${region.width * 100}%`);
    node.style.setProperty("--hotspot-height", `${region.height * 100}%`);
    node.style.setProperty("--hotspot-accent", layerPalette(index)[2]);
    container.append(node);
  });
}

function createSceneVersion() {
  const version = document.createElement("div");
  version.className = "scene-version incoming";
  version.dataset.stage = state.liveStage || "story-pack";
  return version;
}

function waitForVideoFrame(video, signal = null) {
  if (video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) return Promise.resolve();
  return new Promise((resolve, reject) => {
    let settled = false;
    const finish = (callback, value) => {
      if (settled) return;
      settled = true;
      window.clearTimeout(timeout);
      signal?.removeEventListener("abort", onAbort);
      callback(value);
    };
    const onAbort = () => {
      video.pause();
      video.removeAttribute("src");
      video.load();
      finish(reject, new Error("Motion asset load superseded"));
    };
    const timeout = window.setTimeout(
      () => finish(reject, new Error("Motion asset did not become playable")),
      8000,
    );
    video.addEventListener("loadeddata", () => finish(resolve), {once: true});
    video.addEventListener(
      "error",
      () => finish(reject, new Error("Motion asset failed to load")),
      {once: true},
    );
    if (signal?.aborted) {
      onAbort();
      return;
    }
    signal?.addEventListener("abort", onAbort, {once: true});
    video.load();
  });
}

function liveRenderTokenIsCurrent(token) {
  if (!token) return true;
  return token.epoch === state.liveRenderEpoch
    && token.serverInstanceId === state.liveServerInstanceId
    && token.sessionRevision === state.liveSessionRevision
    && token.jobId === state.liveSessionJobId
    && token.revision === state.liveAcceptedRevision
    && !token.signal.aborted;
}

function discardSceneVersion(version, renderer = null) {
  renderer?.destroy();
  version.querySelectorAll("video").forEach((video) => {
    video.pause();
    video.removeAttribute("src");
    video.load();
  });
  version.remove();
}

function commitSceneVersion(nextVersion, mode, nextRenderer = null, renderToken = null) {
  if (!liveRenderTokenIsCurrent(renderToken)) {
    discardSceneVersion(nextVersion, nextRenderer);
    return false;
  }
  const previousVersions = [...elements.generatedScene.querySelectorAll(":scope > .scene-version")];
  const rapidReplacement = previousVersions.some(
    (version) => version.classList.contains("incoming") || version.classList.contains("retiring"),
  );
  const previousRenderer = state.depthRenderer;
  state.depthRenderer = nextRenderer;
  elements.generatedScene.classList.remove(
    "depth-composed",
    "hero-composed",
    "motion-composed",
    "draft-composed",
    "preview-composed",
  );
  if (mode) elements.generatedScene.classList.add(mode);
  elements.generatedScene.prepend(nextVersion);
  elements.fixtureScene.hidden = true;
  previousVersions.forEach((version) => {
    version.classList.remove("current");
    version.classList.add("retiring");
    version.querySelectorAll("[data-layer-id]").forEach((layer) => {
      layer.dataset.retiredLayerId = layer.dataset.layerId;
      delete layer.dataset.layerId;
    });
  });
  let resolveFirstPaint;
  const firstPaint = new Promise((resolve) => {
    resolveFirstPaint = resolve;
  });
  state.lastSceneCommitPaint = {
    version: nextVersion,
    committedAt: performance.now(),
    promise: firstPaint,
  };
  // Flush the incoming state once, then transition on the next display frame.
  // The former nested rAF added a full refresh interval to every scene upgrade.
  void nextVersion.offsetWidth;
  requestAnimationFrame(() => {
    if (rapidReplacement) nextVersion.classList.add("rapid-replacement");
    nextVersion.classList.remove("incoming");
    nextVersion.classList.add("current");
    resolveFirstPaint(performance.now());
  });
  window.setTimeout(() => {
    previousVersions.forEach((version) => version.remove());
    previousRenderer?.destroy();
  }, rapidReplacement ? 50 : SCENE_RETIRE_GRACE_MS);
  // Acceptance telemetry belongs to the scene currently on screen, not to the
  // authoring tab's entire lifetime.
  resetFrameSampling({resetDropped: true});
  return true;
}

async function renderPackLayers(pack, page, renderToken = null, timings = null) {
  const readyAssets = (pack.assets || []).filter(
    (asset) => asset.page_id === page.page_id && asset.state === "ready" && asset.kind !== "procedural",
  );
  const motionAsset = readyAssets.find(
    (asset) => asset.role === "motion" && asset.kind === "video_loop",
  );
  const previewAsset = readyAssets.find(
    (asset) => asset.role === "preview" && asset.kind === "image",
  );
  if (previewAsset?.local_uri?.startsWith("/v1/assets/")) {
    const version = createSceneVersion();
    const scene = document.createElement("div");
    scene.className = "preview-scene";
    const mediaStartedAt = performance.now();
    const preview = await loadSceneImage(previewAsset.local_uri, {
      signal: renderToken?.signal,
    });
    if (timings) timings.mediaReadyMs = performance.now() - mediaStartedAt;
    preview.className = "preview-scene-image";
    preview.alt = `${page.scene_summary} provisional visual sketch`;
    scene.append(preview);
    appendAmbientEffects(scene, page.scene_spec);
    appendSceneHotspots(scene, page);
    version.append(scene);
    const commitStartedAt = performance.now();
    if (!commitSceneVersion(version, "preview-composed", null, renderToken)) return false;
    if (timings) timings.commitMs = performance.now() - commitStartedAt;
    return "preview-composed";
  }
  if (motionAsset?.local_uri?.startsWith("/v1/assets/")) {
    const version = createSceneVersion();
    const scene = document.createElement("div");
    scene.className = "motion-scene";
    const video = document.createElement("video");
    video.src = motionAsset.local_uri;
    video.muted = true;
    video.loop = true;
    video.autoplay = true;
    video.playsInline = true;
    video.setAttribute("aria-label", page.scene_summary);
    scene.append(video);
    appendAmbientEffects(scene, page.scene_spec);
    appendSceneHotspots(scene, page);
    version.append(scene);
    try {
      const mediaStartedAt = performance.now();
      await waitForVideoFrame(video, renderToken?.signal);
      if (timings) timings.mediaReadyMs = performance.now() - mediaStartedAt;
      const commitStartedAt = performance.now();
      if (!commitSceneVersion(version, "motion-composed", null, renderToken)) return false;
      if (timings) timings.commitMs = performance.now() - commitStartedAt;
      video.play().catch(() => setEvent("renderer.waiting", "Tap once to allow motion playback"));
      return "motion-composed";
    } catch (error) {
      if (!liveRenderTokenIsCurrent(renderToken)) {
        discardSceneVersion(version);
        return false;
      }
      discardSceneVersion(version);
      setEvent("renderer.fallback", `${error.message}; using provider artwork`);
    }
  }
  const masterAsset = readyAssets.find((asset) => asset.role === "master");
  const depthAsset = readyAssets.find((asset) => asset.role === "depth");
  if (
    masterAsset?.local_uri?.startsWith("/v1/assets/")
    && depthAsset?.local_uri?.startsWith("/v1/assets/")
  ) {
    const version = createSceneVersion();
    const scene = document.createElement("div");
    scene.className = "depth-scene";
    const canvas = document.createElement("canvas");
    canvas.className = "depth-scene-canvas";
    try {
      // Load and decode each provider asset exactly once. The decoded master image
      // doubles as the always-visible fallback and the WebGL source texture.
      const mediaStartedAt = performance.now();
      const [masterResult, depthResult] = await Promise.allSettled([
        loadSceneImage(masterAsset.local_uri, {signal: renderToken?.signal}),
        loadSceneImage(depthAsset.local_uri, {signal: renderToken?.signal}),
      ]);
      if (timings) timings.mediaReadyMs = performance.now() - mediaStartedAt;
      if (masterResult.status === "rejected") throw masterResult.reason;
      const fallback = masterResult.value;
      fallback.className = "depth-scene-fallback";
      fallback.alt = page.scene_summary;
      const renderSize = sizeDepthCanvasToSource(canvas, fallback);
      scene.append(fallback, canvas);
      appendAmbientEffects(scene, page.scene_spec);
      appendSceneHotspots(scene, page);
      version.append(scene);
      if (depthResult.status === "rejected") throw depthResult.reason;
      const depthImage = depthResult.value;
      const rendererStartedAt = performance.now();
      const renderer = await startDepthRenderer(
        canvas,
        fallback,
        depthImage,
        page.scene_spec,
      );
      if (timings) timings.rendererSetupMs = performance.now() - rendererStartedAt;
      scene.style.setProperty("--projection-exposure", renderer.projectionExposure.toFixed(3));
      scene.dataset.projectionGamma = renderer.projectionGamma.toFixed(3);
      if (renderer.projectionMeanLuma !== null) {
        scene.dataset.projectionMeanLuma = renderer.projectionMeanLuma.toFixed(3);
      }
      if (timings) {
        timings.projectionGamma = renderer.projectionGamma;
        timings.projectionMeanLuma = renderer.projectionMeanLuma;
        timings.renderWidth = renderSize.width;
        timings.renderHeight = renderSize.height;
        timings.renderPixels = renderSize.pixels;
        timings.logicalPixels = renderSize.logicalPixels;
      }
      const commitStartedAt = performance.now();
      if (!commitSceneVersion(version, "depth-composed", renderer, renderToken)) return false;
      if (timings) timings.commitMs = performance.now() - commitStartedAt;
      return "depth-composed";
    } catch (error) {
      if (!liveRenderTokenIsCurrent(renderToken)) {
        discardSceneVersion(version);
        return false;
      }
      setEvent("renderer.fallback", error.message);
      const masterImage = scene.querySelector(".depth-scene-fallback");
      if (!masterImage) {
        discardSceneVersion(version);
        throw error;
      }
      scene.style.setProperty(
        "--projection-exposure",
        projectionExposureForImage(masterImage).toFixed(3),
      );
      const commitStartedAt = performance.now();
      if (!commitSceneVersion(version, "depth-composed", null, renderToken)) return false;
      if (timings) timings.commitMs = performance.now() - commitStartedAt;
      return "master-fallback";
    }
  }

  const assets = new Map(
    readyAssets
      .filter((asset) => asset.role !== "depth")
      .map((asset) => [asset.layer_id, asset]),
  );
  const bundledHero = assets.size === 0 ? bundledHeroForPage(page) : null;
  const version = createSceneVersion();
  const draftTheme = passageDraftTheme(page);
  version.dataset.passageTheme = draftTheme.name;
  version.dataset.draftFocus = draftTheme.focusMotif;
  version.dataset.draftEffect = draftTheme.effectMotif;
  version.style.setProperty("--passage-theme-name", draftTheme.name);
  if (bundledHero) {
    const hero = document.createElement("div");
    hero.className = "visual-layer bundled-hero-scene";
    const video = document.createElement("video");
    video.src = bundledHero.video;
    if (bundledHero.poster) video.poster = bundledHero.poster;
    video.muted = true;
    video.loop = true;
    video.autoplay = true;
    video.playsInline = true;
    video.setAttribute("aria-label", bundledHero.label);
    hero.append(video);
    version.append(hero);
  }
  [...page.layers].sort((left, right) => left.z_index - right.z_index).forEach((layer, index) => {
    const node = document.createElement("div");
    const [start, end, accent] = draftTheme.palettes[
      (index + draftTheme.paletteOffset) % draftTheme.palettes.length
    ] || layerPalette(index);
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
      node.classList.add("has-asset");
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
    } else if (bundledHero) {
      node.classList.add("hero-trigger-layer");
    } else {
      const layout = sceneCompositionLayout(page.scene_spec, layer, index);
      node.classList.add("development-layer");
      node.style.setProperty("--placeholder-x", `${layout.x}%`);
      node.style.setProperty("--placeholder-y", `${layout.y}%`);
      node.style.setProperty("--placeholder-width", `${layout.width}%`);
      node.style.setProperty("--placeholder-height", `${layout.height}%`);
      const label = document.createElement("span");
      label.className = "layer-development-label";
      const kind = document.createElement("small");
      kind.textContent = `${layer.kind} · scene plan`;
      const prompt = document.createElement("strong");
      prompt.textContent = layer.prompt;
      label.append(kind, prompt);
      node.append(label);
    }
    version.append(node);
  });
  appendAmbientEffects(version, page.scene_spec);
  const mode = bundledHero ? "hero-composed" : "draft-composed";
  const commitStartedAt = performance.now();
  const committed = commitSceneVersion(
    version,
    mode,
    null,
    renderToken,
  );
  if (timings) timings.commitMs = performance.now() - commitStartedAt;
  return committed ? mode : false;
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
  const selector = `[data-layer-id="${CSS.escape(trigger.target_layer_id)}"]`;
  const target = elements.generatedScene.querySelector(selector) || elements.scene.querySelector(selector);
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
  const response = await localFetch(`/v1/reader-sessions/${encodeURIComponent(SESSION_ID)}`, {
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
  const response = await localFetch(`/v1/reader-sessions/${encodeURIComponent(SESSION_ID)}/transcripts:simulate`, {
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
  const response = await localFetch(`/v1/reader-sessions/${encodeURIComponent(SESSION_ID)}`, {
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
  state.readerConfiguredPageId = state.page.page_id;
  state.readerConfiguredPageText = state.page.source_text;
  return status;
}

async function resetReaderSession() {
  const response = await localFetch(`/v1/reader-sessions/${encodeURIComponent(SESSION_ID)}:reset`, {
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

function updatePageControls() {
  const total = state.pack?.pages.length || 0;
  elements.pageLabel.textContent = `Page ${state.pageIndex + 1} of ${total}`;
  elements.previousPage.disabled = state.pageIndex <= 0;
  elements.nextPage.disabled = state.pageIndex >= total - 1;
}

async function activatePage(nextIndex, renderToken = null) {
  if (!state.pack || nextIndex < 0 || nextIndex >= state.pack.pages.length) return false;
  if (!liveRenderTokenIsCurrent(renderToken)) return false;
  const nextPage = state.pack.pages[nextIndex];
  const readerSessionReusable = Boolean(
    state.generation !== null
    && state.readerConfiguredPageId === nextPage.page_id
    && state.readerConfiguredPageText === nextPage.source_text
  );
  state.pageIndex = nextIndex;
  state.page = nextPage;
  if (!readerSessionReusable) {
    state.pendingReaderEvents = [];
    state.cursor = -1;
  }
  state.tokens = tokenize(state.page.source_text);
  state.triggerIndices = indexTriggers(state.page);
  clearLayerState();
  renderTimeline();
  updatePageControls();
  const activationStartedAt = performance.now();
  const timings = {
    mediaReadyMs: 0,
    rendererSetupMs: 0,
    commitMs: 0,
    firstPaintMs: 0,
    readerSyncMs: 0,
    visualReadyMs: 0,
    totalMs: 0,
    readerSessionReused: readerSessionReusable,
  };
  const renderedMode = await renderPackLayers(state.pack, state.page, renderToken, timings);
  if (!renderedMode || !liveRenderTokenIsCurrent(renderToken)) return false;
  const committedPaint = state.lastSceneCommitPaint;
  const firstPaintAt = committedPaint?.version?.isConnected
    ? await committedPaint.promise
    : performance.now();
  if (!liveRenderTokenIsCurrent(renderToken)) return false;
  timings.visualReadyMs = performance.now() - activationStartedAt;
  timings.firstPaintMs = Math.max(0, firstPaintAt - (committedPaint?.committedAt || firstPaintAt));
  let readerCursor = state.cursor;
  if (readerSessionReusable) {
    rebuildScene();
  } else {
    const readerSyncStartedAt = performance.now();
    const session = await configureReaderSession();
    timings.readerSyncMs = performance.now() - readerSyncStartedAt;
    readerCursor = session.last_reached_index ?? -1;
  }
  timings.totalMs = performance.now() - activationStartedAt;
  if (!liveRenderTokenIsCurrent(renderToken)) return false;
  state.liveActivationBreakdown = timings;
  if (!readerSessionReusable) goToWord(readerCursor);
  const currentUrl = new URL(window.location.href);
  currentUrl.searchParams.set("page", String(nextIndex + 1));
  window.history.replaceState({}, "", currentUrl);
  publish("page.loaded", {
    pageId: state.page.page_id,
    pageIndex: nextIndex,
    pageCount: state.pack.pages.length,
    assetCount: state.pack.assets.length,
  });
  setEvent(
    "page.loaded",
    `${nextIndex + 1}/${state.pack.pages.length} · ${state.tokens.length} words`,
  );
  return renderedMode;
}

function requestPage(nextIndex) {
  if (state.pageTransition) return state.pageTransition;
  const transition = activatePage(nextIndex)
    .catch((error) => setEvent("page.error", error.message))
    .finally(() => {
      if (state.pageTransition === transition) state.pageTransition = null;
    });
  state.pageTransition = transition;
  return transition;
}

function liveProviderLabel(provider) {
  if (!provider) return "provider pending";
  if (typeof provider === "string") return provider;
  return [
    provider.name || provider.provider || provider.backend,
    provider.model || provider.model_id || provider.variant,
  ].filter(Boolean).join(" · ") || "generation provider";
}

function liveArtifactRoles(snapshot) {
  const roles = new Set();
  const artifacts = snapshot?.artifacts;
  if (Array.isArray(artifacts)) {
    artifacts.forEach((artifact) => roles.add(artifact.role || artifact.kind));
  } else if (artifacts && typeof artifacts === "object") {
    Object.entries(artifacts).forEach(([role, artifact]) => {
      if (artifact) roles.add(role);
    });
  }
  (snapshot?.story_pack?.assets || []).forEach((asset) => {
    if (asset.state === "ready") roles.add(asset.role);
  });
  if (snapshot?.story_pack) roles.add("draft");
  return [...roles].filter(Boolean);
}

function updateLiveGenerationClock() {
  if (!state.liveJobId) return;
  const continued = state.liveTerminal ? 0 : Math.max(0, performance.now() - state.liveUpdatedAt);
  elements.liveGenerationElapsed.textContent = `${((state.liveElapsedMs + continued) / 1000).toFixed(1)} s`;
}

function renderLiveGenerationBadge(snapshot, {activated = true, fallbackMode = null} = {}) {
  const readyStageLabels = {
    queued: "Generation job queued",
    planning: "Planning visual world",
    draft_ready: "Animated draft live",
    preview_ready: "Generated visual sketch live",
    master_ready: "Artwork + depth live",
    motion_ready: "Motion loop live",
    failed: "Generation failed",
  };
  const pendingStageLabels = {
    draft_ready: "Preparing animated draft",
    preview_ready: "Loading generated visual sketch",
    master_ready: "Loading artwork + depth",
    motion_ready: "Loading motion loop",
  };
  state.liveStage = snapshot.stage || "queued";
  const snapshotTerminal = state.liveStage === "motion_ready"
    || state.liveStage === "failed"
    || snapshot.complete === true
    || snapshot.terminal === true;
  state.liveTerminal = snapshotTerminal && activated;
  const timestampElapsed = Date.parse(snapshot.updated_at || "") - Date.parse(snapshot.created_at || "");
  state.liveElapsedMs = Number.isFinite(snapshot.metrics?.elapsed_ms)
    ? snapshot.metrics.elapsed_ms
    : Number.isFinite(snapshot.elapsed_ms)
      ? snapshot.elapsed_ms
    : Number.isFinite(snapshot.latency_ms)
      ? snapshot.latency_ms
      : Number.isFinite(timestampElapsed) ? Math.max(0, timestampElapsed) : state.liveElapsedMs;
  state.liveUpdatedAt = performance.now();
  elements.liveGenerationBadge.classList.remove("hidden");
  elements.liveGenerationBadge.dataset.stage = state.liveStage;
  elements.liveGenerationBadge.dataset.terminal = String(state.liveTerminal);
  elements.liveGenerationStage.textContent = (
    activated ? readyStageLabels[state.liveStage] : pendingStageLabels[state.liveStage]
  ) || readyStageLabels[state.liveStage] || state.liveStage.replaceAll("_", " ");
  const roles = liveArtifactRoles(snapshot);
  const provenance = liveProviderLabel(snapshot.provider);
  const warning = snapshot.warning?.message || snapshot.warning;
  const detail = roles.length
    ? `${provenance} · ${roles.join(" + ")}`
    : provenance;
  const hardware = snapshot.metrics?.gpu
    ? `${snapshot.metrics.warm_state || "unknown"} · ${snapshot.metrics.gpu}`
    : null;
  const fallback = fallbackMode === "master-fallback"
    ? "artwork fallback · depth retrying"
    : fallbackMode === "depth-composed"
      ? "depth fallback · video retrying"
      : !activated && snapshot.story_pack ? "media retrying" : null;
  const evidence = [
    detail,
    snapshot.metrics?.scene_cache_hit ? "verified local replay · no new GPU work" : null,
    hardware,
    Number.isFinite(state.liveActivationMs)
      ? `client activate ${Math.round(state.liveActivationMs)} ms`
      : null,
    Number.isFinite(state.liveActivationMs) ? `visual blend ${SCENE_CROSSFADE_MS} ms` : null,
    fallback,
    warning ? "motion skipped" : null,
  ].filter(Boolean).join(" · ");
  elements.liveGenerationDetail.textContent = evidence;
  if (Number.isFinite(state.liveActivationMs)) {
    elements.liveGenerationBadge.dataset.activationMs = state.liveActivationMs.toFixed(3);
    const breakdown = state.liveActivationBreakdown;
    if (breakdown) {
      elements.liveGenerationBadge.dataset.mediaReadyMs = breakdown.mediaReadyMs.toFixed(3);
      elements.liveGenerationBadge.dataset.rendererSetupMs = breakdown.rendererSetupMs.toFixed(3);
      elements.liveGenerationBadge.dataset.firstPaintMs = breakdown.firstPaintMs.toFixed(3);
      elements.liveGenerationBadge.dataset.readerSyncMs = breakdown.readerSyncMs.toFixed(3);
      elements.liveGenerationBadge.dataset.readerSessionReused = String(
        Boolean(breakdown.readerSessionReused)
      );
    } else {
      delete elements.liveGenerationBadge.dataset.mediaReadyMs;
      delete elements.liveGenerationBadge.dataset.rendererSetupMs;
      delete elements.liveGenerationBadge.dataset.firstPaintMs;
      delete elements.liveGenerationBadge.dataset.readerSyncMs;
      delete elements.liveGenerationBadge.dataset.readerSessionReused;
    }
  } else {
    delete elements.liveGenerationBadge.dataset.activationMs;
    delete elements.liveGenerationBadge.dataset.mediaReadyMs;
    delete elements.liveGenerationBadge.dataset.rendererSetupMs;
    delete elements.liveGenerationBadge.dataset.firstPaintMs;
    delete elements.liveGenerationBadge.dataset.readerSyncMs;
    delete elements.liveGenerationBadge.dataset.readerSessionReused;
  }
  elements.liveGenerationBadge.classList.toggle("has-warning", Boolean(warning));
  updateLiveGenerationClock();
}

function livePageAssetFingerprint(pack, page) {
  const assets = (pack.assets || [])
    .filter((asset) => asset.page_id === page.page_id && asset.state === "ready")
    .map((asset) => [
      asset.asset_id,
      asset.role,
      asset.checksum_sha256,
      asset.local_uri,
    ].join(":"))
    .sort();
  return `${page.page_id}|${assets.join("|")}`;
}

function invalidateLiveRender({jobId, revision, serverInstanceId, sessionRevision}) {
  state.liveRenderAbortController?.abort();
  state.liveRenderAbortController = new AbortController();
  state.liveRenderEpoch += 1;
  state.liveAcceptedJobId = jobId;
  state.liveAcceptedRevision = revision;
  state.liveRenderPending = true;
  return {
    epoch: state.liveRenderEpoch,
    serverInstanceId,
    sessionRevision,
    jobId,
    revision,
    signal: state.liveRenderAbortController.signal,
  };
}

function liveModeSatisfiesStage(stage, mode) {
  if (stage === "motion_ready") return mode === "motion-composed";
  if (stage === "master_ready") return mode === "depth-composed";
  if (stage === "preview_ready") return mode === "preview-composed";
  if (stage === "draft_ready") return mode === "draft-composed" || mode === "hero-composed";
  return true;
}

function queueLiveSceneSnapshot(envelope) {
  if (!LIVE_MODE || envelope?.type !== "bookforge.live-scene") return;
  if (envelope.sessionId && envelope.sessionId !== SESSION_ID) return;
  const snapshot = envelope.snapshot;
  if (!snapshot || typeof snapshot !== "object") return;
  const jobId = snapshot.job_id || state.liveJobId || "live-scene";
  const serverInstanceId = envelope.serverInstanceId || null;
  const sessionRevision = Number(envelope.sessionRevision || 0);
  if (!state.liveServerInstanceId) {
    // The envelope is only a wake-up hint. Fetch the authoritative server epoch
    // immediately so the first scene does not wait for the one-second poll.
    void rendezvousLiveScene();
    return;
  }
  // Only the server-issued epoch is authoritative. This also ignores legacy
  // localStorage envelopes whose process-local revision could pin a kiosk after restart.
  if (!serverInstanceId || !Number.isInteger(sessionRevision) || sessionRevision < 1) return;
  if (state.liveServerInstanceId && state.liveServerInstanceId !== serverInstanceId) return;
  if (!state.liveServerInstanceId) return;
  if (sessionRevision < state.liveSessionRevision) return;
  if (
    sessionRevision === state.liveSessionRevision
    && state.liveSessionJobId
    && state.liveSessionJobId !== jobId
  ) return;
  if (sessionRevision > state.liveSessionRevision) return;
  if (state.liveSessionJobId !== jobId) return;

  const revision = Number.isFinite(Number(snapshot.revision)) ? Number(snapshot.revision) : 0;
  if (state.liveAcceptedJobId === jobId && revision < state.liveAcceptedRevision) return;
  if (state.liveAcceptedJobId !== jobId) {
    state.liveAssetFingerprint = null;
    state.liveCommittedJobId = null;
    state.liveCommittedRevision = -1;
    state.liveRevision = -1;
    state.liveElapsedMs = 0;
    state.liveActivationMs = null;
    state.liveActivationBreakdown = null;
    state.liveStartedAt = performance.now();
  } else if (revision === state.liveAcceptedRevision) {
    if (state.liveRenderPending) return;
    if (!snapshot.story_pack && snapshot.stage === state.liveStage) return;
    if (
      state.liveCommittedJobId === jobId
      && state.liveCommittedRevision === revision
    ) return;
  }
  const renderToken = invalidateLiveRender({
    jobId,
    revision,
    serverInstanceId,
    sessionRevision,
  });
  state.liveTransition = state.liveTransition.then(async () => {
    if (!liveRenderTokenIsCurrent(renderToken)) return;
    if (state.liveJobId !== jobId) {
      const sentAt = Number(envelope.sentAt || 0);
      if (state.liveJobId && sentAt && sentAt < state.liveLastEnvelopeAt) return;
      state.liveJobId = jobId;
    }
    if (revision < state.liveRevision || !liveRenderTokenIsCurrent(renderToken)) return;
    state.liveLastEnvelopeAt = Math.max(state.liveLastEnvelopeAt, Number(envelope.sentAt || 0));
    state.liveRevision = revision;
    if (!snapshot.story_pack) {
      state.liveRenderPending = false;
      renderLiveGenerationBadge(snapshot);
      setEvent("scene.generating", `${snapshot.stage || "queued"} · revision ${revision}`);
      return;
    }
    renderLiveGenerationBadge(snapshot, {activated: false});
    const pack = assertStoryPack(snapshot.story_pack);
    const currentPageId = state.page?.page_id;
    state.pack = pack;
    const nextIndex = Math.max(0, pack.pages.findIndex((page) => page.page_id === currentPageId));
    elements.packLabel.textContent = `${pack.title} · live ${snapshot.stage} · r${revision}`;
    const nextPage = pack.pages[nextIndex];
    const fingerprint = livePageAssetFingerprint(pack, nextPage);
    if (state.liveAssetFingerprint === fingerprint) {
      state.page = nextPage;
      state.liveActivationMs = 0;
      state.liveActivationBreakdown = null;
      state.liveRenderPending = false;
      state.liveCommittedJobId = jobId;
      state.liveCommittedRevision = revision;
      renderLiveGenerationBadge(snapshot);
      setEvent("scene.status-updated", `${snapshot.stage} · revision ${revision} · media unchanged`);
      return;
    }
    const activationStartedAt = performance.now();
    const renderedMode = await activatePage(nextIndex, renderToken);
    if (!renderedMode || !liveRenderTokenIsCurrent(renderToken)) return;
    state.liveActivationMs = performance.now() - activationStartedAt;
    state.liveRenderPending = false;
    if (!liveModeSatisfiesStage(snapshot.stage, renderedMode)) {
      renderLiveGenerationBadge(snapshot, {activated: false, fallbackMode: renderedMode});
      setEvent("scene.retrying", `${snapshot.stage} · revision ${revision} · ${renderedMode}`);
      return;
    }
    state.liveAssetFingerprint = fingerprint;
    state.liveCommittedJobId = jobId;
    state.liveCommittedRevision = revision;
    renderLiveGenerationBadge(snapshot);
    publish("scene.activated", {
      jobId,
      revision,
      stage: snapshot.stage,
      renderedMode,
      activationMs: state.liveActivationMs,
      crossfadeMs: SCENE_CROSSFADE_MS,
      activationBreakdown: state.liveActivationBreakdown,
    });
    setEvent("scene.upgraded", `${snapshot.stage} · revision ${revision} · no reload`);
  }).catch((error) => {
    if (renderToken.signal.aborted) return;
    if (renderToken.epoch === state.liveRenderEpoch) {
      state.liveRenderPending = false;
      renderLiveGenerationBadge(snapshot, {activated: false});
    }
    setEvent("scene.upgrade-error", error.message);
  });
}

function liveSnapshotIsTerminal(snapshot) {
  return snapshot?.complete === true
    || snapshot?.terminal === true
    || snapshot?.stage === "motion_ready"
    || snapshot?.stage === "failed";
}

function acceptLiveSceneSessionPointer(payload) {
  const serverInstanceId = payload?.server_instance_id;
  const sessionRevision = Number(payload?.session_revision);
  const snapshot = payload?.job || null;
  const jobId = snapshot?.job_id || null;
  if (
    payload?.session_id !== SESSION_ID
    || typeof serverInstanceId !== "string"
    || !serverInstanceId
    || !Number.isInteger(sessionRevision)
    || sessionRevision < 0
    || (snapshot && (!jobId || sessionRevision < 1))
    || (!snapshot && sessionRevision !== 0)
  ) throw new Error("rendezvous returned an invalid session pointer");

  if (!state.liveServerInstanceId || state.liveServerInstanceId !== serverInstanceId) {
    state.liveRenderAbortController?.abort();
    state.liveRenderAbortController = null;
    state.liveServerInstanceId = serverInstanceId;
    state.liveSessionRevision = 0;
    state.liveSessionJobId = null;
    state.liveAcceptedJobId = null;
    state.liveAcceptedRevision = -1;
    state.liveCommittedJobId = null;
    state.liveCommittedRevision = -1;
    state.liveRenderPending = false;
    state.liveAssetFingerprint = null;
    state.generation = null;
    state.readerConfiguredPageId = null;
    state.readerConfiguredPageText = null;
    state.liveRenderEpoch += 1;
  }
  if (!snapshot) return true;
  if (sessionRevision < state.liveSessionRevision) return false;
  if (
    sessionRevision === state.liveSessionRevision
    && state.liveSessionJobId
    && state.liveSessionJobId !== jobId
  ) return false;

  state.liveSessionRevision = sessionRevision;
  state.liveSessionJobId = jobId;
  queueLiveSceneSnapshot({
    type: "bookforge.live-scene",
    sessionId: SESSION_ID,
    serverInstanceId,
    sessionRevision,
    sentAt: Date.now(),
    snapshot,
  });
  return true;
}

function connectLiveSceneSessionEvents() {
  state.liveSessionEventSource?.close();
  state.liveSessionStreamHealthy = false;
  const source = new EventSource(
    `/v1/live-scene-sessions/${encodeURIComponent(SESSION_ID)}/events`,
  );
  state.liveSessionEventSource = source;
  const receive = (event) => {
    try {
      acceptLiveSceneSessionPointer(JSON.parse(event.data));
      if (source === state.liveSessionEventSource) {
        state.liveSessionStreamHealthy = true;
        stopLiveSceneRendezvous();
      }
    } catch (error) {
      if (source === state.liveSessionEventSource) {
        state.liveSessionStreamHealthy = false;
        scheduleLiveSceneRendezvous(0);
      }
      setEvent("scene.session-stream-error", error.message);
    }
  };
  source.addEventListener("scene.session", receive);
  source.addEventListener("message", receive);
  source.addEventListener("error", () => {
    if (source !== state.liveSessionEventSource) return;
    state.liveSessionStreamHealthy = false;
    scheduleLiveSceneRendezvous(0);
    setEvent("scene.session-reconnecting", "Server session stream reconnecting; polling is active");
  });
}

function liveSessionStreamIsHealthy() {
  const source = state.liveSessionEventSource;
  return Boolean(
    state.liveSessionStreamHealthy
    && source
    && source.readyState === EventSource.OPEN
  );
}

function stopLiveSceneRendezvous() {
  window.clearTimeout(state.liveRendezvousTimer);
  state.liveRendezvousTimer = null;
}

function scheduleLiveSceneRendezvous(delayMs = 1000) {
  stopLiveSceneRendezvous();
  if (!LIVE_MODE || liveSessionStreamIsHealthy()) return;
  state.liveRendezvousTimer = window.setTimeout(rendezvousLiveScene, delayMs);
}

async function rendezvousLiveScene() {
  if (!LIVE_MODE || state.liveRendezvousInFlight || liveSessionStreamIsHealthy()) return;
  state.liveRendezvousInFlight = true;
  try {
    const response = await localFetch(
      `/v1/live-scene-sessions/${encodeURIComponent(SESSION_ID)}`,
      {cache: "no-store"},
    );
    if (response.status === 404) return;
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.detail || `rendezvous failed (${response.status})`);
    acceptLiveSceneSessionPointer(payload);
  } catch (error) {
    setEvent("scene.rendezvous-error", error.message);
  } finally {
    state.liveRendezvousInFlight = false;
    scheduleLiveSceneRendezvous();
  }
}

function setupLiveSceneTransport() {
  if (!LIVE_MODE) return;
  connectLiveSceneSessionEvents();
  if (!state.liveServerInstanceId) rendezvousLiveScene();
  window.setInterval(updateLiveGenerationClock, 100);
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
  state.frameSamples.push(delta);
  if (state.frameSamples.length > 60) state.frameSamples.shift();
  const sampling = timestamp - monitorFrames.startedAt > 1500 && document.visibilityState === "visible";
  if (sampling && !monitorFrames.baselineMs) {
    const calibration = state.frameSamples
      .filter((sample) => sample > 2 && sample < 100)
      .sort((left, right) => left - right);
    monitorFrames.baselineMs = calibration[Math.floor(calibration.length / 2)] || 16.67;
  }
  const baseline = monitorFrames.baselineMs || 16.67;
  if (sampling && delta > baseline * 1.5) {
    state.droppedFrames += Math.max(1, Math.round(delta / baseline) - 1);
  }
  const average = state.frameSamples.reduce((sum, value) => sum + value, 0) / state.frameSamples.length;
  elements.frameRate.textContent = average > 0 ? `${Math.min(240, 1000 / average).toFixed(0)} fps` : "—";
  elements.droppedFrames.textContent = String(state.droppedFrames);
  requestAnimationFrame(monitorFrames);
}

function resetFrameSampling({resetDropped = false} = {}) {
  // requestAnimationFrame pauses in a background tab. Do not count that pause
  // as millions of dropped display frames when the projector becomes visible.
  monitorFrames.previous = null;
  monitorFrames.startedAt = null;
  monitorFrames.baselineMs = null;
  state.frameSamples = [];
  if (resetDropped) {
    state.droppedFrames = 0;
    elements.droppedFrames.textContent = "0";
  }
}

async function loadStoryPack() {
  try {
    if (PACK_SOURCE === "latest") {
      const response = await localFetch("/v1/story-packs/latest", {cache: "no-store"});
      if (response.ok) {
        state.pack = assertStoryPack(await response.json());
      } else {
        const saved = localStorage.getItem("bookforge.latestStoryPack");
        if (saved) {
          state.pack = assertStoryPack(JSON.parse(saved));
        } else {
          const fixture = await localFetch("/workbench-assets/moon-gate.story-pack.json", {cache: "no-store"});
          if (!fixture.ok) throw new Error(`Fallback Story Pack failed to load (${fixture.status})`);
          state.pack = assertStoryPack(await fixture.json());
        }
      }
    } else {
      const response = await localFetch("/workbench-assets/moon-gate.story-pack.json", {cache: "no-store"});
      if (!response.ok) throw new Error(`Story Pack failed to load (${response.status})`);
      state.pack = assertStoryPack(await response.json());
    }
    const replayLabel = OFFLINE_REPLAY ? " · offline cache" : "";
    elements.packLabel.textContent = `${state.pack.title} · ${state.pack.schema_version}${replayLabel}`;
    const requestedPage = Number.parseInt(query.get("page") || "1", 10) - 1;
    const initialPage = Number.isInteger(requestedPage)
      ? Math.max(0, Math.min(state.pack.pages.length - 1, requestedPage))
      : 0;
    await activatePage(initialPage);
  } catch (error) {
    elements.startupError.textContent = `Bookforge could not start: ${error.message}`;
    elements.startupError.classList.remove("hidden");
  }
}

async function restoreAuthoritativeLiveSession() {
  if (!LIVE_MODE) return false;
  try {
    const response = await localFetch(
      `/v1/live-scene-sessions/${encodeURIComponent(SESSION_ID)}`,
      {cache: "no-store"},
    );
    if (response.status === 404) return false;
    const pointer = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(pointer.detail || `session restore failed (${response.status})`);
    }
    if (!pointer.job?.story_pack) return false;
    acceptLiveSceneSessionPointer(pointer);
    await state.liveTransition;
    return state.liveCommittedJobId === pointer.job.job_id;
  } catch (error) {
    setEvent("scene.session-restore-error", error.message);
    return false;
  }
}

async function startProjector() {
  try {
    // Restore an already-generated session directly. If it has no usable pack,
    // activate the device fallback before SSE is allowed to replace state.pack.
    // The stream replays its current pointer, so a job created during startup cannot be missed.
    if (!await restoreAuthoritativeLiveSession()) await loadStoryPack();
  } finally {
    setupLiveSceneTransport();
    if (state.page) connectReaderSession();
  }
}

elements.previous.addEventListener("click", () => goToWord(state.cursor - 1));
elements.next.addEventListener("click", () => goToWord(state.cursor + 1));
elements.previousPage.addEventListener("click", () => requestPage(state.pageIndex - 1));
elements.nextPage.addEventListener("click", () => requestPage(state.pageIndex + 1));
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
  } else if (event.key === "[") {
    event.preventDefault();
    requestPage(state.pageIndex - 1);
  } else if (event.key === "]") {
    event.preventDefault();
    requestPage(state.pageIndex + 1);
  } else if (key === "r") elements.reset.click();
  else if (key === "f") elements.fullscreen.click();
  else if (key === "c") toggleCalibration();
  else if (key === "b") document.body.classList.toggle("blackout");
  else if (key === "h") document.body.classList.toggle("hud-hidden");
});

window.addEventListener("resize", updateProjection);
document.addEventListener("visibilitychange", resetFrameSampling);
window.addEventListener("beforeunload", () => {
  clearTimeout(state.reconnectTimer);
  stopLiveSceneRendezvous();
  state.liveSessionEventSource?.close();
  state.liveSessionStreamHealthy = false;
  state.depthRenderer?.destroy();
  state.socket?.close();
  void state.screenWakeLock?.release();
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
setupProjectorWakeLock();
void startProjector();
requestAnimationFrame(monitorFrames);
