const assert = require("assert").strict;
const fs = require("fs");
const vm = require("vm");

const source = fs.readFileSync("src/bookforge/static/workbench.js", "utf8");
for (const [query, expected] of [
  ["?voice=1", "voice-demo"], ["?demo=1", "bookforge-live"],
  ["?voice=1&session=my-projection", "my-projection"],
]) {
  const context = {URLSearchParams, window: {location: {search: query}}, document: {body: {dataset: {}}}};
  vm.createContext(context);
  const configuration = source.slice(source.indexOf("const workbenchQuery ="), source.indexOf("const restoreLatestScene ="));
  assert.equal(vm.runInContext(`${configuration}\nreaderSessionId`, context), expected);
}
function harness() {
  const timers = new Map();
  const requests = [];
  const events = [];
  const element = () => ({classList: {add() {}, remove() {}, toggle() {}},
    setAttribute() {}, style: {}, textContent: ""});
  const elements = Object.fromEntries([
    "micButton", "micButtonText", "micLevel", "compileButton", "interim",
    "story", "style", "projectorLink", "projectorFrame", "projectionPreview", "voiceReview",
  ].map((key) => [key, element()]));
  elements.story.value = "A fox carries a lantern.";
  class Recorder {
    static isTypeSupported() { return true; }
    constructor() { this.handlers = {}; this.mimeType = "audio/webm"; }
    addEventListener(name, callback) { this.handlers[name] = callback; }
    removeEventListener(name) { delete this.handlers[name]; }
    start() { this.state = "recording"; }
    stop() { this.state = "inactive"; events.push("recorder-stop"); }
  }
  const context = {
    elements, AbortController, Blob, Uint8Array, MediaRecorder: Recorder,
    canRecordAudio: true, sceneReady: true, demoMode: false, voiceMode: false,
    voiceProjectionLive: false, projectorPreviewUrl: null,
    generationSubmitting: false, pendingSubmission: null, pendingVoiceReview: null, generationReconciling: false,
    activeLiveJobId: null, latestLiveSnapshot: null,
    listening: false, starting: false, finalizing: false, stream: null,
    analyser: null, audioContext: null, mediaRecorder: null, audioChunks: [],
    partialTimer: null, partialBusy: false, partialInFlight: Promise.resolve(),
    partialBytes: 0, recordingEpoch: 0, readerGeneration: null, activePageText: null,
    readerSessionId: "reader/example", requestAnimationFrame() {},
    navigator: {mediaDevices: {async getUserMedia() {
      events.push("permission");
      return {getTracks: () => [{stop: () => events.push("track-stop")}]};
    }}},
    window: {
      MediaRecorder: Recorder,
      AudioContext: class {
        createAnalyser() { return {}; }
        createMediaStreamSource() { return {connect() {}}; }
        close() { events.push("context-close"); }
      },
      setInterval() { return 1; }, clearInterval() {},
      setTimeout(callback, milliseconds) { const id = Symbol(); timers.set(id, {callback, milliseconds}); return id; },
      clearTimeout(id) { timers.delete(id); },
    },
    async fetch(url, options) {
      requests.push(url);
      return context.respond(url, options);
    },
    respond: async () => ({ok: true, json: async () => ({asr: {ready: true}, generation: 1, text: "A fox", total_ms: 10})}),
  };
  vm.createContext(context);
  for (const [start, end] of [
    ["function updateMicAvailability(", "function safeText("],
    ["async function startAudioMeter(", "function liveRevision("],
    ["function setSceneInputsDisabled(", "function finishLiveJob("],
  ]) vm.runInContext(source.slice(source.indexOf(start), source.indexOf(end)), context);
  return {context, elements, timers, requests, events};
}

async function lifecycle() {
  const {context: c, elements, timers, requests, events} = harness();
  let respondStatus;
  c.respond = () => new Promise((resolve) => { respondStatus = resolve; });
  const starting = c.startSpeaking();
  c.setSceneReady(true);
  assert.equal(elements.micButton.disabled, true);
  assert.equal(elements.story.disabled, true);
  assert.deepEqual(events, []);
  respondStatus({ok: true, json: async () => ({asr: {ready: false}})});
  await starting;
  assert.equal(events.includes("permission"), false);
  assert.match(elements.interim.textContent, /disabled or unavailable/);
  assert.equal(elements.story.disabled, false);

  c.respond = async () => ({ok: true, json: async () => ({asr: {ready: true}, generation: 1})});
  await c.startSpeaking();
  assert.equal(c.listening, true);
  const recorder = c.mediaRecorder;
  recorder.handlers.dataavailable({data: new Blob(["a".repeat(1200)])});
  c.stopSpeaking();
  assert.deepEqual(events, ["permission", "recorder-stop", "track-stop", "context-close"]);
  assert.equal(c.stream, null);
  c.setSceneReady(true);
  assert.equal(elements.micButton.disabled, true);
  assert.equal(elements.interim.textContent, "Finishing the local transcript…");
  const before = requests.length;
  await c.startSpeaking();
  assert.equal(requests.length, before);
  recorder.handlers.dataavailable({data: new Blob(["b".repeat(100)])});

  c.respond = (_url, options) => new Promise((_resolve, reject) => {
    assert.equal(options.body.size, 1300); // Include the recorder's final chunk after Stop.
    options.signal.addEventListener("abort", () => reject(new Error("aborted")));
  });
  const finishing = recorder.handlers.stop();
  await new Promise(setImmediate);
  assert.equal(timers.size, 1);
  assert.equal([...timers.values()][0].milliseconds, 45000);
  [...timers.values()][0].callback();
  await finishing;
  assert.equal(c.finalizing, false);
  assert.equal(c.mediaRecorder, null);
  assert.equal(elements.micButton.disabled, false);
  assert.equal(elements.story.disabled, false);
  assert.match(elements.interim.textContent, /timed out/);
  assert.equal(timers.size, 0);

  c.demoMode = true;
  c.setSceneInputsDisabled(false);
  c.ensureProjectionPreview();
  const preview = new URL(elements.projectorFrame.src, "http://localhost");
  assert.equal(preview.searchParams.get("reader"), "1");
  assert.equal(preview.searchParams.get("session"), c.readerSessionId);
  assert.equal(preview.searchParams.has("live"), false);
  assert.equal(elements.story.disabled, true);
  assert.equal(elements.style.disabled, true);
  c.demoMode = false;
  assert.match(c.projectorUrl(), /&live=1$/);

  const permission = harness();
  let grant;
  permission.context.navigator.mediaDevices.getUserMedia = () => new Promise((resolve) => { grant = resolve; });
  const pending = permission.context.startSpeaking();
  await new Promise(setImmediate);
  const timer = [...permission.timers.values()][0];
  assert.equal(timer.milliseconds, 20000);
  timer.callback();
  await pending;
  assert.equal(permission.context.starting, false);
  assert.equal(permission.elements.micButton.disabled, false);
  assert.match(permission.elements.interim.textContent, /permission timed out/);
  let stopped = 0;
  grant({getTracks: () => [{stop() { stopped += 1; }}]});
  await new Promise(setImmediate);
  assert.equal(stopped, 1);
  assert.equal(permission.context.stream, null);
}

async function voiceToScene() {
  const {context: c, elements, requests, events, timers} = harness();
  c.voiceMode = true;
  c.sceneReady = false;
  elements.story.value = "";
  elements.style.value = "rich watercolor";
  elements.compileButton.dataset = {};
  elements.error = {classList: {add() {}, remove() {}}};
  const submitted = [];
  let finishAsr;
  let finishGeneration;
  let rejectPreflight = false;
  let requiresFactReview = false;
  let parsedAction = "floating";
  const preflight = () => ({ok: !rejectPreflight, status: rejectPreflight ? 422 : 200,
    json: async () => rejectPreflight
      ? {detail: {code: "reviewed_description_unsupported", message: "Clarify the action."}}
      : {revision: "reviewed-language-v1", requires_fact_review: requiresFactReview,
        visual_fact_digest: "a".repeat(64),
        local_omissions: requiresFactReview ? [{local_text: "London"}] : [],
        visual_facts: {subjects: [{ref: "whale", label: "whale", attributes: [], actions: [parsedAction]}],
          objects: [], relationships: [], negatives: [], setting: {label: "unspecified"}}},
  });
  Object.assign(c, {
    edgePlanPreparationTimer: null, liveRequestEpoch: 0,
    performance: {now: () => 0}, invalidatePreparation() {},
    stopLiveJobTransport() { events.push("transport-stop"); }, renderGenerationProgress() {}, updateElapsedClock() {}, setStatus() {},
    setGenerateButtonForStage() {}, isTerminalSnapshot: () => true,
    acceptedLiveScenePointer: (_response, snapshot) => snapshot,
    handleLiveSceneSessionPointer(snapshot) { c.activeLiveJobId = snapshot.job_id; },
    fetchLiveSceneSession: async () => null,
    respond(url, options) {
      if (url === "/v1/runtime:status") return {ok: true, json: async () => ({asr: {ready: true}})};
      if (url === "/v1/audio:transcribe") return new Promise((resolve) => { finishAsr = resolve; });
      if (url === "/v1/live-scene-planner/prepare") return preflight();
      assert.equal(url, "/v1/live-scenes");
      submitted.push(JSON.parse(options.body));
      return new Promise((resolve) => { finishGeneration = resolve; });
    },
  });
  vm.runInContext(source.slice(source.indexOf("async function reconcileGeneration("), source.indexOf("async function loadLatestScene(")), c);
  c.ensureProjectionPreview();
  const previousPreview = elements.projectorFrame.src;
  assert.match(previousPreview, /reader=0$/);
  await c.startSpeaking();
  assert.equal(c.partialTimer, null);
  assert.deepEqual(requests, ["/v1/runtime:status"]);
  assert.equal(elements.micButtonText.textContent, "Finish recording");
  const recorder = c.mediaRecorder;
  recorder.handlers.dataavailable({data: new Blob(["a".repeat(1200)])});
  c.stopSpeaking();
  const final = recorder.handlers.stop();
  c.stopSpeaking();
  await c.startSpeaking();
  await new Promise(setImmediate);
  assert.equal(requests.filter((url) => url === "/v1/audio:transcribe").length, 1);
  assert.equal(events.filter((event) => event === "track-stop").length, 1);
  finishAsr({ok: true, json: async () => ({text: "  A blue whale above a forest.  "})});
  await final;
  assert.equal(elements.story.value, "A blue whale above a forest.");
  assert.equal(elements.projectorFrame.src, previousPreview);
  assert.equal(submitted.length, 0);
  assert.equal(c.finalizing, false);
  assert.equal(elements.story.disabled, false);
  assert.equal(elements.compileButton.disabled, false);
  assert.equal(elements.compileButton.textContent, "Generate scene");
  assert.match(elements.interim.textContent, /Review or edit/);
  assert.deepEqual(requests, ["/v1/runtime:status", "/v1/audio:transcribe"]);
  elements.story.value = " ";
  await c.compileStory();
  assert.equal(submitted.length, 0);
  elements.story.value = "A golden whale above a forest.";
  rejectPreflight = true;
  const previousSnapshot = {job_id: "previous-scene", stage: "master_ready"};
  c.latestLiveSnapshot = previousSnapshot;
  await c.compileStory();
  assert.equal(submitted.length, 0);
  assert.equal(c.latestLiveSnapshot, previousSnapshot);
  assert.equal(elements.projectorFrame.src, previousPreview);
  assert.equal(elements.voiceReview.textContent, "Clarify the action.");
  assert.equal(elements.story.disabled, false);
  assert.equal(c.generationSubmitting, false);
  rejectPreflight = false;

  // Validation must leave the existing scene and transport intact until it succeeds.
  const normalRespond = c.respond;
  const stopsBeforeCheck = events.filter((event) => event === "transport-stop").length;
  const assertUnsubmitted = () => {
    assert.equal(requests.includes("/v1/live-scenes"), false);
    assert.equal(c.latestLiveSnapshot, previousSnapshot);
    assert.equal(elements.projectorFrame.src, previousPreview);
    assert.equal(events.filter((event) => event === "transport-stop").length, stopsBeforeCheck);
    assert.equal(c.liveRequestEpoch, 0);
    assert.equal(c.generationSubmitting, false);
    assert.equal(elements.story.disabled, false);
    assert.equal(elements.style.disabled, false);
    assert.equal(elements.compileButton.disabled, false);
    assert.equal(timers.size, 0);
  };
  c.respond = async (url) => {
    assert.equal(url, "/v1/live-scene-planner/prepare");
    return {ok: true, json: async () => ({ready: true})};
  };
  await c.compileStory();
  assertUnsubmitted();
  assert.match(elements.voiceReview.textContent, /checker needs updating/);

  c.respond = (url, options) => {
    assert.equal(url, "/v1/live-scene-planner/prepare");
    return new Promise((_resolve, reject) => {
      options.signal.addEventListener("abort", () => reject(new Error("aborted")));
    });
  };
  const checking = c.compileStory();
  await new Promise(setImmediate);
  const checksBeforeDoubleClick = requests.length;
  assert.equal(elements.story.disabled, true);
  assert.equal(elements.style.disabled, true);
  assert.equal(elements.compileButton.disabled, true);
  await c.compileStory();
  await c.startSpeaking();
  assert.equal(requests.length, checksBeforeDoubleClick);
  assert.equal(c.latestLiveSnapshot, previousSnapshot);
  assert.equal(timers.size, 1);
  const checkTimer = [...timers.values()][0];
  assert.equal(checkTimer.milliseconds, 10000);
  checkTimer.callback();
  await checking;
  assertUnsubmitted();
  assert.match(elements.voiceReview.textContent, /timed out/);

  // Disabled fields can still be changed by restored state or another script.
  let finishPreflight;
  c.respond = (url) => {
    assert.equal(url, "/v1/live-scene-planner/prepare");
    return new Promise((resolve) => { finishPreflight = resolve; });
  };
  for (const [field, edited] of [["story", "A red whale above a forest."], ["style", "paper theater"]]) {
    const original = elements[field].value;
    const staleCheck = c.compileStory();
    await new Promise(setImmediate);
    elements[field].value = edited;
    finishPreflight(preflight());
    await staleCheck;
    assertUnsubmitted();
    assert.match(elements.voiceReview.textContent, /changed|check again|review again/i);
    elements[field].value = original;
  }
  c.respond = normalRespond;
  requiresFactReview = true;
  await c.compileStory();
  assert.equal(submitted.length, 0);
  assert.equal(c.latestLiveSnapshot, previousSnapshot);
  assert.match(elements.voiceReview.textContent, /Kept on this device: London/);
  assert.equal(elements.compileButton.textContent, "Generate this scene");
  parsedAction = "swimming";
  await c.compileStory();
  assert.equal(submitted.length, 0); // Changed facts need a fresh confirmation.
  assert.match(elements.voiceReview.textContent, /swimming/);
  const generation = c.compileStory();
  await new Promise(setImmediate);
  assert.equal(submitted.length, 1);
  assert.equal(submitted[0].text, elements.story.value);
  assert.equal(submitted[0].visual_style, "rich watercolor");
  assert.equal(submitted[0].reviewed_description, true);
  assert.equal(submitted[0].confirm_visual_facts, true);
  assert.equal(submitted[0].visual_fact_digest, "a".repeat(64));
  await c.compileStory();
  await c.startSpeaking();
  assert.equal(submitted.length, 1);
  finishGeneration({status: 400, json: async () => ({detail: "Description rejected"})});
  await generation;
  assert.equal(elements.story.disabled, false);
  assert.equal(elements.compileButton.disabled, false);
  assert.equal(c.generationSubmitting, false);
  assert.equal(elements.projectorFrame.src, previousPreview);
  requiresFactReview = false;
  elements.story.value = "A red whale above a forest.";
  const retry = c.compileStory();
  await c.compileStory();
  await new Promise(setImmediate);
  assert.equal(submitted.length, 2);
  assert.equal(submitted[1].text, elements.story.value);
  finishGeneration({status: 202, json: async () => ({job_id: "voice-scene"})});
  await retry;
  assert.equal(elements.micButton.disabled, true);
  assert.equal(elements.projectorFrame.src, previousPreview);
  c.voiceProjectionLive = true; // renderPack switches the frame only after the new draft exists.
  c.ensureProjectionPreview();
  assert.match(elements.projectorFrame.src, /reader=0&live=1$/);
  assert.equal(requests.some((url) => url.includes("reader-sessions")), false);
  assert.equal(requests.filter((url) => url === "/v1/audio:transcribe").length, 1);

  // A lost acceptance response must not offer a second paid submission.
  c.activeLiveJobId = null;
  c.latestLiveSnapshot = null; // Fresh startup can still have historical work on the server.
  elements.story.value = "A pink fox in a forest.";
  let accepted = {
    session_id: c.readerSessionId, server_instance_id: "server-1", session_revision: 4,
    job: {job_id: "historical", request: {
      text: elements.story.value, visual_style: elements.style.value,
    }},
  };
  c.respond = (url, options) => {
    if (url === "/v1/live-scene-planner/prepare") return preflight();
    if (url === "/v1/live-scenes") {
      submitted.push(JSON.parse(options.body));
      return new Promise((_resolve, reject) => {
        options.signal.addEventListener("abort", () => reject(new Error("timeout")));
      });
    }
    assert.equal(url, `/v1/live-scene-sessions/${encodeURIComponent(c.readerSessionId)}`);
    return accepted
      ? {ok: true, status: 200, json: async () => accepted}
      : {status: 404};
  };
  vm.runInContext(source.slice(source.indexOf("async function fetchLiveSceneSession("), source.indexOf("function acceptedLiveScenePointer(")), c);
  const uncertain = c.compileStory();
  await new Promise(setImmediate);
  const acceptanceTimer = [...timers.values()].find((timer) => timer.milliseconds === 15000);
  assert.ok(acceptanceTimer);
  acceptanceTimer.callback();
  await uncertain;
  assert.equal(submitted.length, 3);
  assert.equal(c.generationSubmitting, true);
  assert.equal(elements.micButton.disabled, true);
  assert.equal(elements.compileButton.textContent, "Check generation status");
  await c.compileStory();
  assert.equal(submitted.length, 3);
  accepted = {session_id: c.readerSessionId, server_instance_id: "server-1", session_revision: 5,
    job: {job_id: "different-review", request: {...submitted[2], visual_fact_digest: "a".repeat(64)}}};
  await c.compileStory();
  assert.notEqual(c.pendingSubmission, null);
  assert.equal(submitted.length, 3);
  accepted = {session_id: c.readerSessionId, server_instance_id: "server-1", session_revision: 6,
    job: {job_id: "accepted-after-timeout", request: submitted[2]}};
  c.handleLiveSceneSessionPointer = (pointer) => { c.activeLiveJobId = pointer.job.job_id; };
  await c.compileStory();
  assert.equal(c.pendingSubmission, null);
  assert.equal(c.activeLiveJobId, "accepted-after-timeout");
  assert.equal(submitted.length, 3);

  const empty = harness();
  empty.context.voiceMode = true;
  empty.context.ensureProjectionPreview();
  const retainedImage = empty.elements.projectorFrame.src;
  const retainedDescription = empty.elements.story.value;
  empty.context.respond = async () => ({ok: true, json: async () => ({asr: {ready: true}, text: "  "})});
  empty.context.compileStory = () => { throw new Error("Empty speech must not generate"); };
  await empty.context.startSpeaking();
  const emptyRecorder = empty.context.mediaRecorder;
  emptyRecorder.handlers.dataavailable({data: new Blob(["a".repeat(1200)])});
  empty.context.stopSpeaking();
  await emptyRecorder.handlers.stop();
  assert.match(empty.elements.interim.textContent, /No speech was recognized/);
  assert.equal(empty.context.finalizing, false);
  assert.equal(empty.elements.micButton.disabled, false);
  assert.equal(empty.elements.projectorFrame.src, retainedImage);
  assert.equal(empty.elements.story.value, retainedDescription);
}

(async () => { await lifecycle(); await voiceToScene(); })().then(() => console.log("Workbench microphone: read-aloud lifecycle, transcript review and explicit generation/retry passed."))
  .catch((error) => { console.error(error); process.exitCode = 1; });
