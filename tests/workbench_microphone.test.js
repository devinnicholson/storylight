const assert = require("assert").strict;
const fs = require("fs");
const vm = require("vm");

const source = fs.readFileSync("src/bookforge/static/workbench.js", "utf8");
function harness() {
  const timers = new Map();
  const requests = [];
  const events = [];
  const element = () => ({classList: {add() {}, remove() {}, toggle() {}},
    setAttribute() {}, style: {}, textContent: ""});
  const elements = Object.fromEntries([
    "micButton", "micButtonText", "micLevel", "compileButton", "interim",
    "story", "style", "projectorLink", "projectorFrame", "projectionPreview",
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
    canRecordAudio: true, sceneReady: true, demoMode: false, activeLiveJobId: null,
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
    ["function setSceneReady(", "function safeText("],
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

lifecycle().then(() => console.log("Workbench microphone: release, finalization, timeout, ASR availability and demo projection passed."))
  .catch((error) => { console.error(error); process.exitCode = 1; });
