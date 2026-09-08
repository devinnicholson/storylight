const assert = require("assert").strict;
const fs = require("fs");
const vm = require("vm");
const SERVER_ID = `server_${"a".repeat(32)}`;

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
  let now = 100;
  const timers = new Map();
  const intervals = new Map();
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
    constructor() { this.handlers = {}; this.listeners = {}; this.mimeType = "audio/webm"; }
    addEventListener(name, callback, options = {}) {
      (this.listeners[name] ||= []).push({callback, once: options.once});
      this.handlers[name] = (event) => {
        let result;
        for (const listener of [...this.listeners[name]]) {
          if (!this.listeners[name].includes(listener)) continue;
          if (listener.once) this.removeEventListener(name, listener.callback);
          result = listener.callback(event);
        }
        return result;
      };
    }
    removeEventListener(name, callback) {
      this.listeners[name] = (this.listeners[name] || []).filter((entry) => entry.callback !== callback);
      if (!this.listeners[name].length) delete this.handlers[name];
    }
    start() { this.state = "recording"; }
    stop() { this.state = "inactive"; events.push("recorder-stop"); }
  }
  const context = {
    elements, AbortController, Blob, Uint8Array, MediaRecorder: Recorder,
    performance: {now: () => now},
    canRecordAudio: true, sceneReady: true, demoMode: false, voiceMode: false,
    voiceProjectionLive: false, projectorPreviewUrl: null,
    generationSubmitting: false, pendingSubmission: null, pendingVoiceReview: null, generationReconciling: false,
    activeLiveJobId: null, latestLiveSnapshot: null,
    listening: false, starting: false, finalizing: false, stream: null,
    analyser: null, audioContext: null, mediaRecorder: null, audioChunks: [],
    partialTimer: null, partialBusy: false, partialInFlight: Promise.resolve(),
    partialBytes: 0, recordingEpoch: 0, readerGeneration: null, activePageText: null,
    voiceTiming: null, partialSchedule: null,
    readerSessionId: "reader/example", requestAnimationFrame() {},
    navigator: {mediaDevices: {async getUserMedia() {
      events.push("permission");
      return {getTracks: () => [{stop: () => events.push("track-stop")}]};
    }}},
    window: {
      crypto: require("crypto").webcrypto,
      MediaRecorder: Recorder,
      AudioContext: class {
        createAnalyser() { return {}; }
        createMediaStreamSource() { return {connect() {}}; }
        close() { events.push("context-close"); }
      },
      setInterval(callback, milliseconds) { const id = Symbol(); intervals.set(id, {callback, milliseconds, at: now + milliseconds}); return id; },
      clearInterval(id) { intervals.delete(id); },
      setTimeout(callback, milliseconds) { const id = Symbol(); timers.set(id, {callback, milliseconds, at: now + milliseconds}); return id; },
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
    ["function voiceTimingEvent(", "function liveRevision("],
    ["function setSceneInputsDisabled(", "function finishLiveJob("],
  ]) vm.runInContext(source.slice(source.indexOf(start), source.indexOf(end)), context);
  async function advance(milliseconds) {
    const end = now + milliseconds;
    let callbacks = 0;
    for (;;) {
      const due = [...timers, ...intervals].filter(([, timer]) => timer.at <= end)
        .sort((a, b) => a[1].at - b[1].at)[0];
      if (!due) break;
      assert.ok(++callbacks < 1000, "Timer loop did not settle");
      const [id, timer] = due;
      now = timer.at;
      if (intervals.has(id)) timer.at += timer.milliseconds;
      else timers.delete(id);
      timer.callback();
      await new Promise(setImmediate);
    }
    now = end;
    await new Promise(setImmediate);
  }
  return {context, elements, timers, intervals, requests, events, advance, now: () => now};
}

async function lifecycle() {
  const {context: c, elements, timers, intervals, requests, events} = harness();
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
  assert.equal(intervals.size, 1);
  assert.equal([...intervals.values()][0].milliseconds, 2000); // Read-aloud mode is unchanged.
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

function generationHarness() {
  const h = harness();
  const c = h.context;
  const displayed = [];
  const submitted = [];
  const presented = [];
  const checks = [];
  let pointer = null;
  Object.assign(c, {
    voiceMode: true, sceneReady: false,
    voiceGeneration: {latest: null, active: null, completed: null, timer: null,
      publishing: false, presentationInFlight: null, attempted: new Set()},
    edgePlanPreparationTimer: null, liveRequestEpoch: 0,
    liveSessionRevision: 0, liveServerInstanceId: null, lastLiveRevision: -1,
    liveElapsedTimer: null, livePollTimer: null, liveStartedAt: 0,
    liveElapsedBaseMs: 0, liveElapsedBaseAt: 0, liveIsTerminal: false,
    invalidatePreparation() {},
    stopLiveJobTransport() {}, renderGenerationProgress() {}, updateElapsedClock() {}, setStatus() {},
    setGenerateButtonForStage() {}, liveSessionStreamIsHealthy: () => true,
    isTerminalSnapshot: (snapshot) => snapshot.complete === true || snapshot.stage === "failed",
  });
  h.elements.story.value = "";
  h.elements.style.value = "rich watercolor";
  h.elements.compileButton.dataset = {};
  h.elements.error = {classList: {add() {}, remove() {}}};
  h.elements.generationProgress = {classList: {add() {}, remove() {}}};
  vm.runInContext(source.slice(source.indexOf("function liveRevision("), source.indexOf("function providerLabel(")), c);
  vm.runInContext(source.slice(source.indexOf("function finishLiveJob("), source.indexOf("async function loadLatestScene(")), c);
  c.renderPack = (payload) => displayed.push(payload.live_snapshot.job_id);
  const ready = (request) => {
    const dog = request.text?.includes("dog");
    const count = request.text?.includes("three") ? 3 : request.text?.includes("two") ? 2 : 1;
    return {revision: "dependency-scene-draft-v1", requires_fact_review: true,
      visual_fact_digest: "a".repeat(64), local_omissions: [],
      visual_facts: {subjects: [{ref: "actor", label: dog ? "dog" : "cat", attributes: [],
        actions: [dog ? "chasing ball" : "chasing mouse"]}],
        objects: [{ref: "target", label: dog ? "ball" : "mouse", count, attributes: []}],
        relationships: [], negatives: [], setting: {label: "unspecified"}},
    };
  };
  const response = (status, payload) => ({ok: status < 400, status, json: async () => payload,
    headers: {get: (name) => name === "X-Bookforge-Server-Instance-Id" ? SERVER_ID : String(pointer?.session_revision || 0)},
  });
  h.state = {rejectCheck: false, ready};
  c.respond = async (url, options = {}) => {
    if (url === "/v1/runtime:status") return response(200, {asr: {ready: true}});
    if (url === "/v1/live-scene-planner/prepare") {
      const request = JSON.parse(options.body);
      assert.deepEqual(Object.keys(request).sort(), [
        "reviewed_description", "seed", "session_id", "text", "visual_style",
      ]); // The real prepare endpoint rejects render/presentation/confirmation fields.
      checks.push(request);
      return h.state.rejectCheck
        ? response(422, {detail: {message: "Clarify the action."}})
        : response(200, h.state.ready(request));
    }
    if (url === `/v1/live-scene-sessions/${encodeURIComponent(c.readerSessionId)}`) {
      return pointer ? response(200, pointer) : response(404, {});
    }
    if (url === "/v1/live-scenes") {
      const request = JSON.parse(options.body);
      submitted.push(request);
      assert.equal(request.expected_server_instance_id, SERVER_ID);
      assert.equal(request.expected_session_revision, pointer?.session_revision || 0);
      assert.match(request.submission_id, /^[a-f0-9]{8}(?:-[a-f0-9]{4}){3}-[a-f0-9]{12}$/);
      pointer = {session_id: c.readerSessionId, server_instance_id: SERVER_ID,
        session_revision: (pointer?.session_revision || 0) + 1,
        job: {job_id: `job-${submitted.length}`, request, revision: 1,
          stage: "draft_ready", complete: false, presentation_ready: false, story_pack: {pages: []}}};
      return response(202, pointer.job);
    }
    if (url.endsWith("/present")) {
      presented.push({url, body: JSON.parse(options.body)});
      assert.equal(url, `/v1/live-scenes/${pointer.job.job_id}/present`);
      assert.deepEqual(presented.at(-1).body, {server_instance_id: SERVER_ID, session_revision: pointer.session_revision});
      pointer = {...pointer, job: {...pointer.job, presentation_ready: true, revision: pointer.job.revision + 1}};
      return response(200, pointer.job);
    }
    throw new Error(`Unexpected request: ${url}`);
  };
  h.complete = () => {
    pointer = {...pointer, job: {...pointer.job, stage: "master_ready", complete: true,
      revision: pointer.job.revision + 1, artifacts: [{kind: "master"}, {kind: "depth"}]}};
    c.handleLiveSceneSessionPointer(pointer);
  };
  h.response = response;
  h.pointer = () => pointer;
  h.setPointer = (value) => { pointer = value; };
  return {...h, displayed, submitted, presented, checks};
}

const flush = () => new Promise(setImmediate);
function fireTimer(h, milliseconds) {
  const found = [...h.timers.entries()].find(([, timer]) => timer.milliseconds === milliseconds);
  assert.ok(found, `Expected ${milliseconds} ms timer`);
  h.timers.delete(found[0]);
  found[1].callback();
}

async function voiceToScene() {
  // A completed final transcript starts exactly one generation without a click.
  const h = generationHarness();
  const c = h.context;
  c.ensureProjectionPreview();
  const previousPreview = h.elements.projectorFrame.src;
  const original = c.respond;
  let finishAsr;
  c.respond = (url, options) => url === "/v1/audio:transcribe"
    ? new Promise((resolve) => { finishAsr = resolve; }) : original(url, options);
  await c.startSpeaking();
  assert.equal(h.intervals.size, 0);
  assert.equal(h.timers.get(c.partialTimer).milliseconds, 1200);
  const recorder = c.mediaRecorder;
  recorder.handlers.dataavailable({data: new Blob(["a".repeat(1200)])});
  c.stopSpeaking();
  const final = recorder.handlers.stop();
  c.stopSpeaking();
  await c.startSpeaking();
  await flush();
  assert.equal(h.requests.filter((url) => url === "/v1/audio:transcribe").length, 1);
  assert.equal(h.events.filter((event) => event === "track-stop").length, 1);
  finishAsr(h.response(200, {text: "  A cat chasing a mouse.  "}));
  await final;
  await flush();
  assert.equal(c.finalizing, false);
  assert.equal(h.elements.story.value, "A cat chasing a mouse.");
  assert.equal(h.submitted.length, 1);
  assert.equal(h.submitted[0].text, "A cat chasing a mouse.");
  assert.equal(h.submitted[0].confirm_visual_facts, true);
  assert.equal(h.submitted[0].visual_fact_digest, "a".repeat(64));
  assert.equal(h.submitted[0].display_when_complete, true);
  assert.equal(h.submitted[0].defer_presentation, true);
  assert.deepEqual(h.displayed, []);
  assert.equal(h.elements.projectorFrame.src, previousPreview);
  await c.pumpVoiceGeneration();
  assert.equal(h.submitted.length, 1);
  h.complete();
  await flush();
  assert.equal(h.presented.length, 1);
  assert.deepEqual(h.displayed, ["job-1"]);
  await c.tryPresentVoiceGeneration();
  assert.equal(h.presented.length, 1);
  const completed = c.voiceGeneration.completed;
  const {sourceKey, semanticKey} = completed;
  assert.equal(completed.snapshot.presentation_ready, true);
  c.offerVoiceTranscript("A cat is chasing a mouse.", {final: true});
  await flush();
  assert.equal(h.submitted.length, 1); // This fixture supplies exactly equal validated facts.
  assert.equal(h.checks.length, 2);
  assert.equal(c.voiceGeneration.completed.sourceKey, sourceKey);
  assert.equal(c.voiceGeneration.completed.semanticKey, semanticKey);
  assert.equal(c.voiceGeneration.completed.snapshot.job_id, "job-1");
  c.offerVoiceTranscript("A cat chasing two mice.", {final: true});
  await flush();
  assert.equal(h.submitted.length, 2); // Changed object count is a different scene.
  h.complete();
  await flush();
  h.elements.style.value = "ink drawing";
  c.offerVoiceTranscript("A cat chasing two mice.", {final: true});
  await flush();
  assert.equal(h.submitted.length, 3); // Style also belongs to the semantic key.

  // One partial starts paid work. A second identical ASR observation allows
  // the complete scene to appear while the microphone is still listening.
  const live = generationHarness();
  await live.context.startSpeaking();
  const liveRecorder = live.context.mediaRecorder;
  liveRecorder.handlers.dataavailable({data: new Blob(["a".repeat(1200)])});
  const liveRespond = live.context.respond;
  live.context.respond = (url, options) => url === "/v1/audio:transcribe"
    ? live.response(200, {text: "A cat chasing a mouse."}) : liveRespond(url, options);
  await live.context.transcribePartialRecording(live.context.recordingEpoch);
  assert.equal(live.submitted.length, 0);
  fireTimer(live, 350);
  await flush();
  assert.equal(live.submitted.length, 1);
  assert.equal(live.context.listening, true);
  assert.equal(live.elements.micButton.disabled, false); // Stop stays available during generation.
  live.complete();
  await flush();
  assert.equal(live.presented.length, 0);
  liveRecorder.handlers.dataavailable({data: new Blob(["b".repeat(100)])});
  await live.context.transcribePartialRecording(live.context.recordingEpoch);
  await flush();
  assert.equal(live.presented.length, 1);
  assert.equal(live.context.listening, true);
  assert.deepEqual(live.displayed, ["job-1"]);
  assert.equal(live.submitted.length, 1);
  assert.equal(live.requests.some((url) => url.includes("reader-sessions")), false);

  // Intermediate intents coalesce. A superseded complete scene is never shown,
  // and only the latest final wording runs after the first active job ends.
  const queue = generationHarness();
  queue.context.listening = true;
  queue.context.offerVoiceTranscript("A cat chasing a mouse.");
  fireTimer(queue, 350);
  await flush();
  queue.context.offerVoiceTranscript("A cat chasing two mice.");
  queue.context.offerVoiceTranscript("A cat chasing three mice.", {final: true});
  await queue.context.pumpVoiceGeneration();
  assert.equal(queue.submitted.length, 1);
  queue.complete();
  await flush();
  assert.deepEqual(queue.displayed, []);
  assert.equal(queue.presented.length, 0);
  assert.equal(queue.submitted.length, 2);
  assert.equal(queue.submitted[1].text, "A cat chasing three mice.");
  queue.context.listening = false;
  queue.complete();
  await flush();
  assert.deepEqual(queue.displayed, ["job-2"]);
  assert.equal(queue.presented.length, 1);

  // Bad partials and empty speech retain the previous artwork and do not retry
  // automatically. A newer phrase invalidates an in-flight local check.
  const bad = generationHarness();
  bad.state.rejectCheck = true;
  bad.context.offerVoiceTranscript("A cat she's seeing a mouse.", {final: true});
  await flush();
  await bad.context.pumpVoiceGeneration();
  assert.equal(bad.submitted.length, 0);
  assert.equal(bad.checks.length, 1);
  assert.deepEqual(bad.displayed, []);
  assert.match(bad.elements.voiceReview.textContent, /Clarify/);
  bad.context.offerVoiceTranscript("   ", {final: true});
  assert.equal(bad.submitted.length, 0);
  assert.equal(bad.checks.length, 1);
  const latest = bad.context.voiceGeneration.latest;
  bad.context.offerVoiceTranscript("A fox jumps over a dog.", {
    final: true, epoch: bad.context.recordingEpoch - 1,
  });
  await flush();
  assert.equal(bad.context.voiceGeneration.latest, latest);
  assert.equal(bad.checks.length, 1);

  const stale = generationHarness();
  const staleRespond = stale.context.respond;
  let finishCheck;
  stale.context.respond = (url, options) => url === "/v1/live-scene-planner/prepare" && !finishCheck
    ? new Promise((resolve) => { finishCheck = resolve; }) : staleRespond(url, options);
  stale.context.offerVoiceTranscript("A cat chasing a mouse.", {final: true});
  await flush();
  stale.context.offerVoiceTranscript("A dog chasing a ball.", {final: true});
  finishCheck(stale.response(200, stale.state.ready({})));
  await flush();
  await stale.context.pumpVoiceGeneration();
  assert.equal(stale.submitted.length, 1);
  assert.equal(stale.submitted[0].text, "A dog chasing a ball.");
  assert.deepEqual(stale.displayed, []);

  // Lost acceptance stays locked. A later transcript cannot cause another POST
  // while the first may already have incurred cost.
  const lost = generationHarness();
  const lostRespond = lost.context.respond;
  const historical = {session_id: lost.context.readerSessionId, server_instance_id: SERVER_ID,
    session_revision: 7, job: {job_id: "historical", complete: true, stage: "master_ready",
      request: {text: "A cat chasing a mouse.", visual_style: "rich watercolor"}}};
  lost.setPointer(historical);
  lost.context.respond = (url, options) => {
    if (url === "/v1/live-scenes") {
      lost.submitted.push(JSON.parse(options.body));
      return new Promise((_resolve, reject) => options.signal.addEventListener("abort", () => reject(new Error("timeout"))));
    }
    return lostRespond(url, options);
  };
  lost.context.offerVoiceTranscript("A cat chasing a mouse.", {final: true});
  await flush();
  fireTimer(lost, 15000);
  await flush();
  assert.notEqual(lost.context.pendingSubmission, null);
  lost.context.offerVoiceTranscript("A dog chasing a ball.", {final: true});
  await lost.context.pumpVoiceGeneration();
  await lost.context.reconcileGeneration();
  assert.equal(lost.submitted.length, 1);
  assert.deepEqual(lost.displayed, []);
  assert.equal(lost.presented.length, 0);
  lost.setPointer({...historical, session_revision: 8,
    job: {job_id: "different-digest", revision: 1, stage: "draft_ready", complete: false,
      request: {...lost.submitted[0], visual_fact_digest: "b".repeat(64)}}});
  await lost.context.reconcileGeneration();
  assert.notEqual(lost.context.pendingSubmission, null);
  lost.setPointer({...historical, session_revision: 8,
    job: {job_id: "other-tab-same-text", revision: 1, stage: "draft_ready", complete: false,
      request: {...lost.submitted[0], submission_id: require("crypto").randomUUID()}}});
  await lost.context.reconcileGeneration();
  assert.notEqual(lost.context.pendingSubmission, null); // Matching text cannot claim another POST.
  lost.setPointer({...historical, session_revision: 9,
    job: {job_id: "recovered", revision: 1, stage: "draft_ready", complete: false,
      request: lost.submitted[0]}});
  await lost.context.reconcileGeneration();
  assert.equal(lost.context.pendingSubmission, null);
  assert.equal(lost.context.activeLiveJobId, "recovered");
  assert.equal(lost.submitted.length, 1);

  // An old API must not make a stale remembered server epoch safe to reuse.
  const missingEpoch = generationHarness();
  const epochRespond = missingEpoch.context.respond;
  missingEpoch.context.liveServerInstanceId = SERVER_ID;
  missingEpoch.context.respond = (url, options) => url.includes("/live-scene-sessions/")
    ? {status: 404, headers: {get: () => null}, json: async () => ({})}
    : epochRespond(url, options);
  missingEpoch.context.offerVoiceTranscript("A cat chasing a mouse.", {final: true});
  await flush();
  assert.equal(missingEpoch.submitted.length, 0);
  assert.match(missingEpoch.elements.error.textContent, /current server ID/);
  assert.equal(missingEpoch.context.pendingSubmission, null);

  // A competing tab wins after the empty-session read. Adopt its pointer without
  // cancelling it or resubmitting this rejected intent when that job finishes.
  const conflict = generationHarness();
  const conflictRespond = conflict.context.respond;
  conflict.context.respond = (url, options) => {
    if (url === "/v1/live-scenes") {
      conflict.submitted.push(JSON.parse(options.body));
      conflict.setPointer({session_id: conflict.context.readerSessionId,
        server_instance_id: SERVER_ID, session_revision: 1,
        job: {job_id: "competing-tab", revision: 1, complete: false, stage: "draft_ready",
          request: {text: "A dog chasing a ball.", visual_style: "rich watercolor"}}});
      return conflict.response(409, {detail: "Generation session changed before submission"});
    }
    return conflictRespond(url, options);
  };
  conflict.context.offerVoiceTranscript("A cat chasing a mouse.", {final: true});
  await flush();
  assert.equal(conflict.context.activeLiveJobId, "competing-tab");
  assert.equal(conflict.context.pendingSubmission, null);
  conflict.complete();
  await flush();
  await conflict.context.pumpVoiceGeneration();
  assert.equal(conflict.submitted.length, 1);
  assert.equal(conflict.presented.length, 0);

  // A failed presentation read is one bounded attempt, not a recursive retry
  // loop or a second image request. Keep a finite fake even if this regresses.
  const publishFailure = generationHarness();
  publishFailure.context.offerVoiceTranscript("A cat chasing a mouse.", {final: true});
  await flush();
  const publishRespond = publishFailure.context.respond;
  let presentationReads = 0;
  publishFailure.context.respond = (url, options) => {
    if (url.includes("/live-scene-sessions/")) {
      presentationReads += 1;
      if (presentationReads < 3) throw new Error("unavailable");
    }
    return publishRespond(url, options);
  };
  publishFailure.complete();
  await flush();
  assert.equal(presentationReads, 1);
  assert.equal(publishFailure.submitted.length, 1);
  assert.equal(publishFailure.presented.length, 0);
  assert.deepEqual(publishFailure.displayed, []);

  // If ASR resumes while the presentation pointer is being read, no display
  // occurs yet; the same completed image remains eligible after ASR finishes.
  const publishRace = generationHarness();
  publishRace.context.offerVoiceTranscript("A cat chasing a mouse.", {final: true});
  await flush();
  const raceRespond = publishRace.context.respond;
  let finishPointer;
  publishRace.context.respond = (url, options) => url.includes("/live-scene-sessions/")
    ? new Promise((resolve) => { finishPointer = resolve; }) : raceRespond(url, options);
  publishRace.complete();
  await flush();
  publishRace.context.partialBusy = true;
  finishPointer(publishRace.response(200, publishRace.pointer()));
  await flush();
  assert.equal(publishRace.presented.length, 0);
  assert.equal(publishRace.context.voiceGeneration.completed.presentationAttempted, false);
  publishRace.context.respond = raceRespond;
  publishRace.context.partialBusy = false;
  await publishRace.context.tryPresentVoiceGeneration();
  assert.equal(publishRace.presented.length, 1);

  // Discovering another active job must leave the unsubmitted new intent
  // eligible, then start it exactly once when that older job finishes.
  const prior = generationHarness();
  prior.setPointer({session_id: prior.context.readerSessionId, server_instance_id: SERVER_ID,
    session_revision: 1, job: {job_id: "older", revision: 1, stage: "draft_ready", complete: false,
      presentation_ready: false, request: {text: "A cat chasing a mouse.", visual_style: "rich watercolor"}}});
  prior.context.offerVoiceTranscript("A dog chasing a ball.", {final: true});
  await flush();
  assert.equal(prior.submitted.length, 0);
  assert.equal(prior.context.activeLiveJobId, "older");
  prior.complete();
  await flush();
  assert.equal(prior.submitted.length, 1);
  assert.equal(prior.submitted[0].text, "A dog chasing a ball.");
  assert.deepEqual(prior.displayed, []);

  // A new intent arriving during the pre-POST pointer read cancels only the
  // unsubmitted intent; if the user returns to it later it must remain eligible.
  const beforePost = generationHarness();
  const beforeRespond = beforePost.context.respond;
  let releasePrior;
  beforePost.context.respond = (url, options) => url.includes("/live-scene-sessions/") && !releasePrior
    ? new Promise((resolve) => { releasePrior = resolve; }) : beforeRespond(url, options);
  beforePost.context.offerVoiceTranscript("A cat chasing a mouse.", {final: true});
  await flush();
  beforePost.context.offerVoiceTranscript("A dog chasing a ball.", {final: true});
  releasePrior(beforePost.response(404, {}));
  await flush();
  assert.equal(beforePost.submitted.length, 1);
  assert.equal(beforePost.submitted[0].text, "A dog chasing a ball.");
  beforePost.context.offerVoiceTranscript("A cat chasing a mouse.", {final: true});
  beforePost.complete();
  await flush();
  assert.equal(beforePost.submitted.length, 2);
  assert.equal(beforePost.submitted[1].text, "A cat chasing a mouse.");

  const failed = generationHarness();
  failed.context.listening = true;
  failed.context.offerVoiceTranscript("A cat chasing a mouse.", {final: true});
  await flush();
  failed.context.offerVoiceTranscript("A dog chasing a ball.", {final: true});
  const failedPointer = failed.pointer();
  failed.setPointer({...failedPointer, job: {...failedPointer.job, revision: 2, stage: "failed"}});
  failed.context.handleLiveSceneSessionPointer(failed.pointer());
  await flush();
  assert.equal(failed.submitted.length, 2);
  assert.equal(failed.submitted[1].text, "A dog chasing a ball.");
  assert.deepEqual(failed.displayed, []);
  assert.equal(failed.elements.compileButton.disabled, true);
  await failed.context.pumpVoiceGeneration();
  assert.equal(failed.submitted.length, 2);

  // Stop releases the hardware immediately, but final ASR waits for an
  // already-started presentation CAS so it cannot race that publication.
  const stopping = generationHarness();
  await stopping.context.startSpeaking();
  const stoppingRecorder = stopping.context.mediaRecorder;
  stoppingRecorder.handlers.dataavailable({data: new Blob(["a".repeat(1200)])});
  stopping.context.offerVoiceTranscript("A cat chasing a mouse.");
  fireTimer(stopping, 350);
  await flush();
  stopping.context.offerVoiceTranscript("A cat chasing a mouse.");
  const stopRespond = stopping.context.respond;
  let releasePresentation;
  stopping.context.respond = (url, options) => {
    if (url.endsWith("/present")) {
      return new Promise((resolve) => { releasePresentation = () => resolve(stopRespond(url, options)); });
    }
    if (url === "/v1/audio:transcribe") return stopping.response(200, {text: "A cat chasing a mouse."});
    return stopRespond(url, options);
  };
  stopping.complete();
  await flush();
  assert.equal(typeof releasePresentation, "function");
  stopping.context.stopSpeaking();
  const stopped = stoppingRecorder.handlers.stop();
  await flush();
  assert.equal(stopping.events.filter((event) => event === "track-stop").length, 1);
  assert.equal(stopping.requests.filter((url) => url === "/v1/audio:transcribe").length, 0);
  releasePresentation();
  await stopped;
  await flush();
  assert.equal(stopping.requests.filter((url) => url === "/v1/audio:transcribe").length, 1);
  assert.equal(stopping.submitted.length, 1);
}

async function partialScheduling() {
  // A pause with no new recorder bytes does not send the same audio again.
  // A slow partial may span several scheduling opportunities, but Stop must
  // drain it before the one final request includes the recorder's final chunk.
  const h = generationHarness();
  const c = h.context;
  await c.startSpeaking();
  const recorder = c.mediaRecorder;
  const respond = c.respond;
  const asr = [];
  let inFlight = 0;
  let maximumInFlight = 0;
  c.respond = (url, options) => {
    if (url !== "/v1/audio:transcribe") return respond(url, options);
    inFlight += 1;
    maximumInFlight = Math.max(maximumInFlight, inFlight);
    return new Promise((resolve) => asr.push({at: h.now(), bytes: options.body.size,
      finish(text) { inFlight -= 1; resolve(h.response(200, {text, total_ms: 25})); }}));
  };
  await h.advance(6000);
  assert.equal(asr.length, 0);
  recorder.handlers.dataavailable({data: new Blob(["a".repeat(1200)])});
  await h.advance(2000);
  assert.equal(asr.length, 1);
  recorder.handlers.dataavailable({data: new Blob(["b".repeat(100)])});
  await h.advance(6000);
  assert.equal(asr.length, 1);
  assert.equal(maximumInFlight, 1);
  c.stopSpeaking();
  recorder.handlers.dataavailable({data: new Blob(["c".repeat(100)])});
  const finalizing = recorder.handlers.stop();
  await flush();
  assert.equal(h.events.filter((event) => event === "track-stop").length, 1);
  assert.equal(asr.length, 1);
  asr[0].finish("A cat chasing a mouse.");
  await flush();
  assert.equal(asr.length, 2);
  assert.deepEqual(asr.map((request) => request.bytes), [1200, 1400]);
  assert.equal(h.submitted.length, 0); // The stopped partial is stale.
  asr[1].finish("A dog chasing a ball.");
  await finalizing;
  await flush();
  assert.equal(h.submitted.length, 1);
  assert.equal(h.submitted[0].text, "A dog chasing a ball.");
  assert.equal(maximumInFlight, 1);
  assert.equal(c.finalizing, false);
  await h.advance(6000);
  assert.equal(asr.length, 2);

  // Punctuation and whitespace changes must still pass the local facts check,
  // but equal validated meaning must not purchase another image, including
  // when the updated wording arrives while the first image is rendering.
  const punctuation = generationHarness();
  punctuation.context.listening = true;
  punctuation.context.offerVoiceTranscript("A cat chasing a mouse");
  await punctuation.advance(350);
  assert.equal(punctuation.submitted.length, 1);
  punctuation.context.offerVoiceTranscript("  A cat chasing a mouse.  ", {final: true});
  await flush();
  assert.equal(punctuation.submitted.length, 1);
  punctuation.complete();
  await flush();
  assert.equal(punctuation.submitted.length, 1);
  assert.equal(punctuation.checks.length, 2);
  assert.equal(punctuation.presented.length, 1);
  assert.deepEqual(punctuation.displayed, ["job-1"]);
  punctuation.context.offerVoiceTranscript("A cat chasing a mouse!", {final: true});
  await flush();
  assert.equal(punctuation.checks.length, 3);
  assert.equal(punctuation.submitted.length, 1);
  assert.equal(punctuation.presented.length, 1);
}

async function adaptiveCadence() {
  const h = generationHarness();
  const c = h.context;
  c.partialSchedule = {lastActivityAt: h.now()};
  for (const [duration, expected] of [[0, 750], [200, 550], [400, 400],
    [850, 850], [2400, 2000]]) {
    assert.equal(c.nextVoicePartialDelay(duration), expected);
    assert.ok(expected >= 350);
    assert.ok(duration + expected >= 750);
  }
  await h.advance(1201);
  assert.equal(c.nextVoicePartialDelay(200), 2000);
  c.partialSchedule.lastActivityAt = null;
  assert.equal(c.nextVoicePartialDelay(200), 2000);

  await c.startSpeaking();
  const startedAt = h.now();
  const recorder = c.mediaRecorder;
  const respond = c.respond;
  h.state.rejectCheck = true; // This cadence scenario never needs image generation.
  const asr = [];
  c.respond = (url, options) => url === "/v1/audio:transcribe"
    ? new Promise((resolve) => asr.push({at: h.now(), bytes: options.body.size,
      finish: (text) => resolve(h.response(200, {text, total_ms: 200}))}))
    : respond(url, options);
  const chunk = (bytes) => recorder.handlers.dataavailable({data: new Blob(["a".repeat(bytes)])});
  chunk(1200);
  await h.advance(1199);
  assert.equal(asr.length, 0);
  await h.advance(1);
  assert.equal(asr[0].at - startedAt, 1200);
  await h.advance(200);
  c.partialSchedule.lastActivityAt = h.now();
  asr[0].finish("A cat chasing a mouse.");
  await flush();
  assert.equal(h.timers.get(c.partialTimer).milliseconds, 550);
  chunk(100);
  await h.advance(549);
  assert.equal(asr.length, 1);
  await h.advance(1);
  assert.equal(asr.length, 2);
  assert.equal(asr[1].at - asr[0].at, 750);
  assert.equal(c.partialTimer, null); // No periodic ticks accumulate behind ASR.
  await h.advance(1200);
  assert.equal(asr.length, 2);
  c.partialSchedule.lastActivityAt = h.now();
  asr[1].finish("A cat chasing a mouse.");
  await flush();
  assert.equal(h.timers.get(c.partialTimer).milliseconds, 1200);
  chunk(100);
  await h.advance(1199);
  assert.equal(asr.length, 2);
  await h.advance(1);
  assert.equal(asr.length, 3);
  assert.equal(asr[2].at - asr[1].at, 2400);
  await h.advance(1); // Activity is now older than the quiet-period boundary.
  const latest = c.voiceGeneration.latest;
  asr[2].finish(null);
  await flush();
  assert.equal(c.voiceGeneration.latest, latest); // Malformed partial is not a scene correction.
  assert.equal(h.timers.get(c.partialTimer).milliseconds, 2000);
  assert.equal(h.submitted.length, 0);
  assert.deepEqual(asr.map((request) => request.bytes), [1200, 1300, 1400]);
  c.stopSpeaking();
  await h.advance(6000);
  assert.equal(asr.length, 3);

  // Diagnostics retain bounded event data, never the synthetic transcript.
  const events = c.voiceTiming.events;
  assert.ok(events.some((event) => event.type === "recording_started"));
  assert.ok(events.some((event) => event.type === "asr_started"));
  assert.ok(events.some((event) => event.type === "asr_completed"));
  assert.ok(events.every((event) => Number.isFinite(event.ms) && event.ms >= 0));
  assert.equal(JSON.stringify(events).includes("A cat chasing a mouse"), false);
  for (let index = 0; index < 300; index += 1) c.voiceTimingEvent("test_sample");
  assert.equal(c.voiceTiming.events.length, 256);
}

async function timingIsolation() {
  const h = generationHarness();
  const c = h.context;
  await c.startSpeaking();
  c.voiceTimingEvent("asr_completed", {asrMs: {text: "private synthetic words"},
    serverMs: Infinity, providerMs: "private synthetic words", bytes: 1200,
    final: true, jobId: "private synthetic words", arbitrary: "private synthetic words"});
  assert.deepEqual(JSON.parse(JSON.stringify(c.voiceTiming.events.at(-1))), {
    type: "asr_completed", ms: 0, bytes: 1200, final: true,
  });
  let onMessage;
  c.window.location = {origin: "http://localhost"};
  c.window.addEventListener = (type, callback) => { assert.equal(type, "message"); onMessage = callback; };
  c.elements.projectorFrame.contentWindow = {};
  vm.runInContext(source.slice(source.indexOf("window.bookforgeVoiceTiming =")), c);
  const message = {origin: "http://localhost", source: c.elements.projectorFrame.contentWindow,
    data: {type: "bookforge.preview-activated", sessionId: c.readerSessionId, jobId: "job-1"}};
  c.voiceGeneration.completed = {snapshot: {job_id: "job-1"}};
  c.voiceTimingEvent("generation_response", {status: 202, jobId: "job-1"});
  const before = c.voiceTiming.events.length;
  for (const wrong of [
    {...message, origin: "https://elsewhere.invalid"}, {...message, source: {}},
    {...message, data: {...message.data, sessionId: "another-session"}},
    {...message, data: {...message.data, jobId: "another-job"}},
  ]) onMessage(wrong);
  assert.equal(c.voiceTiming.events.length, before);
  onMessage(message);
  onMessage(message);
  assert.equal(c.voiceTiming.events.length, before + 1);
  assert.equal(c.voiceTiming.events.at(-1).type, "preview_activated");
  const snapshot = c.window.bookforgeVoiceTiming();
  snapshot.events[0].type = "mutated";
  assert.equal(c.voiceTiming.events[0].type, "recording_started");

  const oldEpoch = c.recordingEpoch;
  c.recordingEpoch += 1;
  c.voiceTiming = {epoch: c.recordingEpoch, startedAt: h.now(), events: [], jobIds: new Set()};
  // The old completed pointer can remain visible during a new recording, but
  // its late display/completion cannot become that recording's latency result.
  onMessage(message);
  c.voiceTimingEvent("generation_completed", {jobId: "job-1"});
  c.voiceTimingEvent("generation_response", {status: 202, jobId: "job-2"}, oldEpoch);
  assert.equal(c.voiceTiming.events.length, 0);
  assert.equal(c.voiceTiming.jobIds.size, 0);
  c.voiceTimingEvent("generation_response", {status: 202, jobId: "job-2"});
  c.voiceTimingEvent("generation_completed", {jobId: "job-2", serverMs: 300});
  assert.equal(c.voiceTiming.events.length, 2);
  assert.equal(c.voiceTiming.events.at(-1).serverMs, 300);
  c.resetMicControls();
}

async function recorderFlush() {
  async function setup() {
    const h = generationHarness();
    const c = h.context;
    await c.startSpeaking();
    h.state.rejectCheck = true;
    const recorder = c.mediaRecorder;
    recorder.handlers.dataavailable({data: new Blob(["a".repeat(1200)])});
    const respond = c.respond;
    h.asrBytes = [];
    c.respond = (url, options) => {
      if (url !== "/v1/audio:transcribe") return respond(url, options);
      h.asrBytes.push(options.body.size);
      assert.equal(recorder.listeners.dataavailable.length, 1); // Only the original chunk collector remains.
      return h.response(200, {text: "A cat chasing a mouse.", total_ms: 50});
    };
    h.flushRequests = [];
    recorder.requestData = () => { h.flushRequests.push(h.now()); };
    return {...h, recorder};
  }
  const h = await setup();
  const c = h.context;
  c.partialInFlight = c.transcribePartialRecording(c.recordingEpoch);
  await flush();
  assert.equal(c.partialBusy, true);
  assert.equal(h.recorder.listeners.dataavailable.length, 2);
  assert.equal(h.flushRequests.length, 1);
  assert.deepEqual(h.asrBytes, []);
  await c.transcribePartialRecording(c.recordingEpoch);
  assert.equal(h.flushRequests.length, 1); // Flush waiting already owns ASR admission.
  h.recorder.handlers.dataavailable({data: new Blob(["b".repeat(150)])});
  await c.partialInFlight;
  assert.deepEqual(h.asrBytes, [1350]); // Original collector ran before the flush listener.
  assert.equal(h.recorder.listeners.dataavailable.length, 1);
  assert.equal([...h.timers.values()].some((timer) => timer.milliseconds === 250), false);
  c.resetMicControls();

  const stopping = await setup();
  const sc = stopping.context;
  sc.partialInFlight = sc.transcribePartialRecording(sc.recordingEpoch);
  await flush();
  sc.stopSpeaking();
  assert.equal(stopping.events.filter((event) => event === "track-stop").length, 1);
  // The requested chunk precedes MediaRecorder's final data and stop event.
  stopping.recorder.handlers.dataavailable({data: new Blob(["b".repeat(100)])});
  stopping.recorder.handlers.dataavailable({data: new Blob(["c".repeat(200)])});
  const final = stopping.recorder.handlers.stop();
  await final;
  assert.deepEqual(stopping.asrBytes, [1500]); // No stale partial POST, one complete final POST.
  assert.equal(sc.partialBusy, false);
  assert.equal(sc.finalizing, false);
  assert.equal(stopping.recorder.listeners.dataavailable.length, 1);
  assert.equal(stopping.submitted.length, 0);

  const timeout = await setup();
  const pending = timeout.context.transcribePartialRecording(timeout.context.recordingEpoch);
  await timeout.advance(249);
  assert.deepEqual(timeout.asrBytes, []);
  await timeout.advance(1);
  await pending;
  assert.deepEqual(timeout.asrBytes, [1200]); // Bounded fallback uses existing audio.
  assert.equal(timeout.context.partialBusy, false);
  assert.equal(timeout.recorder.listeners.dataavailable.length, 1);
  timeout.recorder.handlers.dataavailable({data: new Blob(["late".repeat(100)])});
  await flush();
  assert.equal(timeout.context.audioChunks.length, 2); // Late data is retained, never a second ASR dispatch.
  assert.deepEqual(timeout.asrBytes, [1200]);
  timeout.context.resetMicControls();
}

async function finalRefusalStatus() {
  const h = generationHarness();
  const c = h.context;
  const text = "The English cream golden retriever ran down the cobblestone path in the city of Paris.";
  const detail = "reviewed_description_unsupported: Clarify the actor, action and place.";
  const respond = c.respond;
  let refuse = true;
  c.respond = (url, options) => {
    if (url === "/v1/audio:transcribe") return h.response(200, {text, total_ms: 50});
    if (url === "/v1/live-scene-planner/prepare" && refuse) {
      h.checks.push(JSON.parse(options.body));
      return h.response(422, {detail: {message: detail}});
    }
    return respond(url, options);
  };
  await c.startSpeaking();
  const recorder = c.mediaRecorder;
  recorder.handlers.dataavailable({data: new Blob(["a".repeat(1200)])});
  await c.transcribePartialRecording(c.recordingEpoch);
  await h.advance(350);
  assert.equal(h.checks.length, 1);
  assert.equal(h.submitted.length, 0);
  assert.equal(c.voiceGeneration.latest.refusal, detail);
  // Another identical partial keeps the same refusal and does not recheck it.
  recorder.handlers.dataavailable({data: new Blob(["b".repeat(100)])});
  await c.transcribePartialRecording(c.recordingEpoch);
  await h.advance(350);
  assert.equal(h.checks.length, 1);
  c.stopSpeaking();
  await recorder.handlers.stop();
  await flush();
  assert.equal(c.finalizing, false);
  assert.equal(h.checks.length, 1);
  assert.equal(h.submitted.length, 0);
  assert.deepEqual(h.displayed, []);
  assert.equal(h.elements.voiceReview.textContent, detail);
  assert.match(h.elements.interim.textContent, /could not be verified.*previous scene is unchanged/i);
  assert.doesNotMatch(h.elements.interim.textContent, /Finishing/);
  assert.equal(h.elements.compileButton.textContent, "Check description again");
  assert.equal(h.elements.compileButton.disabled, false);
  refuse = false;
  c.offerVoiceTranscript("A dog chasing a ball.", {final: true});
  await flush();
  assert.equal(c.voiceGeneration.latest.refusal, null);
  assert.equal(h.checks.length, 2);
  assert.equal(h.submitted.length, 1);
  assert.equal(h.submitted[0].text, "A dog chasing a ball.");
}

async function staticPartialAdmission() {
  function staticHarness() {
    const h = generationHarness();
    h.context.listening = true;
    const ready = h.state.ready;
    h.state.ready = (request) => {
      const result = ready(request);
      if (request.text === "A cat") {
        result.visual_facts.subjects[0].actions = [];
        result.visual_facts.objects = [];
      }
      return result;
    };
    return h;
  }
  const correction = staticHarness();
  correction.context.offerVoiceTranscript("A cat");
  await correction.advance(350);
  assert.equal(correction.checks.length, 1);
  assert.equal(correction.submitted.length, 0);
  assert.equal(correction.context.voiceGeneration.attempted.size, 0);
  assert.doesNotMatch(correction.elements.voiceReview.textContent, /cat:/);
  correction.context.offerVoiceTranscript("A cat chasing a mouse.");
  await correction.advance(350);
  assert.equal(correction.submitted.length, 1);
  assert.equal(correction.submitted[0].text, "A cat chasing a mouse.");

  const stable = staticHarness();
  stable.context.offerVoiceTranscript("A cat");
  await stable.advance(350);
  assert.equal(stable.submitted.length, 0);
  stable.context.offerVoiceTranscript("A cat");
  await stable.advance(350);
  assert.equal(stable.submitted.length, 0);
  await stable.context.pumpVoiceGeneration();
  assert.equal(stable.submitted.length, 0);
  assert.match(stable.elements.interim.textContent, /finish recording to generate this subject/);

  const final = staticHarness();
  final.context.offerVoiceTranscript("A cat");
  await final.advance(350);
  final.context.finalizing = true;
  final.context.offerVoiceTranscript("A cat", {final: true});
  assert.equal(final.submitted.length, 0);
  final.context.finalizing = false;
  await final.context.pumpVoiceGeneration();
  assert.equal(final.submitted.length, 1);

  const manual = staticHarness();
  manual.context.listening = false;
  manual.elements.story.value = "A cat";
  await manual.context.compileStory();
  assert.equal(manual.submitted.length, 1); // Explicit Generate is already final intent.

  // Repetition arriving during a local check is still not a final utterance.
  // It must neither purchase an image nor leave a stuck attempted claim.
  const during = staticHarness();
  const respond = during.context.respond;
  let finishCheck;
  during.context.respond = (url, options) => url === "/v1/live-scene-planner/prepare"
    ? new Promise((resolve) => { finishCheck = () => resolve(respond(url, options)); })
    : respond(url, options);
  during.context.offerVoiceTranscript("A cat");
  await during.advance(350);
  during.context.offerVoiceTranscript("A cat");
  finishCheck();
  await flush();
  assert.equal(during.submitted.length, 0);
  assert.equal(during.context.voiceGeneration.attempted.size, 0);
}

(async () => { await lifecycle(); await voiceToScene(); await partialScheduling(); await adaptiveCadence(); await timingIsolation(); await recorderFlush(); await finalRefusalStatus(); await staticPartialAdmission(); })().then(() => console.log("Workbench microphone: cleanup, recorder flush, adaptive ASR cadence, latest-only presentation and no duplicate paid requests passed."))
  .catch((error) => { console.error(error); process.exitCode = 1; });
