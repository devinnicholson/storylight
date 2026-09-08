const assert = require("assert").strict;
const fs = require("fs");
const vm = require("vm");

function harness() {
  let now = 0;
  let finishPlanner;
  const requests = [];
  const timers = new Map();
  const planner = new Promise((resolve) => { finishPlanner = resolve; });
  const elements = {
    story: {value: "A fox carries a lantern."}, style: {value: "watercolor"},
    prewarmButton: {}, rendererPreflight: {dataset: {}},
    rendererReadiness: {}, rendererReadinessDetail: {},
  };
  const context = {
    voiceMode: false,
    elements, readerSessionId: "synthetic-reader", rendererPrewarming: false,
    rendererWarmUntil: 0, rendererWarmExpiryTimer: null, preparedPlanKey: null,
    Date: {now: () => now},
    window: {
      clearTimeout(id) { timers.delete(id); },
      setTimeout(callback, milliseconds) {
        const id = Symbol();
        timers.set(id, {callback, deadline: now + milliseconds});
        return id;
      },
    },
    async fetch(url) {
      requests.push(url);
      if (url.endsWith("/prewarm")) {
        return {ok: true, json: async () => ({expires_in_seconds: 90})};
      }
      assert.ok(url.endsWith("/prepare"));
      return planner;
    },
  };
  const source = fs.readFileSync("src/bookforge/static/workbench.js", "utf8");
  vm.createContext(context);
  // Run the real readiness flow without starting the page's microphone/projector wiring.
  for (const [start, end] of [
    ["function setRendererReadiness(", "async function warmEdgePlanner("],
    ["async function prewarmRenderer(", "function invalidatePreparation("],
  ]) vm.runInContext(source.slice(source.indexOf(start), source.indexOf(end)), context);
  return {
    context, elements, requests, timers,
    async start() {
      const completion = context.prewarmRenderer();
      await new Promise(setImmediate); // Renderer response and its expiry capture finish first.
      return {completion};
    },
    advance(milliseconds) {
      now += milliseconds;
      for (const [id, timer] of timers) {
        if (timer.deadline <= now) { timers.delete(id); timer.callback(); }
      }
    },
    finish(ok = true) {
      finishPlanner({ok, status: ok ? 200 : 503, json: async () => (
        ok ? {planning_ms: now} : {detail: "Local planner unavailable"}
      )});
    },
  };
}

async function slowPlannerPreservesRendererExpiry() {
  const app = harness();
  const {completion} = await app.start();
  app.advance(111550);
  app.finish();
  await completion;
  assert.equal(app.context.rendererWarmUntil, 0);
  assert.equal(app.elements.rendererPreflight.dataset.state, "idle");
  assert.equal(app.context.preparedPlanKey, app.elements.story.value);
  assert.equal(app.elements.prewarmButton.disabled, false);
  assert.equal(app.timers.size, 0);
  assert.equal(app.requests.filter((url) => url.endsWith("/prewarm")).length, 1);

  const unexpired = harness();
  const pending = await unexpired.start();
  unexpired.advance(30000);
  unexpired.finish();
  await pending.completion;
  assert.equal(unexpired.context.rendererWarmUntil, 90000);
  assert.equal([...unexpired.timers.values()][0].deadline, 90000);
  assert.equal(unexpired.elements.rendererPreflight.dataset.state, "ready");
}

async function warmReuseDoesNotPrewarmOrExtendExpiry() {
  const app = harness();
  app.context.markRendererReady(90000);
  app.advance(30000);
  const {completion} = await app.start();
  app.advance(20000);
  app.finish();
  await completion;
  assert.equal(app.context.rendererWarmUntil, 90000);
  assert.equal([...app.timers.values()][0].deadline, 90000);
  assert.deepEqual(app.requests, ["/v1/live-scene-planner/prepare"]);
  app.advance(40000);
  assert.equal(app.elements.rendererPreflight.dataset.state, "idle");

  const expired = harness();
  expired.context.markRendererReady(90000);
  const pending = await expired.start();
  expired.advance(111550);
  expired.finish(false);
  await pending.completion;
  assert.equal(expired.context.rendererWarmUntil, 0);
  assert.equal(expired.elements.rendererPreflight.dataset.state, "idle");
  assert.equal(expired.timers.size, 0);
  assert.deepEqual(expired.requests, ["/v1/live-scene-planner/prepare"]);
}

async function generateDoesNotWaitForBackgroundPlannerWarmup() {
  const app = harness();
  const {context, elements, requests} = app;
  let finishWarmup;
  const warmup = new Promise((resolve) => { finishWarmup = resolve; });
  const submitted = [];
  elements.compileButton = {dataset: {}};
  elements.error = {classList: {add() {}, remove() {}}};
  elements.interim = {};
  Object.assign(context, {
    edgePlannerWarmUntil: 0, EDGE_PLANNER_KEEP_WARM_MS: 480000,
    edgePlanPreparationTimer: null, listening: false, liveRequestEpoch: 0,
    starting: false, finalizing: false, generationSubmitting: false, activeLiveJobId: null,
    pendingSubmission: null, latestLiveSnapshot: null, AbortController,
    updateMicAvailability() {},
    fetchLiveSceneSession: async () => null,
    performance: {now: () => 0},
    stopLiveJobTransport() {}, setSceneInputsDisabled() {},
    renderGenerationProgress() {}, updateElapsedClock() {}, ensureProjectionPreview() {},
    setStatus() {},
    acceptedLiveScenePointer: (_response, snapshot) => snapshot,
    handleLiveSceneSessionPointer(snapshot) { context.activeLiveJobId = snapshot.job_id; },
    async fetch(url, options) {
      requests.push(url);
      if (url.endsWith("/warmup")) return warmup;
      assert.equal(url, "/v1/live-scenes");
      submitted.push(JSON.parse(options.body));
      return {status: 202, json: async () => ({job_id: "scene-cached"})};
    },
  });
  context.window.setInterval = () => 1;
  const source = fs.readFileSync("src/bookforge/static/workbench.js", "utf8");
  for (const [start, end] of [
    ["async function warmEdgePlanner(", "function scheduleEdgePlanPreparation("],
    ["async function compileStory(", "async function loadLatestScene("],
  ]) vm.runInContext(source.slice(source.indexOf(start), source.indexOf(end)), context);

  const background = context.warmEdgePlanner();
  const generation = context.compileStory();
  await new Promise(setImmediate);
  assert.deepEqual(requests, ["/v1/live-scene-planner/warmup", "/v1/live-scenes"]);
  assert.deepEqual(submitted, [{
    text: elements.story.value, visual_style: elements.style.value, session_id: context.readerSessionId,
  }]);
  await generation;
  assert.equal(context.activeLiveJobId, "scene-cached");
  assert.equal(context.edgePlannerWarmUntil, 0);
  finishWarmup({ok: true, json: async () => ({ready: true})});
  assert.equal(await background, true);
  assert.equal(context.edgePlannerWarmUntil, context.EDGE_PLANNER_KEEP_WARM_MS);
}

(async () => {
  await slowPlannerPreservesRendererExpiry();
  await warmReuseDoesNotPrewarmOrExtendExpiry();
  await generateDoesNotWaitForBackgroundPlannerWarmup();
  console.log("Workbench readiness: expiry preserved; Generate does not await background warmup.");
})().catch((error) => { console.error(error); process.exitCode = 1; });
