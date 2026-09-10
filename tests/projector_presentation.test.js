const assert = require("assert").strict;
const fs = require("fs");
const vm = require("vm");
const source = fs.readFileSync("src/storylight/static/projector.js", "utf8");

async function presentationGate() {
  const previous = {title: "previous", pages: [{page_id: "old"}], assets: []};
  const pack = {title: "candidate", pages: [{page_id: "new"}], assets: []};
  const state = {
    pack: previous, page: previous.pages[0], liveServerInstanceId: "server", liveSessionRevision: 2,
    liveSessionJobId: "job", liveAcceptedRevision: -1, liveTransition: Promise.resolve(),
    liveLastEnvelopeAt: 0,
  };
  const activations = [];
  const frameMessages = [];
  const events = [];
  const timers = new Set();
  let now = 0;
  let finishActivation;
  const c = {state, LIVE_MODE: true, COMPLETE_ONLY: true, SESSION_ID: "voice", SCENE_CROSSFADE_MS: 320,
    window: {location: {origin: "http://localhost"},
      clearInterval(id) { timers.delete(id); },
      setInterval() { const id = Symbol(); timers.add(id); return id; }, parent: {
      postMessage(data, origin) { frameMessages.push({data, origin}); },
    }},
    elements: {packLabel: {textContent: "previous"},
      liveGenerationBadge: {dataset: {}, classList: {remove() {}, toggle() {}}},
      liveGenerationStage: {}, liveGenerationDetail: {}, liveGenerationElapsed: {},
    }, performance: {now: () => now},
    setEvent(type) { events.push(type); }, publish() {}, liveProviderLabel: () => "fixture renderer",
    assertStoryPack: (value) => value, livePageAssetFingerprint: () => "new-media",
    liveRenderTokenIsCurrent: () => true,
    invalidateLiveRender(value) {
      state.liveAcceptedJobId = value.jobId; state.liveAcceptedRevision = value.revision;
      return {...value, signal: {aborted: false}};
    },
    async activatePage(index, token, value) {
      activations.push({index, token, pack: value});
      await new Promise((resolve) => { finishActivation = resolve; });
      state.pack = value;
      return "depth-composed";
    },
  };
  vm.createContext(c);
  for (const [start, end] of [
    ["function liveArtifactRoles(", "function updateLiveGenerationClock("],
    ["function updateLiveGenerationClock(", "function livePageAssetFingerprint("],
    ["function liveSnapshotAwaitsPresentation(", "function liveSnapshotIsTerminal("],
  ]) vm.runInContext(source.slice(source.indexOf(start), source.indexOf(end)), c);
  const snapshot = {job_id: "job", story_pack: pack, request: {
    display_when_complete: true, defer_presentation: true,
  }, metrics: {elapsed_ms: 1500}, artifacts: [{kind: "master"}, {kind: "depth"}]};
  let revision = 0;
  for (const delta of [
    {stage: "draft_ready", complete: false},
    {stage: "master_ready", complete: false},
    {stage: "master_ready", complete: true, presentation_ready: false},
    {stage: "failed", complete: true, presentation_ready: false},
  ]) {
    c.queueLiveSceneSnapshot({type: "storylight.live-scene", sessionId: "voice",
      serverInstanceId: "server", sessionRevision: 2,
      snapshot: {...snapshot, ...delta, revision: ++revision}});
    await state.liveTransition;
    assert.equal(activations.length, 0);
    assert.equal(frameMessages.length, 0);
    assert.equal(state.pack, previous);
    assert.equal(c.elements.packLabel.textContent, "previous");
    if (delta.complete && delta.stage === "master_ready") {
      assert.equal(c.elements.liveGenerationStage.textContent, "Waiting for verified description");
      assert.match(c.elements.liveGenerationDetail.textContent, /held for presentation approval/);
      assert.doesNotMatch(c.elements.liveGenerationDetail.textContent, /retrying/);
      assert.equal(events.at(-1), "scene.awaiting-presentation");
      assert.equal(timers.size, 0);
      assert.equal(c.elements.liveGenerationBadge.dataset.clockRunning, "false");
      now += 10000;
      c.updateLiveGenerationClock();
      assert.equal(c.elements.liveGenerationElapsed.textContent, "1.5 s");
      c.queueLiveSceneSnapshot({type: "storylight.live-scene", sessionId: "voice",
        serverInstanceId: "server", sessionRevision: 2,
        snapshot: {...snapshot, ...delta, revision}});
      await state.liveTransition;
      assert.equal(activations.length, 0);
      assert.equal(timers.size, 0);
      assert.equal(c.elements.liveGenerationStage.textContent, "Waiting for verified description");
    }
  }
  c.queueLiveSceneSnapshot({type: "storylight.live-scene", sessionId: "voice",
    serverInstanceId: "server", sessionRevision: 2,
    snapshot: {...snapshot, revision: ++revision, stage: "master_ready", complete: true,
      presentation_ready: true}});
  await new Promise(setImmediate);
  assert.equal(frameMessages.length, 0);
  assert.equal(state.pack, previous);
  assert.equal(c.elements.liveGenerationStage.textContent, "Loading artwork + depth");
  assert.match(c.elements.liveGenerationDetail.textContent, /media retrying/);
  finishActivation();
  await state.liveTransition;
  assert.equal(activations.length, 1);
  assert.equal(activations[0].token.displayWhenComplete, true);
  assert.equal(state.pack, pack);
  assert.equal(c.elements.liveGenerationStage.textContent, "Artwork + depth live");
  assert.doesNotMatch(c.elements.liveGenerationDetail.textContent, /held|retrying/);
  assert.equal(timers.size, 0);
  assert.deepEqual(JSON.parse(JSON.stringify(frameMessages)), [{
    data: {type: "storylight.preview-activated", sessionId: "voice", jobId: "job"},
    origin: "http://localhost",
  }]);
  assert.equal(c.liveSnapshotCanDisplay({...snapshot, request: {}, stage: "draft_ready"}), false);
  assert.equal(c.liveSnapshotCanDisplay({...snapshot, request: {}, stage: "master_ready",
    complete: true, artifacts: [{kind: "master"}]}), false);
}

async function noIncompleteFallback() {
  const node = () => ({style: {setProperty() {}}, dataset: {}, append() {}});
  const commits = [];
  let discarded = 0;
  const c = {
    document: {createElement: node}, performance: {now: () => 0},
    createSceneVersion: node, discardSceneVersion() { discarded += 1; },
    commitSceneVersion(_version, mode) { commits.push(mode); return true; },
    liveRenderTokenIsCurrent: () => true, appendSceneHotspots() {},
    sizeDepthCanvasToSource: () => ({}),
    loadSceneImage: async (uri) => {
      if (uri.endsWith("depth")) throw new Error("depth unavailable");
      return node();
    },
    startDepthRenderer: async () => ({projectionExposure: 1, projectionGamma: 1,
      projectionMeanLuma: null}),
  };
  vm.runInNewContext(source.slice(source.indexOf("async function renderPackLayers("),
    source.indexOf("function renderTimeline(")), c);
  const assets = ["master", "depth"].map((role) => ({page_id: "p", state: "ready",
    kind: role === "master" ? "image" : "depth_map", role, local_uri: `/v1/assets/${role}`}));
  const page = {page_id: "p", scene_summary: "new scene", scene_spec: {}};
  const token = {displayWhenComplete: true};
  await assert.rejects(c.renderPackLayers({assets: assets.slice(0, 1)}, page, token), /not available/);
  await assert.rejects(c.renderPackLayers({assets}, page, token), /depth unavailable/);
  assert.equal(commits.length, 0);
  assert.equal(discarded, 1);
  let finishDepth;
  c.loadSceneImage = (uri) => uri.endsWith("depth")
    ? new Promise((resolve) => { finishDepth = resolve; }) : Promise.resolve(node());
  const pending = c.renderPackLayers({assets}, page, token);
  await new Promise(setImmediate);
  assert.equal(commits.length, 0);
  finishDepth(node());
  assert.equal(await pending, "depth-composed");
  assert.deepEqual(commits, ["depth-composed"]);
}

(async () => { await presentationGate(); await noIncompleteFallback(); })()
  .then(() => console.log("Projector deferred presentation and complete media transaction passed."))
  .catch((error) => { console.error(error); process.exitCode = 1; });
