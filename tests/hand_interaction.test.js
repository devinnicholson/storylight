const assert = require("assert").strict;
const fs = require("fs");
const vm = require("vm");

function harness() {
  let now = 0, blocked = false, cameraRequests = 0, bitmapRequests = 0, stopped = 0;
  let getCamera = async () => stream;
  const timers = new Map(), frames = new Map(), workers = [], events = {}, nodes = new Map();
  const stream = {getTracks: () => [track], getVideoTracks: () => [track]};
  const cameraSettings = {deviceId: "test", width: 640, height: 480}, trackEvents = {};
  let projection = "projection", stored = null;
  const track = {stop() { stopped++; }, getSettings: () => cameraSettings, addEventListener(type, fn) { trackEvents[type] = fn; }};
  const context2d = {createRadialGradient: () => ({addColorStop() {}}), fillRect() {}, clearRect() {}, drawImage() {}};
  function node(id) {
    if (!nodes.has(id)) nodes.set(id, {
      hidden: true, events: {}, style: {}, width: 960, height: 540, readyState: 4, videoWidth: 640, videoHeight: 480,
      addEventListener(type, fn) { this.events[type] = fn; }, getContext: () => context2d,
      play: async () => {}, getBoundingClientRect: () => ({left: 0, top: 0, width: 400, height: 300}),
      querySelectorAll: () => [], appendChild() {},
    });
    return nodes.get(id);
  }
  class Worker {
    constructor() { this.posts = []; workers.push(this); }
    postMessage(message) { this.posts.push(message); }
    terminate() { this.terminated = true; }
    send(data) { this.onmessage({data}); }
  }
  const schedule = map => (fn, ms) => { const id = Symbol(); map.set(id, {fn, ms}); return id; };
  const context = {
    window: {isSecureContext: true, Worker, OffscreenCanvas: true, createImageBitmap: true,
      addEventListener(type, fn) { events[type] = fn; }},
    Worker,
    navigator: {mediaDevices: {getUserMedia() { cameraRequests++; return getCamera(); }}},
    document: {hidden: false, body: {classList: {add() {}, remove() {}}}, getElementById: node, createElement: node,
      addEventListener(type, fn) { events[type] = fn; }},
    performance: {now: () => now},
    localStorage: {getItem: () => stored, setItem(key, value) { stored = value; }},
    fetch: async () => ({ok: true, json: async () => ({version: "0.10.32"})}),
    createImageBitmap: async () => { bitmapRequests++; return {close() {}}; },
    setTimeout: schedule(timers), clearTimeout: id => timers.delete(id),
    requestAnimationFrame: schedule(frames), cancelAnimationFrame: id => frames.delete(id),
  };
  vm.runInNewContext(fs.readFileSync("src/bookforge/static/hand-interaction.js", "utf8"), context);
  const app = context.window.BookforgeHands.init({stage: node("stage"), solve: () => [1, 0, 0, 0, 1, 0, 0, 0], screenToStage: () => ({x: .5, y: .5}), blocked: () => blocked, projectionIdentity: () => projection});
  async function flush() { for (let i = 0; i < 12; i++) await Promise.resolve(); }
  async function tick(value) {
    now = value;
    const pending = [...frames.values()]; frames.clear(); pending.forEach(({fn}) => fn(now));
    await flush();
  }
  return {app, context, workers, timers, frames, events, node, tick, flush,
    click: id => node(id).events.click(),
    counters: () => ({cameraRequests, bitmapRequests, stopped}),
    block: value => { blocked = value; }, camera: value => { getCamera = value; }, stream,
    cameraSettings, trackEvents, projection: value => { projection = value; }, storage: value => { stored = value; },
  };
}

(async () => {
  const h = harness();
  assert.equal(h.app.metrics.mode, "off");
  assert.equal(h.workers.length, 0);
  assert.equal(h.timers.size + h.frames.size, 0);
  assert.equal(h.counters().cameraRequests, 0);
  const {validCorners, project} = h.context.window.BookforgeHands;
  const square = [{x: .1, y: .1}, {x: .9, y: .1}, {x: .9, y: .9}, {x: .1, y: .9}];
  assert.equal(validCorners(square), true);
  assert.equal(validCorners([square[0], square[2], square[1], square[3]]), false);
  assert.equal(validCorners(square.map(() => ({x: .1, y: .1}))), false);
  assert.equal(validCorners([{x: NaN, y: 0}, ...square.slice(1)]), false);
  assert.equal(validCorners([{x: -1, y: 0}, ...square.slice(1)]), false);
  assert.equal(validCorners([null, ...square.slice(1)]), false);
  assert.equal(project([1,0,0,0,1,0,-1,0], {x: 1, y: 0}), null);
  assert.equal(project([1,0,0,0,1,0,0,0], {x: .4, y: .6}).x, .4);

  h.click("handDemo"); assert.equal(h.app.metrics.mode, "pointer-demo");
  assert.equal(h.counters().cameraRequests, 0);
  h.click("handStop"); assert.equal(h.frames.size, 0);

  const starting = h.click("handStart"); await h.flush(); await h.tick(1100); await starting;
  assert.equal(h.workers.length, 1);
  const worker = h.workers[0];
  worker.send({type: "ready"}); await h.flush();
  assert.equal(h.app.metrics.mode, "camera");
  assert.equal(h.counters().bitmapRequests, 1);
  const poll = [...h.timers.values()].find(t => t.ms === 100);
  await poll.fn(); assert.equal(h.counters().bitmapRequests, 1); // one in flight, no queue
  worker.send({type: "result", id: 999, point: null, inferenceMs: 2});
  assert.equal(h.app.metrics.frames, 0); // mismatched worker result cannot release credit
  for (let i = 1; i <= 5; i++) {
    worker.send({type: "result", id: i, point: null, inferenceMs: 151});
    if (i < 5) await poll.fn();
  }
  assert.equal(h.app.metrics.mode, "off"); assert.ok(h.counters().stopped > 0);
  assert.match(h.node("handStatus").textContent, /150 ms/);
  assert.equal(worker.terminated, true);

  const race = harness(); let resolveCamera;
  race.camera(() => new Promise(resolve => { resolveCamera = resolve; }));
  const pending = race.click("handStart"); await race.flush(); await race.tick(1100);
  race.click("handStop"); resolveCamera(race.stream); await pending;
  assert.equal(race.app.metrics.mode, "off"); assert.equal(race.workers.length, 0);
  assert.equal(race.counters().stopped, 1); // late permission grant never leaks a camera

  const hidden = harness();
  const startHidden = hidden.click("handStart"); await hidden.flush(); await hidden.tick(1100); await startHidden;
  hidden.workers[0].send({type: "ready"}); await hidden.flush();
  hidden.context.document.hidden = true; hidden.events.visibilitychange();
  assert.equal(hidden.app.metrics.mode, "off"); assert.equal(hidden.counters().stopped, 1);

  async function startCamera(test) {
    const pending = test.click("handStart"); await test.flush(); await test.tick(1100); await pending;
    test.workers[test.workers.length - 1].send({type: "ready"}); await test.flush();
  }
  const initialization = harness();
  const initializing = initialization.click("handStart");
  await initialization.flush(); await initialization.tick(1100); await initializing;
  initialization.workers[0].onerror();
  assert.equal(initialization.counters().stopped, 1);
  assert.equal(initialization.workers[0].terminated, true);
  assert.equal(initialization.app.metrics.mode, "off");

  const disconnect = harness(); await startCamera(disconnect);
  disconnect.trackEvents.ended();
  assert.equal(disconnect.app.metrics.mode, "off");
  assert.match(disconnect.node("handStatus").textContent, /disconnected/);

  const changing = harness();
  changing.storage(JSON.stringify({identity: JSON.stringify(["test", 640, 480]), projection: "projection", points: square}));
  await startCamera(changing);
  changing.workers[0].send({type: "result", id: 1, point: {x: .5, y: .5}, inferenceMs: 10});
  changing.workers[0].send({type: "result", id: 1, point: null, inferenceMs: 10});
  assert.equal(changing.app.metrics.frames, 1); // duplicate result must not consume a second frame
  changing.projection("changed");
  await [...changing.timers.values()].find(t => t.ms === 100).fn();
  assert.match(changing.node("handStatus").textContent, /Projection alignment changed/);
  changing.cameraSettings.deviceId = "replacement";
  await [...changing.timers.values()].find(t => t.ms === 100).fn();
  assert.match(changing.node("handStatus").textContent, /Camera or resolution changed/);
  changing.click("handStop");

  const paused = harness(); await startCamera(paused);
  paused.workers[0].send({type: "result", id: 1, point: null, inferenceMs: 10});
  paused.block(true);
  await [...paused.timers.values()].find(t => t.ms === 100).fn();
  assert.equal(paused.counters().bitmapRequests, 1);
  paused.block(false);
  await [...paused.timers.values()].find(t => t.ms === 100).fn();
  assert.equal(paused.counters().bitmapRequests, 2);
  paused.click("handStop");

  const repeats = harness();
  for (let i = 0; i < 30; i++) {
    const pending = repeats.click("handStart"); await repeats.flush(); await repeats.tick((i + 1) * 1100); await pending;
    repeats.workers[i].send({type: "ready"}); await repeats.flush();
    repeats.click("handStop");
    assert.equal(repeats.timers.size + repeats.frames.size, 0);
    assert.equal(repeats.workers[i].terminated, true);
  }
  assert.equal(repeats.counters().stopped, 30);
  const soak = harness(); await startCamera(soak);
  for (let id = 1; id <= 1000; id++) {
    soak.workers[0].send({type: "result", id, point: null, inferenceMs: 10});
    const [timerId, timer] = [...soak.timers.entries()].find(([, t]) => t.ms === 100);
    soak.timers.delete(timerId); await timer.fn();
    await soak.tick(1100 + id * 17);
    assert.ok(soak.timers.size <= 2);
    assert.equal(soak.frames.size, 1);
  }
  assert.equal(soak.workers.length, 1);
  assert.equal(soak.app.metrics.frames, 1000);
  soak.click("handStop");
  assert.equal(soak.timers.size + soak.frames.size, 0);
  console.log("Hand interaction: opt-in, geometry, backpressure, performance stop, late-camera and hidden-tab cleanup passed.");
})().catch(error => { console.error(error); process.exitCode = 1; });
