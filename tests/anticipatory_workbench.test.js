const assert = require("assert").strict;
const fs = require("fs");
const vm = require("vm");

function harness(respond, enabled = true) {
  const elements = new Map();
  const requests = [];
  const timers = new Map();
  const delays = [];
  const shown = [];
  const context = {
    document: {getElementById(id) {
      if (!elements.has(id)) elements.set(id, {
        value: "A silver fox enters the garden.", disabled: false, textContent: "", events: {},
        addEventListener(name, fn) { this.events[name] = fn; },
      });
      return elements.get(id);
    }},
    window: {
      setTimeout(fn, ms) { const id = Symbol(); timers.set(id, fn); delays.push(ms); return id; },
      clearTimeout(id) { timers.delete(id); }, addEventListener() {},
    },
    fetch: async (path, options) => {
      const request = {path, method: options.method, body: options.body && JSON.parse(options.body)};
      requests.push(request);
      const result = path.endsWith("/runtime")
        ? {enabled, server_instance_id: "server_fixture"} : respond(request);
      const status = result.status || 200;
      return {ok: status < 400, status, json: async () => result};
    },
    Date,
  };
  vm.runInNewContext(fs.readFileSync("src/bookforge/static/anticipatory-workbench.js", "utf8"), context);
  return {
    requests, timers, shown, delays,
    element: (name) => elements.get(`nextPage${name}`),
    start: () => context.window.BookforgeAnticipation.init({
      sessionId: "reader", visualStyle: () => "watercolor",
      currentProjection: () => ({server_instance_id: "server_fixture", session_revision: 7}),
      onShow: (pointer) => shown.push(pointer),
    }),
  };
}

const snapshot = (state) => ({
  prepared_id: "prepared_" + "a".repeat(24), state, detail: state,
  planning_ms: 12, render_ms: 400, critic_ms: 6200, assets_verified: state === "staged",
});

(async () => {
  let current = "preparing";
  const app = harness(({path, method, body}) => {
    if (path.endsWith(":activate")) {
      assert.equal(body.expected_session_revision, 7);
      return {session_revision: 8};
    }
    if (method === "DELETE") return {status: 204};
    if (path.endsWith("/stage")) return snapshot("staged");
    return snapshot(current);
  });
  await app.start();
  assert.equal(app.requests.length, 1); // Opening the panel does not warm or generate.
  assert.equal(app.element("Show").disabled, true);
  await app.element("Prepare").events.click();
  assert.equal(app.timers.size, 1);
  assert.deepEqual(app.delays, [100]);
  assert.ok(app.requests[app.requests.length - 1].path.endsWith("?wait_seconds=20"));
  assert.equal(app.element("Text").disabled, true);
  assert.equal(app.shown.length, 0);
  current = "approved";
  await [...app.timers.values()][0]();
  assert.equal(app.element("Show").disabled, false);
  assert.equal(app.requests.filter((r) => r.path.endsWith("/prepared-projections") && r.method === "POST").length, 1);
  assert.equal(app.shown.length, 0); // Even staging does not switch the projection.
  const before = app.requests.length;
  await app.element("Show").events.click();
  assert.equal(app.requests.length, before + 1);
  assert.equal(app.shown.length, 1);
  assert.equal(app.element("Show").disabled, true);
  await app.element("Discard").events.click();
  assert.equal(app.element("Text").disabled, false);

  const disconnected = harness(() => { throw Error("unexpected cloud request"); }, false);
  await disconnected.start();
  assert.equal(disconnected.element("Prepare").disabled, true);
  assert.equal(disconnected.element("Warm").disabled, true);
  assert.equal(disconnected.requests.length, 1);

  const conflict = harness(({path}) => path.endsWith(":activate")
    ? {status: 409, detail: "Projection changed"} : snapshot("staged"));
  await conflict.start();
  await conflict.element("Prepare").events.click();
  await conflict.element("Show").events.click();
  assert.equal(conflict.shown.length, 0);
  assert.equal(conflict.element("Status").textContent, "Projection changed");
  assert.equal(conflict.timers.size, 0); // Errors never schedule another billable operation.

  const cached = harness(({path}) => snapshot(path.endsWith("/stage") ? "staged" : "approved"));
  await cached.start();
  await cached.element("Prepare").events.click();
  assert.equal(cached.requests.length, 3); // Runtime, prepare cache hit, and stage; no status round trip.
  assert.equal(cached.element("Show").disabled, false);
  assert.equal(cached.timers.size, 0);
  console.log("Next-page workbench: lifecycle, polling, disabled setup, and conflict checks passed.");
})().catch((error) => { console.error(error); process.exitCode = 1; });
