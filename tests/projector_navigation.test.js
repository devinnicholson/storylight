const assert = require("assert").strict;
const fs = require("fs");
const vm = require("vm");

const source = fs.readFileSync("src/bookforge/static/projector.js", "utf8");
const navigation = source.slice(
  source.indexOf("function updatePageControls()"),
  source.indexOf("function liveProviderLabel("),
);

function harness() {
  const pages = ["before", "after", "next-page"].map((page_id) => ({
    page_id, source_text: page_id, triggers: [],
  }));
  const pendingReaderEvents = [{page_id: "before"}];
  const state = {
    pack: {pages, assets: []}, page: pages[0], pageIndex: 0,
    cursor: 3, tokens: ["old-word"], triggerIndices: new Map(),
    pendingReaderEvents, pageTransition: null,
  };
  const controls = {
    pageLabel: {textContent: "Page 1 of 3"},
    previousPage: {disabled: true}, nextPage: {disabled: false},
  };
  const loads = [];
  const events = [];
  let visiblePage = "before";
  let cleared = 0;
  let url = "http://localhost/projector?page=1";
  const context = {
    state, elements: controls, READER_MODE: false, URL,
    performance: {now: () => 100},
    window: {
      location: {href: url},
      history: {replaceState: (_state, _title, value) => { url = String(value); }},
    },
    liveRenderTokenIsCurrent: (token) => !token?.stale,
    tokenize: (text) => text.split(" "),
    indexTriggers: () => new Map(),
    clearLayerState: () => { cleared += 1; },
    renderTimeline() {}, rebuildScene() {},
    publish: (type, detail) => events.push({type, detail}),
    setEvent: (type, detail) => events.push({type, detail}),
    renderPackLayers: (_pack, page) => new Promise((resolve, reject) => {
      loads.push({
        page,
        reject,
        commit() { visiblePage = page.page_id; resolve("hero-composed"); },
        refuse() { resolve(false); },
      });
    }),
  };
  vm.runInNewContext(navigation, context);
  return {
    context, state, pages, pendingReaderEvents, controls, loads, events,
    visible: () => visiblePage, clears: () => cleared, url: () => url,
  };
}

(async () => {
  const app = harness();
  const failed = app.context.requestPage(1);
  assert.equal(app.context.requestPage(2), failed);
  assert.equal(app.loads.length, 1);
  assert.equal(app.state.page, app.pages[0]);
  assert.equal(app.state.pageIndex, 0);
  assert.equal(app.state.cursor, 3);
  assert.equal(app.state.pendingReaderEvents, app.pendingReaderEvents);
  assert.equal(app.controls.pageLabel.textContent, "Page 1 of 3");
  assert.equal(app.clears(), 0);
  assert.equal(app.visible(), "before");
  app.loads[0].reject(new Error("cached after image unavailable"));
  await failed;
  assert.equal(app.state.page, app.pages[0]);
  assert.equal(app.state.cursor, 3);
  assert.equal(app.state.pageTransition, null);
  assert.equal(app.url(), "http://localhost/projector?page=1");
  assert.equal(app.events.filter((event) => event.type === "page.loaded").length, 0);

  const retried = app.context.requestPage(app.state.pageIndex + 1);
  assert.equal(app.loads[1].page.page_id, "after");
  app.loads[1].commit();
  await retried;
  assert.equal(app.visible(), "after");
  assert.equal(app.state.page, app.pages[1]);
  assert.equal(app.state.pageIndex, 1);
  assert.equal(app.state.cursor, -1);
  assert.equal(app.controls.pageLabel.textContent, "Page 2 of 3");
  assert.equal(app.controls.previousPage.disabled, false);
  assert.equal(app.clears(), 1);
  assert.equal(new URL(app.url()).searchParams.get("page"), "2");

  const refused = app.context.requestPage(2);
  app.loads[2].refuse();
  await refused;
  assert.equal(app.state.page, app.pages[1]);
  assert.equal(app.visible(), "after");
  assert.equal(app.clears(), 1);
  console.log("projector navigation transaction passed");
})().catch((error) => { console.error(error); process.exitCode = 1; });
