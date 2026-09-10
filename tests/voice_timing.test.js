const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const {summarize} = require("../src/storylight/static/voice-timing.js");

const events = [
  {type: "recording_started", ms: 0},
  {type: "provider_preconnect_started", ms: 1},
  {type: "audio_activity", ms: 100},
  {type: "provider_preconnect_completed", ms: 800, failed: false},
  {type: "generation_requested", ms: 1200, submissionId: "old"},
  {type: "generation_response", ms: 1250, submissionId: "old", jobId: "old-job"},
  {type: "generation_requested", ms: 2200, submissionId: "new"},
  {type: "generation_response", ms: 2250, submissionId: "new", jobId: "new-job"},
  {type: "recording_stopped", ms: 3000},
  {type: "generation_completed", ms: 5200, jobId: "old-job"},
  {type: "generation_completed", ms: 6200, jobId: "new-job"},
  {type: "preview_activated", ms: 6400, jobId: "new-job"},
];
assert.deepEqual(summarize(events), {
  previewJobId: "new-job",
  recordingToPreviewMs: 6400,
  detectedAudioToPreviewMs: 6300,
  finishButtonToPreviewMs: 3400,
  requestToPreviewMs: 4200,
  completionToPreviewMs: 200,
  connectionPreparationMs: 799,
  connectionPreparationFailed: false,
});
// Incomplete/truncated traces must not invent latency or join a different job.
assert.equal(summarize(events.slice(0, -1)).requestToPreviewMs, null);
assert.equal(summarize(events.slice(5)).recordingToPreviewMs, null);
assert.equal(summarize(events.filter((e) => e.type !== "generation_response")).requestToPreviewMs, null);
assert.equal(summarize([...events, {type: "recording_stopped", ms: 7000}]).finishButtonToPreviewMs, null);
assert.equal(summarize([{type: "provider_preconnect_completed", ms: 10, failed: true}])
  .connectionPreparationFailed, true);
const panel = {hidden: true};
const output = {textContent: "", closest: () => panel};
const browser = {window: {}, document: {getElementById: () => output}};
vm.runInNewContext(fs.readFileSync("src/storylight/static/voice-timing.js", "utf8"), browser);
browser.window.renderStorylightVoiceTiming(events);
assert.equal(panel.hidden, false);
assert.deepEqual(JSON.parse(output.textContent), {summary: summarize(events), events});
console.log("Voice timing: exact job correlation and incomplete traces passed.");
