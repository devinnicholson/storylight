const elements = {
  title: document.querySelector("#titleInput"),
  level: document.querySelector("#levelInput"),
  style: document.querySelector("#styleInput"),
  story: document.querySelector("#storyInput"),
  micButton: document.querySelector("#micButton"),
  micButtonText: document.querySelector("#micButtonText"),
  compileButton: document.querySelector("#compileButton"),
  interim: document.querySelector("#interimText"),
  status: document.querySelector("#status"),
  empty: document.querySelector("#emptyState"),
  results: document.querySelector("#results"),
  error: document.querySelector("#errorBox"),
  model: document.querySelector("#modelMetric"),
  time: document.querySelector("#timeMetric"),
  tokens: document.querySelector("#tokenMetric"),
  summary: document.querySelector("#sceneSummary"),
  layers: document.querySelector("#layerList"),
  layerCount: document.querySelector("#layerCount"),
  triggers: document.querySelector("#triggerList"),
  triggerCount: document.querySelector("#triggerCount"),
  supports: document.querySelector("#supportList"),
  questions: document.querySelector("#questionList"),
  raw: document.querySelector("#rawOutput"),
  micLevel: document.querySelector("#micLevel"),
  browserNote: document.querySelector("#browserNote"),
};

let listening = false;
let audioContext = null;
let analyser = null;
let stream = null;
let mediaRecorder = null;
let audioChunks = [];
let partialTimer = null;
let partialBusy = false;
let partialInFlight = Promise.resolve();
let partialBytes = 0;
let recordingEpoch = 0;
let readerGeneration = null;
let activePageText = null;
let starting = false;
const readerSessionId = "moon-gate-demo";

function setStatus(state, text) {
  elements.status.dataset.state = state;
  elements.status.lastChild.textContent = ` ${text}`;
}

function safeText(value) {
  const node = document.createElement("span");
  node.textContent = value == null ? "" : String(value);
  return node.innerHTML;
}

async function startAudioMeter() {
  if (stream) return;
  stream = await navigator.mediaDevices.getUserMedia({audio: true, video: false});
  const Context = window.AudioContext || window.webkitAudioContext;
  audioContext = new Context();
  analyser = audioContext.createAnalyser();
  analyser.fftSize = 512;
  audioContext.createMediaStreamSource(stream).connect(analyser);
  const values = new Uint8Array(analyser.fftSize);

  function meter() {
    if (!analyser) return;
    analyser.getByteTimeDomainData(values);
    let energy = 0;
    values.forEach((value) => {
      const normalized = (value - 128) / 128;
      energy += normalized * normalized;
    });
    const rms = Math.sqrt(energy / values.length);
    elements.micLevel.style.width = `${Math.max(6, Math.min(100, rms * 620))}%`;
    requestAnimationFrame(meter);
  }
  requestAnimationFrame(meter);
}

function releaseMicrophone() {
  analyser = null;
  if (stream) stream.getTracks().forEach((track) => track.stop());
  stream = null;
  if (audioContext) audioContext.close();
  audioContext = null;
  elements.micLevel.style.width = "6%";
}

async function startSpeaking() {
  if (starting || listening) return;
  if (!window.MediaRecorder || !navigator.mediaDevices?.getUserMedia) {
    elements.interim.textContent = "Audio recording is unavailable in this browser.";
    return;
  }
  starting = true;
  elements.micButton.disabled = true;
  elements.micButtonText.textContent = "Starting…";
  elements.compileButton.disabled = true;
  try {
    const pageText = elements.story.value.trim();
    if (!pageText) throw new Error("Enter the trusted page text before starting the reader.");
    activePageText = pageText;
    await configureReaderSession(pageText);
    readerGeneration = (await resetReaderSession()).generation;
    await startAudioMeter();
    audioChunks = [];
    partialBytes = 0;
    partialInFlight = Promise.resolve();
    recordingEpoch += 1;
    const epoch = recordingEpoch;
    const preferredType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
      ? "audio/webm;codecs=opus"
      : "";
    mediaRecorder = preferredType
      ? new MediaRecorder(stream, {mimeType: preferredType})
      : new MediaRecorder(stream);
    mediaRecorder.addEventListener("dataavailable", (event) => {
      if (event.data.size > 0) audioChunks.push(event.data);
    });
    mediaRecorder.addEventListener("stop", transcribeRecording, {once: true});
    mediaRecorder.start(500);
    listening = true;
    elements.micButton.disabled = false;
    elements.micButton.classList.add("listening");
    elements.micButtonText.textContent = "Stop speaking";
    elements.compileButton.disabled = true;
    elements.interim.textContent = "Listening locally… Partial words will drive the live projector.";
    partialTimer = window.setInterval(() => {
      if (!partialBusy) partialInFlight = transcribePartialRecording(epoch);
    }, 2000);
  } catch (error) {
    elements.interim.textContent = `Microphone unavailable: ${error.message}`;
    if (mediaRecorder?.state === "recording") {
      mediaRecorder.removeEventListener("stop", transcribeRecording);
      mediaRecorder.stop();
    }
    mediaRecorder = null;
    audioChunks = [];
    recordingEpoch += 1;
  } finally {
    starting = false;
    if (!listening) resetMicControls();
  }
}

function stopSpeaking() {
  listening = false;
  elements.micButton.classList.remove("listening");
  elements.micButton.disabled = true;
  elements.micButtonText.textContent = "Transcribing…";
  elements.interim.textContent = "Whisper is transcribing locally…";
  window.clearInterval(partialTimer);
  partialTimer = null;
  if (mediaRecorder?.state === "recording") mediaRecorder.stop();
  else resetMicControls();
}

function resetMicControls() {
  elements.micButton.classList.remove("listening");
  elements.micButton.disabled = false;
  elements.micButtonText.textContent = "Start speaking";
  elements.compileButton.disabled = false;
  window.clearInterval(partialTimer);
  partialTimer = null;
  activePageText = null;
  readerGeneration = null;
  releaseMicrophone();
}

async function transcribeBlob(recording, mimeType) {
  const response = await fetch("/v1/audio:transcribe", {
      method: "POST",
      headers: {"Content-Type": mimeType},
      body: recording,
  });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.detail || `Transcription failed (${response.status})`);
  return payload;
}

async function publishReaderTranscript(text, isFinal, generation = readerGeneration) {
  if (!activePageText || generation === null) {
    throw new Error("The reader session changed; stop and start this reading again.");
  }
  const response = await fetch(`/v1/reader-sessions/${readerSessionId}/transcripts:simulate`, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({
      source: "asr",
      text,
      page_id: "page-01",
      language: "en",
      is_final: isFinal,
      generation,
    }),
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.detail || `Reader event failed (${response.status})`);
  }
}

async function configureReaderSession(pageText) {
  const configureResponse = await fetch(`/v1/reader-sessions/${readerSessionId}`, {
    method: "PUT",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({page_id: "page-01", page_text: pageText}),
  });
  if (!configureResponse.ok) {
    const payload = await configureResponse.json().catch(() => ({}));
    throw new Error(payload.detail || `Reader setup failed (${configureResponse.status})`);
  }
  return configureResponse.json();
}

async function resetReaderSession() {
  const response = await fetch(`/v1/reader-sessions/${readerSessionId}:reset`, {
    method: "POST",
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.detail || `Reader reset failed (${response.status})`);
  }
  return response.json();
}

async function transcribePartialRecording(epoch) {
  if (partialBusy || !listening || epoch !== recordingEpoch || audioChunks.length === 0) return;
  const generation = readerGeneration;
  const mimeType = mediaRecorder?.mimeType || "audio/webm";
  const recording = new Blob(audioChunks, {type: mimeType});
  if (recording.size < 1000 || recording.size === partialBytes) return;
  partialBusy = true;
  partialBytes = recording.size;
  try {
    const payload = await transcribeBlob(recording, mimeType);
    if (!listening || epoch !== recordingEpoch) return;
    elements.interim.textContent = `Live transcript · ${payload.text}`;
    await publishReaderTranscript(payload.text, false, generation);
  } catch (error) {
    if (listening && epoch === recordingEpoch) {
      elements.interim.textContent = `Live transcript retrying: ${error.message}`;
    }
  } finally {
    partialBusy = false;
  }
}

async function transcribeRecording() {
  const mimeType = mediaRecorder?.mimeType || "audio/webm";
  const recording = new Blob(audioChunks, {type: mimeType});
  const generation = readerGeneration;
  try {
    await partialInFlight;
    if (recording.size < 1000) throw new Error("Recording was too short. Try speaking for a little longer.");
    const payload = await transcribeBlob(recording, mimeType);
    await publishReaderTranscript(payload.text, true, generation);
    elements.interim.textContent = `Final transcript ready in ${(payload.total_ms / 1000).toFixed(1)} s. Review it, then compile.`;
  } catch (error) {
    elements.interim.textContent = error.message;
  } finally {
    mediaRecorder = null;
    audioChunks = [];
    partialInFlight = Promise.resolve();
    resetMicControls();
  }
}

function renderPack(payload) {
  const pack = payload.story_pack;
  localStorage.setItem("bookforge.latestStoryPack", JSON.stringify(pack));
  const page = pack.pages[0];
  elements.model.textContent = payload.metrics.model;
  elements.time.textContent = `${(payload.metrics.total_ms / 1000).toFixed(1)} s`;
  elements.tokens.textContent = String(payload.metrics.output_tokens);
  elements.summary.textContent = page.scene_summary;
  elements.layerCount.textContent = `${page.layers.length} layers`;
  elements.triggerCount.textContent = `${page.triggers.length} triggers`;
  elements.layers.innerHTML = page.layers.map((layer) => `
    <article class="layer">
      <div class="layer-top"><b>${safeText(layer.kind)}</b><small>z ${layer.z_index}</small></div>
      <p>${safeText(layer.prompt)}</p>
      <em>${safeText(layer.motion)}</em>
    </article>
  `).join("");
  elements.triggers.innerHTML = page.triggers.length ? page.triggers.map((trigger) => `
    <div class="trigger">
      <b>“${safeText(trigger.word)}”</b>
      <span>${safeText(trigger.action)} → ${safeText(trigger.target_layer_id)}</span>
      <small>${trigger.duration_ms} ms</small>
    </div>
  `).join("") : '<div class="trigger"><span>No word triggers generated.</span></div>';
  elements.supports.innerHTML = page.literacy_support.length ? page.literacy_support.map((support) => `
    <div class="support"><b>${safeText(support.word)}</b><span>${support.hint_ladder.map(safeText).join(" → ")}</span></div>
  `).join("") : '<div class="support">No scaffolds generated.</div>';
  elements.questions.innerHTML = page.comprehension.length ? page.comprehension.map((item) => `
    <div class="question">${safeText(item.question)}<span>${item.expected_concepts.map(safeText).join(", ")}</span></div>
  `).join("") : '<div class="question">No questions generated.</div>';
  elements.raw.textContent = JSON.stringify(pack, null, 2);
  elements.empty.classList.add("hidden");
  elements.error.classList.add("hidden");
  elements.results.classList.remove("hidden");
}

async function compileStory() {
  const text = elements.story.value.trim();
  if (!text) {
    elements.interim.textContent = "Speak or type at least one sentence first.";
    return;
  }
  if (listening) stopSpeaking();
  elements.compileButton.disabled = true;
  elements.error.classList.add("hidden");
  setStatus("working", "Gemma is compiling");

  try {
    const response = await fetch("/v1/story-packs:compile", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        story_id: `voice-${Date.now()}`,
        title: elements.title.value.trim() || "Untitled Story",
        reading_level: Number(elements.level.value),
        visual_style: elements.style.value.trim() || "luminous paper theater",
        pages: [{page_id: "page-01", text, art_direction: ""}],
      }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || `Request failed (${response.status})`);
    renderPack(payload);
    setStatus("idle", "Compilation complete");
  } catch (error) {
    elements.error.textContent = error.message;
    elements.error.classList.remove("hidden");
    setStatus("error", "Compilation failed");
  } finally {
    elements.compileButton.disabled = false;
  }
}

elements.micButton.addEventListener("click", () => {
  if (starting) return;
  if (listening) stopSpeaking();
  else startSpeaking();
});
elements.compileButton.addEventListener("click", compileStory);

const canRecordAudio = Boolean(window.MediaRecorder && navigator.mediaDevices?.getUserMedia);
document.body.dataset.audioSupport = canRecordAudio ? "available" : "unavailable";
if (!canRecordAudio) {
  elements.micButton.disabled = true;
  elements.interim.textContent = "This embedded browser cannot capture audio. Open this URL in Chrome to speak, or type the page text here.";
  elements.browserNote.textContent = "Local Whisper is ready, but this browser does not expose microphone recording. Chrome on localhost supports the full private audio flow.";
}
