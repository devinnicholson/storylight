const EXPECTED_WORDS = ["the", "small", "moth", "went", "through", "the", "red", "gate"];
const DISPLAY_WORDS = ["The", "small", "moth", "went", "through", "the", "red", "gate."];
const TARGET_INDEX = 4;
const HESITATION_MS = 1200;

const elements = {
  sentence: document.querySelector("#sentence"),
  stagePanel: document.querySelector("#stagePanel"),
  storyScene: document.querySelector("#storyScene"),
  moth: document.querySelector("#moth"),
  gate: document.querySelector("#gate"),
  hintCard: document.querySelector("#hintCard"),
  readerPrompt: document.querySelector("#readerPrompt"),
  transcript: document.querySelector("#transcript"),
  statusPill: document.querySelector("#statusPill"),
  statusText: document.querySelector("#statusText"),
  micLevelBar: document.querySelector("#micLevelBar"),
  fpsMetric: document.querySelector("#fpsMetric"),
  latencyMetric: document.querySelector("#latencyMetric"),
  wordMetric: document.querySelector("#wordMetric"),
  eventLog: document.querySelector("#eventLog"),
  startButton: document.querySelector("#startButton"),
  stopButton: document.querySelector("#stopButton"),
  simulateButton: document.querySelector("#simulateButton"),
  resetButton: document.querySelector("#resetButton"),
  fullscreenButton: document.querySelector("#fullscreenButton"),
  calibrationButton: document.querySelector("#calibrationButton"),
  calibrationOverlay: document.querySelector("#calibrationOverlay"),
};

let recognition = null;
let audioContext = null;
let analyser = null;
let micStream = null;
let listening = false;
let accumulatedFinal = "";
let interimText = "";
let currentIndex = -1;
let matchedIndices = new Set();
let hintVisible = false;
let hintShownForRun = false;
let lastVoiceAt = performance.now();
let lastRecognitionAt = performance.now();
let latencySamples = [];
let simulationTimers = [];
let frameCounter = 0;
let fpsWindowStarted = performance.now();

function buildSentence() {
  elements.sentence.replaceChildren();
  DISPLAY_WORDS.forEach((displayWord, index) => {
    const span = document.createElement("span");
    span.className = "word";
    span.dataset.index = String(index);
    if (index === TARGET_INDEX) {
      span.innerHTML = '<span class="grapheme">th</span>rough';
    } else {
      span.textContent = displayWord;
    }
    elements.sentence.append(span);
  });
  renderWordState();
}

function normalizeText(text) {
  return text
    .toLowerCase()
    .replace(/[^a-z'\s]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function editDistance(a, b) {
  const rows = a.length + 1;
  const cols = b.length + 1;
  const table = Array.from({ length: rows }, () => Array(cols).fill(0));
  for (let row = 0; row < rows; row += 1) table[row][0] = row;
  for (let col = 0; col < cols; col += 1) table[0][col] = col;
  for (let row = 1; row < rows; row += 1) {
    for (let col = 1; col < cols; col += 1) {
      const substitution = a[row - 1] === b[col - 1] ? 0 : 1;
      table[row][col] = Math.min(
        table[row - 1][col] + 1,
        table[row][col - 1] + 1,
        table[row - 1][col - 1] + substitution,
      );
    }
  }
  return table[a.length][b.length];
}

function wordsAreSimilar(expected, heard) {
  if (expected === heard) return true;
  if (!heard || Math.min(expected.length, heard.length) < 3) return false;
  const distance = editDistance(expected, heard);
  return distance <= Math.max(1, Math.floor(expected.length * 0.28));
}

// Aligns the recognized transcript to the known sentence while tolerating repeats,
// insertions, and a single omitted word. This is intentionally deterministic.
function alignTranscript(transcript) {
  const heard = normalizeText(transcript).split(" ").filter(Boolean);
  const matches = [];
  let expectedCursor = 0;

  for (const heardWord of heard) {
    let matchAt = -1;
    const searchEnd = Math.min(EXPECTED_WORDS.length, expectedCursor + 3);
    for (let index = expectedCursor; index < searchEnd; index += 1) {
      if (wordsAreSimilar(EXPECTED_WORDS[index], heardWord)) {
        matchAt = index;
        break;
      }
    }
    if (matchAt >= 0) {
      matches.push(matchAt);
      expectedCursor = matchAt + 1;
      continue;
    }

    // A repeated word may refer to the most recently matched token.
    const previousIndex = Math.max(0, expectedCursor - 1);
    if (wordsAreSimilar(EXPECTED_WORDS[previousIndex], heardWord)) {
      matches.push(previousIndex);
    }
  }
  return matches;
}

function updateFromTranscript(transcript, source = "speech") {
  const nextMatches = alignTranscript(transcript);
  const uniqueMatches = [...new Set(nextMatches)];
  const nextHighest = uniqueMatches.length ? Math.max(...uniqueMatches) : -1;

  uniqueMatches.forEach((index) => {
    if (!matchedIndices.has(index)) {
      matchedIndices.add(index);
      const measuredLag = Math.max(0, performance.now() - lastVoiceAt);
      if (source === "speech" && measuredLag < 3000) {
        latencySamples.push(measuredLag);
        updateLatencyMetric();
      }
      logEvent(`word confirmed: ${EXPECTED_WORDS[index]} (${source})`);
      triggerStoryEvent(index);
    }
  });

  if (nextHighest > currentIndex) currentIndex = nextHighest;
  if (currentIndex >= TARGET_INDEX) hideHint();
  if (currentIndex === EXPECTED_WORDS.length - 1) completeReading();
  renderWordState();
}

function renderWordState() {
  const words = [...elements.sentence.querySelectorAll(".word")];
  words.forEach((word, index) => {
    word.classList.toggle("matched", matchedIndices.has(index));
    word.classList.toggle("active", index === Math.min(currentIndex + 1, EXPECTED_WORDS.length - 1) && currentIndex < EXPECTED_WORDS.length - 1);
    word.classList.toggle("hinting", hintVisible && index === TARGET_INDEX);
  });
  elements.wordMetric.textContent = `${matchedIndices.size}/${EXPECTED_WORDS.length}`;
}

function triggerStoryEvent(index) {
  if (index >= 2) elements.moth.classList.add("moth-visible");
  if (index >= TARGET_INDEX && !elements.moth.classList.contains("moth-flying")) {
    requestAnimationFrame(() => elements.moth.classList.add("moth-flying"));
  }
  if (index >= 6) elements.gate.classList.add("gate-open");
}

function showHint(reason = "hesitation detected") {
  if (hintVisible || currentIndex >= TARGET_INDEX) return;
  hintVisible = true;
  hintShownForRun = true;
  elements.hintCard.classList.add("visible");
  elements.hintCard.setAttribute("aria-hidden", "false");
  elements.readerPrompt.textContent = "A subtle scaffold appeared after the pause. Try the word again.";
  setStatus("hint", "Offering a subtle hint");
  renderWordState();
  logEvent(`support shown: ${reason}`);
}

function hideHint() {
  if (!hintVisible) return;
  hintVisible = false;
  elements.hintCard.classList.remove("visible");
  elements.hintCard.setAttribute("aria-hidden", "true");
  renderWordState();
  logEvent("support faded after confirmation");
  if (listening) setStatus("listening", "Listening");
}

function completeReading() {
  setStatus("complete", "Reading complete");
  elements.readerPrompt.textContent = hintShownForRun
    ? "The story continued after support—and the support disappeared."
    : "The sentence was completed independently.";
  logEvent("sentence complete");
}

function setStatus(state, text) {
  elements.statusPill.dataset.state = state;
  elements.statusText.textContent = text;
}

function logEvent(message) {
  const row = document.createElement("div");
  row.className = "event-row";
  const elapsed = ((performance.now()) / 1000).toFixed(2);
  row.innerHTML = `<span class="event-time">${elapsed}s</span><span>${message}</span>`;
  elements.eventLog.prepend(row);
  while (elements.eventLog.children.length > 24) elements.eventLog.lastElementChild.remove();
}

function percentile(values, percentileValue) {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const index = Math.min(sorted.length - 1, Math.ceil((percentileValue / 100) * sorted.length) - 1);
  return sorted[index];
}

function updateLatencyMetric() {
  const p95 = percentile(latencySamples, 95);
  elements.latencyMetric.textContent = p95 === null ? "—" : String(Math.round(p95));
}

async function startMicrophone() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error("Microphone access is unavailable. Open the app from localhost in a modern browser.");
  }
  micStream = await navigator.mediaDevices.getUserMedia({
    audio: {
      channelCount: 1,
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
    },
    video: false,
  });
  const BrowserAudioContext = window.AudioContext || window.webkitAudioContext;
  audioContext = new BrowserAudioContext();
  const source = audioContext.createMediaStreamSource(micStream);
  analyser = audioContext.createAnalyser();
  analyser.fftSize = 1024;
  analyser.smoothingTimeConstant = 0.72;
  source.connect(analyser);
  monitorAudioLevel();
}

function monitorAudioLevel() {
  if (!analyser) return;
  const samples = new Uint8Array(analyser.fftSize);
  const tick = () => {
    if (!analyser) return;
    analyser.getByteTimeDomainData(samples);
    let energy = 0;
    for (const value of samples) {
      const normalized = (value - 128) / 128;
      energy += normalized * normalized;
    }
    const rms = Math.sqrt(energy / samples.length);
    const level = Math.min(100, Math.max(3, rms * 580));
    elements.micLevelBar.style.width = `${level}%`;
    if (rms > 0.035) lastVoiceAt = performance.now();

    const waitingForTarget = currentIndex === TARGET_INDEX - 1;
    if (listening && waitingForTarget && !hintShownForRun && performance.now() - lastVoiceAt > HESITATION_MS) {
      showHint();
    }
    requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

function createSpeechRecognition() {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) return null;
  const instance = new SpeechRecognition();
  instance.continuous = true;
  instance.interimResults = true;
  instance.lang = "en-US";
  instance.maxAlternatives = 3;

  instance.onresult = (event) => {
    interimText = "";
    let newlyFinal = "";
    for (let index = event.resultIndex; index < event.results.length; index += 1) {
      const result = event.results[index];
      const text = result[0].transcript;
      if (result.isFinal) newlyFinal += `${text} `;
      else interimText += `${text} `;
    }
    if (newlyFinal) accumulatedFinal += newlyFinal;
    const combined = `${accumulatedFinal} ${interimText}`.trim();
    elements.transcript.textContent = combined || "Listening…";
    lastRecognitionAt = performance.now();
    updateFromTranscript(combined, "speech");
  };

  instance.onerror = (event) => {
    if (event.error === "no-speech" || event.error === "aborted") return;
    logEvent(`speech recognition error: ${event.error}`);
    setStatus("error", "Speech service unavailable");
  };

  instance.onend = () => {
    if (!listening) return;
    // Chrome briefly remains in a stopping state. A short retry is more stable
    // than immediately calling start() from inside onend.
    window.setTimeout(() => {
      if (!listening) return;
      try { instance.start(); } catch { /* A later user click can restart it. */ }
    }, 180);
  };
  return instance;
}

async function startListening() {
  clearSimulation();
  if (!micStream) {
    try {
      await startMicrophone();
    } catch (error) {
      setStatus("error", "Microphone blocked");
      elements.transcript.textContent = error.message;
      logEvent(error.message);
      return;
    }
  }

  if (!recognition) recognition = createSpeechRecognition();
  listening = true;
  elements.startButton.disabled = true;
  elements.stopButton.disabled = false;
  lastVoiceAt = performance.now();
  setStatus("listening", recognition ? "Listening" : "Microphone active — manual alignment mode");
  logEvent("microphone started");

  if (recognition) {
    try { recognition.start(); } catch { /* It may already be active. */ }
  } else {
    elements.transcript.textContent = "SpeechRecognition is unavailable. Use Space to advance words or run the simulation.";
  }
}

function stopListening() {
  listening = false;
  elements.startButton.disabled = false;
  elements.stopButton.disabled = true;
  if (recognition) {
    try { recognition.stop(); } catch { /* Already stopped. */ }
  }
  setStatus("idle", "Paused");
  logEvent("listening stopped");
}

function advanceManual(source = "manual") {
  if (currentIndex >= EXPECTED_WORDS.length - 1) return;
  const nextIndex = currentIndex + 1;
  const syntheticTranscript = EXPECTED_WORDS.slice(0, nextIndex + 1).join(" ");
  elements.transcript.textContent = syntheticTranscript;
  updateFromTranscript(syntheticTranscript, source);
}

function runSimulation() {
  if (listening) stopListening();
  resetExperience();
  setStatus("listening", "Simulating a read-aloud");
  logEvent("simulation started");
  const schedule = [500, 1050, 1600, 2150, 4100, 4700, 5250, 5850];
  schedule.forEach((delay, index) => {
    simulationTimers.push(window.setTimeout(() => {
      if (index === TARGET_INDEX) hideHint();
      advanceManual("simulation");
    }, delay));
  });
  simulationTimers.push(window.setTimeout(() => showHint("simulated hesitation"), 3450));
}

function clearSimulation() {
  simulationTimers.forEach((timer) => clearTimeout(timer));
  simulationTimers = [];
}

function resetExperience() {
  clearSimulation();
  accumulatedFinal = "";
  interimText = "";
  currentIndex = -1;
  matchedIndices = new Set();
  hintVisible = false;
  hintShownForRun = false;
  latencySamples = [];
  elements.transcript.textContent = "Waiting for speech…";
  elements.hintCard.classList.remove("visible");
  elements.moth.classList.remove("moth-visible", "moth-flying");
  elements.gate.classList.remove("gate-open");
  elements.readerPrompt.textContent = "Press “Start listening,” then read naturally.";
  updateLatencyMetric();
  renderWordState();
  if (!listening) setStatus("idle", "Ready");
  logEvent("experience reset");
}

function toggleFullscreen() {
  if (document.fullscreenElement) document.exitFullscreen();
  else elements.stagePanel.requestFullscreen();
}

function toggleCalibration() {
  const visible = elements.calibrationOverlay.classList.toggle("visible");
  elements.calibrationOverlay.setAttribute("aria-hidden", String(!visible));
  logEvent(`calibration overlay ${visible ? "shown" : "hidden"}`);
}

function monitorFps(now) {
  frameCounter += 1;
  const elapsed = now - fpsWindowStarted;
  if (elapsed >= 1000) {
    const fps = Math.round((frameCounter * 1000) / elapsed);
    elements.fpsMetric.textContent = String(fps);
    frameCounter = 0;
    fpsWindowStarted = now;
  }
  requestAnimationFrame(monitorFps);
}

elements.startButton.addEventListener("click", startListening);
elements.stopButton.addEventListener("click", stopListening);
elements.simulateButton.addEventListener("click", runSimulation);
elements.resetButton.addEventListener("click", resetExperience);
elements.fullscreenButton.addEventListener("click", toggleFullscreen);
elements.calibrationButton.addEventListener("click", toggleCalibration);

document.addEventListener("keydown", (event) => {
  if (event.target instanceof HTMLInputElement || event.target instanceof HTMLTextAreaElement) return;
  if (event.code === "Space") {
    event.preventDefault();
    advanceManual();
  } else if (event.key.toLowerCase() === "h") {
    showHint("manual test");
  } else if (event.key.toLowerCase() === "r") {
    resetExperience();
  }
});

buildSentence();
logEvent("POC initialized");
requestAnimationFrame(monitorFps);
