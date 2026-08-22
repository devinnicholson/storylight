# Bookforge

Bookforge is the cloud/edge foundation for **The Book That Listens Back**. The live system keeps
audio, video, reading behavior, and intervention decisions on the Jetson. Google Cloud compiles
publisher-supplied book pages into validated, projection-ready Story Packs.

This repository now contains working model-facing services rather than only a visual mockup:

- a structured Gemma client for Ollama on the development Mac;
- the same client contract for vLLM or NVIDIA NIM-compatible cloud endpoints;
- a deterministic sub-900 ms intervention fast path;
- schema-constrained Gemma intervention decisions;
- a Story Compiler that returns versioned scene layers, triggers, scaffolds, and questions;
- immutable asset manifests with provider, seed, checksum, dimensions, state, and location;
- a fullscreen 1920×1080 projector runtime with deterministic cached visuals;
- monotonic known-text alignment and a local WebSocket event stream for live word progression;
- rolling local partial transcription, typed recovery controls, four-corner calibration, blackout,
  fullscreen, and live latency telemetry;
- direct playback of the latest workbench-compiled Story Pack, including ready image/video assets
  and visibly labeled development layers when generation is still pending;
- private atomic Story Pack persistence that survives browser and API restarts;
- verified offline book-package installation and loopback-only cached media serving;
- runtime readiness, edge preflight, WhisperTRT adapter, process privacy audit, and machine-readable
  Jetson hardware acceptance evidence;
- strict validation that rejects invented pages, missing visual layers, and trigger words not in
  the source text;
- a Docker image and guarded GKE/NVIDIA L4 deployment path;
- deterministic tests that run without downloading a model.

The original browser synchronization experiment remains in `index.html`, `styles.css`, and
`app.js`. It is no longer the architecture.

## Local setup

Prerequisites already present on the development laptop: Python, `uv`, and Ollama.

```bash
cp .env.example .env
make install
```

In one terminal:

```bash
make model-serve
```

In another terminal, pull the pinned 4.3 GB instruction-tuned model and start the API:

```bash
make model-pull
make dev
```

Verify the runtime:

```bash
make model-probe
make compile-demo
```

Interactive API documentation is available at `http://127.0.0.1:8080/docs`.

The local voice workbench is available at `http://127.0.0.1:8080/workbench`. It records a spoken
story page, transcribes it locally with MLX Whisper, and compiles it into a Story Pack with the
configured Gemma model. Install the Mac ASR extra with `uv sync --extra mac-asr`, then download the
small local model with `make asr-model-pull`. GCP is not required for this flow.

The projection POC is available at `http://127.0.0.1:8080/projector`. It deliberately does not call
Gemma or GCP during playback: it loads a validated cached Moon Gate fixture, connects to a local
reader-session WebSocket, and responds instantly to aligned word events. Enter a cumulative phrase
in the typed transcript simulator to exercise the complete recovery path. Compile in the workbench
and use its **Open this Story Pack** link to load the new pack with `pack=latest`. Use Space or Right
Arrow to advance, Left Arrow to rewind, `R` to reset, `F` for fullscreen, `C` for calibration, `B`
for blackout, and `H` to hide controls. Calibration is saved locally as the `yaber-t1-pro` profile.

Microphone capture remains private to the local API. While recording, the workbench sends rolling
cumulative clips every two seconds, aligns each partial transcript to the trusted page text, and
emits only reader events to the projector. This laptop POC is rolling-batch partial ASR—not yet a
frame-streaming Jetson ASR engine—and the typed/manual paths remain the deterministic demo fallback.

## API surface

| Endpoint | Live model required | Purpose |
| --- | --- | --- |
| `GET /healthz` | No | Process health |
| `GET /readyz` | No | Device storage and playback readiness |
| `GET /v1/runtime:status` | Probe only | Local model, ASR, storage, and path readiness |
| `GET /projector` | No | Fullscreen deterministic projection stage |
| `GET /v1/models:probe` | No generation | Runtime and model readiness |
| `POST /v1/audio:transcribe` | No | Transcribe recorded audio with local Whisper |
| `POST /v1/interventions:select` | Only after the fast path | Select constrained support |
| `POST /v1/story-packs:compile` | Yes | Compile book pages into a Story Pack |
| `GET /v1/story-packs/latest` | No | Recover the latest private device-stored Story Pack |
| `PUT /v1/reader-sessions/{id}` | No | Configure trusted page text for local alignment |
| `GET /v1/reader-sessions/{id}` | No | Recover the current generation and aligned position after reconnect |
| `POST /v1/reader-sessions/{id}:reset` | No | Rewind the aligner and every connected projector for another reading |
| `POST /v1/reader-sessions/{id}/transcripts:simulate` | No | Align a generation-bound typed or ASR transcript and publish word events |
| `WS /v1/reader-sessions/{id}/events` | No | Stream ordered local transcript and word events |

## Run without model weights

Tests and API integration can use the deterministic fake backend:

```bash
BOOKFORGE_MODEL_BACKEND=fake BOOKFORGE_MODEL_NAME=fake make dev
make test
```

## Google Cloud

The GCP path is documented in [`infra/gcp/README.md`](infra/gcp/README.md). The scripts default to
a non-mutating cost guard; they will not create a cluster or GPU workload without an explicit
environment variable acknowledging billable resources.

No GCP project or account is configured on this machine yet, so cloud resources have not been
created.

## Jetson Orin Nano

The Jetson deployment track targets JetPack 7.2.1 / Jetson Linux 39.2.1 without performing or
automating a device flash. Start with its read-only hardware and runtime report:

```bash
./deploy/jetson/check-device.sh
```

The diagnostic covers L4T, CUDA, TensorRT, Docker, Python, power mode, NVMe, camera, microphone,
display, Chromium, and thermal zones. `bootstrap.sh` is also diagnostic-only unless an explicit
installation option is supplied. System and graphical-user service templates provide a loopback API
and Chromium projector kiosk, respectively.

See [`deploy/jetson/README.md`](deploy/jetson/README.md) for the guarded setup, interactive smoke
test, systemd installation, kiosk configuration, WhisperTRT activation, privacy audit, and hardware
acceptance commands. Jetson speech recognition remains disabled for first boot; the adapter is
implemented, but it does not become accepted until the exact board passes the real I/O and latency
run on JetPack 7.2.1.

## Design documents

- [`docs/architecture.md`](docs/architecture.md): data boundary and cloud/edge responsibilities
- [`infra/gcp/README.md`](infra/gcp/README.md): guarded GKE and Gemma serving workflow
- [`benchmarks/macbook-m4-smoke-2026-08-20.json`](benchmarks/macbook-m4-smoke-2026-08-20.json): first real-model latency measurements
- [`benchmarks/live-reader-laptop-2026-08-21.json`](benchmarks/live-reader-laptop-2026-08-21.json): clean-wheel, browser, persistence, and live event latency evidence
