# Bookforge

Bookforge is the cloud/edge foundation for **The Book That Listens Back**. The live system keeps
audio, video, reading behavior, and intervention decisions on the Jetson. Google Cloud compiles
publisher-supplied book pages into validated, projection-ready Story Packs.

This repository now contains working model-facing services rather than only a visual mockup:

- a structured Gemma client for Ollama on the development Mac;
- the same client contract for vLLM or NVIDIA NIM-compatible cloud endpoints;
- a deterministic sub-900 ms intervention fast path;
- schema-constrained Gemma intervention decisions;
- a Story Compiler that returns validated SceneSpec v2 composition, motion, triggers, scaffolds,
  and questions;
- a provider-neutral asynchronous Scene Foundry with real Modal GPU and local MFLUX backends;
- immutable asset manifests with provider, seed, checksum, dimensions, state, and location;
- a fullscreen 1920×1080 projector runtime with depth-aware WebGL parallax, authored ambience,
  localized word effects, and deterministic cached visuals;
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
Gemma, Modal, or GCP during playback: it loads a validated cached Story Pack, connects to a local
reader-session WebSocket, and responds instantly to aligned word events. A ready schema 2.0 pack
with master and depth assets uses the WebGL 2 renderer; the master PNG remains the automatic
fallback. Enter a cumulative phrase in the typed transcript simulator to exercise the complete
recovery path. Compile in the workbench and use its **Open this Story Pack** link to load the new
pack with `pack=latest`. Add `offline=1` to enforce a same-origin resource boundary during rehearsal.
Use Space or Right Arrow to advance, Left Arrow to rewind, `R` to reset, `F` for fullscreen, `C` for
calibration, `B` for blackout, and `H` to hide controls. Calibration is saved locally as the
`yaber-t1-pro` profile.

### Generate scenes on Modal while GCP is pending

The Modal backend uses an NVIDIA T4 on the Starter plan, SDXL-Turbo for the 16:9 master, and Depth
Anything V2 for the depth sidecar. Both assets return in one remote call and are stored under their
SHA-256 checksums before the Story Pack becomes `latest`.

```bash
modal profile current
BOOKFORGE_ASSET_BACKEND=modal make dev
```

Then create a scene in the workbench. The first run builds the reusable container and populates the
`bookforge-model-cache` Modal Volume. The backend boundary is provider-neutral: GCP can replace
Modal later without changing SceneSpec, Story Pack storage, the Jetson cache, or the projector.

For higher-quality offline scene R&D, `deploy/modal_visual_lab.py` provides finite `modal run`
jobs on an NVIDIA L4. It pins SANA 1.5 and LTX-Video revisions, records prompts, seeds, checksums,
generation time, and estimated GPU cost, and never deploys a persistent endpoint. The experiment
envelope in `experiments/visual-lab/plan.json` caps this month's lab work at $15 while retaining a
$1.66 credit reserve. Raw candidates stay under the ignored `artifacts/visual-lab/` directory;
selected production loops are promoted into checksum-verified Story Packs.

```bash
modal run deploy/modal_visual_lab.py::master_batch_cli \
  --prompt-file experiments/visual-lab/moon-gate-prompts.json \
  --output-dir artifacts/visual-lab/masters

modal run deploy/modal_visual_lab.py::motion_batch_cli \
  --image-path artifacts/visual-lab/masters/moon-gate-lantern-a__2026082205.png \
  --prompt-file experiments/visual-lab/moon-gate-motion-prompts.json \
  --output-dir artifacts/visual-lab/motion

python -m bookforge.motion_promotion SOURCE.story-pack.json LOOP.mp4 \
  --asset-root artifacts/visual-lab \
  --output artifacts/visual-lab/promoted.story-pack.json \
  --prompt "Locked camera, subtle ambient motion" \
  --seed 2026082211 \
  --generation-ms 34653
python -m bookforge.pack_installer artifacts/visual-lab/promoted.story-pack.json \
  --asset-root artifacts/visual-lab
```

The visual lab intentionally stops instead of silently falling back when the selected GPU is not
available. Premium GPU access was not enabled on the no-payment Modal workspace, so the planned
Cosmos comparison remains gated rather than adding a payment method or risking overage.

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
| `POST /v1/story-packs:build` | Yes + asset GPU | Compile, generate master/depth assets, checksum, cache, and save the ready pack |
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
- [`benchmarks/scene-engine-modal-t4-2026-08-22.json`](benchmarks/scene-engine-modal-t4-2026-08-22.json): real Gemma → Modal master/depth → local ASR → WebGL trigger evidence
- [`benchmarks/visual-lab-modal-l4-2026-08-22.json`](benchmarks/visual-lab-modal-l4-2026-08-22.json): pinned SANA/LTX generation, loop quality, cost, and projector playback evidence
