# Bookforge

Bookforge is the cloud/edge foundation for **The Book That Listens Back**. The live system keeps
audio, video, reading behavior, and intervention decisions on the Jetson. Google Cloud compiles
publisher-supplied book pages into validated, projection-ready Story Packs.

This repository now contains working model-facing services rather than only a visual mockup:

- a structured Gemma client verified on both the development Mac and an NVIDIA Jetson Orin Nano;
- the same client contract for vLLM or NVIDIA NIM-compatible cloud endpoints;
- a deterministic sub-900 ms intervention fast path;
- schema-constrained Gemma intervention decisions;
- a Story Compiler that returns validated SceneSpec v2 composition, motion, triggers, scaffolds,
  and questions;
- a provider-neutral asynchronous Scene Foundry with real Modal GPU and local MFLUX backends;
- a bounded, text-only live-scene job API with revisioned status and Server-Sent Events;
- progressive typed-text projection: an immediate animated draft, a generated master plus depth
  scene, and an optional generative video upgrade without reloading or blanking the projector;
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

The workbench is available at `http://127.0.0.1:8080/workbench`. Its primary path is now typed story
text becoming the exact moving projector output in progressive stages: an immediate procedural
draft, generated artwork with depth-aware WebGL movement, and an optional cinematic loop. Stage,
revision, provider, latency, and artifact provenance remain visible throughout. The microphone is
intentionally a later, collapsed milestone; GCP is not required to exercise the typed-text flow.

The accepted edge path runs a real 1B Gemma 3 instruction model on the Orin Nano, emits a strict
semantic SceneSpec, and sends only that visual direction to the remote image renderer. The first
integrated acceptance planned on Jetson in 9.165 seconds and reached moving master-plus-depth output
in 15.005 seconds total. A local outbound gate now rejects source-text echoes, obvious contact data,
and source proper names before any rendering request; rejected plans use the cloud-safe fallback.

Run the complete interface without model weights or cloud spend:

```bash
BOOKFORGE_MODEL_BACKEND=fake \
BOOKFORGE_ASSET_BACKEND=fake \
BOOKFORGE_LIVE_SCENE_BACKEND=fake \
BOOKFORGE_ASR_BACKEND=disabled \
make dev
```

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

### Generate live scenes on Modal while GCP is pending

The live Modal backend makes one finite NVIDIA L4 call for a SANA 1.5 16:9 master and Depth Anything
V2 sidecar. The projector is already moving from its procedural draft while that call runs, then
crossfades to depth-aware WebGL motion as soon as both checksum-addressed files are ready. In finite
mode no GPU service is deployed. The optional warm mode uses authenticated, scale-to-zero Modal
classes with no public web endpoint and no permanently warm container. LTX-Video is an explicit
opt-in upgrade; it is off by default so every typed sentence does not silently start a slower,
costlier video job.

```bash
modal profile current
BOOKFORGE_MODEL_BACKEND=fake \
BOOKFORGE_ASSET_BACKEND=modal \
BOOKFORGE_LIVE_SCENE_BACKEND=modal \
BOOKFORGE_LIVE_SCENE_ENABLE_MOTION=false \
BOOKFORGE_ASR_BACKEND=disabled \
make dev
```

Then type a new passage and select **Generate moving scene**. The first run builds the reusable
container and populates the `bookforge-model-cache` Modal Volume. Every call is bounded by the
finite-provider timeout, per-stage maximum, session maximum, and the monthly envelope in
`experiments/live-scenes/modal-plan.json`. The backend boundary is provider-neutral: GCP can replace
Modal later without changing the live-scene API, SceneSpec, Story Pack, Jetson cache, or projector.

For an intentional warm demo run, install the authoring extra and deploy the named classes first.
`modal app list` verifies the app exists without allocating a GPU; the API's warm-status route then
performs an authenticated metadata lookup for both class names. Select
`BOOKFORGE_LIVE_SCENE_BACKEND=modal_warm`, start the API, and explicitly prewarm master/depth. The
route is loopback-only, budget-checked, expires with Modal's scale-down window, and is never invoked
automatically:

```bash
uv sync --extra modal-authoring
modal profile current
modal deploy deploy/modal_fast_scene.py
modal app list

BOOKFORGE_MODEL_BACKEND=fake \
BOOKFORGE_ASSET_BACKEND=modal \
BOOKFORGE_LIVE_SCENE_BACKEND=modal_warm \
BOOKFORGE_LIVE_SCENE_ENABLE_MOTION=false \
BOOKFORGE_ASR_BACKEND=disabled \
make dev

curl -sS http://127.0.0.1:8080/v1/live-scene-provider/warm-status
curl -sS -X POST http://127.0.0.1:8080/v1/live-scene-provider/prewarm \
  -H 'content-type: application/json' \
  -d '{"prewarm_id":"bookforge-demo","include_motion":false}'
# Submit one scene from the workbench immediately after prewarm returns.

# Explicit teardown after the demo; this terminates any remaining containers.
modal app stop bookforge-fast-scene --yes
```

On the August 23 acceptance, fast-only prewarm took 28.459 seconds and the following master/depth
job completed in 5.256 seconds end to end (3.528 seconds inference and 2.87 ms cache promotion).
The authoritative Modal delta was $0.01350024 under the atomic $0.12 session ceiling. These are
measurements, not a pricing guarantee; the provider still checks current billing and reserves the
full ceiling before prewarm. On a Mac that must retain local microphone support, sync both optional
groups with `uv sync --extra modal-authoring --extra mac-asr`.

For higher-quality offline scene R&D, `deploy/modal_visual_lab.py` provides finite `modal run`
jobs on an NVIDIA L4. It pins SANA 1.5, SigLIP, and LTX-Video revisions; records prompts, seeds,
checksums, generation time, and estimated GPU cost; and never deploys a persistent endpoint. The
experiment envelope in `experiments/visual-lab/plan.json` enforces a hard monthly stop and a retained
credit reserve. Raw candidates stay under the ignored `artifacts/visual-lab/` directory; selected
production loops are promoted into checksum-verified Story Packs.

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
Cosmos comparison remains gated rather than adding a payment method or risking overage. The accepted
six-page milestone used $0.48878335 of Modal compute and platform services and left $16.17606214 of
the monthly credit unused.

The six-page literacy pipeline adds two independent quality gates. Local FFmpeg measurements cover
checksum integrity, projection luminance and contrast, frame rate, temporal change, and loop-end
SSIM. A pinned SigLIP scorer on the same bounded Modal L4 measures prompt fidelity, character
continuity, child-safety contrast, and accidental text. The selector refuses mismatched checksums,
missing pages, unsafe candidates, or text-contaminated candidates before ranking the remaining art.

The exact accepted inputs, human decisions, final benchmark, and GCP handoff are tracked in the repo.
The large generated files remain ignored and are rebuilt with the following pipeline:

```bash
modal run deploy/modal_visual_lab.py::master_batch_cli \
  --prompt-file experiments/visual-lab/silver-fox-literacy-masters.json \
  --output-dir artifacts/visual-lab/lost-words-masters
modal run deploy/modal_visual_lab.py::master_batch_cli \
  --prompt-file experiments/visual-lab/silver-fox-literacy-remediation.json \
  --output-dir artifacts/visual-lab/lost-words-remediation
python -m bookforge.visual_evaluation artifacts/visual-lab/lost-words-masters/*.png \
  --output artifacts/visual-lab/lost-words-technical.json
python -m bookforge.visual_evaluation artifacts/visual-lab/lost-words-remediation/*.png \
  --output artifacts/visual-lab/lost-words-remediation-technical.json
modal run deploy/modal_visual_lab.py::score_batch_cli \
  --manifest-path artifacts/visual-lab/lost-words-masters/manifest.json \
  --reference-image-path \
  artifacts/visual-lab/silver-fox-masters/silver-fox-watercolor-theater-a__2026082231.png \
  --output-path artifacts/visual-lab/lost-words-semantic.json
modal run deploy/modal_visual_lab.py::score_batch_cli \
  --manifest-path artifacts/visual-lab/lost-words-remediation/manifest.json \
  --reference-image-path \
  artifacts/visual-lab/silver-fox-masters/silver-fox-watercolor-theater-a__2026082231.png \
  --output-path artifacts/visual-lab/lost-words-remediation-semantic.json

python -m bookforge.visual_selection \
  artifacts/visual-lab/lost-words-technical.json \
  artifacts/visual-lab/lost-words-semantic.json \
  --additional-technical artifacts/visual-lab/lost-words-remediation-technical.json \
  --additional-semantic artifacts/visual-lab/lost-words-remediation-semantic.json \
  --human-review experiments/visual-lab/silver-fox-literacy-human-review.json \
  --expected-pages 6 \
  --output artifacts/visual-lab/lost-words-master-selection.json
modal run deploy/modal_visual_lab.py::multi_motion_batch_cli \
  --master-manifest-path \
  "artifacts/visual-lab/lost-words-masters/manifest.json,\
artifacts/visual-lab/lost-words-remediation/manifest.json" \
  --selection-path artifacts/visual-lab/lost-words-master-selection.json \
  --output-dir artifacts/visual-lab/lost-words-motion-projection
python -m bookforge.visual_evaluation \
  artifacts/visual-lab/lost-words-motion-projection/*.mp4 \
  --output artifacts/visual-lab/lost-words-motion-projection-technical.json
python -m bookforge.motion_selection \
  artifacts/visual-lab/lost-words-motion-projection-technical.json \
  --human-review experiments/visual-lab/silver-fox-literacy-motion-review.json \
  --expected-pages 6 \
  --output artifacts/visual-lab/lost-words-motion-projection-selection.json
python -m bookforge.final_asset_manifest \
  --master-manifest artifacts/visual-lab/lost-words-masters/manifest.json \
  --master-manifest artifacts/visual-lab/lost-words-remediation/manifest.json \
  --master-selection artifacts/visual-lab/lost-words-master-selection.json \
  --motion-manifest artifacts/visual-lab/lost-words-motion-projection/manifest.json \
  --motion-selection artifacts/visual-lab/lost-words-motion-projection-selection.json \
  --output artifacts/visual-lab/lost-words-final-assets.json
python -m bookforge.literacy_pack \
  experiments/visual-lab/silver-fox-literacy-story.json \
  artifacts/visual-lab/lost-words-final-assets.json \
  --asset-root . \
  --output artifacts/visual-lab/silver-fox-lost-words.story-pack.json
python -m bookforge.handoff_bundle \
  artifacts/visual-lab/silver-fox-lost-words.story-pack.json \
  --asset-root . \
  --output-dir artifacts/visual-lab/handoff/silver-fox-lost-words
```

The projector validates every page in a Story Pack. Its page controls and `[` / `]` shortcuts swap
the motion scene, trusted reading text, trigger index, URL page number, and local reader generation
without restarting the service.

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
| `POST /v1/live-scenes` | Asset provider | Submit typed text and receive a bounded progressive generation job immediately |
| `GET /v1/live-scenes/{job_id}` | No | Read the newest revision, stage, artifacts, completion, and nonfatal warning state |
| `GET /v1/live-scenes/{job_id}/events` | No | Stream full revisioned job snapshots as `scene.job` Server-Sent Events |
| `GET /v1/live-scene-sessions/{session_id}` | No | Recover the authoritative current live job after a browser reconnect |
| `GET /v1/live-scene-sessions/{session_id}/events` | No | Stream the server epoch, current job, revisions, and same-session replacements |
| `GET /v1/live-scene-provider/warm-status` | No | Inspect explicit Modal warm-provider readiness without allocating a GPU |
| `POST /v1/live-scene-provider/prewarm` | Modal warm provider | Budget-check and intentionally prewarm the selected Modal classes |
| `PUT /v1/reader-sessions/{id}` | No | Configure trusted page text for local alignment |
| `GET /v1/reader-sessions/{id}` | No | Recover the current generation and aligned position after reconnect |
| `POST /v1/reader-sessions/{id}:reset` | No | Rewind the aligner and every connected projector for another reading |
| `POST /v1/reader-sessions/{id}/transcripts:simulate` | No | Align a generation-bound typed or ASR transcript and publish word events |
| `WS /v1/reader-sessions/{id}/events` | No | Stream ordered local transcript and word events |

## Run without model weights

Tests and API integration can use the deterministic fake backend:

```bash
BOOKFORGE_MODEL_BACKEND=fake \
BOOKFORGE_ASSET_BACKEND=fake \
BOOKFORGE_LIVE_SCENE_BACKEND=fake \
BOOKFORGE_ASR_BACKEND=disabled \
make dev
make test
```

## Google Cloud

The GCP path is documented in [`infra/gcp/README.md`](infra/gcp/README.md). The scripts default to
a non-mutating cost guard; configuring a project does not create a cluster or GPU workload, and
provisioning still requires an explicit environment variable acknowledging billable resources.

## Jetson Orin Nano

The Jetson deployment track targets JetPack 7.2.1 / Jetson Linux 39.2.1 without performing or
automating a device flash. Start with its read-only hardware and runtime report:

```bash
./deploy/jetson/check-device.sh
```

The diagnostic covers L4T, CUDA, TensorRT, Docker, Python, power mode, NVMe, camera, microphone,
display, the projector browser, and thermal zones. `bootstrap.sh` is also diagnostic-only unless an
explicit installation option is supplied. System and graphical-user service templates provide a
loopback API and Chromium-first projector kiosk with a Firefox fallback, respectively.

See [`deploy/jetson/README.md`](deploy/jetson/README.md) for the guarded setup, interactive smoke
test, local Gemma service, systemd installation, kiosk configuration, WhisperTRT activation,
privacy audit, and hardware acceptance commands. Jetson speech recognition remains disabled for
first boot; the adapter is implemented, but it does not become accepted until the exact board
passes the real I/O and latency run on JetPack 7.2.1.

## Design documents

- [`docs/architecture.md`](docs/architecture.md): data boundary and cloud/edge responsibilities
- [`docs/gcp-handoff.md`](docs/gcp-handoff.md): portable offline bundle and GCP promotion boundary
- [`infra/gcp/README.md`](infra/gcp/README.md): guarded GKE and Gemma serving workflow
- [`benchmarks/macbook-m4-smoke-2026-08-20.json`](benchmarks/macbook-m4-smoke-2026-08-20.json): first real-model latency measurements
- [`benchmarks/live-reader-laptop-2026-08-21.json`](benchmarks/live-reader-laptop-2026-08-21.json): clean-wheel, browser, persistence, and live event latency evidence
- [`benchmarks/scene-engine-modal-t4-2026-08-22.json`](benchmarks/scene-engine-modal-t4-2026-08-22.json): real Gemma → Modal master/depth → local ASR → WebGL trigger evidence
- [`benchmarks/visual-lab-modal-l4-2026-08-22.json`](benchmarks/visual-lab-modal-l4-2026-08-22.json): pinned SANA/LTX generation, loop quality, cost, and projector playback evidence
- [`benchmarks/visual-technical-existing-2026-08-23.json`](benchmarks/visual-technical-existing-2026-08-23.json): reproducible FFmpeg projection and loop measurements for the two accepted scenes
- [`benchmarks/literacy-navigation-2026-08-23.json`](benchmarks/literacy-navigation-2026-08-23.json): six-page session, media lifecycle, and final-page live-trigger browser acceptance
- [`benchmarks/modal-billing-gate-2026-08-23.json`](benchmarks/modal-billing-gate-2026-08-23.json): authoritative monthly usage, remaining credit, and paid-generation gate evidence
- [`benchmarks/lost-words-modal-acceptance-2026-08-23.json`](benchmarks/lost-words-modal-acceptance-2026-08-23.json): final six-page generation, quality, billing, bundle, and read-aloud acceptance
- [`benchmarks/winged-library-modal-live-scene-2026-08-23.json`](benchmarks/winged-library-modal-live-scene-2026-08-23.json): pinned cold SANA/depth/LTX smoke, loop quality, checksums, guardrails, and authoritative billing evidence
- [`benchmarks/winged-library-modal-warm-acceptance-2026-08-23.json`](benchmarks/winged-library-modal-warm-acceptance-2026-08-23.json): authenticated scale-to-zero prewarm, 5.256-second live master, cache/StoryPack validation, and $0.01350024 authoritative cost
- [`benchmarks/jetson-projector-live-scene-2026-08-23.json`](benchmarks/jetson-projector-live-scene-2026-08-23.json): physical-Jetson cross-device generation and asset-delivery pass, with the projector display gate explicitly pending an unlocked screenshot or video
- [`benchmarks/jetson-gemma3-ollama-2026-08-23.json`](benchmarks/jetson-gemma3-ollama-2026-08-23.json): pinned Jetson Gemma runtime, GPU/memory/privacy evidence, exact planning latency, and the complete Gemma-to-SANA-to-depth acceptance
