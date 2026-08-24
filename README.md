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

The authenticated warm live backend selects a budget-checked NVIDIA L40S variant for its SANA-Sprint
16:9 master and Depth Anything V2 sidecar. The deployable base class and finite CLI remain on L4 so
the credit-only workspace needs no payment method; optional LTX motion also remains on its accepted
L4 path. The projector is already moving from its procedural draft while the fast call runs, then
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
automatically unless `BOOKFORGE_LIVE_SCENE_AUTO_PREWARM_ON_SUBMIT=true` is deliberately enabled.
With that flag, a scene submission starts the same budget-checked, text-free prewarm concurrently
with private Gemma planning; image generation still waits for the locally validated semantic plan.
The selected master/depth class keeps a 90-second warm window so Jetson planning and operator
handoff cannot accidentally trigger a second cold start:

```bash
uv sync --extra modal-authoring
modal profile current
modal deploy deploy/modal_fast_scene.py
modal app list

BOOKFORGE_MODEL_BACKEND=fake \
BOOKFORGE_ASSET_BACKEND=modal \
BOOKFORGE_LIVE_SCENE_BACKEND=modal_warm \
BOOKFORGE_LIVE_SCENE_AUTO_PREWARM_ON_SUBMIT=true \
BOOKFORGE_LIVE_SCENE_ENABLE_MOTION=false \
BOOKFORGE_ASR_BACKEND=disabled \
make dev

curl -sS http://127.0.0.1:8080/v1/live-scene-provider/warm-status
# Submit one scene from the workbench. Automatic prewarm and local planning overlap.

# Optional rehearsal/showcase window: retain the fast master/depth container for up to 10 minutes.
curl -sS -X POST http://127.0.0.1:8080/v1/live-scene-provider/prewarm \
  -H 'content-type: application/json' \
  -d '{"prewarm_id":"bookforge-showcase","include_motion":false,"scaledown_window_seconds":600}'

# Or open the operator workbench in explicit rehearsal mode. It prepares the renderer and the
# private local Gemma plan concurrently; only the local planner request contains the passage.
# http://127.0.0.1:8080/workbench?session=bookforge-live&rehearsal=1

# Explicit teardown after the demo; this terminates any remaining containers.
modal app stop bookforge-fast-scene --yes
```

The presentation window is explicit, master/depth-only, and bounded to 90–900 seconds. It changes
Modal's idle scale-down policy rather than creating an always-on minimum container, so a crashed
local API still scales the GPU to zero. Extended sessions reserve a conservative $0.70 ceiling
before starting; use the normal 90-second window outside rehearsals and stop the app after a demo.
The workbench also exposes a **Prepare full path** control. It starts the text-free Modal prewarm and
the loopback-only Gemma plan concurrently, then holds the privacy-gated semantic plan in bounded
memory and permission-restricted local caches. An unchanged passage can therefore skip Gemma inference
when **Generate** is pressed—even after a local API restart; only editing the passage invalidates the
prepared semantics, while style auditions reuse them. Concurrent duplicate preparations are
coalesced into one Jetson inference. Automatic page-load preparation occurs only when the operator
adds `rehearsal=1`; ordinary workbench visits never allocate a GPU or run Gemma.

On the August 23 acceptance, fast-only prewarm took 28.459 seconds and the following master/depth
job completed in 5.256 seconds end to end (3.528 seconds inference and 2.87 ms cache promotion).
The authoritative Modal delta was $0.01350024 under the normal atomic $0.12 session ceiling. These are
measurements, not a pricing guarantee; the provider still checks current billing and reserves the
full ceiling before prewarm. On a Mac that must retain local microphone support, sync both optional
groups with `uv sync --extra modal-authoring --extra mac-asr`.

For later warm scenes, Bookforge performs the same fresh authoritative billing check and creates an
independent one-scene reservation, but overlaps that work with local Gemma planning. A real acceptance
hid 850.856 ms of billing latency under planning and reduced projected warm end-to-end time from
6.194 seconds to 5.350 seconds. This is latency overlap, not a cached or weakened spend guard.
Completed live scenes are also persisted locally before the terminal event is published, so an API
restart restores the last finished projection instead of paying to generate it again.

The prior full-device speed pass coupled the real Jetson Gemma planner to an 896x512, 8-step warm
SANA 1.5 renderer. Its exact-code accepted scene reached `master_ready` in **6.971 seconds**: 4.285
seconds for local Gemma,
2.673 seconds for SANA plus Depth Anything, and 2.7 ms of cache work. That is 53.5% faster than the
original 15.005-second real Gemma acceptance. The 270M Gemma experiment was faster but rejected for
inventing and omitting story elements; the production choice remains the quality-preserving 1B
model. See the speed benchmark below for the rejected configurations, seed-variance evidence, and
authoritative billing reconciliation.

The production fast renderer is now the pinned two-step SANA-Sprint 1.6B checkpoint. Across three
scene types, steady image inference averaged 730 ms; the exact warm production API path reached
`master_ready` in 1.188 seconds, including depth, encoding, cache promotion, and control overhead.
Combining that measured provider result with the last exact Jetson Gemma time projects a 5.464-second
full path (21.6% faster), but that number remains explicitly projected until the Jetson is back
online for the exact combined run. Six visual trials were presentation-grade, while composition
misses in the fox and orrery prompts keep seed-level semantic validation on the backlog. SANA-Sprint
has no separate negative-prompt channel; no-text and safety direction is carried in the positive
visual prompt and provenance reports that boundary.

A same-prompt, same-seed one-step experiment was rejected. Diffusers' documented non-two-step
override reduced image inference by 17.1%, but provider end-to-end improved only 65.6 ms (5.5%) and
the central subject duplicated from one figure to two. Production settings therefore reject fewer
than two steps; the finite CLI retains the explicit override only for bounded research.

Modal's documented `us-west` routing gateway was also tested with a separately deployed function.
The master and depth outputs remained byte-identical, but provider overhead increased by 132.5 ms
and full API wall regressed by 113.1 ms. Default routing remains selected; no experimental routing
configuration remains in production source.

An exact completed-scene retry now bypasses both Gemma and Modal. Bookforge matches the normalized
passage, visual style, full session identity, and resolved master seed, then re-hashes every local
asset before replay. A valid hit publishes the usual draft → master → optional-motion stages in
milliseconds with `scene_cache_hit=true`, zero provider/inference time, and zero new GPU cost. Any
mismatch, missing file, corrupt checksum, unsupported asset, or provider change falls through to a
normal new generation.
The Story Pack store builds a newest-wins SHA-256 replay index once at startup instead of rescanning
the entire library on every request. A 1,000-scene synthetic benchmark indexed in 64.3 ms, then
averaged 0.134 ms for an exact hit and 0.066 ms for a miss; the selected file is still checksum- and
schema-validated before use, and a corrupt entry becomes a clean miss.

The projector handoff no longer spends another 680–720 ms on every artwork blend. Normal decoded,
first-painted scenes crossfade in 320 ms; depth-canvas reveal is 180 ms. A burst-safe path detects
overlapping draft/master/motion upgrades and promotes the newest prepared scene on the next display
frame instead of stacking opacity transitions. Browser stress evidence held minimum combined opacity
at 1.0, finished with one scene version at 30 fps/zero dropped frames, and never reloaded.

An opt-in short-key Gemma response contract is also ready for the next Jetson session. It reduces a
representative structured response from 272 to 226 bytes without changing the scene plan. It is
disabled by default because output size alone does not prove lower latency or equivalent semantics;
the production switch requires a real multi-passage edge benchmark.

The uncached live path now generates a labeled 512×288 provisional SANA plate while local Gemma plans
the authoritative scene. It runs only with the authenticated warm Modal provider, never receives the
source passage, and is skipped for cached/prepared plans. A warm acceptance returned the generated
JPEG in 0.972 seconds (0.614 seconds of image inference) and reconciled to $0.00701723. The projector
then replaces `preview_ready` with the Gemma-authored SANA/depth master; any preview failure is
non-fatal. The finite cold baseline remains honest at 31.892 seconds, so explicit prewarm is still a
showcase requirement. Physical Jetson-to-projector transition timing remains pending.

Modal GPU Memory Snapshots were tested separately and rejected. Snapshot creation exceeded 210
seconds without reaching preview generation, versus the proven 32.514-second explicit prewarm. The
isolated app was stopped without retry, cost $0.01308926, and never touched the production renderer.
BookForge therefore retains explicit prewarm plus the bounded 90-second warm window.

The fast warm path uses an invocation-specific L40S variant over that L4 base. A controlled A/B cut
generated-preview wall time 40.3% and full master/depth wall time 35.8%. The exact production provider
then attested `gpu=L40S`, prewarmed in 27.261 seconds, and returned a checksum-valid 896×512 master
plus depth map in 0.716 seconds. That full provider acceptance cost $0.05229226 including its bounded
90-second window; no payment method was added. If Modal does not honor L40S, BookForge fails closed
before presenting the output rather than silently weakening provenance or budget accounting.

Repeated passages now reuse bounded, content-addressed memory and local-disk Gemma plan caches even
when the visual style changes or the API restarts. Style is applied later during local SceneSpec
compilation, so it no longer consumes Gemma input tokens or invalidates the private semantic plan.
The disk record stores no source passage, uses a SHA-256 key plus model/contract revisions, is written
atomically under `0700`/`0600` permissions, and is revalidated through the current source privacy gate
on every restore. A local restart test restored a plan in 0.541 ms with zero model calls. A new seed is
still applied when the final SceneSpec is built; first-time passages remain genuinely generated on the
spot. The workbench labels cache hits explicitly.

The sub-100ms procedural draft is passage-aware rather than a generic loader. Local keyword cues
select five environmental treatments plus reader, fox, whale, and turtle silhouettes and flock,
jellyfish, fish-school, swarm, bloom, and constellation accents. Browser acceptance exercised
three-layer fox and whale drafts at 30 fps with zero dropped frames and no console errors; generated
SANA/depth artwork still replaces the draft in place when ready.

Depth Anything remains in explicit FP16. On the prior L4 baseline, both the master and depth JPEGs
were byte-identical to the FP32 baseline; model load improved 8.4%, depth
inference improved 24.3%, and the measured cold wall improved by 726 ms. A parallel checkpoint-load
experiment was 35.8% slower at prewarm and was reverted.

The automatic-prewarm acceptance then measured the cold scale-to-zero case honestly. A real Modal
prewarm took 30.744 seconds at the API boundary while the 4.288-second last-measured Jetson planning
delay ran inside that window; the subsequent two-step master/depth call took 1.318 seconds. The
overlap removed 4.288 seconds, or 11.8%, from the serialized cold-stage projection. This run uses a
measured planning surrogate because the Jetson was powered off, so the evidence does not replace a
current exact-device acceptance. `preparation_ms` is now a first-class metric and the reported
32.063-second backend elapsed time matched the 32.070-second observed wall time within 6.9 ms.

The earlier delivery pass kept that SANA 1.5 profile but changed the master plate to quality-95
4:4:4 JPEG.
On the same prompt and seed it reduced the master from 599,489 to 163,709 bytes (72.7%) at 0.9904
decoded SSIM, materially reducing cache and Mac-to-Jetson tunnel traffic. It did not measurably
improve the Modal RPC itself, so the 6.971-second result remained the last exact full-device claim.
The projector now samples a 32x18 copy of the decoded master and applies a bounded 1.0-1.22 exposure
to both WebGL and still-image fallback; bright scenes remain unchanged and no extra generation pass
is added. The linked delivery benchmark records the rejected smaller render, `torch.compile`, and
prompt A/B experiments as well as their spend.

The follow-up packaging pass also makes the depth plate a quality-85 grayscale JPEG. Across ten
real depth maps it reduced 686,418 bytes to 222,272 bytes (67.6%) while the worst decoded SSIM was
0.9944; the deployed acceptance depth was only 12,545 bytes. Master and depth encoding now run
concurrently and report `packaging_ms` through the live API. The projector loads and decodes each
plate once, reusing the master image as both its WebGL texture and fail-safe still. Modal wall time
was noisy, so this is a delivery/decoder win—not a replacement for the 6.971-second end-to-end
record until the current Sprint renderer is rerun with the Jetson online.

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
| `POST /v1/live-scene-planner/prepare` | Local Gemma | Privacy-gate and cache one passage semantic plan without starting a cloud render; alternate styles reuse it |
| `POST /v1/live-scene-planner/warmup` | Local Gemma | Optionally hide model load with fixed synthetic input and no story text |
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
- [`benchmarks/jetson-gemma3-planner-optimization-2026-08-23.json`](benchmarks/jetson-gemma3-planner-optimization-2026-08-23.json): five-passage compact-planner acceptance with a 5.24-second mean plus two bounded warm visual comparisons; the fastest full Gemma-to-master result was 9.37 seconds, the selected character-preserving result was 10.55 seconds, and both paid comparisons cost $0.03883467 combined
- [`benchmarks/bookforge-speed-optimization-2026-08-23.json`](benchmarks/bookforge-speed-optimization-2026-08-23.json): final 6.971-second Jetson Gemma → warm Modal SANA/depth acceptance, renderer sweep, rejected tiny-model/low-step configurations, seed-quality evidence, and billing reconciliation
- [`benchmarks/bookforge-resolution-ab-2026-08-24.json`](benchmarks/bookforge-resolution-ab-2026-08-24.json): same-seed warm 896x512 versus 768x448 renderer A/B; the lower plate was rejected because a 25% pixel reduction saved only 18.8 ms and did not improve inference
- [`benchmarks/bookforge-sana-sprint-one-step-ab-2026-08-24.json`](benchmarks/bookforge-sana-sprint-one-step-ab-2026-08-24.json): same-prompt, same-seed warm one-step versus two-step A/B; one step was rejected after a duplicate-subject regression for only 65.6 ms end-to-end gain
- [`benchmarks/bookforge-modal-routing-ab-2026-08-24.json`](benchmarks/bookforge-modal-routing-ab-2026-08-24.json): byte-identical default versus us-west Modal routing A/B; west routing was rejected after a 113.1 ms end-to-end regression
- [`benchmarks/bookforge-exact-scene-replay-2026-08-24.json`](benchmarks/bookforge-exact-scene-replay-2026-08-24.json): exact request replay through POST/SSE in 4.73 ms client wall with full SHA-256 asset validation and zero new provider work
- [`benchmarks/bookforge-cross-session-scene-cache-2026-08-24.json`](benchmarks/bookforge-cross-session-scene-cache-2026-08-24.json): identical text/style/default-seed replay across transport sessions in 4.088 ms with zero provider work, matching asset hashes, and current-session StoryPack rebinding
- [`benchmarks/bookforge-projector-crossfade-2026-08-24.json`](benchmarks/bookforge-projector-crossfade-2026-08-24.json): live-browser burst test of the 320 ms crossfade and immediate replacement guard; minimum scene opacity 1.0, 30 fps, zero dropped frames
- [`benchmarks/bookforge-projector-tone-mapping-2026-08-24.json`](benchmarks/bookforge-projector-tone-mapping-2026-08-24.json): zero-generation-cost adaptive gamma for dark masters; real WebGL browser pass at 31 fps with zero dropped frames, while physical-projector validation remains explicit
- [`benchmarks/bookforge-projector-native-resolution-2026-08-24.json`](benchmarks/bookforge-projector-native-resolution-2026-08-24.json): source-resolution WebGL rendering with full-size display compositing; browser pass at 30 fps/zero drops and 4.52× fewer render pixels for the 896×512 production plate, with Jetson power/FPS measurement pending
- [`benchmarks/bookforge-projector-frame-pacing-2026-08-24.json`](benchmarks/bookforge-projector-frame-pacing-2026-08-24.json): 30 Hz pacing for only the continuous WebGL depth pass, retaining native-refresh CSS/video/compositing; browser validation measured 29.2 depth draws/sec with zero drops, while the expected 50% draw reduction on Jetson's 60 Hz output remains pending physical measurement
- [`benchmarks/bookforge-persistent-edge-plan-cache-2026-08-24.json`](benchmarks/bookforge-persistent-edge-plan-cache-2026-08-24.json): restart-safe, privacy-gated semantic plan caching; a fresh planner instance restored the synthetic plan in 0.541 ms with zero model calls, while the exact Jetson restart acceptance remains pending
- [`benchmarks/bookforge-progressive-preview-smoke-2026-08-24.json`](benchmarks/bookforge-progressive-preview-smoke-2026-08-24.json): finite cold 512×288 baseline; strong visual quality and 1.475-second inference, but 31.892-second remote wall time proves cold calls are unsuitable for the live path
- [`benchmarks/bookforge-progressive-preview-warm-2026-08-24.json`](benchmarks/bookforge-progressive-preview-warm-2026-08-24.json): authenticated warm preview acceptance; 0.972-second generated JPEG, 0.614-second inference, $0.00701723 reconciled cost, explicit `preview_ready` replacement contract, and physical-projector validation still pending
- [`benchmarks/bookforge-modal-gpu-snapshot-rejection-2026-08-24.json`](benchmarks/bookforge-modal-gpu-snapshot-rejection-2026-08-24.json): isolated Modal GPU-snapshot rejection; snapshot creation exceeded 210 seconds without an artifact, cost $0.01308926, and was stopped without retry or production changes
- [`benchmarks/bookforge-modal-l40s-acceptance-2026-08-24.json`](benchmarks/bookforge-modal-l40s-acceptance-2026-08-24.json): controlled L40S acceptance for preview and master/depth; 0.580-second preview, 0.755-second full render, presentation-grade assets, and $0.01456318 reconciled A/B cost while motion remains on L4
- [`benchmarks/bookforge-concise-renderer-prompt-rejection-2026-08-24.json`](benchmarks/bookforge-concise-renderer-prompt-rejection-2026-08-24.json): same-seed L40S prompt compression A/B; shorter prompts improved brightness but repeatedly duplicated the child, so the structured production prompt remains
- [`benchmarks/bookforge-story-replay-index-2026-08-24.json`](benchmarks/bookforge-story-replay-index-2026-08-24.json): startup-built exact-replay index over 1,000 synthetic completed scenes; 64.3 ms startup, 0.134 ms mean validated hit, and 0.066 ms mean miss without repeated directory scans
- [`benchmarks/bookforge-mac-gemma-wire-ab-2026-08-24.json`](benchmarks/bookforge-mac-gemma-wire-ab-2026-08-24.json): counterbalanced offline five-scene standard-versus-compact Gemma contract A/B plus style-independent semantic caching; compact was 10.4% faster, and an alternate style reused the local plan in 0.218 ms with zero new model tokens, but the exact Jetson acceptance remains required before enabling compact mode
- [`benchmarks/bookforge-gemma-tuple-wire-ab-2026-08-24.json`](benchmarks/bookforge-gemma-tuple-wire-ab-2026-08-24.json): two independent five-scene local A/B runs of the tuple-based compact contract; all outputs passed schema/privacy/semantic review and mean planning fell 25.3%, with exact Jetson acceptance still required
- [`benchmarks/bookforge-gemma-positional-wire-rejection-2026-08-24.json`](benchmarks/bookforge-gemma-positional-wire-rejection-2026-08-24.json): zero-cloud positional-array follow-up; rejected and reverted because its explicit-slot repair saved only 1.6% warm latency while four of five passages mixed actor, action, and transformation roles
- [`benchmarks/bookforge-master-chroma-subsampling-rejection-2026-08-24.json`](benchmarks/bookforge-master-chroma-subsampling-rejection-2026-08-24.json): zero-GPU 12-scene JPEG 4:2:2 evaluation; rejected because a 15.1% average byte reduction did not demonstrate latency improvement and sacrificed chroma fidelity
- [`benchmarks/bookforge-browser-live-path-audit-2026-08-24.json`](benchmarks/bookforge-browser-live-path-audit-2026-08-24.json): isolated real-browser hot-swap and recovery audit; 64.6 ms activation, 104–130 ms reload recovery, 30 fps/zero drops, and an explicit image-decode candidate rejected after a 178 ms regression
- [`benchmarks/bookforge-local-planner-warmup-ab-2026-08-24.json`](benchmarks/bookforge-local-planner-warmup-ab-2026-08-24.json): local cold-start A/B; a fixed text-free background warmup converted a 12-second planner timeout into a 3.53-second uncached plan, with exact Jetson latency/memory acceptance still required
- [`benchmarks/bookforge-session-sse-consolidation-2026-08-24.json`](benchmarks/bookforge-session-sse-consolidation-2026-08-24.json): authoritative session-stream consolidation; 50% fewer persistent streams, no post-acceptance GET or healthy polling, 0 ms reader sync on visual upgrades, preserved cursor, and session-first projector restore without a wrong-scene fetch
- [`benchmarks/bookforge-render-delivery-optimization-2026-08-23.json`](benchmarks/bookforge-render-delivery-optimization-2026-08-23.json): JPEG delivery, adaptive projector exposure, and measured rejections for smaller renders, compilation, and prompt changes
- [`benchmarks/bookforge-packaging-optimization-2026-08-24.json`](benchmarks/bookforge-packaging-optimization-2026-08-24.json): parallel packaging telemetry, depth-JPEG quality/transfer evidence, single-decode projector delivery, and reconciled spend
- [`benchmarks/bookforge-sana-sprint-optimization-2026-08-24.json`](benchmarks/bookforge-sana-sprint-optimization-2026-08-24.json): six-image SANA-Sprint quality/speed sweep, exact 1.188-second production API acceptance, honest projected full-path boundary, and billing reconciliation
- [`benchmarks/bookforge-auto-prewarm-overlap-2026-08-24.json`](benchmarks/bookforge-auto-prewarm-overlap-2026-08-24.json): real cold Modal prewarm/generation overlap, accepted FP16 depth, a 71.1% client first-paint activation reduction, bounded presentation-window reuse, rejected shared-reservation evidence, visual/checksum review, and billing reconciliation
