# Bookforge living implementation plan

Last updated: 2026-09-03

## Latest increment — next-page rehearsal

The exact-page workbench now prepares a privacy-gated scene through GKE/Nemotron without changing
the current projection, verifies and caches master/depth assets locally, and switches on an explicit
user action with no cloud call. Unit, HTTP, and browser-fixture verification cover the new flow.
The live six-case GKE v5 benchmark remains separate cloud evidence; the combined physical
Jetson/GKE/projector run and laptop-independent authenticated ingress are the next gates.
See [the current architecture and verification record](anticipatory-story-engine.md).

Connection checkpoint: the supervised Mac → private GKE CPU API → SSH loopback bridge was verified
from the Jetson at `127.0.0.1:18082`. GPU-off readiness now blocks new rendering and renderer warmup
before any paid request. The one-time root configuration helper is staged on the device at
`/home/operator/bookforge-connect-gke-301d14ec6270.py`; enabling it requires the user's `sudo`
authentication because the restricted administrator intentionally cannot edit configuration.
The live GPU remains at zero replicas until that step and the supervised acceptance run.

The original milestones below are the August 22 snapshot, retained as project history; their
deployment statuses are not the current cloud/hardware inventory.

## North-star demonstration

A child reads a physical storybook aloud. Private edge sensing follows the reading and offers a
minimal literacy intervention only when needed. Important words trigger an already-prepared,
projected world with no perceptible delay. Gemma compiles the book into a validated Story Pack;
Cosmos turns selected page art into coherent motion; Google Cloud prepares and distributes assets;
the Jetson controls the live experience without sending raw audio or video to the cloud.

## Current system

| Capability | Status | Evidence |
| --- | --- | --- |
| Local FastAPI runtime | Complete | Health, probe, intervention, compiler, and transcription APIs |
| Local speech capture | Complete | Browser MediaRecorder uploads only to the laptop |
| Local speech-to-text | Complete | MLX Whisper Base English smoke-tested on Apple Silicon |
| Local Gemma compiler | Complete | `gemma4:e2b-it-qat` produces validated Story Packs |
| Visual prompt and trigger plan | Complete | Workbench renders layers, triggers, scaffolds, and questions |
| Visual asset contract | Complete | Immutable provider, seed, checksum, dimensions, URI, state, and latency records |
| Finished visual assets | Modal vertical slice complete | Gemma-authored silver-fox page generated a checksummed 16:9 master and depth map on NVIDIA T4 |
| Cosmos integration | Not started | Target is Image2World through a provider abstraction |
| Google Cloud deployment | Scaffolded | Infrastructure exists but no project has been deployed |
| Projector runtime | Depth scene POC complete | WebGL 2 depth parallax, authored ambience, localized effects, cached fallback, and live telemetry |
| Target projector | Selected | Yaber T1 Pro; 1920x1080 input, 40-inch minimum image, 1.18:1 throw |
| Target Jetson OS | Selected with gate | JetPack 7.2.1 / L4T r39.2.1 after confirming UEFI 36.x+ |
| Jetson runtime | Software hardware-ready; device proof pending | Wheel install, preflight, persistence, WhisperTRT adapter, services, privacy and evidence tooling pass without Jetson-only imports |

## Delivery plan

### M0 — Local story compiler

Status: complete

- Record audio locally and provide visible recording state.
- Transcribe with local Whisper.
- Compile one story page with local Gemma.
- Validate all model output against the Story Pack schema.
- Preserve a deterministic intervention fast path.

### M1 — Visual asset contract

Status: vertical slice complete; durable background jobs remain

- [x] Add immutable asset records to the Story Pack: provider, prompt, seed, dimensions, duration,
  checksum, storage URI, generation state, and measured latency.
- [x] Define an `AssetGenerator` interface so Modal, MFLUX, Cosmos, and deterministic test
  fixtures share the same contract.
- Add asynchronous generation jobs and idempotency keys.
- [x] Render real generated master/depth assets in the exact projector surface.
- [x] Persist the generated manifest beside the Story Pack.

Exit evidence: the non-fixture silver-fox page became SceneSpec v2 plus a 1024×576 SDXL-Turbo
master and Depth Anything V2 map in 38.5 seconds on Modal T4. Both were checksum-addressed and
promoted atomically to the latest Story Pack. Local Whisper transcribed a spoken phrase in 412.8 ms;
the aligned word fired a spatial effect in 8.4 ms at 120 fps with zero observed dropped frames.

### M1A — Laptop projection vertical slice

Status: complete

- Add a fullscreen `/projector` route with a fixed 1920x1080 logical canvas.
- Load a Story Pack and its asset manifest without invoking Gemma during playback.
- Render deterministic fixture layers first so projection work is not blocked by cloud generation.
- Add a local session event bus with `page.loaded`, `word.reached`, `trigger.fired`, and
  `intervention.selected` events.
- Support keyboard/manual trigger simulation before connecting live word alignment.
- Add four-corner crop/perspective calibration, page bounds, blackout outside the active surface,
  and a saved `yaber-t1-pro` projection profile.
- Add frame-time, trigger-delay, preload, and dropped-frame instrumentation.

Exit criterion: the laptop can load the Moon Gate fixture, enter fullscreen projector mode, and
play every word trigger at 60 fps with under 50 ms application-side trigger delay.

Evidence: browser verification loaded five ready cached layers, advanced all eight source words,
executed all seven trigger events, saved the Yaber calibration profile, and reported 8.5 ms average
visual-onset delay at 120 fps. All projector resources returned HTTP 200 and the browser console was
clean. The manual path is now the deterministic demo fallback while live word alignment is built.

### M1B — Live reader loop on the laptop

Status: laptop POC complete; device streaming and intervention timing remain

- [x] Send rolling cumulative microphone clips while the reader is still speaking.
- [x] Produce partial local transcripts and align them monotonically to the known page text.
- [x] Emit ordered `word.reached` events without asking Gemma to decide ordinary progression.
- [x] Preserve typed and manual controls so the projector demo remains recoverable if ASR fails.
- [x] Load the latest workbench-compiled Story Pack into the same live projector runtime.
- [ ] Replace rolling two-second clips with a true streaming Jetson ASR backend.
- [ ] Detect silence/retries and invoke the intervention policy only after measured thresholds.

Laptop exit evidence: a cumulative transcript advanced all eight trusted Moon Gate words, fired all
seven cached effects in sequence, and produced a measured 7.5 ms average application-side trigger
delay with no browser console warnings/errors. A newly compiled workbench pack also rendered and
responded to its trigger through `pack=latest`. Natural child read-aloud accuracy remains a separate
ASR evaluation milestone.

### M2 — Cosmos hero-page spike

Status: pending

- Use a locked page illustration as the Image2World conditioning frame.
- Generate a short, loopable world clip with Cosmos Predict.
- Compare text-only generation against image-conditioned generation for character consistency.
- Test at least three seeds and record latency, GPU, cost, prompt adherence, temporal coherence, and
  child-appropriate visual quality.
- Reject Cosmos for the primary renderer if it cannot preserve the selected illustrated style;
  retain it for physics-rich educational scenes if that use is materially stronger.

Exit criterion: a reproducible hero clip whose use of Cosmos is visually obvious and technically
defensible.

### M3 — Google Cloud control plane

Status: pending

- Confirm the GCP project, region, billing account, quota, and contest-eligible account.
- Store source frames, clips, manifests, and checksums in Cloud Storage.
- Deploy the Bookforge orchestration API and asynchronous job queue.
- Run the NVIDIA/Cosmos workload on an appropriately sized GKE GPU node or approved managed GPU
  surface; do not force it onto hardware below the published memory requirement.
- Add authentication, structured logs, tracing, budget alerts, retry limits, and a kill switch.
- Produce a one-command deployment and one-command teardown path.

Exit criterion: a clean environment can compile a page, request a Cosmos asset, persist it, and
return a signed or authenticated asset reference.

### M4 — Instant reading and projection runtime

Status: pending

- Cache the complete book package on the Samsung 9100 Pro before reading begins.
- Support three interchangeable output modes: an on-screen camera compositor, a compact-projector
  mode for a controlled dark enclosure, and a professional installation-projector mode.
- Map word occurrences to deterministic media time ranges and visual layers.
- Add preloading, double buffering, dropped-frame telemetry, and graceful fallbacks.
- Build projector calibration for crop, keystone, page bounds, color, and ambient light.
- Ship a `yaber-t1-pro` profile targeting a 40-inch 16:9 canvas, downward gimbal placement, and
  controlled low-ambient-light presentation.
- Show an immediate lightweight effect for uncached improvisation while a richer asset generates.

Exit criterion: cached word-trigger feedback begins within 50 ms in every output mode and the
experience does not stall when the network is disconnected. A professional projector is not a
prerequisite for completing this milestone.

### M5 — Jetson private edge runtime

Status: software migration complete; physical device proof pending

#### M5A — Device provisioning and hardware proof

- Install the bare Samsung 9100 Pro in the M.2 Key-M 2280 slot and use it as the Jetson Linux,
  model, container, asset-cache, and telemetry drive. Expect PCIe 3.0 x4 rather than the SSD's
  desktop Gen5 peak speed.
- Check the board UEFI/QSPI version before installation. Version 36.x or newer can use the current
  JetPack 7.2.1 USB-ISO path; older factory firmware must complete NVIDIA's JetPack 6.x firmware
  update path first.
- Write the JetPack 7.2.1 ISO to a 16GB+ USB flash drive and install Jetson Linux directly to the
  NVMe. A microSD card is not required when the NVMe is the selected target.
- Enable MAXN SUPER after first boot, apply updates, install JetPack SDK components, and capture a
  version manifest for L4T, CUDA, TensorRT, Docker, Python, kernel, and power mode.
- Prove the real I/O path independently: USB webcam video, webcam or USB microphone audio,
  DisplayPort-to-HDMI projector output, Ethernet/Wi-Fi, and NVMe read throughput.
- Run thermal and power telemetry under a ten-minute GPU load before installing Bookforge.

Exit criterion: the Jetson boots reproducibly from NVMe and a checked-in diagnostic command proves
GPU/TensorRT, camera, microphone, projector, network, disk, temperature, and power-mode readiness.

#### M5B — Bookforge edge migration

- [x] Package speech recognition, word alignment, reader events, and intervention selection behind
  a loopback-only supervised Jetson runtime.
- [x] Implement a lazy NVIDIA-AI-IOT WhisperTRT `AsrBackend` with persistent engine path and
  serialized GPU access; keep it disabled until JetPack 7 compatibility is measured on this board.
- [x] Keep Gemma compilation and Cosmos generation out of the live loop. The Jetson loads validated
  Story Packs from private persistent storage and emits deterministic `word.reached` events locally.
- Add webcam page/hand tracking and projection-safe occlusion masks.
- Send only Story Pack requests, explicit user-authored additions, and aggregate telemetry to GCP.
- [x] Start and play cached books without network-online ordering; retain visible local connection
  state and deterministic typed/manual recovery.
- [x] Install Bookforge as a hardened supervised boot service with liveness, readiness, preflight,
  persistent state/cache, crash restart, and health-gated Chromium kiosk.
- [x] Add a process socket privacy audit and one-command JSON hardware evidence collector.

Exit criterion: raw microphone and webcam data never leave the device, verified by an outbound
traffic test.

### M6 — Evidence and evaluation

Status: pending

- Benchmark cold and warm latency for ASR, Gemma, generation, download, and word triggers.
- Evaluate transcription accuracy using scripted read-aloud samples and realistic child errors.
- Measure trigger alignment, visual consistency, intervention restraint, and recovery behavior.
- Test network loss, model timeout, corrupt assets, page changes, background speech, and silence.
- Conduct a small, consented usability study if eligibility and timing permit.

Exit criterion: every major claim in the presentation links to a measured result or recorded test.

### M7 — Contest submission

Status: pending

- Record a 60–90 second uninterrupted hero demonstration.
- Document the edge/cloud boundary and why each model is used.
- Publish reproducible setup, architecture, privacy, benchmarks, limitations, and cost.
- Map the final evidence explicitly to innovation, NVIDIA/GCP use, impact, and presentation quality.
- Complete the required learning pathway and publish the qualifying social post before the deadline.

Exit criterion: a reviewer can understand the value in ten seconds and reproduce the technical
core from the repository.

## Performance budgets

| Interaction | Target | Current evidence |
| --- | --- | --- |
| Recording-state feedback | Under 100 ms | Immediate UI state change observed |
| Local ASR, short page | Under 4 s warm | Synthetic sentence transcribed correctly in about 3 s |
| Warm one-page Gemma compile | Under 8 s | 16.0 s warm; 24.1 s cold with 8.1 s load; optimization required |
| Cached word-trigger response | Under 50 ms | 8.4 ms on the real depth scene through the local ASR/alignment path |
| Cached page transition | Under 100 ms | Not implemented |
| Cosmos generation | Offline/background job | Not measured |
| Network-loss reading mode | No interruption | Projector has a same-origin-only offline replay mode; physical Jetson unplugged rehearsal pending |

Targets are engineering budgets, not reported achievements. Only the evidence column records
observed results.

## Immediate next actions

1. [x] Extend the Story Pack schema with immutable asset records, generation states, and checksums.
2. [x] Implement deterministic fixture assets and a fullscreen 1920x1080 projector renderer.
3. [x] Add the local playback event bus and keyboard/manual trigger simulator.
4. [x] Add saved four-corner calibration and the `yaber-t1-pro` projection profile.
5. [x] Replace stop-only transcription with rolling local partial transcription, monotonic known-
   text alignment, and a local ordered event stream.
6. Provision the Jetson from the current JetPack USB ISO onto the 9100 Pro and capture the device
   manifest plus camera, microphone, projector, and thermal diagnostics.
7. [x] Add and connect a cross-platform `AsrBackend` plus WhisperTRT adapter; benchmark it on the
   Jetson against the laptop alignment fixture before enabling it in the service environment.
8. [x] Add a blocking vertical-slice build endpoint behind the asset contract; next make it a
   durable asynchronous job with idempotency.
9. Create one fixed hero-page keyframe and acceptance rubric for the Cosmos spike.
10. Confirm GCP project, region, GPU quota, and budget before creating billable resources.

## Product backlog

### Projected chess tutor

Priority: post-submission expansion; does not displace the storybook milestones.

- Calibrate the same camera/projector runtime to a physical 8x8 board.
- Let the player declare a side, point to a piece, and request legal-move overlays.
- Use Stockfish as the sole authority for legality, evaluation, and candidate lines.
- Use a constrained open model to turn engine output into progressive, age-appropriate Socratic
  hints; never allow the model to invent or approve moves.
- Track board changes locally, keep raw audio and video on the edge, and send only FEN/PGN plus
  approved learner settings to the cloud.
- Use NVIDIA GPUs on Google Cloud to turn completed games into validated personalized lesson and
  puzzle packs that download for instant offline projection.

Reason retained: this is a strong second demonstration of the reusable physical-learning runtime,
but the storybook remains the contest focus because it is more emotionally distinctive and gives
Gemma, Cosmos, Google Cloud, and the edge device naturally central roles.

## Decision log

- 2026-08-20: Keep raw audio and video on the edge; cloud receives text and approved visual inputs.
- 2026-08-20: Use Gemma as the Story Pack compiler rather than using a large cloud model for every
  live decision.
- 2026-08-20: Use Cosmos as a background world-generation layer, not in the latency-critical word
  trigger loop.
- 2026-08-20: Prefer Image2World from locked keyframes to improve character and composition
  consistency.
- 2026-08-20: Treat the Jetson as the private real-time controller; current Cosmos models require
  larger cloud GPUs.
- 2026-08-20: Treat projector brightness as progressive enhancement rather than a system
  dependency. The complete experience must remain demonstrable on a screen or with a compact
  projector in controlled lighting; professional projection is optional final-stage polish.
- 2026-08-21: Select the Yaber T1 Pro as the compact demo target. Treat 1920x1080 as the logical
  render canvas, calibrate into its 40-inch minimum optical image, and design for controlled low
  ambient light.
- 2026-08-21: Build and benchmark deterministic projector playback before connecting cloud asset
  generation. This separates live-loop correctness and latency from generative-model latency.
- 2026-08-21: Preserve the projected chess tutor as a post-submission backlog concept. Do not split
  implementation effort before the storybook contest entry is complete.
- 2026-08-21: Complete the laptop projection vertical slice before live ASR alignment. The measured
  cached visual-onset delay is 8.5 ms average, so M1B can preserve this renderer and replace only
  the manual `word.reached` event source.
- 2026-08-21: Target JetPack 7.2.1 / L4T r39.2.1 on the Orin Nano and install directly to the
  Samsung 9100 Pro using the current USB-ISO path, subject to the mandatory UEFI 36.x firmware
  check. Do not pin Bookforge to a discontinued Riva-on-Orin stack; benchmark WhisperTRT behind a
  replaceable ASR interface instead.
- 2026-08-21: Keep trusted page text as the alignment authority. ASR supplies cumulative hypotheses;
  a deterministic monotonic aligner emits progression, and Gemma remains outside the ordinary
  word-trigger latency path.
- 2026-08-21: Treat typed transcript simulation as a first-class recovery control. The verified
  browser path uses the same API, event hub, aligner, and projector renderer as local ASR.
- 2026-08-21: Allow the projector to recover `pack=latest` from device storage, then browser storage,
  then the bundled fixture. During playback, only checksum-verified loopback-cache media renders;
  cloud storage URIs and missing generation remain visibly unavailable.
- 2026-08-21: Move `pack=latest` authority to an atomic, private device store with browser storage
  as fallback. A clean installed wheel recovered the same pack after a full process restart.
- 2026-08-21: Implement WhisperTRT against its published Python API but keep activation gated. The
  upstream project does not state JetPack 7.2 compatibility, so only exact-device evidence can
  promote it from candidate to primary ASR.
- 2026-08-21: Define hardware readiness as a machine-readable acceptance artifact plus a live
  process socket audit. Configuration claims alone are not privacy or performance evidence.
- 2026-08-21: Bind every transcript and word event to a monotonically increasing reading generation.
  Reset invalidates delayed ASR, and reconnect restores the authoritative position before rendering.
- 2026-08-21: Fail privacy evidence closed on missing `/proc` visibility and require a separate
  network-disabled or packet-captured rehearsal for whole-device offline claims.
- 2026-08-21: Warm WhisperTRT and its upstream checkpoint under the persistent service cache before
  timing acceptance; the first engine build is not a live-demo latency measurement.
- 2026-08-22: Use Modal T4 as the temporary cloud Scene Foundry while GCP account setup is pending.
  It generates master and depth assets in one call; GCP will replace the provider behind the same
  interface rather than changing Story Packs or playback.
- 2026-08-22: Require SceneSpec v2 structurally in the authoring JSON schema while retaining legacy
  schema 1.1 playback. Interpret depth as an ordered non-negative plane value; only screen-space
  composition coordinates are normalized.
- 2026-08-22: Render generated master/depth pairs with a WebGL 2 depth shader and keep the master
  PNG as the zero-setup fallback. All reading-time resources remain checksum-verified and local.

## Plan maintenance rule

Update this file and the active task plan whenever a milestone changes state, a measured baseline
changes, a significant architectural decision is made, or a blocker changes the delivery sequence.
