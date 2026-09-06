# Bookforge living implementation plan

Last updated: 2026-09-05

## Latest increment — renderer latency measurement

The [eastern-region comparison](renderer-region-east-2026-09-05.md) completed 14 operations in
matched AWS `us-east-1`, with six byte-identical image/depth pairs and zero request failures.
HTTP artifact-ready median was 2.200 s versus SDK 2.611 s: 15.72% faster, below the 25% gate.
SDK container replacement also failed the warm-bucket and stability gates. Both apps are stopped
and the temporary credential is revoked. Reported C charges are $0.26456376, provisionally.

The workbench now preserves the renderer's original warm deadline while the planner runs; slow
planning or warm reuse can no longer extend displayed readiness. Expiry makes no extra paid call.
Warm image inference was approximately 1.6 s on either transport, so startup and useful warm
lifetime are the next performance decision. Promotion and automatic session prewarming remain
gated. Updated reported usage plus unreleased holds exceeds the $35 conservative stop, so no
additional paid dispatch is allowed until billing reconciliation restores headroom.

The [corrected regional recovery](renderer-region-recovery-2026-09-05.md) completed its bounded
attempt but timed out after a capacity wait and late warmup compilation. Both apps are stopped;
no images completed. Closed-run reservation reductions allowed the subsequent eastern comparison
within the already approved $35 stop.

The approved [region-controlled follow-up](renderer-region-comparison-2026-09-05.md) failed before
model initialization because a shared deployment module was missing from the container. The mount
is fixed and an isolated import regression reproduces the failure. No images completed; both
temporary apps are stopped. Its hold was later reduced to $1.89, retaining setup and the sole
dispatched operation. The repair draft is inactive; no provider promotion or session prewarming
has occurred.

The [Jetson transport comparison](renderer-latency-results-2026-09-05.md) completed 14 bounded
operations with identical paired artwork and no failures. HTTP artifact-ready median was 2.586 s
against SDK 3.154 s, an 18.02% improvement that missed the frozen 25% gate. Different compute
locations also prevent attribution to transport alone. Both temporary apps stopped with zero
containers; the accepted appliance remains unchanged. Control region/cloud placement before the
next paid comparison. Session prewarming remains gated, and the visual correctness gate is still
unresolved.

## Earlier product milestone — faithful six-page projection

The [product fidelity plan](product-fidelity-plan-2026-09-04.md) defines the next increment:
freeze a short story and visible-fact criteria, fix one bounded source-binding failure,
pass a matched live semantic gate, then evaluate a finite image comparison and complete
projector rehearsal. The story is frozen and the bounded passive construction change passes
local controls; see the [increment results](product-fidelity-results-2026-09-04.md).
The resident seven-request smoke produced zero accepted graphs and four valid unchanged fallbacks.
The development and visual gates did not advance, and the accepted appliance is unchanged.

## Latest increment — construction, grounding, and evaluator repairs

Bounded result clauses, explicit transformation counts, complete action/object grounding, and
shared-subject temporal proofs repair deterministic losses in the opt-in graph path. Semantic
comparison now tolerates articles and equivalent descriptor placement while preserving entity
bindings. Typed renderer evidence must match the actual source-validated compiler output.

The frozen first 32 training targets now produce two integrated graphs, one exact, against zero
before the repair. The accepted path remains 28 valid and four refused. Coverage is still low;
the parser retains a finite grammar, and an unsafe open-ended predicate extension was removed.
No new model inference or paid rendering ran, and the accepted appliance remains unchanged.

Stricter evaluation exposes missing watched-object, posture, and extent proof in authored public
targets. Valid target counts remain 4,096 training and 509 development; exact counts are now
3,684 and 460 under the new evaluator. Historical evidence stays pinned to its original evaluator.
See [repairs, paired controls, remaining blockers, and reproduction](live-scene-underlying-fixes-2026-09-04.md).

## Latest increment — accepted-first single-request gate

The opt-in `tensorrt_accepted_graph` backend attempts graph binding on the unchanged accepted
response and reuses that same response on refusal. It preserves cache behavior and safe long
visual styles, with graph/source/style privacy validation. The finite grammar gained bounded
explicit simultaneous clauses and directly stated result motion after synthetic diagnostics.

The separate frozen 512-record Jetson run completed without request failures but produced zero
accepted graphs. All final hashes match the accepted renderer: 346 nonempty contracts, 166
empty/refused results, zero exact cases and zero final privacy-screen failures. Median final
planning is 1.046 seconds, with p95 1.237 seconds and maximum 1.378 seconds. This meets latency
but fails semantic improvement, so promotion is rejected and no paid image comparison ran.

Scoring positive controls expose article/modifier and separate-clause representation penalties;
they do not excuse construction refusals or revise this gate. The original hybrid evidence remains
unchanged and reproducible. All 1,560 Python tests, both JavaScript suites, scoped lint, public
target coverage and retained aggregates pass. See the
[accepted-first contract, controls, results and decision](accepted-first-scene-facts-2026-09-04.md).
The next diagnostic work belongs on synthetic/public training fixtures and frozen evaluator
controls before another inference experiment. The accepted appliance remains unchanged.

## Latest increment — live SceneFacts adapter and matched Jetson gate

An opt-in `tensorrt_graph` path now converts the existing four-line hybrid response and local
source into a validated graph. The finite parser refuses unsupported or ambiguous binding;
graphs survive the local cache and compile through the existing renderer interface. A refused
graph uses a separately measured accepted-protocol fallback. Independent adversarial review
closed additional name, credential, printed-payload, motion, event, negation, and temporal-order
bypasses. The accepted appliance remains unchanged.

The 512-record development comparison uses an isolated candidate checkout on the resident
Jetson engine. Its journal retains hashes, scores, refusal codes, per-case timings/tokens, and
sampled memory/thermal evidence without source passages or raw model payloads. The installed
wheel and candidate code have separate provenance. See
[adapter, benchmark, and decision](live-scene-facts-gate-2026-09-04.md).

The completed 512-pair run rejects promotion: zero valid graphs, no final-contract improvement,
and a 2.449-second reconstructed median fallback path against the 1.5-second gate. Ten raw
hybrid privacy failures were contained; 346 final contracts passed the nonempty schema screen,
166 were empty/refused, and none failed the final privacy screen. No paid rendering was run.
All 1,509 Python tests, both JavaScript suites, scoped lint, and aggregate reproduction pass.
The read-only Jetson check still fails camera presence/enumeration; it does not block the typed
text benchmark and was not repaired in this scope. The next hypothesis is a separately gated
accepted-prompt-first adapter that preserves its original accepted output on refusal.

## Latest increment — Story Fidelity V2 fact graph

A strict, source-grounded SceneFacts V2 contract now represents entity-bound counts, attributes,
actions, motion, salience, ordered events, relations, negations, and transformations on the
private edge. The existing fidelity
evaluator now checks graph associations rather than allowing unrelated nodes to satisfy a relation
through bag-of-words overlap. Public-corpus exactness also rejects source-mentioned distractor
nodes and contradictory graph facts. The existing 4,096/512/512 corpus and hidden-split custody
remain the canonical evaluation foundation.

The deterministic public target adapter produces evaluator-exact graphs for 4,096/4,096 train
records and 509/512 development records within the 64-token estimate. The three refusals are
deliberate ambiguous same-label containment cases; no hidden record was read. Adversarial gates
reject negated positive facts, cross-entity binding, printed source payloads, prompt injection,
contact data, and conservative marked, Unicode, or lowercase clause-head name candidates. The
aggregate, value-free evidence is in
[`benchmarks/story-fidelity-v2-graph-coverage-2026-09-04.json`](../benchmarks/story-fidelity-v2-graph-coverage-2026-09-04.json).

Direct five- and six-line graph decoding was rejected on the real Jetson Gemma/TensorRT engine.
A backward-compatible four-line hybrid candidate achieved 12/12 outer-schema adherence and a
1.375-second median on 12 public development probes. In a separate five-case integrated screen it
improved automatic semantic passes from 2/5 to 3/5, while median planning rose from 1.035 to
1.399 seconds. It remains opt-in because secondary-detail accuracy is below the promotion gate.
No paid renderer or cloud service was called. The graph adapter is public evaluation/training
tooling, not yet the live kiosk parser. See
[implementation and evidence](story-fidelity-v2-runtime-2026-09-04.md).

## Latest increment — optional local hand interaction and Nsight

MediaPipe fingertip-driven fireflies are implemented behind explicit camera activation, with
local pinned assets, worker isolation, single-frame backpressure, planar calibration, and
performance stops. The actual CPU worker passed 20 repeated official-fixture frames on the Mac
at a 14.5 ms warm median; physical Jetson camera/30 FPS acceptance remains open.
A real Nsight Systems capture of the Jetson TensorRT planner completed and restored the planner
and kiosk. It confirms 91 CUDA Graph executions; the synthetic baseline/profile medians were
746.8/815.0 ms, not a production-speed improvement. No cloud service or live model route changed.
See [implementation, measurements, and hardware gate](hand-interaction-and-nsight.md).

## Latest increment — offline renderer loading and restartable compilation

Pinned Klein/DepthAnything weights now build on CPU and load offline in an isolated, portable
runtime. Three single-use L4 containers establish compiler-artifact reuse: first 128-token render
fell from 23.072 s during compilation to 8.624 / 8.355 s after restoration; model loading is a
separate 20.446 / 6.804 / 6.759 s. Warm full contracts run around 1.7–1.9 s including depth/encoding.
A paired production-parser contract experiment removed duplicate owls, foxes, and boats with
concise instructions, at 1.644 s versus 1.728 s warm median. These are small synthetic diagnostics,
not a live provider promotion or end-to-end latency claim. Exactly/count privacy false positives
and negation deletion are fixed, with both semantic and completed-pack cache versions advanced.
See [results, deployment, costs, and remaining gates](renderer-restart-optimization-2026-09-03.md).

## Latest increment — warmup-independent cache and regional compilation

The Jetson now returns validated memory/disk semantic-cache hits without waiting for model warmup;
cache-miss inference still waits safely. Cache timing now includes disk lookup. All 1,115 tests
pass. The compiled GPU-snapshot probe timed out at 480 seconds without a scene and was stopped.
Regional compilation with CUDA graphs also failed; standard regional compilation completed,
reducing warm image/depth/encoding median from 2.007 s to 1.638 s across six synthetic cases.
Its first compiled warmup was 21.940 s, with a separate 84.643 s model download/load. This remains
experimental, not a live provider switch or a fast cold-start claim. All test apps are stopped.
See [measurements, failures, and next gate](cache-snapshot-optimization-2026-09-03.md).

## Latest increment — planner integrity and compiled renderer

The second September 3 pass deployed count/name, open-book state, negation, unfinished-response,
and cache-identity fixes on the Jetson. Absence constraints no longer become required objects.
The full suite passes 1,112 tests. A bounded L4 comparison reduced warm Klein image/depth/encoding
median from 2.070 s to 1.651 s; compilation adds 42 s to initial warmup, so snapshot restoration
remains a promotion gate. Current Vertex routing is unchanged. The live planner baseline is
16/20 on the existing lexical screen; historical 20/20 used a different prompt. Experimental
prompt changes were rejected because actor/relationship errors persisted.
See [results, limits, and remaining work](planner-renderer-optimization-2026-09-03.md).

## Latest performance pass

The September 3 follow-up fixed a real uint32-to-int32 seed mismatch that unnecessarily sent
Vertex requests into a cold Modal fallback. The failing browser path took 60.4 seconds; a new
high-seed scene completed on Vertex in 5.3 seconds after the fix. The passages match, but styles
differ, so this is routing-recovery evidence, not a controlled model speedup. Repair review now
retains the original visual contract and rejects contradictory/truncated verdicts. The tested
Cloud Run worker uses direct CUDA loading with byte-identical outputs; one cold comparison fell
from 75.9 to 65.1 seconds, while warm HTTP medians remained about 0.32 seconds.

Accuracy remains an open release gate: the existing critic missed five of eight holdout contracts;
the proposed two-stage critic rejected every image and was not promoted. An isolated same-L4
comparison found Klein matched six of six tested requirements versus SANA's two of six, at about
2.37 seconds per warm image; non-truncating shorter encoding brought that to 2.02 seconds in a
12-scene diagnostic with unchanged core-requirement pass counts. It remains a candidate.
Duplicate-detail assembly and forced-single-actor wording are now fixed: a same-seed real UI
retest generated exactly two boats without the prior duplicate inset in 3.49 seconds. Artifact
reuse took 54 ms with no provider inference. Older compiler-contract caches cannot mask the fix.
The edge planner still loses some secondary objects, including the moon; its alternate instruction
failed the expanded test and was not promoted. Full suite: 1,097 passing tests.
See [current results, limitations, and release identities](performance-accuracy-2026-09-03.md).

### Earlier September 3 checkpoint

Persistent NIM engine reuse now passes a real restart test without model or vendor-code changes:
container startup fell from 631 to 229 seconds, and all 12 baseline review verdicts stayed identical.
The workbench uses bounded event-driven readiness waits, skips an unnecessary cache-hit request,
and the NIM client avoids reusing sockets beyond the server's five-second idle lifetime. These
transport changes are deployed to GKE and the Jetson; 1,070 tests pass.

An observation-first review experiment was 19.2% faster and improved count checks, but still missed
the known carrying and relative-position errors. It is not the production default. The real browser
test successfully staged a new scene, while exposing a 26.1-second cold renderer call as a remaining
latency target. See [the measurements and rejected experiments](nemotron-performance.md).

Next: independently measure renderer cold versus warm latency, test persistent CUDA extension
caches for NIM's first request, and evaluate unbiased observation plus explicit action verification
before treating Nemotron approval as reliable fidelity evidence. Cached playback stays local-only.

## Latest increment — next-page rehearsal

The exact-page workbench now prepares a privacy-gated scene through GKE/Nemotron without changing
the current projection, verifies and caches master/depth assets locally, and switches on an explicit
user action with no cloud call. Unit, HTTP, and browser-fixture verification cover the new flow.
The live six-case GKE v5 benchmark remains separate cloud evidence. The combined real Jetson/GKE
delivery test now passed offline activation; visual review found an actor-object fidelity gap.
Laptop-independent authenticated ingress and stronger action review remain open gates.
See [the current architecture and verification record](anticipatory-story-engine.md).

Connection checkpoint: the user completed the one-time privileged setup. The Jetson reports the
next-page backend enabled, and the supervised Mac → private GKE CPU API → SSH loopback bridge is
reachable from the Jetson at `127.0.0.1:18082`. The workbench's separate Mac → Jetson tunnel was
restored too. A real browser preparation attempt while Nemotron was unavailable failed at the
readiness gate without submitting a scene. The next-page panel's warmup is explicit; the existing
regular-generation panel still performs its own renderer warmup on opening the workbench.
The real next-page test rendered in 426 ms and passed Nemotron review in 5,676 ms. With the GKE
GPU scaled to zero and the cloud bridge disconnected, the Jetson activated its cached scene in
233 ms plus a configured 320 ms blend, then reported approximately 30 depth-rendered FPS.
This is one sample, not p95 or click-to-photon latency. Cold NIM startup took 1,001 seconds.
See `benchmarks/anticipatory-jetson-playback-2026-09-03.json` for the measured boundaries.

Next optimization gates:

- Treat actor-object actions as explicit acceptance requirements: this run preserved “carrying”
  in its contract, but Nemotron accepted a fox standing beside an oversized lantern. Add this
  known false positive to the next critic evaluation before claiming stronger fidelity.
- Avoid rebuilding the NIM engine for every rehearsal if a supported, reproducible persistence
  path can be validated. Do not trade an unbounded warm GPU bill for a shorter startup.
- Provide authenticated, laptop-independent ingress without exposing the Jetson's local API.
- Obtain observer confirmation of the physical projection. Kiosk telemetry is not a camera or
  photodiode measurement of the projector output.

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
