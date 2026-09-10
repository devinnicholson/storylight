# Storylight architecture

Storylight is a local-first visual reading system. It turns spoken scene descriptions into projected artwork, and it can synchronize prepared artwork with a reader moving through a known passage. Those paths share an artifact model and projector, but they have different latency, privacy, and failure semantics.

The production path is deliberately conservative around inference. Speech stays on the local machine, scene facts are validated before a renderer sees them, paid image work is serialized and idempotent, and a completed scene is never replaced by a stale result. Experimental model planners, compiled Gemma decoding, speculative rendering, and visual criticism live behind explicit boundaries so their measurements cannot be confused with deployed end-to-end performance.

## Design constraints

The architecture follows six constraints that shape nearly every interface:

1. **Keep the reading interaction local.** Microphone audio, alignment state, projector control, and physical-display telemetry remain on loopback interfaces.
2. **Send the smallest useful representation to a renderer.** The cloud receives a reviewed visual brief rather than audio or an unrestricted transcript.
3. **Fail closed at semantic boundaries.** Unsupported scene facts, changed review digests, mismatched assets, stale sessions, and ambiguous provider failures stop the request.
4. **Keep the current projection stable.** Partial work is prepared off-screen. Publication is a separate, fenced operation.
5. **Bound cost and resource use.** Paid renders are serialized, queues and caches have fixed capacities, and automatic cross-provider failover ends once billing may have begun.
6. **Preserve evidence.** Jobs retain model provenance, cache status, stage timing, provider timing, artifact checksums, and cost-source metadata.

## Deployment topology

The demonstrated installation splits latency-sensitive local work between a Mac and an NVIDIA Jetson. All externally reachable components remain behind loopback or SSH forwarding.

```text
┌──────────────────────────────── Mac ────────────────────────────────┐
│                                                                     │
│  Browser workbench                                                  │
│  ├─ MediaRecorder ───────► local MLX Whisper                        │
│  ├─ reviewed description                                            │
│  └─ session and job streams                                         │
│              │                                                      │
│              ▼                                                      │
│  loopback voice gateway                                             │
│  ├─ /v1/audio:transcribe ──► Mac API                                │
│  └─ scene routes ──────────► SSH loopback forward ───────────┐      │
└───────────────────────────────────────────────────────────────┼──────┘
                                                                │
┌──────────────────────────── NVIDIA Jetson ─────────────────────▼─────┐
│  FastAPI service                                                     │
│  ├─ deterministic scene compiler / optional local syntax service     │
│  ├─ live-scene job registry and session fences                       │
│  ├─ Story Pack store and content-addressed asset cache                │
│  ├─ reader alignment and event fan-out                               │
│  └─ provider router ───────────────────────────────────────────┐      │
│                                                               │      │
│  Projector browser ◄── local session events and cached assets │      │
└───────────────────────────────────────────────────────────────┼──────┘
                                                                │
                          reviewed visual brief only             │
               ┌──────────────────────┬─────────────────────────┘
               ▼                      ▼
       managed Google route     GPU render workers
       Vertex / Cloud Run       Modal / experimental GKE
```

A single-machine deployment exposes the same FastAPI contract without the gateway or SSH hop. The split topology is operational, not architectural: browser clients still address one API surface, and the gateway decides which loopback upstream owns each route.

The voice gateway is intentionally narrow. It validates the socket peer, `Host`, and `Origin`; rejects forwarded identity headers; strips cookies, authorization, proxy, and hop-by-hop headers; disables redirects and environment proxies; caps audio bodies at 20 MB and other bodies at 64 KB; and refuses GPU prewarm routes in the voice demo. Audio requests terminate at the Mac ASR service. Scene requests cross the SSH tunnel as reviewed text.

## Runtime surfaces

The FastAPI application serves four cooperating surfaces:

| Surface | Responsibility | Transport |
| --- | --- | --- |
| Workbench | Microphone capture, transcript review, scene submission, and operator controls | HTTP and server-sent events |
| Live-scene API | Planning, cache recovery, provider execution, progressive artifacts, and presentation | JSON HTTP and server-sent events |
| Reader runtime | Align cumulative or partial transcripts against known page text | JSON HTTP and WebSocket events |
| Projector | Hold the last valid scene, load local assets, animate supported depth or motion, and report physical-renderer telemetry | Browser rendering and local HTTP |

Every control endpoint that changes a reading or projection session performs a loopback check. Requests carrying forwarded client-address headers are rejected because this service is not an internet-facing, multi-user trust boundary.

## Live voice-to-scene lifecycle

```text
audio
  │ local transcription
  ▼
partial/final text
  │ bounded scene compilation + privacy review
  ▼
SceneFactsV2 + reviewed digest
  │ guarded submission
  ▼
queued → planning → draft_ready → preview_ready? → master_ready → motion_ready?
                     │                │                │
                     └──────────── local artifacts ────┘
                                              │ explicit presentation
                                              ▼
                                       projector session
```

### 1. Local transcription

The browser records audio with `MediaRecorder` and uploads it to the local ASR endpoint. The default Mac backend uses MLX Whisper; the Jetson-compatible alternative uses Whisper TensorRT. Transcription is serialized behind an async lock so concurrent requests do not contend for one model instance. A temporary audio directory is removed after each request, and cancellation waits for the inference worker to drain before releasing the lock.

The ASR contract accepts English, French, or automatic language detection. It bounds upload size and file type and disables Whisper's cross-request text conditioning, which prevents one reader's prior phrase from becoming context for the next request.

### 2. Scene compilation and review

Voice text enters a revisioned, deterministic compiler. The compiler extracts a typed `SceneFactsV2` representation containing supported subjects, actions, relationships, spatial details, and negative constraints. An optional spaCy-based syntax service can add bounded linguistic analysis through a Unix-domain socket; it runs under strict payload and response limits and does not widen the cloud boundary.

Compilation enforces two invariants:

- Every positive visual fact must be grounded in the source text.
- The renderer prompt must be a faithful serialization of the validated facts.

Recognized names and contact-like details become local omissions rather than prompt content. Proper-name omissions force an explicit operator review. The server hashes the source text, style, compiler revision, facts, and omissions into a review digest; a client confirming an older digest cannot submit changed content.

An optional model planner exists for controlled experiments. The deterministic compiler remains the fallback, and model output must pass the same schema and grounding checks. Planner backends include configured model calls and TensorRT slot, hybrid, graph, and accepted-graph experiments. A scene-wide planner scope is allowed only with the accepted graph path.

### 3. Guarded job submission

A live-scene request includes a submission ID, expected server-instance ID, and expected session revision. The registry treats them as one guard:

- Replaying the same submission and payload returns the original job.
- Reusing a submission ID with changed content is a conflict.
- A server restart invalidates the old server-instance ID.
- A stale browser cannot write through a newer session revision.
- A newer request supersedes unfinished work for the same session.

The registry retains bounded submission claims even after old jobs are evicted, preventing a delayed browser retry from accidentally creating duplicate paid work. Capacity exhaustion returns a distinct error instead of silently dropping work.

### 4. Exact cache recovery

Before invoking a provider, Storylight looks for a completed scene keyed by source text, visual style, seed, and planning scope. A cache hit restores only a Story Pack whose required master and depth assets still exist and whose bytes match their SHA-256 checksums. Missing, corrupt, or mismatched data becomes a clean cache miss.

Cache hits report zero provider work and no new cost. This is enforced by the metrics schema, not left as a logging convention.

### 5. Progressive generation

Providers may emit several artifact stages. A draft is lightweight planning output; a preview is an optional early raster; a master and depth map form the minimum complete projected scene; and an optional motion loop upgrades the completed scene. State transitions are validated for legal order, monotonic progress, coherent artifacts, and timing evidence.

`master_ready` can be complete when motion is disabled or unavailable. If a post-master motion step fails, the usable master can complete with a warning. A failed job always carries a structured error and never masquerades as partial success.

The configured fidelity mode changes when the first plate may be exposed. Inline fidelity keeps the Grounding DINO gate in the render path. Deferred fidelity can return a first plate before an optional critic finishes. The mode is recorded in configuration and must not be inferred from a latency number alone.

### 6. Explicit presentation

Generation and publication are separate operations when deferred presentation is enabled. `present` verifies the server instance, session revision, current job pointer, completion state, and the request's presentation policy. It never starts or retries inference. This separation allows the workbench to prepare a scene while the projector continues displaying the last complete one.

The frontend applies the same identity discipline to progressive updates. A job ID and revision accompany every event, so late preview or master responses from a superseded description cannot replace the current scene.

## Concurrency and backpressure

Storylight assumes that inference capacity is smaller than browser concurrency.

- The job registry has a bounded active and retained-job capacity.
- Finite paid jobs are serialized, even when a provider supports broader request concurrency.
- Intermediate voice descriptions can be superseded; the latest description wins without publishing stale output.
- ASR access is serialized around the local model.
- Server-sent event subscribers receive revisioned snapshots rather than owning job execution.
- Reader event queues are bounded. When a slow subscriber falls behind, the oldest event is dropped and the next event reports `dropped_before_sequence`, allowing the client to detect the gap.

Cancellation is a state transition, not proof that remote compute or billing stopped. That distinction is why routing behavior changes after a provider begins generation.

## Provider routing and the billing boundary

The live-scene provider interface supports local fixtures, Modal GPU workers, a warm Modal route, the constrained Klein research route, Google Cloud Run, and a resilient Google-first composition that can include Vertex and Modal.

| Route | Intended role | Architectural notes |
| --- | --- | --- |
| `fake` | Deterministic tests and UI development | No external inference or cost |
| `modal` / `modal_warm` | GPU-worker rendering | Cold and warmed execution are reported separately |
| `modal_klein` | Constrained FLUX.2 Klein candidate | Fixed dimensions and sampling settings; incompatible options are rejected |
| `gcp_cloud_run` | Managed container inference | Readiness is probed before work begins |
| Vertex image route | Managed Google image generation | Credentials remain server-side |
| `gcp_resilient` | Ordered Google routes with optional GPU fallback | Failover is legal only before billing may have started |

The resilient router probes readiness before selecting a route and caches healthy or unavailable states for short TTL and cooldown windows. Connection preparation may prime the preferred Vertex endpoint with a `HEAD` request, but it never starts inference.

The critical rule is simple: fallback is permitted after a failed readiness probe or an explicit safe-fallback signal, because no billable generation began. Once a generation call starts, a timeout, disconnect, or unavailable response is billably ambiguous. Storylight marks that attempt terminal and does not automatically invoke another provider. This avoids duplicate spend and two conflicting images for one submission.

## Story Packs and artifact integrity

A Story Pack is the durable contract between inference and playback. Strict Pydantic models reject unknown fields and validate page geometry, layers, triggers, camera motion, ambient effects, literacy support, comprehension prompts, and asset provenance.

Generated bytes enter a content-addressed cache:

```text
provider bytes
  ├─ validate kind, dimensions, and allowed suffix
  ├─ compute SHA-256
  ├─ write private temporary file
  ├─ fsync + atomic replace
  └─ expose /v1/assets/{checksum}/{filename}
```

Cache directories use private permissions. Asset paths are checked against traversal and package escape, and persisted Story Packs use checksum-derived filenames with an atomically updated `latest` pointer. Artifact and Story Pack records must agree on checksum, dimensions, duration, seed, URI, role, provider, model, and compiler provenance.

This makes the local asset boundary independently verifiable. A provider response is insufficient by itself; playback consumes bytes that have been validated, stored, and linked through a typed pack.

## Known-text reading and projection

Prepared reading avoids fresh inference in the word loop.

1. A trusted Story Pack supplies page text, word triggers, and local asset references.
2. The reader receives cumulative or partial transcript updates.
3. A deterministic longest-common-subsequence aligner maps normalized, case-folded tokens onto the known page.
4. Word progress is monotonic. Interpolated timestamps produce `word.reached` events.
5. A per-session generation number rejects events from a reset or prior page.
6. The event hub publishes bounded, monotonically sequenced updates over WebSocket.
7. The projector applies trigger actions to cached artwork.

The event vocabulary includes partial transcript updates, word reaches, and session resets. Direct event injection is disabled on the Jetson deployment. Because assets are already local, advancing a highlight, animation, or parallax effect does not require a model call for every word.

Depth assets are provider-dependent. An estimated depth map and an authored projection gradient are both valid typed assets, but they are not interchangeable geometric measurements. The projector reports local FPS, dropped frames, active depth mode, and job stage so physical display behavior can be separated from server-side inference timing.

## Prepared and anticipatory scenes

The anticipatory path moves optional next-page work outside the reading-critical interval. It is separate from ordinary live generation and never publishes during preparation.

```text
local known-next text
  │ local planner removes raw interaction data
  ▼
sanitized AnticipatorySceneSpec
  │ bounded remote batch
  ▼
queued → rendering → critiquing → repairing? → ready
                                                │ commit one branch
                                                ▼
                              download + verify master/depth
                                                │ local stage
                                                ▼
                                   fenced projector activation
```

The edge specification carries a privacy attestation stating that raw text, audio, camera data, and reader identity were removed. Candidate batches have expiration times, render-cost ceilings, session-cost ceilings, fixed capacity, and bounded concurrency. Exact cached candidates can bypass rendering only if their referenced result is still available.

A visual critic may approve, reject, or request one bounded repair. Committing a branch cancels siblings from the same sequence. For an exact known-next page, the playback layer downloads the selected master and depth assets, verifies their digests and dimensions, stores them locally, constructs a Story Pack, and asks the remote coordinator to commit that exact scene. Only then can a separate local activation update the projector session. Once staged, activation does not require cloud connectivity.

The anticipatory coordinator is intentionally in-memory: losing it discards optional lookahead work, while committed local Story Packs and the current projection remain usable.

## Failure semantics

| Failure | Result |
| --- | --- |
| Unsupported or ungrounded scene fact | Compilation fails closed before provider access |
| Changed facts after operator review | Digest mismatch; submission rejected |
| Stale browser or restarted server | Instance or session-revision conflict |
| Replayed identical submission | Original job returned idempotently |
| Replayed submission with changed payload | Conflict; no new render |
| Missing or corrupt cached asset | Cache miss; corrupt result is never presented |
| Provider unavailable before generation | Next eligible route may be tried |
| Timeout or disconnect after generation starts | Terminal ambiguous failure; no automatic cross-provider retry |
| Preview or motion failure after a valid master | Previous scene remains, or master completes with a warning where supported |
| Superseded partial description | Old job cannot become the session's current scene |
| Slow reader-event subscriber | Oldest queued event dropped and the sequence gap is reported |
| Prepared candidate expires | Candidate cannot be committed or activated |

## Privacy and trust boundaries

| Data | Default location | Persistence | May cross a cloud boundary? |
| --- | --- | --- | --- |
| Microphone audio | Browser and local ASR process | Temporary file removed after transcription | No in the demonstrated split setup |
| Raw transcript | Local workbench and API | Not part of generated asset payloads | Other non-voice configurations may differ |
| Reviewed scene facts and visual brief | Local compiler | Stored with scene metadata where applicable | Yes, when a remote renderer is configured |
| Names and contact-like details | Local omission record | May appear in local review evidence | Excluded by the bounded voice compiler |
| Generated image, depth, and motion | Provider, then local cache | Content-addressed local assets | Produced remotely when a cloud provider is selected |
| Reader alignment and projector telemetry | Local API | Runtime state; timing traces omit content | No by default |

The compiler's bounded vocabulary and pattern checks reduce disclosure; they are not a general privacy proof. The repository makes no claim of child-data regulatory compliance, a complete security audit, or suitability as an internet-facing multi-tenant service. See [Privacy and data flow](privacy.md) for the exact demonstrated boundary.

## Measurement and provenance

Each live job carries structured metrics rather than a single latency field:

- elapsed, planning, preparation, provider inference, provider overhead, packaging, and cache time;
- per-stage milestones;
- cache-hit status and warm-state classification;
- GPU and provider identity;
- model role, model name, revision, and backend provenance;
- estimated cost and whether it came from a fixture, provider manifest, or was unavailable.

Validators reject internally inconsistent evidence. A cache hit cannot claim fresh inference or new provider cost, stage timing cannot move backward, and a Story Pack's compiler model must match recorded scene-plan provenance. This matters when comparing a warm kernel benchmark, a prepared-scene activation, and microphone-to-projection latency: they measure different systems.

## Experimental inference architecture

Research code is isolated from the default voice path and must pass explicit promotion gates.

### Structured planning with Gemma

The experimental planner fine-tunes Gemma 4 E2B with NF4 QLoRA on synthetic scene-extraction examples. At inference time, a grammar constrains output to the scene schema. The optimized decoder separates dynamic prefill from compiled token generation:

```text
prompt + grammar state
        │ dynamic prefill
        ▼
   model KV state
        │ transfer into fixed-shape buffers
        ▼
 compiled decode step ── grammar mask ──► next token
        ▲                                  │
        └──────── fixed-shape KV update ───┘
```

The hybrid design avoids compiling variable-length prompt work while keeping the repeated decode step graphable. Tensor identity, grammar acceptance, and refusal behavior are independent gates. The current adapter improved strict positive extraction and preserved token identity in the measured hybrid comparison, but it regressed the refusal gate and has not replaced the deterministic compiler.

Compiler caches are keyed by actual graph inputs and shapes. Experiments showed that changing the random seed could change attention or rotary input order and invalidate reuse, so a fast cache hit is recorded as a narrow compiler result rather than attributed to model quality or end-to-end latency.

### Open image generation and criticism

Open image candidates include FLUX.2 Klein 4B and SANA-Sprint on GPU workers. Managed Vertex generation is a separate route and is labeled separately in provenance. A Nemotron visual critic can run through NVIDIA NIM on a GKE L4 worker; persisted TensorRT engines reduce same-node restart preparation time, while verdict stability is checked independently from startup speed.

These components demonstrate the intended promotion path:

```text
offline experiment
  → reproducible artifact and provenance
  → semantic, refusal, fidelity, and cost gates
  → constrained runtime adapter
  → deployment-specific validation
  → optional production route
```

Model-only throughput, prepared-scene latency, and same-node engine startup are never reported as live microphone-to-image performance. Current measurements and their limitations are documented in [Research results](research-results.md).

## Deployment profiles

| Profile | Compute placement | Intended use |
| --- | --- | --- |
| Local fixture | One machine, fake renderer | Tests, UI work, deterministic demos |
| Single host | Browser, ASR, API, cache, and optional local models together | Development and compact installations |
| Mac + Jetson | Mac microphone/ASR, Jetson orchestration/cache/projector, SSH loopback link | Demonstrated local-first installation |
| Managed image route | Local control plane with Google or Modal rendering | Live high-quality generation with server-side credentials |
| Research GPU route | TensorRT, open image models, or NIM on provisioned GPUs | Benchmarks and promotion-gated experiments |

## Module map

| Module | Responsibility |
| --- | --- |
| [`api.py`](../src/storylight/api.py) | HTTP, SSE, and WebSocket surface; loopback enforcement |
| [`asr.py`](../src/storylight/asr.py) | Serialized local speech recognition and temporary-file lifecycle |
| [`voice_gateway.py`](../src/storylight/voice_gateway.py) | Route allowlist and loopback Mac/Jetson split |
| [`reviewed_description.py`](../src/storylight/reviewed_description.py) | Source grounding, local omissions, review digests, and fail-closed compilation |
| [`scene_facts.py`](../src/storylight/scene_facts.py) | Typed scene representation and renderer-prompt contract |
| [`live_scene.py`](../src/storylight/live_scene.py) | Job state machine, idempotency, session fencing, cache recovery, and presentation |
| [`provider_router.py`](../src/storylight/provider_router.py) | Readiness-aware routing and pre-billing fallback policy |
| [`asset_cache.py`](../src/storylight/asset_cache.py) | Content-addressed, checksum-verified, atomic asset storage |
| [`story_store.py`](../src/storylight/story_store.py) | Story Pack persistence, latest pointer, and exact-scene index |
| [`reader.py`](../src/storylight/reader.py) | Deterministic transcript-to-page alignment |
| [`reader_runtime.py`](../src/storylight/reader_runtime.py) | Per-session alignment state and generation fencing |
| [`event_hub.py`](../src/storylight/event_hub.py) | Bounded, sequenced local event fan-out |

## Explicit limits

- Automatic camera page tracking is not an established feature.
- Reading-outcome improvements have not been demonstrated by this repository.
- A generated depth estimate is not calibrated scene geometry.
- The loopback gateway is not an authentication layer for hosted deployment.
- The current learned planner has not passed every refusal gate.
- Research inference timings do not include the entire microphone-to-projector path.
- Open-model weights, compiled engines, cloud credentials, and generated caches are not distributed with the source tree.

## Related documentation

- [Privacy and data flow](privacy.md)
- [Research results](research-results.md)
- [Demo guide](demo.md)
