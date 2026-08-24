# Live scene generation performance contract

Bookforge's central experience is typed story text becoming a new moving projected world while the
reader watches. Prebuilt Story Packs remain the offline fallback and rehearsal format; they are not
the primary live-generation claim.

## Progressive pipeline

| Stage | Warm target | User-visible result |
| --- | ---: | --- |
| Accepted | 50 ms | A generation job exists and can be replaced by a newer job in the same session. |
| Draft ready | 100 ms | A deterministic animated paper-theater composition moves immediately. |
| Master ready | 8 s | A text-conditioned master and depth map hot-swap into WebGL parallax. |
| Motion ready | 40 s | An optional generative video loop replaces the depth scene when it passes gates. |

Cold GPU/container startup is reported separately and must never be hidden inside the warm number.
The projector keeps the prior accepted stage visible during every upgrade; no stage may blank or
reload the projection page.

## Job contract

`POST /v1/live-scenes` accepts typed text, visual style, an optional seed, and an optional local
session identifier. It never accepts microphone audio or camera frames. A bounded local registry
returns a job immediately. Submitting another job with the same session ID supersedes the prior job
and cancels its provider task. `GET /v1/live-scenes/{job_id}` returns the current revision, while
the matching `/events` endpoint streams Server-Sent Event snapshots through:

`queued → planning → draft_ready → master_ready → motion_ready`

Every ready snapshot contains a complete one-page Story Pack. This deliberately reuses the same
validated SceneSpec, checksum-addressed asset, and projector contracts as offline playback.

## Rendering strategy

The first moving draft is procedural and derived from the generated SceneSpec: composition boxes,
camera motion, ambience, palette, and semantic trigger regions. It is an honest preview, visibly
labeled as the draft stage.

The first cloud result is a 16:9 master plus depth sidecar. The existing WebGL renderer uses depth
for a slow camera push, parallax, focus breathing, and ambient particles. This is already a moving
generated scene and is the required live milestone.

The accepted speed profile is 896x512 at two SANA-Sprint steps. Explicit prewarm runs the exact
two-step SANA-Sprint shape and Depth Anything inference, not just model loading, and the
authenticated class remains warm for 90 seconds before scaling to zero. The prior SANA 1.5 profile
needed eight steps; seven saved only about 0.2 seconds and reduced anatomical stability, while six
visibly collapsed the central subject. SANA-Sprint is distilled for one-to-four-step generation and
cut the exact warm provider path from 2.673 seconds to 1.176 seconds. Master and depth plates use
bounded JPEG encodings, and the live request never empties the CUDA cache between scenes.

For a rehearsal or judged presentation, the loopback-only prewarm request may explicitly extend
the master/depth idle window from 90 seconds to at most 900 seconds. This updates the deployed
class's scale-down window; it does not set an always-on minimum container. The GPU therefore still
returns to zero if the local API crashes or the operator walks away. Extended sessions reserve a
$0.30 conservative ceiling before launch and do not support the optional LTX class.
The workbench provides the same operation as **Prepare renderer**. Appending `rehearsal=1` to its
URL is the explicit operator opt-in that starts preparation on page load, allowing the 30–50 second
cold start to happen while the demo is being introduced. The request contains only a generated
prewarm ID, the motion flag, and the bounded window—never story text, style, audio, or imagery.

An exact 180-second rehearsal-window acceptance prewarmed in 29.869 seconds. Its first scene used
1.363 seconds of provider time; a second independently authorized scene reused the live container
and used 1.050 seconds, 23.0% faster. A follow-up attempt to share one reservation
across multiple scenes was rejected after two production replays cleared the local session after
scene one despite isolated tests passing. That code was removed, both uncertain reservations remain
fail-closed, the remote window was restored to 90 seconds, and the app reached zero tasks.

The accepted replacement overlaps a fresh, one-scene billing authorization with local planning.
It never caches authoritative spend and does not start a GPU request until both operations finish.
In the real acceptance, the billing read took 850.856 ms while the planning surrogate took 4.285
seconds; combined preparation took 4.291 seconds. The warm generation then took 1.058 seconds wall
(1.051 seconds provider), producing a 5.350-second total instead of the 6.194-second serialized
projection—a measured 844.576 ms or 13.6% reduction. Cancellation before any remote call releases
the unused reservation; once a remote call might exist, the reservation remains fail-closed. The
authoritative current-app and workspace interval delta was $0.03089237. The app was explicitly
stopped after the proof and reached zero tasks.

Every successfully completed live master is now atomically written to the existing local Story Pack
store before its terminal event is delivered. The in-progress registry and session sequence remain
memory-only, but a restart or browser reload can immediately restore the last finished projection
without another Gemma or Modal call. A storage failure never converts already-generated artwork into
a failed job; it only forfeits this restart shortcut.

When `BOOKFORGE_LIVE_SCENE_AUTO_PREWARM_ON_SUBMIT=true`, the deployed class prewarm begins at the
same time as local Gemma planning. The prewarm request contains no story text or visual prompt;
only the later generation call receives the locally validated, privacy-gated visual direction.
The feature remains explicit because prewarm allocates a bounded GPU session and retains its full
reservation after cancellation or an uncertain provider failure.

LTX-Video is an optional refinement. It must preserve identity and composition, pass endpoint and
motion-stability checks, and arrive without interrupting the depth scene. A failed or malformed
video leaves the accepted depth scene in place.

## Performance and cost evidence

Each revision records backend elapsed time, provider time, inference time, packaging time, cache
time, measured overhead, milestone times, warm-state evidence, model and immutable revision, GPU class, seed,
dimensions, checksums, and the provider-manifest cost estimate. Authoritative workspace billing is
checked before a paid call; post-call deltas are reconciled in benchmark evidence rather than
misrepresented as real-time SSE fields. Benchmarks report p50, p95, cold, and warm runs separately
once enough repeated samples exist.

The August Modal workspace had $16.17606214 of promotional credit remaining when this milestone
started. New jobs retain a $1 billing-delay reserve, never add a payment method, deploy no idle GPU,
and stop scheduling work when the authoritative workspace total reaches $29.00. Visual completion,
not credit consumption, is the goal.

## Edge/cloud boundary

During development the Mac can invoke finite Modal jobs. The resulting checksum-addressed assets
are served through the local Bookforge API and can be promoted to the Jetson cache. The provider
interface remains HTTP/job based so GCP can replace Modal without changing the live-scene API or
projector.

Raw typed story text, microphone audio, camera data, reader alignment, and reader telemetry remain
local. The Jetson Gemma planner emits a compact visual plan; a fail-closed local privacy gate blocks
distinctive source overlap and common PII patterns before only that semantic visual direction,
style, seed, and renderer configuration may reach the generation provider. This is content
minimization rather than semantic confidentiality.

The final exact-code warm acceptance reached `master_ready` in 6.971 seconds: 4.285 seconds local
Gemma planning, 2.673 seconds provider time, 2.402 seconds of GPU inference within that provider
call, and 2.7 ms cache promotion. The selected output retained the child, open book, and multiple
origami birds. A second seed at comparable speed omitted the birds and was rejected, so seed-level
semantic image validation remains a documented quality frontier rather than a hidden success.

The subsequent SANA-Sprint production-provider acceptance reached `master_ready` in 1.188 seconds
with a deterministic integration planner: 1.176 seconds provider time, 813 ms inference, 33 ms
packaging, and 3.1 ms cache promotion. Reusing the last measured 4.285-second Jetson Gemma result
projects 5.464 seconds end to end, but it is not labeled an exact combined result until the offline
Jetson reruns the current code. Across six visual samples, the book/origami scenes preserved the
central semantics while fox and orrery compositions showed bounded misses; seed-level semantic
selection remains a quality frontier.

The next planner candidate replaces only the model-facing JSON property names with short aliases.
Static contract measurement reduces a representative response by 16.9% and the schema by 31.5%
without changing the normalized `LiveScenePlan`. It is opt-in and remains disabled until the Jetson
can run semantic and truncation acceptance; these byte reductions are not presented as measured
latency savings.

The automatic-prewarm acceptance exercised a real cold deployed-class prewarm and real two-step
generation while substituting the last measured 4.285-second Jetson planning delay because the
board was off. API preparation took 30.744 seconds, planning took 4.288 seconds concurrently, and
provider generation took 1.318 seconds. The critical-path metric is therefore
`max(planning_ms, preparation_ms) + provider_ms + cache_ms`: 32.063 seconds versus 32.070 seconds
observed wall. A serialized path would have been 36.351 seconds, so concurrency saved 4.288 seconds
(11.8%) without sending story text during prewarm. This is cold-path concurrency evidence, not a
replacement for the pending exact current-code Jetson run.

A bounded Modal GPU-memory-snapshot experiment was rejected. Snapshot creation was still active
after more than 210 seconds, outside the 180-second live-function budget, so the task was stopped,
the app was verified at zero tasks, and the proven non-snapshot class was redeployed. The provider
app/day billing interval increased by $0.05521155 during that experiment. Bookforge does not claim
snapshot acceleration and does not carry the experimental snapshot flags in production.

Loading the independent SANA and depth checkpoints in parallel was also rejected: it increased
model load by 52.9% and prewarm by 35.8% on the exact L4 path, indicating resource contention.
Serial loading remains selected. A narrower Depth Anything FP16 change passed: master and depth
files were byte-identical to the FP32 baseline, while model load improved 8.4%, depth inference
improved 24.3%, and observed cold wall improved 726 ms (2.26%). Manifests pin the depth precision as
`float16` rather than leaving it implicit.

Projector activation is now decomposed into media-ready, renderer-setup, first-paint, reader-sync,
and total client timing. In the isolated no-cost browser harness, replacing the nested two-frame
crossfade wait with a forced incoming-style flush plus one display frame reduced motion-stage client
activation from 81.6 ms to 23.6 ms (71.1%); first-paint wait fell from 61.3 ms to 3.5 ms. Frame-loss
telemetry calibrates to the display's observed refresh interval instead of assuming 60 Hz. This is
browser motion-handoff evidence; the physical Jetson depth-WebGL timing remains an acceptance gate.

SANA-Sprint does not expose a separate negative-prompt input. Bookforge therefore carries no-text,
no-logo, and projection constraints in the positive semantic visual direction and records
`negative_prompt_supported=false` in the provider manifest instead of implying that a discarded
negative string influenced the pixels.

### Mac-to-Jetson development topology

During Modal development, the API and authenticated provider run on the Mac while the Jetson drives
the physical projector. The API remains bound to Mac loopback. An outbound SSH session from the Mac
creates a Jetson-loopback reverse forward instead of opening either machine to the LAN:

```bash
ssh -N \
  -o ExitOnForwardFailure=yes \
  -R 127.0.0.1:18081:127.0.0.1:18081 \
  USER@JETSON
```

The Mac workbench uses
`http://127.0.0.1:18081/workbench?session=bookforge-live`; the Jetson kiosk uses
`http://127.0.0.1:18081/projector?pack=latest&live=1&session=bookforge-live&present=1`. Override both
launcher endpoints for this bridge:

```bash
BOOKFORGE_KIOSK_URL='http://127.0.0.1:18081/projector?pack=latest&session=bookforge-live&live=1&present=1'
BOOKFORGE_READY_URL=http://127.0.0.1:18081/readyz
```

The readiness override is mandatory: leaving it on port 8080 can start the kiosk against a stale
Jetson fallback service even when the Mac tunnel is unavailable. The workbench and projector are
separate browsers, so browser-only channels are merely a same-browser optimization. Both discover
the authoritative server epoch and latest job through the long-lived server-backed session SSE
endpoint, which emits the queued job and every current-job revision—including replacements—without
waiting for polling.
The one-second session GET remains recovery-only. Assets remain checksum-addressed on the same API
host.

Before physical display acceptance, run `deploy/jetson/check-kiosk-session.sh` in the graphical
user's environment. A green API, fetched assets, and successful WebGL rendering are insufficient
proof when logind still reports a locked/idle session or X11 reports `Monitor is Off`; screen
captures in that state can be entirely black. Bookforge fails closed and asks the operator to
unlock and wake the real display. It never bypasses the lock or changes DPMS. Capture evidence only
after the preflight reports active, unlocked, non-idle, and visible, and pair the display recording
with the matching session SSE revisions.

This tunnel is a development bridge, not the final cloud topology. On GCP the Jetson-side local
service remains the privacy boundary and sends only typed scene intent to the remote generation
provider. Audio, camera input, alignment, and learner telemetry do not traverse this path.
