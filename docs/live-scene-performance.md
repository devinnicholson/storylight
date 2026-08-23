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

LTX-Video is an optional refinement. It must preserve identity and composition, pass endpoint and
motion-stability checks, and arrive without interrupting the depth scene. A failed or malformed
video leaves the accepted depth scene in place.

## Performance and cost evidence

Each revision records backend elapsed time, provider time, inference time, cache time, measured
overhead, milestone times, warm-state evidence, model and immutable revision, GPU class, seed,
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

Typed story text, style, seed, and model configuration may reach the generation provider. Live
microphone audio, camera data, reader alignment, and reader telemetry remain local. Microphone work
is intentionally deferred until typed-text generation meets this performance contract.

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
