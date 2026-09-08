# Prepared inference sessions — 8 September 2026

The repaired r/s/t cohort returned **30 of 30 measured results in under one
second**: pooled median **0.512 seconds**, observed maximum **0.981 seconds**.
Preparation took **68–83 seconds** per session. Supervised lifecycles took
**254–275 seconds**, including provisioning, access propagation, measurement
and deletion. All three release checks verified historical zero usage and current
service absence.

| Session | Preparation | First ready delivery | Median delivery | Maximum delivery |
| --- | ---: | ---: | ---: | ---: |
| r | 68.044 s | 0.717 s | 0.583 s | 0.981 s |
| s | 82.730 s | 0.729 s | 0.486 s | 0.729 s |
| t | 81.545 s | 0.736 s | 0.527 s | 0.736 s |

Delivery includes HTTP transfer, complete master/depth JPEG decoding, validation
and saving. It excludes microphone capture, speech recognition, scene planning
and browser/projector presentation. Preparation is the warmup request's elapsed
time, excluding service provisioning. These are prepared-renderer results;
they do not establish microphone-to-projector latency.

All **72 JPEGs** match their corresponding images from successful session o
byte-for-byte. Rich watercolor detail remains, along with the known extra silver
fox and lantern. The prepared serving adapter remains inactive; the existing
production image provider is unchanged.

## Configuration and measurement

Each fresh private Cloud Run service used one RTX PRO 6000 Blackwell GPU,
20 vCPU, 80 GiB memory, concurrency one, and minimum/maximum one instance.
The unchanged renderer used FLUX.2 Klein 4B, BF16, four steps, guidance 1,
1024 × 576 output, and the existing depth model and watercolor prompts.

Two actual renders prepared the 128- and 256-token execution buckets. After
30 seconds idle, each service handled the frozen eight-case corpus followed
by repeats of its first two cases: two warmups plus ten measured deliveries.
Each session had a distinct process identity. Physical host placement and image
caches were uncontrolled; these were fresh services, not verified hardware cold boots.

All three sessions passed the predefined speed screen: median at most one second
and observed maximum at most two seconds. Thirty successes are a bounded screen,
not a production reliability guarantee. The earlier successful o session is
retained separately and is not substituted into the repaired cohort.

## Readiness, transport and application boundary

Readiness is bound to the prepared process identity, exact runtime identity and
both warmup receipts. Its 300-second expiry starts with preparation and never
extends. Replacement, expiry, unsupported buckets and ambiguous responses fail
closed. Work is serialized; cancellation drains in-flight GPU work before cleanup.

The reviewed transport wrapper requires fresh, token-free TLS verification for
the renderer and ten control-plane hosts. It keeps original URL hostnames, SNI
and certificate checks while using verified addresses for its finite process tree.
Its 720-second outer limit includes preflight and preserves the supervisor's
600-second work limit plus 60 seconds for cleanup. No failed image is retried.

The separate serving candidate accepts compiled prompts, verifies fixed leases
and request identities, validates complete JPEGs, and can reconstruct saved
bundles under its exact model contract. Persistent local claims precede dispatch;
there is no automatic HTTP retry or fallback. Its pooled HTTP client can reuse
live connections and closes after active work drains. This pooling change was
not part of the frozen cohort and has no measured speed gain.
[HTTPX connection pooling](https://www.python-httpx.org/advanced/clients/).

Original-source privacy validation remains mandatory on the caller's local
reviewed-description path. Raw speech, source text and omitted named locations
stay local; the renderer receives the admitted compiled prompt. The adapter's
obvious-sensitive-data check does not replace source-aware privacy validation.

The browser/API guarded-submission change binds submission IDs, server identity
and session revision. Exact replay reconciles an accepted job; changed payloads,
stale revisions and server restarts reject before a new generation. Existing
scenes remain visible until replacement assets are complete. This application
behavior is separate from the measured renderer endpoint.

## Failures retained and repaired

- **n:** Preparation failed without an image. Source inspection found that the
  wrapper omitted `app.PACK_PROOF_SHA256`, required by the inherited loader.
  An explicit export and actual-helper regression repaired that contract. The
  original failure had no exception trace, so its exact cause is not asserted.
- **p:** Both warmups completed, but the first delivery failed with `ConnectError`.
  The error does not distinguish DNS, TCP or TLS failure. Automatic deletion also
  timed out; a separate recovery deleted the service. Original failed evidence
  remains intact. The transport wrapper repairs a demonstrated operational risk
  without claiming to identify p's precise cause.
- **q:** Deployment succeeded, but local client validation rejected its service
  name before authentication or render dispatch. Manager/client allowlists now
  agree, and pre-build admission runs the actual client validator. Every admitted
  name has a regression. q supplies no renderer-delivery measurement.

The r/s/t cohort used the reviewed repairs with identical workloads. Every failed
attempt remains in the evidence and retains its budget reservation.

## Bounds, verification and remaining work

Seven **$3.50 reservations total $24.50** against the approved $25 allowance.
All seven are consumed for admission accounting; failed attempts are not recycled.
This is reserved allowance, not invoiced cost or a cloud-enforced spending cap.
The configured instance estimate is $0.00088522/second before discounts, including
GPU, CPU and memory. [Cloud Run pricing](https://cloud.google.com/run/pricing).

Each build was bounded to 600 seconds. Cleanup required observed service absence
and terminal creation state. Release observation is separately bounded to
900 seconds. A successful observer decision means historical zero metrics plus
current service absence; it does not prove current instance count, quota
reservation or billing closure. The adapter's `aclose` is not cloud deletion:
an external lifecycle owner remains required.

[Retained evidence](../benchmarks/prepared-gcp-2026-09-08/) binds source snapshots,
external context pins, generation-bound build archives, immutable image digests,
all 17 inherited layers and the exact application overlay, dispatch journals,
images, noise receipts, timings, cleanup and release observations. No new full
model tensor-equivalence claim follows merely from retaining the parent layers.
The [generated results report](../benchmarks/prepared-gcp-2026-09-08/report/results.md)
replays all three matching protocols, distinct workers, route evidence and release
records before marking the 30-delivery screen passed.

The inactive implementation was pushed to main as `ac0716f`. Validation passed
**1,084 project tests, 74 combined
adapter/experiment tests, and six JavaScript suites**, with independent code and
style review. Native prepared serving remains inactive.

The next quality screen should use the actual modern compiler rather than the
older prompt that produced the extra fox. The [local fixture receipt](../benchmarks/prepared-gcp-2026-09-08/support/modern-voice-cases.json)
and [generator](../experiments/renderer-session/voice_cases.py) retain eight typed
fixtures and propose six modern cases alongside the original warmup controls.
They bind compiler/source/tokenizer provenance; they are not ASR accuracy or
rendered-quality results. Raw source text and the London omission stay local.
No GPU execution of that screen is authorized by the remaining allowance.

The historical factual review scored 6/8; a later reviewer passed the unchanged
owl perch (7/8). That disagreement is not a pixel or quality improvement.
The extra fox remains a clear failure, so these speed results do not authorize
production promotion or claim human/projector acceptance.
