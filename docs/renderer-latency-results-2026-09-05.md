# Renderer latency comparison

The HTTP candidate did not qualify. Its median verified-artwork latency was **2.586 seconds**
against **3.154 seconds** over the SDK, an **18.02% improvement**, below the frozen 25% threshold.
Compute placement also differed: SDK used AWS `us-west-2`; HTTP used GCP `asia-northeast3`.
Both routing proxies used `us-east`. This is evidence about these two deployments, not an isolated
measurement of transport overhead. The accepted provider and appliance remain unchanged.

## Implemented and measured

The approved [plan](renderer-latency-plan-2026-09-05.md) produced a separate instrumented runtime,
authenticated SDK and HTTP deployments, a bounded binary response format, and a Jetson comparison
harness. Runtime, compiler cache, weights, seed, precision, four steps, dimensions, depth model,
and JPEG settings remain pinned. Neither original renderer source file changed.

The reduced batch completed **14 calls: 12 measured-image calls and two prewarm calls, producing
16 images in total**.
Each of six frozen sanitized requests ran once per transport. All 12 measured requests used a
previously warmed token bucket. One container served each transport throughout the run. All six
matched master/depth pairs are byte-for-byte identical, with every stored checksum and dimension
verified. There were no failed, missing, retried, or unresolved requests.

| Measurement | SDK | HTTP |
| --- | ---: | ---: |
| Warm artifact-ready median | 3.154 s | 2.586 s |
| Warm artifact-ready p95 / maximum | 4.541 s | 3.470 s |
| Image-stage median | 1.577 s | 1.591 s |
| Depth-stage median | 0.056 s | 0.076 s |
| JPEG encoding median | 0.0049 s | 0.0079 s |
| Local validation median | 0.0017 s | 0.0017 s |
| Durable local storage median | 0.0168 s | 0.0164 s |
| Cold startup plus two-image warmup | 48.202 s | 64.718 s |

With only six measured requests per route, nearest-rank p95 equals the maximum. These observations
are an engineering screen, not a production latency guarantee. The earlier 3.66-second Mac result
used a different population and timer boundary; it is not a matched before/after baseline here.

The image-stage timer includes external tokenization and synchronized pipeline execution.
`pipeline_seconds` includes text encoding, denoising and decode; CUDA event duration is an elapsed
stream interval, not a sum of kernel execution times. The handler's `server_seconds` excludes
the durable request claim and surrounding transport boundary. Per-request artifact-ready minus
handler time has a 1.521-second SDK median and 0.886-second HTTP median. That residual includes
platform, client, transport and storage effects; it is not a direct network measurement.

HTTP submission timing ends when response headers arrive and therefore includes rendering.
SDK submission timing ends when its call handle arrives. Only their complete artifact-ready
durations are directly comparable. Billing preparation happened once on the Mac before dispatch.
No unsynchronized host timestamps were subtracted. Physical projector activation was not measured;
the existing projector activation timings remain a separate surface.

## Operational evidence

The HTTP proxy rejected unauthenticated access with `401`. Cold readiness polling was bounded and
never retried a generation POST. Requests were restricted to the frozen allowlist, claimed durably,
and bounded by deadlines and response limits. Local tests covered ambiguous SDK submission,
late-handle cancellation, HTTP disconnection, timeout, replay and output corruption. These failure
controls were tested locally; this paid run did not inject extra failing generation calls.

Both experimental apps are stopped with zero containers. Their temporary proxy token was revoked
and its local/device files removed. The accepted Jetson API remained ready, with its original
planner process still running. No private story package was retrieved or submitted.
The client summary leaves external shutdown unverified; `cleanup-cost.json` supplies the separate
cloud-state verification.

The $4.58 reservation remains held. Reported workspace usage moved from $13.71088685 to
$13.83014709, a **provisional $0.11926024** delta. The provider currently attributes $0.07042676
to SDK and $0.04883348 to HTTP, including their different lifetimes and warmups. Dividing each by
eight completed images gives about $0.00880 and $0.00610; these are provisional batch averages,
not marginal image prices or settled billing.

The approved additional $8 did not reset earlier reservations. The shared phase cap was explicitly
amended from $10 to $14.85 under the existing ledger lock, preserving previous reservations,
records and estimates. The $30 monthly boundary, $2 reserve and $28 workspace stop remain.
After the new reservation, conservative projected workspace usage was $27.54088685. The initial
deployment packaging error was corrected before any image dispatch; its superseded manifest and
authorization chain are retained. It did not reset an execution attempt.

## Decision and remaining work

Keep HTTP experimental. The next useful comparison must control compute region and cloud; the
current default placement sends the two routes to different continents. A region-constrained
deployment needs its own frozen pricing and placement qualification before another paid run.

Session prewarming remains **unimplemented and gated on transport qualification**, as specified
in the plan. The current “Start reading” control already requires ready artwork, so attaching
paid warmup there would waste compute. A future explicit preparation operation should check for
verified upcoming artwork, deduplicate session starts, serialize with generation, cancel on exit,
and preserve the 90-second idle window. Cached reading and local draft planning must stay free of
warmup. Product session prewarming was not enabled.

The independent visual gate remains failed at two correct candidate pages out of six. Identical
bytes in this transport experiment do not improve that result.

## Evidence and reproduction

The [evidence directory](../benchmarks/renderer-latency-2026-09-05) retains the frozen manifest,
all source-free observations, independently reproduced summary, authorization/amendment chain,
provisional billing, authentication, supervisor and shutdown receipts. Generated JPEGs are retained
locally at `.bookforge/renderer-latency-20260905-a/results` and on the isolated Jetson checkout.

`scripts/benchmark_klein_latency.py --aggregate-only` verifies the retained JPEGs and reproduces
the summary without any provider call. Supply the retained manifest, authorization, its SHA-256
from `predeploy-correction.json`, and the results directory. It writes `recomputed-summary.json`
exclusively; use a fresh copy of the retained results when repeating this check.

Verification: 885 Python tests, both JavaScript suites, scoped Ruff, source-pin checks and whitespace
checks passed. Public graph coverage reproduced byte-for-byte. Independent review reproduced the
latency decision, checked every JPEG pair and confirmed both apps' stopped/zero-container records.
