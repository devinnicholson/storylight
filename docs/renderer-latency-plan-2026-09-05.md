# Plan: faster verified images

Status: the September 5 comparison completed; HTTP did not qualify. See the
[results and remaining gate](renderer-latency-results-2026-09-05.md). Session prewarming remains
gated on transport qualification.

The approved $8 ceiling does not release earlier ledger reservations. The executable comparison
is reduced to one repeat of each of the six requests per transport: 12 measured images plus four
synthetic warmup images, across 14 operations. The paired order and acceptance thresholds below
remain unchanged; six samples per transport provide weaker tail evidence than the proposed 12.
The new reservation is $4.58: 14 operations at $0.22 plus $1.50 for setup. Existing reservations,
the $28 workspace stop, and the $2 billing-delay reserve remain in force.

## Objective and baseline

Reduce the time from a ready visual contract to verified artwork available for display, preserving
the current images and privacy checks. Separately move cold startup ahead of uncached reading.

The [last run](../benchmarks/scene-routing-2026-09-04/render-summary.json) measured a 3.66-second
median warm provider call, 1.63-second median image-stage duration, and a 41.58-second cold call.
These are Mac benchmark measurements, not complete Jetson-to-projector latency.

The provider timer starts after its operation lock and ends before local bundle validation and
storage. Its first call also includes billing reservation and metadata lookup. The image-stage
timer includes tokenization and pipeline execution, not just GPU kernels. The roughly two-second
gap is an investigation target; it has not been attributed to network transfer alone. Calculated
per request, the median warm residual after image, depth and packaging is 1.95 seconds.

The renderer's visual gate remains failed at two correct candidate pages out of six. This work
must not change that result or claim improved fidelity.

## 1. Measure the actual critical path

Add bounded, source-free timing records to the existing provider and benchmark:

- Client: lock wait, billing/preparation, lookup, submission, result wait/download, validation,
  local storage and total artifact-ready time.
- Server: request handling, tokenization, image generation, depth, encoding and total execution.
- Display: receipt of verified assets to successful atomic image activation, reported separately.

Correlate with opaque request IDs and hashes. Use monotonic durations within each process;
never subtract unsynchronized host timestamps. Compute residual overhead per request before
aggregating. Do not add nested model-load, compile and startup measurements together.

Separate cold startup, first use of each token bucket and steady warm requests. Preserve failed
and cancelled attempts. Confirm the client metrics on the Jetson; the Mac run remains historical.

Deliverable: a breakdown identifying the largest avoidable delay and a frozen comparison manifest.

## 2. Build one opt-in transport candidate

Compare the current authenticated SDK job path with an authenticated low-latency serving path.
Use separate experimental deployments for the instrumented SDK baseline and serving candidate,
with the same instrumented, pinned renderer runtime and baked weights. Keep the
model revision, L4 resources, precision, 1024×576 size, four steps, guidance, token buckets, seeds,
depth model and JPEG settings unchanged.

Qualify authentication, response size limits, manifest/hash verification and bounded execution
before deploying. Cover timeout during lookup, submission and result retrieval, caller cancellation,
ambiguous dispatch, duplicate submission and cleanup. Do not retry or switch providers after an
uncertain paid request. Retain its reservation. Reject the candidate if the chosen serving API
cannot preserve these controls.

Use the same default regional-placement policy initially and record actual compute and routing
locations. Different placement is a confounder, not proof of a transport improvement. Region pinning
introduces pricing and routing constraints; qualify it separately if the measured delay supports
it. No accepted deployment or appliance settings change.

Deliverable: a locally tested adapter with an explicit rollback to the existing transport.

## 3. Run one bounded matched comparison

Freeze these six existing sanitized requests from the
[original batch](../benchmarks/scene-routing-2026-09-04/visual-batch.json): candidate pages 1, 2 and
4 for the 128-token bucket; accepted pages 3, 5 and 6 for the 256-token bucket. Their earlier visual
ratings remain unchanged. This subset measures latency, not a new accuracy population.

Run each request twice on each transport: **24 measured images**. Use one explicit warmup operation
per transport, each covering two synthetic prompts: **four additional warmup images**, or **28
images across 26 remote operations** in total. Record warmup time and verify both buckets were
exercised. Alternate paired transport order using a frozen schedule. Run from the Jetson with one
in-flight request per transport and at most one GPU container per deployment.

Proposed qualification gates, frozen before execution:

- Median artifact-ready latency at least 25% below the matched SDK baseline; 2.5 seconds is a
  stretch target, not a promised result.
- No increase in p95 or maximum latency; report per-bucket results and every observation. With
  12 measured requests per transport, tail statistics are an engineering screen, not an SLA.
- Zero failed, missing or ambiguous requests; complete timing and cost evidence.
- Identical image/depth hashes for matched deterministic requests. Any difference blocks a
  transport-only success claim until explained and reviewed; better speed cannot waive fidelity.
- Authentication, integrity checks, cancellation and zero-container cleanup remain intact.

Compare cost per completed image and total experiment usage as well as speed. If the candidate
fails, retain the evidence and fix the measured cause before proposing another run.

## 4. Add bounded session prewarming

After transport qualification, connect the existing prewarm operation to an explicit reading-session
start when uncached artwork is needed. Reuse its two synthetic prompts; no story passage is needed.
Deduplicate session events and coordinate warmup with generation so they cannot race or double-spend.

Keep warmup off the local planning/draft-display path. Cached-only sessions must not allocate a
GPU. Retain the existing 90-second idle window initially, with no paid heartbeat or perpetual
warm container. Expired or failed warmup must not be reported as ready.

Warmup moves startup earlier; it does not eliminate that work. Measure session-start-to-ready and
first-image latency both with sufficient lead time and with an immediate image request. Charge
warmup and idle time to the same session envelope. Repeated UI events, early exit and cancellation
must not trigger extra generation. Additional live prewarm validation requires its own enumerated
calls within the approved budget; it is not silently added to the 26-operation comparison.

## Budget and execution boundary

Propose a maximum **$8 additional experiment envelope**: $6.50 for 26 operations at the existing
$0.25 conservative ceiling, plus $1.50 for bounded build/setup costs. Recalculate this before
execution using the actual candidate resources, startup, request timeout and idle policy. If it
does not fit, reduce scope or obtain a revised envelope; do not assume the old ceiling qualifies
a new serving primitive. Provider billing can lag and this is not a provider-enforced hard cap.

The user subsequently approved this experiment on September 5. Freeze the exact new payload,
destination and qualified cost before billable work. Preserve existing ledger reservations;
never reset the ledger to manufacture headroom. Verify zero GPU containers after the run and retain
provisional billing snapshots without calling them final cost.

## Ownership, verification and completion

| Owner | Work |
| --- | --- |
| Root | Frozen manifest, integration, Jetson execution, budget, documentation and reviewed pushes |
| Agent A | Timing instrumentation and focused boundary checks |
| Agent B | Opt-in transport adapter and comparison harness, after interfaces are frozen |
| Agent C | Independent review of authentication, timing, cancellation, cost and evidence claims |

Use disjoint file ownership. Keep tests focused on consequential behavior; reuse the existing
provider and projector tests. Run the full Python/JavaScript checks, scoped lint, changed-artifact
reproduction, privacy review and deslop before reviewed fast-forward pushes.

Completion means a measured transport decision, verified shutdown, and a bounded session-warmup
implementation with accurate readiness reporting. Report generation time, artifact-ready time and
display time separately. A passing speed experiment leaves the separate visual gate unresolved.

## Provider references

- [Modal Servers](https://modal.com/docs/guide/servers): low-latency serving and lifecycle differences.
- [Region selection](https://modal.com/docs/guide/region-selection): routing restrictions and placement pricing.
- [Cold starts](https://modal.com/docs/guide/cold-start): warm containers, initialization and idle-cost tradeoffs.
