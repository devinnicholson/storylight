# Cold-start work

The user requested a dedicated cold-start effort after the matched transport comparison failed
its speed gate. This increment tests cold initialization independently of transport promotion.
It does not enable automatic session prewarming or change accepted appliance routing.

## What the existing evidence shows

The [eastern comparison](renderer-region-east-2026-09-05.md) separates three problems:

- SDK preparation took 263.849 seconds, including 29.757 seconds of runtime startup and 14.520
  seconds of warmup. The remaining 219.572 seconds includes waiting, platform startup and client
  overhead; it cannot all be labeled GPU queueing without scheduler evidence.
- HTTP preparation took 111.549 seconds between SDK warmup and the first measured SDK request.
  This exceeded the 90-second idle window. A different SDK container then served the request.
- About 11–14 seconds inside runtime startup is outside the existing model-load and cache timers.
  Framework imports occur before the model-load timer. Cache restoration itself took under a
  second; warm image inference was about 1.6 seconds.

The older Klein GPU snapshot probe timed out without a returned image. It used downloads and
full-transformer `reduce-overhead` compilation, unlike the current baked-weight/default regional
runtime. That failure does not establish whether current Klein snapshots work. Modal documents
compiler compatibility and storage-bandwidth limitations, so snapshots remain a separate
candidate rather than an assumed solution. [Memory snapshots](https://modal.com/docs/guide/memory-snapshots).

## Immediate service fix

Repeated explicit preparation requests previously acquired the same operation lock in succession
and each invoked both synthetic renders. The Klein provider now reuses a completed explicit
prewarm while its original deadline remains valid. Overlapping callers share the existing report
and reservation. Reuse does not extend that deadline or allocate another GPU call.

Generation alone does not certify that both token buckets are warm. A cold generation response,
remote failure, cancellation or failed artifact validation clears reusable readiness. Budget
reservations for attempted calls remain retained. Expired preparation still requires an explicit
request. The change adds one focused concurrent regression and extends existing failure/expiry
coverage rather than creating a large new test matrix.

## Frozen memory-request comparison

Requesting 64 GiB may restrict available host placements even when the process uses less. The
candidate requests 16 GiB while retaining the same 64 GiB hard limit; exceeding the request still
depends on available host resources. This is a capacity hypothesis, not a promised improvement.
Modal charges the higher of requested and actual CPU/memory use.
[Resource requests and limits](https://modal.com/docs/guide/resources).

Six sequential single-use SDK calls compare three fresh containers per configuration, in order
baseline/candidate, candidate/baseline, baseline/candidate. Every call loads the same pinned
runtime and compiler cache, then renders the same 128-token and 256-token cases twice: two cold
bucket renders and two warm controls, 24 images total. Cases, seeds and reference hashes come
from the completed eastern comparison. No story passages are sent.

Everything else is fixed: baked image `im-WtXer8GjRPdgMqWAAUSMwJ`, L4, eight CPU cores, AWS
`us-east-1`, model and depth revisions, precision, 1024×576, four steps, JPEG settings and default
regional compilation. The historical runtime and its compiler-cache identity stay byte-identical.
Neither provider has a public endpoint or minimum container count.

The worker flushes source-free phase events before and after framework imports, model loading,
cache restoration and each render. Responses include peak process RSS and monotonic timings.
Peak RSS describes the worker process, not total container memory or compiler subprocesses.
Client totals include lookup, submission, waiting, transfer, validation and storage. The residual
after worker time is reported as outside-worker time, never as a precise queue measurement.

Acceptance requires six distinct containers, exact model/region identity, complete artifacts,
all hashes equal to the frozen references, zero failed or ambiguous calls, and at least a 25%
reduction in median cold-cycle artifact-ready time without a worse maximum. Three cold starts per
configuration are an engineering screen, not reliable tail statistics or an availability SLA.
First-bucket inference durations are reported separately from the four-render cycle; the response
returns all four images together, so cycle artifact-ready time is not first-image display latency.

## Funding and execution

The ceiling is **6 × $0.39 + $0.50 setup = $2.84**. Each call has a 120-second startup limit,
180-second function limit, 310-second client deadline, no retries and a single-use container.
An external supervisor stops the exact app after success, first failure or timeout. Fresh opaque
request IDs, a private manifest-bound authorization, one local attempt marker and durable remote
claims prevent repeated dispatch. Only the reviewed existing image may be used; no package or
weight build is included.

The completed C experiment occupied apps for 520 and 518 seconds. Charging both full lifetimes at
the maximum regional resource rate, adding three 30-second teardown allowances (including its SDK
replacement) and the full $0.50 setup yields $1.42556912. A rounded **$1.50 retained hold** replaces
C's $5.96 ceiling; this is a conservative lifetime bound, not a claim of settled billing. Both
recorded pools had maximum one container, with no autoscaler overrides or parameter pools.
[Scaling limits](https://modal.com/docs/guide/scale), [Server replacement](https://modal.com/docs/guide/servers).

At reported workspace usage $14.12593674, retaining every other hold and reserving this $2.84
projects **$33.49593674**, within the existing $35 stop. The $30 credit amount, $7 paid
authorization, $2 reserve and $21.85 phase cap remain unchanged. Fresh billing and actual
deployment limits must pass recheck before dispatch. No failed request is retried or refunded
based on missing billing rows.

## Local qualification

All 895 Python tests, three JavaScript suites and scoped lint pass. Independent review verified
prewarm concurrency and failure handling, strict six-call authorization, source and reference
hashes, finite timeouts and cleanup. A fresh isolated Python process exercises deployment import
and rejects changed cases, request IDs, placement, expiry and duplicate claims before heavy
framework imports. Offline checks against the installed Modal 1.5.5 SDK verify the actual function
declarations, runtime mounts and asynchronous invocation signatures.

The [evidence directory](../benchmarks/renderer-cold-start-2026-09-05) retains the frozen draft,
active manifest, authorization receipt and conservative reconciliation. Authorization is private
and no proxy credential is needed for the SDK-only run. Local preflight validates the active
manifest with zero generation calls. Status: ready for supervised live measurement.
