# Cold-start work

The memory-request comparison completed all six fresh containers and 24 images with unchanged
reference hashes. Its **6.07%** median improvement fails the frozen 25% gate, so the smaller
request is not promoted. Framework imports are now measured directly at roughly 10–14 seconds.

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
manifest with zero generation calls. The memory comparison is now complete; an import-safety
audit is the next bounded step.

## Memory comparison result

| Boundary | Baseline, 64 GiB requested | Candidate, 16 GiB requested |
| --- | ---: | ---: |
| Four-render cycle artifact-ready median | 43.535 s | 40.894 s |
| Four-render cycle maximum | 52.644 s | 48.725 s |
| Framework import median | 12.463 s | 10.865 s |
| Model load median | 5.361 s | 5.113 s |
| First 128-token render median | 8.686 s | 7.396 s |
| First 256-token render median | 4.912 s | 4.839 s |
| Warm 128-token render median | 1.632 s | 1.631 s |
| Warm 256-token render median | 1.711 s | 1.718 s |

All 48 JPEG files pass size, dimension and hash verification. All three paired sets match, all
six container hashes differ, and no call failed. Lower requested RAM did not provide the required
improvement in this sample. Process RSS peaks were 19.567–19.918 GiB, above the candidate's
guaranteed amount and below its unchanged limit; this does not establish total container memory.

The first pair loaded models in 14.848/11.195 seconds; later load times were much lower. Host or
storage caching may contribute, but fresh containers do not prove fresh physical hosts or cold
storage. The 90 source-free phase events have consistent monotonic ordering and agree with the
response timings. The first 128-token render finished at a median worker offset of 26.183 seconds
for baseline and 23.570 seconds for candidate; these are not image-delivery measurements.

Independent aggregation reproduces summary SHA-256
`68e38672e7c75cefbe28012575af073d645ade688ad3bba8e656ea548ddff111` from the retained local
`.bookforge/renderer-cold-start-20260905-a/results` directory using the pinned manifest and private
authorization with `scripts/benchmark_klein_cold_start.py --aggregate-only`. All artifact bytes
are required. The supervisor completed in 265.470 seconds and stopped the app successfully;
independent inventory confirmed it stopped with zero tasks and no containers. Reported app
charges are **$0.18549067**, provisionally; workspace usage is **$14.31142741** and the full
$2.84 comparison hold remains. See the checked-in summary, phase events and cleanup receipt.

## CPU import audit

One CPU-only call audits the exact installed framework imports before attempting any snapshot.
It installs guards on nine Python-exposed CUDA discovery or initialization functions after
importing Torch, retains attempted calls even if a library catches the exception, and restores
the original functions afterward. Initial Torch import remains explicitly unaudited. A clean
result would be a necessary partial check, not proof that snapshot restoration is safe or fast.

The audit uses the same baked image, CPU request/limit of eight cores and RAM request/limit of
8/16 GiB, with no GPU or model call. One exact request is bound to a source-pinned manifest and
durable claim. The $0.70 reservation includes $0.20 for the call and the full $0.50 setup allowance;
the resource maximum for startup, execution, idle and teardown is $0.06679232. With all existing
holds retained, projected workspace exposure is $34.38142741 under the unchanged $35 stop.
The [audit evidence](../benchmarks/renderer-import-audit-2026-09-05) retains the authorization and
source pins. Implementation and independent review pass; one focused test checks authorization,
duplicate claims, swallowed CUDA attempts and restoration of the patched functions.

The audit **rejected** the import block: `torch.cuda.is_available()` was called and guarded
imports did not complete. Torch alone imported in 4.378 seconds; the whole diagnostic stopped
after 6.677 seconds inside the worker. No GPU or model was requested. The app is stopped with zero
tasks and no containers. Its reported charge is $0.00172050, provisionally; workspace usage is
$14.31314791. The full $0.70 hold remains. The checked-in result and cleanup receipt retain hashes.

Pinned Diffusers 0.39.0 imports `peft_utils`, which imports `torch_utils`. That module initializes
`torch_device = get_device()` at module scope, and `get_device()` calls CUDA availability and caches
the result. Earlier TorchDynamo imports might trigger the first observed call; the diagnostic
does not retain its caller. Forcing CUDA availability to false could preserve CPU device state
across restoration. Restoring patched functions alone is also unsafe because imported libraries
can retain references to them. See [Diffusers device initialization](https://raw.githubusercontent.com/huggingface/diffusers/v0.39.0/src/diffusers/utils/torch_utils.py)
and [TorchDynamo callable tables](https://raw.githubusercontent.com/pytorch/pytorch/v2.8.0/torch/_dynamo/variables/torch.py).

The next candidate under review captures framework imports using a GPU-assisted snapshot so
device discovery runs normally. Model loading, compiler-cache restoration and rendering would
all run after restoration. This is distinct from the older full-model/compiled snapshot that
timed out. A successful image alone cannot qualify it: the probe must distinguish captures,
actual restores and platform fallback, preserve reference hashes, and count the complete
supervised app lifetime against the existing budget. Its completed qualification is recorded below.

## Imports-only GPU snapshot qualification

The candidate keeps the baseline 64 GiB request and limit, the same baked image, L4, CPU and
region. It captures only the existing framework import block. Model construction, compiler-cache
restoration and all four reference renders execute after restoration. No CUDA function is patched.

One class with no parameters admits at most three snapshot captures and five sequential inputs.
Each input must produce the same four master/depth pairs as the cold comparison. The run stops
after two observations reuse an earlier capture ID in a different container with a fresh activation
ID, or at the first failed input. Initial hardware-specific captures are counted separately.
Provider logs must corroborate restoration and show no failed restore or fallback. Modal's runtime
can retry failed GPU restoration without a snapshot even when application retries are zero.

This qualifies snapshot correctness and repeat restoration before a matched speed comparison.
The retained cold baseline used a Jetson client; this probe uses the Mac. Comparing their absolute
client times cannot establish a causal speedup or meet the promotion gate.

The full-app watchdog starts before deployment and stops the app after at most 480 seconds, with
60 seconds allowed for shutdown. The $1.67 reservation budgets two full resource slots for that
540-second interval, eight extra 30-second teardown allowances, and $0.50 setup: **$1.58311280**.
Only one max-one pool is declared; the extra slot covers capture/restore overlap. Unexpected
concurrency stops the probe. This is a conservative engineering envelope, not a provider billing
cap. There are no builds, endpoints, automatic retries, configuration variants or redeployments.

The closed six-call memory experiment now retains **$1.60**: two full 539-second pool lifetimes,
six 30-second teardown allowances and full setup total $1.53223932. This releases $1.24 while
preserving every other hold and the original funding envelope. At current reported usage
$14.31314791, adding the snapshot reservation projects **$34.81314791** under the $35 stop.
The [snapshot evidence directory](../benchmarks/renderer-import-snapshot-2026-09-05) retains this
reconciliation, frozen authorization and result.

Local qualification passes all 900 Python tests, three JavaScript suites, scoped lint and the
staged credential scan. Four focused snapshot tests cover lifecycle separation, finite capture
and request claims, restore evidence, repeated-container rejection, expiry and cancelled
submissions. Independent review also checked the private supervisor and actual Modal 1.5.5
configuration offline. The manifest and private authorization are frozen; preflight made zero
generation calls.

## Snapshot result

The imports-only snapshot **qualifies for further comparison**. One capture was reused in two
later, distinct containers with fresh activation IDs. All 12 images and 24 JPEG files match the
frozen references. Full platform logs corroborate capture and restoration without a failed
checkpoint, retry or fallback. The supervisor stopped after the third input, in 377.156 seconds;
separate inventory confirms the app stopped with zero tasks and no containers.

| Boundary | Initial capture | First later restore | Second later restore |
| --- | ---: | ---: | ---: |
| Complete four-render client cycle | 258.686 s | 76.608 s | 35.767 s |
| Post-restore worker work | 36.713 s | 40.786 s | 27.192 s |
| Model loading | 15.626 s | 18.428 s | 5.795 s |
| First 128-token render, including depth/encoding | 11.381 s | 12.749 s | 10.346 s |

Imports took 14.692 seconds once, before capture. The two later containers skipped that work;
the returned import duration is captured metadata, not time spent again. This does not establish
a 14.692-second net latency improvement: restoration, placement and storage costs still apply.
The second restore's client cycle also includes 4.479 seconds of intentional scale-to-zero
waiting. None of these cycle times measures first-image display.

Three provider messages explicitly report waiting for L4 capacity under the fixed AWS/eastern
placement and 64 GiB request. This confirms capacity waiting occurred, without assigning every
second outside the worker to queueing. Model loading and first-bucket execution also vary widely.
The candidate is not promoted and no causal speed gate has passed.

Independent aggregation reproduces summary SHA-256
`59bc68bf5cdaa8f300da4128fffefa0c5c312db9a9e90c902a5d07b0c712e7e2`, including actual validation
of all 24 local JPEGs. The 44 phase events agree with request/activation bindings and render
durations. The final cleanup receipt resolves the raw summary's pending platform-log audit.
Reported probe charges are $0.11128800 and workspace usage is $14.42443591, provisionally;
the full $1.67 probe hold remains at this closeout.

The next initialization target is first execution. The unchanged runtime's `compile()` restores
cached artifacts and installs wrappers without running a forward pass. Warm controls remain
about 1.6–1.7 seconds, while the first bucket still spends roughly 10–13 seconds rendering after
imports have been restored. A snapshot containing the current default-compiled, warmed runtime
could remove that setup, but roughly 17 GiB of GPU state and compiler subprocesses introduce
additional checkpoint risk. It needs its own finite correctness and latency qualification.
The [warmed-runtime probe](renderer-warmed-snapshot-2026-09-05.md) now implements that next step,
with broader US placement, one capture, three requests and a separate $1.15 ceiling.
