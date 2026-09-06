# Native GCP direct-device loading comparison — September 6, 2026

This is a prospective **P/Q/R/S ABBA screen**, prepared offline. It compares the current
CPU-first loading sequence with `device_map="cuda"` during Klein pipeline loading. Both
arms have the same four synchronized timing stages. Root activated the reviewed plan
under the user's instruction to continue GCP optimization and increase its budget. The
separate gross allowance is **$12**, including GPU release tails after every service.
There are no results or promotion claims at freeze time.

The [research note](gcp-klein-device-loading-research-2026-09-06.md) records the pinned
library support and earlier negative comparisons. The
[comparison plan](../benchmarks/gcp-klein-device-loading-2026-09-06/comparison-plan.json)
binds the source hashes, cases, order, resources and prospective gates. The
[runtime AST proof](../experiments/renderer-device-loading/runtime-diff-proof.json)
verifies that the candidate's only difference from the instrumented baseline is the
literal device-map keyword; all non-constructor code remains original.

## Fixed comparison

Build two small runtime/application/manifest overlays on G's immutable image
`sha256:e18009ca1785394b84340be7f462bd6cb4ee2b4455484a9f3d7769fbc26ffaa8`.
Verify both image digests, unchanged base layers and runtime configuration, and exact
overlay file hashes before P starts. There is no weight rebuild, model download,
compiler-cache import, precision change or inference-method change. Keep both manifests
valid through the last possible generation; final deletion cleanup and the read-only
release guard need no live manifest. A changed expiry requires an explicit new build
artifact, not a silent extension.

The prepared baseline context contains 25,776 bytes and the candidate 25,807 bytes.
Both are pinned file-by-file in the plan, with manifest expiry `1788743092`. Their worker
and runtime copies match the reviewed sources, and all eight original cases match G.
Image digests remain pending until the two builds; they are required before service
dispatch, not before building the already pinned contexts.

Expiry admission requires at least **6,480 seconds before the first build**:
two 600-second builds, three complete 1,560-second service/release cycles, and the last
600-second work window. After both builds, recheck at least **5,280 seconds remaining
before P dispatch**, retaining another 120 seconds of orchestration margin if available.
Stop before dispatch if this bound no longer fits. The worker's maximum future expiry
remains 7,200 seconds; do not extend it to cover the final read-only guard.

| Order | Exact service | Arm |
| --- | --- | --- |
| P | `bookforge-klein-qualification-20260906-p` | Instrumented CPU-first baseline |
| Q | `bookforge-klein-qualification-20260906-q` | Direct-device candidate |
| R | `bookforge-klein-qualification-20260906-r` | Same candidate image |
| S | `bookforge-klein-qualification-20260906-s` | Same baseline image |

Use the existing native defaults: `us-central1`, one RTX PRO 6000 Blackwell Server
Edition, 20 vCPUs, 80 GiB RAM, concurrency one, minimum zero, and both service and
revision maximum one. Keep CPU boost and zonal redundancy off, private IAM, one revision
receiving all traffic, and no fallback provider. Preserve BF16 Klein, FP16 depth,
four steps, 1024×576 output, original model revisions and watercolor prompts. Verify
deployment and IAM metadata before inference; retain the existing 120-second IAM wait
inside the work deadline without treating it as proof of propagation.

Each service receives exactly ten sequential requests: the two retained cases, six
original watercolor cases, then the two retained cases again. Preserve seeds and order.
There is no renderer health request or prewarm before the first generation, no generation
retry and no reroll. Forty completed requests produce eighty JPEGs. A worker replacement,
failed request or incomplete run remains visible and stops the remaining comparison.

## Lifecycle and release evidence

For each service, start the external deadline before deployment: **600 seconds for work
plus 60 seconds for cleanup**. Delete the exact service on completion, failure or deadline.
Verify terminal operations and absence; an absent service alone does not resolve an
ambiguous pending creation. Request timeout does not terminate server work.
[Cloud Run timeout behavior](https://docs.cloud.google.com/run/docs/configuring/request-timeout).

After the deletion audit, run the [read-only release guard](gcp-klein-gpu-release-2026-09-06.md)
for up to **900 seconds**, including a **minimum ten-minute wait after deletion**. Require
fresh, complete active and idle counts across every observed revision, with the two
latest zero samples at least 60 seconds apart and after deletion. The latest sample
must be at most 180 seconds old. Missing, stale, partial, nonzero or paginated evidence
cannot pass. Require this proof before advancing to the next service and after S.

Admission before P alone uses the independently reviewed
[first-admission receipt](../benchmarks/gcp-klein-device-loading-2026-09-06/preflight/first-admission.json),
SHA256 `bad07a04cec1eec81f9e0ccc2386d98c909145116b5575eeb7dbaa00c0bd609f`.
L, M and N must each have complete historical
active/idle zero pairs for their expected and all observed revisions, at least 60 seconds
apart after their exact terminal deletion, with no newer nonzero or partial series.
Current inventory must show them absent, and the bounded lifecycle audit must show no
post-deletion recreation or update. This historical evidence is explicitly **not fresh**;
quota availability remains unknown. It does not reserve a GPU or authorize spending.
The exception does not relax the fresh guard after P, Q, R or S.

Unknown release stops new work. Preserve the incomplete order rather than launching a
replacement or waiting indefinitely under this allowance. Monitoring is sampled and
delayed; a zero observation neither reserves quota nor guarantees GPU availability.
The ten-minute floor is a conservative experiment rule, not a provider release promise.
[Instance-count metric](https://docs.cloud.google.com/monitoring/api/metrics_gcp_p_z).

## Measurements and prospective gates

Both constructors log exactly these synchronized stages: `imports_and_cuda_initialization`,
`pipeline_from_pretrained`, `pipeline_to_cuda`, and `depth_construction`. Each includes
its concluding CUDA synchronization and reports finite nonnegative seconds and Linux
process peak RSS in KiB. Bind all four records to the correct service, revision and
worker. The first stage covers constructor imports and synchronization; the worker's
earlier framework/device checks are outside that stage.

The external factory timer must include those earlier checks and all constructor timing
and logging. Keep it separate from runtime `load_seconds`. Compare pipeline loading
**plus** final placement, because the candidate moves placement into loading. Stage
synchronization and logging change overlap and add overhead relative to production;
only the equally instrumented arms are controls. RSS is a cumulative process high-water
mark. No component-specific Qwen time, PCIe bandwidth or disk bandwidth is inferred.

The timing screen requires all of the following, in addition to complete protocol and
release evidence:

- At least **15% lower median first-client artifact-ready time** across the two arms.
- At least **15% lower median pipeline-loading-plus-placement time**, and improvement
  in each adjacent pair: P→Q and S→R.
- Lower inclusive external factory time in both of those pairs.
- No more than **5% regression** in the candidate arm's median later-worker time. Compute
  each worker's median over fixed zero-based request rows 2–9, then each arm's median
  of its two worker medians. First use of a token bucket is not established warm work.

Report the four stage times, first rendering of each token bucket, complete client
timing, compiler-wrapper time, process peak RSS and later-worker medians per service.
Preserve build, deployment, IAM wait, credential acquisition and cleanup timing separately.
Do not replace the inclusive factory timer with a sum of stage durations. Four fresh
workers provide a screening comparison, not p95, cold-start reliability or evidence of
uncached physical hosts, storage or driver state.

Keep **timing, strict equivalence and human quality as separate verdicts**. Require all
forty paired same-ordinal master/depth comparisons to be byte-equal for strict equivalence:
first occurrences compare with first occurrences, final repeats with final repeats.
Known baseline nondeterminism does not waive a mismatch. Preserve both images on failure.
Frozen scene facts, counts, relationships, watercolor appearance and user visual review
remain required; faster loading cannot compensate for incorrect images. Product promotion
also still requires broader cold reliability and the physical Jetson-to-projector path.

## Separate funding proposal

Earlier phase holds total **$14.7564136** ($9.7224520 plus $5.0339616), separate and
unreleased until their own reconciliation supports a change. This proposal does not reuse
their unused margins or erase unresolved tails. Refresh billing before dispatch and at
closeout; retain reported, estimated and settled amounts separately.

| Component | Proposed allowance |
| --- | ---: |
| Four lifecycles: two resource slots × (600 + 60 + 900) seconds | $11.0475456 |
| Two tiny eight-CPU, 100 GB, ten-minute overlay builds | $0.34 |
| Incremental overlay storage, at most seven days | $0.05 |
| Remaining margin | $0.5624544 |
| **Separate gross phase cap** | **$12.00** |

The full fifteen-minute release-guard window is funded after every lifecycle, exceeding
the ten-minute minimum. The calculation uses $0.00088522 per second: RTX GPU $0.00036522,
20 CPUs at $0.000018 each, and 80 GiB at $0.000002 each, without free-tier credits. Two
resource slots allow for transient overlap; service maximum settings and this estimate
are not hard platform billing caps. Unknown liabilities remain held when the guard ends.
[Cloud Run pricing](https://cloud.google.com/run/pricing),
[maximum-instance limitations](https://docs.cloud.google.com/run/docs/configuring/max-instances).

Each build retains the reviewed eight-CPU rate of $0.0156 per minute, or $0.156 for ten
minutes rounded to $0.17; the first 100 GB disk is included. Shared model-image retention
stays with the earlier phase; this allowance covers only new small overlays. Retain exact
build durations, layers, operations, request journals, artifacts, stage logs, deletion
audits, release proofs and cost observations. Margin does not authorize another build,
service or request. Root must freeze and verify the execution images, manifests, source
bindings and available budget before any paid operation.
[Cloud Build pricing](https://cloud.google.com/build/pricing),
[Artifact Registry pricing](https://cloud.google.com/artifact-registry/pricing).
