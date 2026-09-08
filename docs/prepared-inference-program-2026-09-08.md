# Prepared inference sessions

This program tests whether preparing the existing native GCP renderer before a
voice session can deliver consistently fast complete images. It preserves the
watercolor prompt, model weights, runtime and 1024 × 576 output. Production stays
on the existing provider until both performance and image quality qualify.

## Measurement protocol

Run three fresh private Cloud Run services sequentially. Each uses one RTX PRO
6000, 20 vCPU, 80 GiB memory, concurrency one, minimum one and maximum one instance.
Prepare the actual 128- and 256-token execution buckets, retaining both images and
the complete preparation time. After 30 seconds idle, measure ten requests using
the frozen eight-case corpus followed by its first two cases. Report preparation,
first delivery, all ten delivery times, median and observed maximum separately.

The speed screen requires a median at most one second and observed maximum at
most two seconds in each session. Thirty successes are a qualification screen,
not a production reliability guarantee. Review every image for source facts and
watercolor detail; faster output alone does not qualify a candidate.

Readiness belongs to one process identity and expires 300 seconds after preparation
starts. Replacement, changed source identity, unsupported token bucket, expiry,
duplicate ordinal and ambiguous responses fail the attempt. A failed request is
never silently submitted again. Keep failed attempts and diagnose them before
starting a separately identified experiment.

The browser/API work adds guarded submission IDs and session revisions so lost
responses can reconcile with the accepted job. Exact replay returns that job;
changed payloads, stale revisions and restarted servers reject before starting
generation. Existing scenes remain visible until replacement assets are complete.

## Bounds and evidence

The approved incremental allowance is $25. Reserve $3.50 per trial before its build;
do not release that reservation based on delayed billing. Each build is bounded to
600 seconds. Each deployment and measurement lifecycle is bounded to 600 seconds,
with another 60 seconds reserved for deletion and a closure receipt. Observe
resource release for at most 900 seconds before another GPU experiment. Missing
Monitoring data does not prove zero usage. Stop paid dispatch when cleanup or
remaining allowance cannot be established.

At published us-central1 instance rates, one configured instance costs
$0.00088522/second before discounts: 20 × $0.000018 CPU, 80 × $0.000002 memory,
plus $0.00036522 GPU. Reserving two instance equivalents for the 1,560-second
work/cleanup/observation envelope costs $2.7618864; the remaining trial reservation
covers its bounded CPU build, storage and small operational costs. This is an
experiment estimate, not an invoice or a cloud-enforced spending cap.
[Cloud Run pricing](https://cloud.google.com/run/pricing).

Minimum instances can restart, so a minimum of one does not prove model readiness.
Verify actual warmup results and pin subsequent requests to their process identity.
[Minimum instances](https://docs.cloud.google.com/run/docs/configuring/min-instances).

Retain source hashes, build configuration and ID, immutable image digest, exact
overlay verification, request dispatch journal, images, noise receipts, timing,
failure evidence, deletion receipt and release observations. The new image must
preserve all 17 parent layers and add only the reviewed six-file application
overlay. No new tensor-equivalence claim is implied by preserving those layers.

## Status

The guarded-submission change passed independent review, 1,074 project tests and
six JavaScript suites. Its exact API/registry hunks were applied to the installed
Jetson package while preserving its existing compatibility changes. Three mocked
tests passed on the Jetson before restart; the running gateway then returned the
new server identity on a missing-session response. Backup:
`/opt/bookforge/.voice-backups/guarded-submission-20260908`.

The initial prepared worker and measurement tools passed 24 offline tests,
including termination cleanup and ambiguous deployment outcomes. Independent
source and deslop reviews passed. The first build completed successfully and its
source archive and 17 inherited layers were verified before GPU dispatch.

Trial n's first preparation returned HTTP 409 after 8.893 seconds at the platform.
Package and GPU checks passed, and the final application log marked model loading.
No image or successful preparation receipt was returned. The supervisor stopped
without retrying and verified service deletion after 156.069 seconds; subsequent
release checks found historical zeros and current absence. Its $3.50 reservation remains consumed
for admission accounting, regardless of eventual billed cost.

Source inspection found an inherited loader contract missed by the wrapper:
`klein_flashpack.load_transformer` imports `app.PACK_PROOF_SHA256`. The wrapper did
not expose that constant. This defect is consistent with the observed failure
stage; the original response did not retain an exception trace. The next revision
adds the compatibility export, a regression test and bounded error diagnostics.
Trial o passed the first speed screen; trials p and q subsequently failed as
described below. Retain all failures and use the remaining r/s/t reservations for
three identical speed workloads after the repairs. Seven $3.50 reservations total
$24.50; unused reservations are prospective, and failed-trial reservations are not
recycled. The modern voice compiler has a locally validated six-scene fixture
screen, but its GPU quality trial no longer fits this reservation envelope. The
existing image quality defects still prevent production promotion.

The inherited k images also require qualification: they match i exactly, whose
retained factual review passed six of eight cases. Rich watercolor detail passed
all eight, but an extra silver fox/lantern and an unclear owl perch remain known
failures. Preserving these images would not establish product quality acceptance.
Trial o completed both preparation renders and all ten measured deliveries.
Preparation took **79.643 seconds**. After 30 seconds idle, the first complete
delivery took **0.635 seconds**, the median was **0.510 seconds**, and the observed
maximum was **0.635 seconds**. The service was deleted after a 273.435-second
supervised lifecycle. This passes one session's speed screen, not the planned
three-session qualification or production reliability.

Independent review compared all 24 JPEGs. Warmups exactly match k; 18 of 20
measured JPEGs match their k counterparts. The remaining master/depth pair matches
k's later repeat of that case. All eight scenes retain rich watercolor detail.
Current visual review passes the owl perch and scores seven of eight factual
cases; this differs from the historical review of the same owl pixels and is not
a quality improvement. The extra silver fox and lantern remain a clear failure.

The failing prompt is from the older `LiveScenePlan.to_page` template. Current
reviewed voice uses `compile_scene_facts_prompt`, which does not emit the generic
upper-right placement instruction. A later quality screen should therefore test
that actual compiler before drawing conclusions about the modern voice path.
BFL recommends specific positive descriptions and documents that Klein does not
upsample prompts automatically. [Prompt guidance](https://docs.bfl.ai/guides/prompting_guide_t2i_negative),
[Klein overview](https://docs.bfl.ai/flux_2/flux2_overview).

No production GPU promotion is claimed. The existing image provider remains live.

Trial p completed both warmups in **75.011 seconds**, then its first measured
request failed with `ConnectError` after 30 seconds idle. No measured image was
returned. The retained error cannot distinguish DNS, TCP and TLS failures. The
client opens a fresh connection after this idle interval; stale connection reuse
does not explain this attempt. No request was retried.

The automatic deletion command also timed out. Its original failed closure is
preserved. A separate recovery command deleted the service, verified absence and
retained the terminal deletion audit at 08:15:21 UTC. Release monitoring later
confirmed historical zero usage and current absence before the next trial.

The repaired cohort uses a separate, reviewed transport wrapper. Before dispatch it
requires a fresh token-free TLS proof for the renderer and ten control-plane
domains. It preserves URL hostnames, SNI and certificate verification while using
the verified network addresses. Its 720-second process bound includes read-only
preflight; the GPU supervisor's 600-second work and 60-second cleanup limits stay
unchanged. This is a test of a transport repair, not proof of p's precise cause.

Trial q deployed successfully but failed local input validation before
authentication or a render dispatch. The client allowed names n/o/p while the
manager also admitted q. The service was automatically deleted after 144.324
seconds. The repair makes the manager run the actual client's input validation
before any build or deployment, with a regression for every admitted name.
Trial q provides no inference or transport-delivery measurement.

The inactive serving adapter uses the reviewed compiled-prompt boundary and
verifies fixed leases, request identity, complete JPEGs and saved bundles. Its
cloud lifecycle remains externally owned. Connection pooling now avoids
setting up a new TCP/TLS connection for every call, following
[HTTPX's client guidance](https://www.python-httpx.org/advanced/clients/). This is
separate from the frozen benchmark and has no measured GPU speed claim yet.

The inactive implementation passed independent code and deslop review, 1,084
project tests, 71 combined adapter/experiment tests and all six JavaScript suites.
The pool's close/drain regression confirms that readiness stops immediately,
active work drains and cancellation still closes the owned client.
