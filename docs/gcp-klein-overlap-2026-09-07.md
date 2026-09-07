# Native GCP transformer-read overlap — September 7, 2026

This comparison tests whether reading the existing transformer pack during
framework imports reduces complete startup time. Both arms retain BF16 Klein4B,
1024×576 watercolor images, four steps, the same prompts and seeds, and unchanged
compilation and depth generation. Production is unchanged.

The candidate validates the pack before starting one reader, joins before model
construction, and assigns the original 169 parameter views. Every constructor
exit joins the reader. The comparison uses the existing whole-factory and client
artifact timers; the private runtime timer excludes imports and cannot establish
a gain. Read and import durations overlap and must not be added.

The [frozen plan](../benchmarks/gcp-klein-overlap-2026-09-07/plan.json) requires
at least 15% less factory time, 10% less first-artifact time, and no more than 5%
regression in the later eight client requests. Tensor equality, valid JPEGs,
image byte equality and visual quality are reported separately. One sequential
pair cannot establish a latency distribution or justify production promotion.

Each image adds a verified small application layer to the existing common model
image: 11,833 compressed bytes for the serial baseline and 12,864 for the overlap
candidate. Both preserve all fourteen inherited layers. The two CPU builds and
two GPU trials share a separate $6.50 allowance, with $5.9137728 reserved and
$49.023164 in total unreconciled reservations. These are neither invoiced spend
nor a platform hard cap.

The baseline completed ten renders and all 169 CUDA tensor checks. Its factory
took 50.213 seconds, first verified artifact 76.331 seconds, and later-eight client
median 0.504 seconds. Agent inspection found rich watercolor treatment in all
eight unique scenes and core prompt facts in seven; the extra silver fox and
lantern remain in the known failing case. Human projector acceptance is pending.
The baseline service was deleted; its release evidence passed independent replay
at 05:06:32 UTC. This proves retained historical zero samples and scoped service
absence, not current quota availability or settled billing.

The candidate's first request failed with platform HTTP 500, “no available
instance,” at 05:10:12 UTC. No worker startup was logged, and no image or tensor
oracle completed. The supervisor verified deletion and absence after 162.610
seconds; the separate release check passed. The original comparison is
incomplete and has no candidate timing or speed verdict.

This error does not identify a quota violation or an overlap-code defect.
Google lists startup/scaling delays and transient service factors among its
possible causes. A separately funded fresh attempt preserves the failed request;
it does not replace it in the original result. [Cloud Run troubleshooting](https://docs.cloud.google.com/run/docs/troubleshooting#abort-request).

The [replayed failure report](../benchmarks/gcp-klein-overlap-2026-09-07/failed-comparison.json)
retains the baseline and failed candidate separately. Its failed supervisor gate
reflects the workload failure; both control-plane deletions and subsequent
release checks succeeded. It reports no candidate timing ratios.

Local verification passed 960 Python tests, both JavaScript suites, eight focused
qualification tests, scoped Ruff and exact reproduction of the current public
graph-coverage artifact.

## Separate continuation

Fresh service g completed all ten renders and 169 CUDA tensor checks using the
unchanged candidate code. Its separate $3.50 allowance raises unreconciled
reservations to $51.9750504. The original f failure remains in the report.

| Measurement | Serial e | Overlap g |
| --- | ---: | ---: |
| Whole factory | 50.213 s | 55.215 s |
| First verified client artifact | 76.331 s | 85.645 s |
| First 128-bucket image work | 15.547 s | 19.795 s |
| First 256-bucket image work | 6.392 s | 6.158 s |
| Later-eight client median | 0.504 s | 0.493 s |

This interrupted comparison fails both cold-speed thresholds. The transformer
read took 11.492 seconds and completed before the join, which waited 9.33
microseconds. Setup took 1.333 seconds. These observations show that the scheduled
overlap occurred; they do not measure time saved against a counterfactual startup.
There is no exact import timer in this candidate. The earlier import diagnostics
also varied substantially between fresh workers without a code change.

One of ten paired masters and one of ten paired depth JPEGs matched exactly.
Agent inspection again found eight rich scenes and seven with core prompt facts;
the known extra fox and lantern remained. Similar appearance does not waive the
failed byte-equivalence check. Human acceptance and promotion remain pending.
Service deletion is verified; the separate release check passed at 05:39:36 UTC.
The [complete continuation report](../benchmarks/gcp-klein-overlap-continuation-2026-09-07/comparison.json)
replays both completed arms and the intervening failed attempt. It keeps the
original phase incomplete and reports no speed-screen pass or promotion.

The next local candidate adds compiler-cache export and restoration to this exact
runtime, with initial-noise and tensor proofs in both arms. The earlier cache
experiment showed shorter first-bucket execution but was incomplete and had
different images. Its cache belongs to another runtime and cannot be reused.
