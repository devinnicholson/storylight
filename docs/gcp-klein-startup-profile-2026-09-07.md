# Native GCP startup profile — September 7, 2026

Framework imports consumed **36.008 seconds of a 69.365-second worker factory**
in the first instrumented startup. This locates the largest measured delay;
it does not establish how much is avoidable or whether image streaming caused it.
Production is unchanged.

| Sequential factory stage | Host wall time |
| --- | ---: |
| Small preflight checks | 0.054 s |
| Torch and runtime-module imports | 11.249 s |
| Device checks | 0.539 s |
| Runtime framework imports | 24.759 s |
| Transformer loading | 12.635 s |
| Remaining pipeline `from_pretrained` | 8.161 s |
| Remaining pipeline `.to("cuda")` | 11.623 s |
| Depth setup | 0.329 s |

These scopes omit small gaps and logging overhead. The enclosing factory receipt
remains authoritative. Within transformer loading, FlashPack assignment took
11.489 seconds; its vendor `read_and_copy` timer reported a rounded 11.39 seconds.
That includes file reads, host staging, CUDA copies and synchronization. Nested
timers must not be added to their parents or called isolated GPU transfer time.

The first verified client artifact took 97.365 seconds. Compile-wrapper setup was
2.046 seconds, first 128-token-bucket image work 15.294 seconds, and first
256-token-bucket image work 6.402 seconds. The later eight client requests had a
0.449-second median. This single diagnostic is not a matched speed comparison or
a latency distribution. It preserves the original BF16 watercolor profile,
1024×576 dimensions, four steps, prompts, seeds and compilation.

All ten renders, twenty JPEG validations and 169 resident CUDA tensor hashes
passed. Agent inspection retained rich watercolor treatment in all eight unique
scenes and core facts in seven; the known extra silver fox and lantern remained.
Human and physical projector acceptance are pending.

The [frozen plan](../benchmarks/gcp-klein-transformer-profile-2026-09-07/plan.json)
and [execution receipt](../benchmarks/gcp-klein-transformer-profile-2026-09-07/execution-receipt.json)
bind the exact source, immutable image and prior release evidence. Independent
review reproduced the actual image layer, render aggregate and
[stage analysis](../benchmarks/gcp-klein-transformer-profile-2026-09-07/stage-analysis.json)
from retained raw receipts. The service was deleted at the terminal audit event
04:03:01.861695 UTC. The separate release check passed at 04:13:20 UTC and was
independently replayed. It establishes retained historical zero samples and
scoped service absence, not current GPU inventory, quota reservation or settled
billing. The [closeout](../benchmarks/gcp-klein-transformer-profile-2026-09-07/phase-closeout.json)
retains these limits.

The separate allowance is $3.50, including $2.9518864 in resource and small-operation
reservations. Total unreconciled reservations are $40.1575048, not reported spend.
The latest retained gross-project report is $25.23 at 03:50:41 UTC, before this
GPU run; it is delayed project-wide usage, not this experiment's invoice.
The CPU build succeeded in 23.017 seconds. No additional model payload was copied:
the verified application overlay is 12,071 compressed bytes over the existing
common image. Starting a worker can still read inherited image layers.

## Finer import trace

A second diagnostic completed ten renders and all 169 CUDA checks. Its
[ordered import trace](../benchmarks/gcp-klein-import-profile-2026-09-07/stage-analysis.json)
measured Torch at 7.906 seconds, Diffusers at 8.080, Transformers at 1.165,
AutoImageProcessor at 2.771, the depth auto-model import at 0.030, and the pipeline
registry at 0.345. Repeated Torch, Triton and runtime-module imports were already
cached within that process. These are marginal costs in the preserved order,
not independent package benchmarks.

Its enclosing import scopes totaled 20.297 seconds, factory 51.851 seconds and
first client artifact 82.792 seconds. The large difference from the first trace
occurred without an optimization; it establishes neither an improvement nor a
cache cause. The packed read/copy timer reported 9.56 seconds. Service deletion
is verified; its separate release check passed at 04:37:23 UTC and root and an
independent reviewer replayed the raw evidence. The same historical-zero and
scoped-absence limits apply.
Its separate allowance is $3.50, bringing unreconciled reservations to
$43.1093912. Neither amount is an invoice.

The pipeline registry is not the largest measured target. Source inspection also
found that generic processor and media utilities are imported independently by
Diffusers and Qwen; changing depth setup alone could move their cost elsewhere.
The earlier standalone depth adapter saved only 0.145 seconds in its CPU screen.

The next candidate overlaps one verified transformer-pack read with framework
imports, then joins before meta-model construction and unchanged parameter
assignment. It preserves the original depth, model and rendering operations.
Package setup and CPU/I/O contention may consume the overlap opportunity;
only full worker and first-artifact timings can establish a gain. Text-encoder
packing and dependency patches remain deferred.
