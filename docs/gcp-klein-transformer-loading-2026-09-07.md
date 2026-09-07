# Native GCP transformer loading — September 7, 2026

Transformer-only FlashPack loading passed the actual CUDA weight checks, but its
first sequential comparison missed the speed thresholds. Production is unchanged.
The next experiment will time the loading stages before another optimization is
chosen. A four-service ABBA reporter is prepared but has not been used.

| Measurement | Baseline | Candidate |
| --- | ---: | ---: |
| First client artifact ready | 90.880 s | 86.217 s |
| Whole worker factory | 61.100 s | 54.267 s |
| Compile-wrapper setup | 2.385 s | 2.055 s |
| First 128-token-bucket image | 19.569 s | 22.080 s |
| First 256-token-bucket image | 6.584 s | 6.349 s |
| Later eight client requests, median | 0.478 s | 0.485 s |
| Post-render tensor oracle | 4.274 s | 4.142 s |
| Verified renders / CUDA tensor hashes | 10 / 169 | 10 / 169 |

The first-client reduction was **5.13%**, below the frozen 10% threshold. Factory
time fell **11.18%**, below 15%. The later-request median regressed **1.49%**,
within the 5% allowance. First-client timing ends after verified JPEGs are saved;
it excludes deployment and prior token acquisition. Factory time includes imports
and all model initialization, not just transformer loading. First-bucket timings
include priming work. Two sequential workers do not establish a causal speed gain,
a cold-start distribution, p95 or physical host-cache state.

Both runs used RTX PRO 6000 Blackwell, 20 CPUs, 80 GiB RAM, minimum zero and maximum
one instance. Model revision, BF16, 1024×576 dimensions, four steps, guidance 1,
prompts, seeds and request order were identical. Each worker completed ten renders
before one bounded oracle hashed all 169 resident transformer parameters. Every
hash matched the [CPU producer proof](gcp-klein-transformer-pack-2026-09-07.md).

Only **1/10 master and 1/10 depth JPEG pairs** matched byte-for-byte. In the
unblinded agent screen, both variants retained rich watercolor treatment in 8/8
unique scenes and satisfied core prompt facts in 7/8. Both retained the extra
silver fox and lantern in the same failing prompt. This is not human acceptance,
exact image equivalence or physical projector validation. See the
[baseline](../benchmarks/gcp-klein-transformer-loading-2026-09-07/baseline-visual-review.json)
and [candidate](../benchmarks/gcp-klein-transformer-continuation-2026-09-07/candidate-visual-review.json)
screens.

## Image and execution evidence

Both small application overlays inherit the same
[verified common image](gcp-klein-packed-image-2026-09-07.md), containing original
checkpoints plus the extra 7.75 GB transformer pack. Baseline loads the original
transformer; candidate assigns the pack to CUDA. Qwen, VAE, depth and rendering
remain unchanged. This controls image contents between arms but does not prove a
startup advantage over the original smaller image. The
[execution receipt](../benchmarks/gcp-klein-transformer-loading-2026-09-07/execution-receipt.json)
pins both actual images, manifests, source files and verification receipts.

The first phase stopped after baseline release exposed a replay-clock bug: the
checker sampled its decision and enclosing receipt 311 microseconds apart. The
reviewed repair uses one decision timestamp, preserves the deadline, and supports
only the exact original source version when replaying its recorded decision time.
Every raw observation and final predicate is rechecked. The
[original failure](../benchmarks/gcp-klein-transformer-loading-2026-09-07/release-replay-diagnostic/diagnostic.json)
and incomplete phase outcome remain unchanged. A separately reviewed
[continuation](../benchmarks/gcp-klein-transformer-continuation-2026-09-07/plan.json)
then ran the unchanged candidate.

Both supervisors completed successfully and deleted their services. Both release
results passed offline replay, including the exact legacy compatibility for the
baseline. The [paired report](../benchmarks/gcp-klein-transformer-continuation-2026-09-07/summary.json)
reproduces the results and full admission/deletion chronology. Historical zero
samples and scoped service absence do not reserve quota, prove current GPU count
or settle billing.

The initial allowance was $6.50. The separate candidate continuation allowed
$3.25: $2.7618864 for lifecycle/release capacity, $0.02 for small operations and
$0.4681136 margin. Each service had 600 seconds for work, 60 for cleanup and 900
for release observation, with two resource slots conservatively reserved.
Unreconciled reservations total $37.2056184; this is not reported spend. The latest
retained gross-project report was $23.58 at 03:27:30 UTC and can lag actual usage.
