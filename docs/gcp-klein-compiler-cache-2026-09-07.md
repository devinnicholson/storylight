# Fresh compiler-cache comparison

The cache consumer's measured first 256-token image execution was 74% shorter, but its first artifact was slower. The overall speed screen failed. Both fresh GCP workers used the same restored BF16 Klein 4B model, 1024×576 output, four steps, guidance 1, watercolor prompts and overlap loader. The overlap loader remains unpromoted; this comparison holds its code fixed.

| Measurement | Producer, seconds | Consumer, seconds | Observed change |
| --- | ---: | ---: | ---: |
| First client artifact | 72.191 | 74.990 | 3.88% slower |
| Factory, including cache setup | 50.466 | 52.731 | 4.49% slower |
| First 128-token image execution | 12.069 | 12.765 | 5.77% slower |
| First 256-token image execution | 6.272 | 1.630 | 74.02% faster |
| Later eight requests, median worker time | 0.3523 | 0.3502 | 0.59% faster |

Each arm completed ten renders, all 169 resident transformer tensor checks and ten initial-noise receipts. The producer exported 12,229,653 bytes of compiler artifacts covering the exercised buckets. The consumer contained that exact verified export; only two cache digest constants changed in its worker. All ten paired initial-noise and position-ID hashes matched. The workers were deleted after 226.72 and 225.39 seconds respectively; release evidence is retained separately.

The frozen gates require at least 15% less first-client time, 30% less first execution time for each token bucket, and no more than 5% regression in the later eight requests' median worker time. The first-client and 128-token gates failed; the 256-token and later-request timing thresholds passed. Factory time includes cache validation and restore; restore has no separate timer. One sequential pair cannot establish a reliability rate or a causal estimate.

None of the 20 paired JPEGs were byte-identical. Within each worker, repeating the first case also changed master/depth bytes despite identical initial noise; repeating the second case was byte-exact. This establishes existing output drift, not its cause.

Agent inspection of all 20 master images found rich watercolor detail retained. Basic requested facts passed in seven of eight producer cases and six of eight consumer cases: the long silver-fox prompt added an extra fox and lantern in both, while the consumer owl did not clearly perch on its branch. Human and physical-projector acceptance remain pending. Neither tensor/noise equality nor the narrower timing improvement qualifies this image for promotion.

The next diagnostic records actual PyTorch 2.8 Inductor/AOT cache counters and compiler timing deltas around first execution. Cache restoration returning non-None proves ingestion, not a graph hit. The current result does not distinguish an initial miss from expensive frontend work after a hit; the diagnostic targets that uncertainty before another optimization is chosen. It preserves the model, cache and render profile.

The phase has a separate $6.50 gross allowance for two sequential CPU builds and two fresh services, each bounded to 600 seconds of work, 60 seconds of cleanup and 900 seconds of release observation. Prior holds remain unreconciled; the combined reserved holds are $57.8888232, not an invoice or a platform spending cap. The latest retained billing report was $25.87 gross at 05:32:29 UTC and lags usage.

Evidence: `benchmarks/gcp-klein-overlap-cache-2026-09-07/`. Implementation and offline replay instructions: `experiments/renderer-overlap-cache/README.md`. The phase's pinned plan and execution receipts govern the measured sources and limits. Production configuration remains unchanged.
