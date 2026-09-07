# GCP compiler diagnostic: successful continuation

The September 7 k trial completed ten renders and reproduced all twenty master/depth JPEGs from i byte for byte. All 169 resident CUDA tensor hashes and ten initial-noise records matched. The restored BF16 Klein4B watercolor profile remains 1024×576, four steps, guidance 1. Production is unchanged; this is diagnostic evidence, not a speed qualification or human image acceptance.

The [replayable report](../benchmarks/gcp-klein-cache-diagnostics-continuation-2026-09-07/summary.json) binds the actual build, image, collection, compiler logs, cleanup and release evidence.

| Seconds | Earlier i | Diagnostic k |
| --- | ---: | ---: |
| First client request | 74.990 | 76.011 |
| Factory including cache restore | 52.731 | 53.977 |
| Compile wrapper | 1.779 | 1.776 |
| First 128-token image | 12.765 | 11.933 |
| First 256-token image | 1.630 | 1.614 |
| Later eight worker median | 0.350 | 0.350 |
| Later eight client median | 0.465 | 0.483 |

These are one instrumented worker and one earlier worker, not a causal comparison. Factory, wrapper and image timings overlap with the client request; they must not be added as independent costs.

## What the compiler records establish

Both first bucket uses recorded two AOT and two Inductor cache hits, with zero misses, bypasses, guard misses or recorded cache errors. Each captured two unique graphs. The compile wrapper itself recorded no compiler work because compilation is lazy.

For the first 128-token image, `OutputGraph.call_user_compiler` accumulated 8.266 seconds, bytecode tracing 0.663 seconds and AOT cache loading 0.487 seconds. On first 256-token use, their deltas were 0.127, 0.968 and 0.037 seconds. These are inclusive, overlapping wall timers, including failed regions. They are neither additive nor GPU times. Subtracting them would not establish exclusive compiler-initialization cost.

This rules out cache misses in the recorded calls and makes backend first-use initialization worth investigating. It does not identify the cause of all eight seconds. The [PyTorch 2.8 source audit](../experiments/renderer-cache-diagnostics/backend-startup-research.md) identifies lazy backend imports and compiler-pool warmup. Setting `TORCHINDUCTOR_COMPILE_THREADS=1` skips that pool warmup and is excluded from FX/AOT cache keys. The [local candidate](../experiments/renderer-compiler-pool/README.md) makes that one configuration change and observes existing pool state without creating a pool. It remains unmeasured and unpromoted.

## Execution and cleanup

One CPU build, `a90d728f-92f5-4a08-9dae-48b75974b0df`, ran from 18:32:03.253 to 18:32:31.066 UTC. Its image digest is `d7dc49efcc1ec9e9816a0d02668a3d531e390e98e9dbf90c8c7e265014093571`; all sixteen parent layers were retained with one three-file overlay. Only the admitted manifest expiry changed from j. The old experiment identifier intentionally remains in that manifest; execution and evidence explicitly scope service k.

The finite supervisor completed in 225.311 seconds. The terminal Service deletion was 18:43:35.258306 UTC. Release observation began about 323.7 seconds later, after log retention and analysis. Early metric reads still contained active instances. Later fresh zero pairs passed the unchanged guard; the final pair at 18:52/18:53 was observed at 18:53:59.790, and fresh scoped audit plus service 404 completed at 18:54:01.588. The [release report](../benchmarks/gcp-klein-cache-diagnostics-continuation-2026-09-07/release/summary.json) retains all six reads. This establishes historical zeros and current service absence, not instantaneous capacity, a future reservation or settled billing.

The approved gross allowance was $3.50. Conservative new hold: $2.9518864; cumulative unreconciled holds: $63.792596. These are reservations, not an invoice or actual spend. Earlier j holds remain unreconciled. No further paid candidate run is included in k authorization.

A short-DNS-TTL read-only probe was refused before command dispatch and repeated with fresh evidence. An initial build-log read used the wrong bucket and returned 404; the observed build logs bucket supplied the retained log. Neither caused a paid build or GPU retry. Process-local verified DNS routes preserved original HTTPS hostnames and certificate checks.

## Validation

Independent review replayed build provenance, all collected outputs, the three compiler events, clean closure and the final release proof. Offline tests cover diagnostic records, source derivation, transport routing and the candidate observer's refusal of unavailable or nonempty pool state. The production runtime was not edited.
