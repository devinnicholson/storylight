# Compiler-pool recovery: exact images, no speed gain

The September 7 m trial completed all ten requests with the intended one-thread compiler setting. All 169 resident transformer tensor hashes, ten initial-noise records and twenty master/depth JPEG hashes matched k. First-client latency was 83.818 seconds versus k's 76.011 seconds. This intervention did not improve startup in the observed run; one historical pair cannot establish a causal regression. Production remains unchanged.

| Seconds | k baseline | m |
|---|---:|---:|
| First client artifact-ready | 76.011 | 83.818 |
| Factory including cache restoration | 53.977 | 56.478 |
| Compile wrapper | 1.776 | 1.743 |
| First 128-bucket image | 11.933 | 16.622 |
| First 256-bucket image | 1.614 | 1.624 |
| Later eight worker median | 0.350 | 0.350 |
| Later eight client median | 0.483 | 0.625 |

The [replayable observations](../benchmarks/gcp-klein-compiler-pool-recovery-2026-09-07/diagnostics.json) retain the exact measurements and comparisons. Both first buckets recorded two AOT and two Inductor cache hits, with no misses or bypasses. The pool observation reported effective compile threads of one, empty process/thread pool caches and zero tracked process pools. This observation does not prove that no subprocess existed at any earlier instant.

First-128 backend calls still totaled 12.092 seconds, compared with 8.266 seconds in k. Bytecode tracing was 0.662 seconds, AOT Inductor loading 1.716 seconds and Python code loading 0.055 seconds. These nested wall timers overlap; their differences are not a measurement of import time. Recorded backend code-generation timers were zero.

The [next diagnostic](../benchmarks/gcp-klein-compiler-pool-recovery-2026-09-07/backend-next-step.md) isolates the lazy backend import. PyTorch 2.8 imports `torch._inductor.compile_fx` inside its default compiler wrapper, within the measured backend region. A sequential import timer would distinguish that initialization from subsequent backend execution while keeping the complete first-request timer authoritative. A CPU-only import trace can rank dependencies more cheaply, but cannot reproduce the CUDA factory or attribute the GPU interval. Neither proposal is a measured optimization or authorization for another paid trial.

## Execution and cleanup

The user approved one fresh recovery build and service within $3.50, explicitly accepting l's unknown release metrics. The recovery used experiment ID m, matching its service, and checked all permanent claim namespaces immediately before both build and deployment. Existing claims were not modified. Historical proof replay does not repeat those live absence checks.

Build `20c0beaa-c728-4204-9919-ef1ff5594fcd` succeeded. Its six-file source archive totaled 37,664 bytes. Actual image `bc3cebad8906bd89615ff065a40023cb37a79ac3db9abb6d9641d5dd4fc64e71` preserved all seventeen k parent layers and added the reviewed three-file overlay. Independent review replayed the actual build, image, collection and source bindings.

The supervisor completed in 235.288 seconds, including deployment and IAM propagation, with no failure and verified deletion. That duration is separate from first-client generation latency. The terminal Service deletion was 23:00:26.376650 UTC. The unchanged release guard completed ten metric reads and passed at 23:10:54.704064 UTC, retaining zero samples at 23:09 and 23:10 plus fresh scoped audit and service-absence evidence. This proves historical zeros and current service absence, not reserved capacity or settled billing. It does not change l's unknown result.

The [final closeout](../benchmarks/gcp-klein-compiler-pool-recovery-2026-09-07/summary.json) binds those results. The new conservative hold is $2.9518864; cumulative unreconciled holds are $69.6963688. These are budget holds, not measured spend or an invoice. No retry or additional paid experiment ran.

All 960 project Python tests and both JavaScript suites passed. Focused admission, observation and closeout checks cover the recovery helpers; production runtime and the frozen candidate sources remain unchanged.
