# Cache fast path and compiled-snapshot experiment

## Deployed cache fast path

Validated semantic-cache hits now return before waiting for model warmup. New passages still
wait for the bounded warmup before starting their independent inference timeout. This removes an
unnecessary dependency on model availability for already-prepared scenes without changing
privacy validation, model weights, prompt semantics, or renderer routing.

Two event-controlled regression tests hold warmup open while a memory-cache hit or a disk-cache
hit completes. A third verifies that reported cache latency includes disk lookup; previously the
timer started after disk access and understated restart-time latency. All **1,115 tests pass**;
the existing Starlette deprecation warning remains.

The installed wheel has SHA-256
`b0a4f0628ba7c0566514273fd359442d9a57934549d6a59bc59c1d7389b03dcd`.
Only the Jetson API was restarted. The 25 W setting, resident TensorRT model, renderer routing,
and physical projection session were not changed. Previous wheels remain available for rollback.

Through the Mac-to-Jetson loopback tunnel, the existing synthetic open-book passage returned:

| Request after API restart | Planning time | Input / output tokens |
| --- | ---: | ---: |
| Private disk-cache hydration | 11.718 ms | 0 / 0 |
| Subsequent memory-cache hit | 0.980 ms | 0 / 0 |

These are server-side cache timings, not model inference or click-to-photon measurements.
The live API check did not start a cloud render, nor did it deliberately unload the resident model.
Concurrent warmup bypass is established by the controlled regression tests, not these two samples.

## Compiled GPU snapshot probe

The separate `bookforge-klein-compiled-snapshot-probe` app tests whether the previous warm
compilation gain survives scale-to-zero. It uses one L4, minimum zero containers, five-second
scale-down, no automatic application retries, a 420-second startup bound, and a 900-second
client-run bound. It has no public endpoint and does not modify the production Modal app.

The configuration follows [Modal's GPU snapshot documentation](https://modal.com/docs/guide/memory-snapshots):
compile and warm before capture, use one compiler thread for compatibility, and reset per-container
identities after restoration. Explicit per-request seeds prevent captured random state from
silently producing identical scenes. The planned three cycles verify scale-to-zero through
provider runner counts and compare image/depth hashes for repeated seeds.

This is a development experiment, not a live renderer promotion. The initial model download and
load took 105.589 seconds, first compilation/warmup 121.767 seconds, and second warmup 1.743
seconds. None of those timings establishes snapshot restore performance.

The first request exceeded its **480-second wait bound** without returning a scene. The runner
cancelled the call and stopped app `ap-EPO72Rd75GLs1BmqfSXM24` at 20:24:18 PDT. No second cycle
was attempted, and there is no restore-latency or image-equivalence result. This rejects this
configuration for promotion; it does not establish that all GPU snapshots are incompatible.

Harness: [klein_snapshot.py](../experiments/renderer-fidelity/klein_snapshot.py).
Evidence: [bounded failure](../benchmarks/planner-optimization-2026-09-03/klein-snapshot/results.json).
Function call: `fc-01M1N6NH7MQDEZ0VFEPFFND2E9`.
The subsequent container inventory was empty. Modal's billing report attributed $0.124884 to
this app at the time of inspection; usage reports can still lag final settlement.

## Regional compilation alternative

The existing finite compilation harness now supports `--regional`, following
[Diffusers' repeated-block compilation guidance](https://huggingface.co/docs/diffusers/optimization/fp16#regional-compilation).
This compiles repeated transformer blocks rather than the entire transformer, without using
GPU snapshots. Because the helper modifies blocks in place, the harness finishes all eager
measurements before enabling compilation and explicitly records this ordering limitation.
Prompts, seeds, weights, image dimensions, steps, depth model, and encoding stay fixed.

The first regional run, with `mode="reduce-overhead"`, failed during compiled warmup with
`accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run`.
Its six completed images are **eager baselines only**, not successful regional results.
App `ap-fqX4nbE99mkC8KEV9NoXLH` stopped when the finite entrypoint finished. Rather than change
model behavior to accommodate CUDA graph lifetimes, the next comparison uses the standard
compiler mode, which does not enable those regional CUDA graphs.

Evidence: [regional CUDA-graph failure](../benchmarks/planner-optimization-2026-09-03/klein-regional/results.json).

### Standard regional compilation completed

The `--regional --mode default` run completed all six matched synthetic scenes per variant:

| Boundary | Eager | Regional, standard mode |
| --- | ---: | ---: |
| Median warm image generation | 1.928 s | 1.560 s |
| Median warm image + depth + JPEG encoding | 2.007 s | 1.638 s |
| Maximum image + depth + JPEG encoding | 2.185 s | 1.647 s |
| First warmup, including compilation where applicable | 4.481 s | 21.940 s |
| Peak reserved GPU memory | 17.74 GiB | 17.74 GiB |

The median full processing boundary is **18.4% lower** in this same-GPU run. Ordering is not
interleaved, sample size is six per variant, and this excludes planning, network delivery, and
projector presentation. Do not treat the maximum as a reliable p95.

The earlier full-transformer compilation run needed 45.567 seconds for its first warmup. The
new 21.940-second observation is promising, but it is from a separate run and changes both compile
scope and mode; it is not a controlled attribution of the startup gain to regional compilation alone.
Model download/loading still took **84.643 seconds**, so this is not fast first-request startup.

Visual review of all six compiled results retained the tested requirements: golden boat and moon,
fox left of lantern, lighthouse left of owl, lantern carried in the fox's mouth, exactly two boats,
and a child standing on the bridge holding an open green book. The child scene still adds
unrequested curtains. Images differ from eager outputs, and this is a small development screen,
not general accuracy or an end-to-end planner-fidelity pass. The test sends synthetic briefs
directly to the image model; it does not fix the live planner's omitted bridge relationship.

App `ap-fEUEkIuUTZYRQ28w7sL4Bo`, call `fc-01M1N799GH1EVJ6NGRNA7HQV3N`, completed in 149.199
seconds at the client boundary. All 24 image/depth files passed recorded SHA-256 verification.
The exact executed harness hash was
`019695e9b8f1fc5fd776e44ac5307c29046f9aeadb819a3b9c062a8a3beaf23c`.
It is retained in commit `53b6a4c`. A subsequent harness-only cleanup captures its source hash
before dispatch and returns a failing exit status after saving a failed comparison, so an
automated caller cannot mistake partial eager-only evidence for success.
All three experimental apps were subsequently verified stopped with zero tasks.
Their combined reported Modal usage was **$0.25824**; that is a usage-report observation, not a
guarantee that billing has finished settling. No GCP configuration or budget was changed.

Evidence: [standard regional results](../benchmarks/planner-optimization-2026-09-03/klein-regional-default/results.json),
[carried lantern](../benchmarks/planner-optimization-2026-09-03/klein-regional-default/compiled-3.jpg),
[open book on bridge](../benchmarks/planner-optimization-2026-09-03/klein-regional-default/compiled-5.jpg).
Deployment and cleanup: [audit](../benchmarks/planner-optimization-2026-09-03/cache-snapshot-audit.json).

## Next gate

Keep Vertex live and the existing Modal fallback unchanged. For the Klein candidate, prepackage
pinned weights and measure reusable compilation artifacts across real process restarts, then run
a larger prompt-fidelity and cold/warm latency comparison. Neither an unbounded warm GPU nor an
unverified snapshot is needed to test those alternatives. The local cache improvement is deployed;
the alternative renderer remains experimental.
