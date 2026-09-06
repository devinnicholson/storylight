# Denoiser optimization

The whole-transformer CUDA Graph trial completed with **28 exact JPEGs**, but failed the
5% speed screen: median paired runtime reduction was **0.449%**. No production renderer
changed. The recovered baseline traces show that matrix multiplication dominates this
workload, with little idle space between denoiser kernels.

## Experiment and outcomes

The candidate wraps the existing regional compiler kernels in two independent CUDA Graph
pools. It updates every input tensor before replay and clones each returned output. Capture
forbids recompilation; measurements cannot silently recapture or fall back. Weights, BF16,
1024×576 resolution, four steps, guidance 1.0, and the frozen 128/256-token watercolor cases
remain fixed. Earlier regional `reduce-overhead` had failed with overwritten output storage;
regional `default` is the established baseline.

The fixed schedule is two baseline warmups, two separately profiled baseline renders, two
candidate preparation renders, then eight unprofiled measurements: 128 B/C, 256 C/B, 128 C/B,
256 B/C. Both measured variants run after profiling has stopped. The speed screen requires
at least 5% median paired runtime reduction, positive median reduction in each bucket, no
more than 2% regression in the observed maximum, and peak reserved memory below 23 GiB.
All fourteen master/depth pairs must match historical JPEG hashes exactly.

| Attempt | Observed outcome | Evidence |
| --- | --- | --- |
| A | Three renders and six exact JPEGs; failed at trace processing before graph capture. Worker time 39.997 s; supervisor total 67.383 s. Raw traces were lost, so A's exact diagnostic cause remains unknown. | [Result](../benchmarks/renderer-denoiser-2026-09-06/failed-a/result.json), [platform audit](../benchmarks/renderer-denoiser-2026-09-06/failed-a/platform-audit.json) |
| B | Provider capacity wait consumed the work window. Initialization completed and the first warmup started, but cancellation returned no images. No speed result. | [Platform audit](../benchmarks/renderer-denoiser-2026-09-06/capacity-b/platform-audit.json), [supervisor](../benchmarks/renderer-denoiser-2026-09-06/capacity-b/supervisor.json) |
| C | Broader US placement obtained one GCP `us-east1` L4. All 14 renders and 28 JPEGs verified; two graphs captured and each replayed 12 times. Peak reserved memory 18.186 GiB. Supervisor total 95.801 s. | [Summary](../benchmarks/renderer-denoiser-2026-09-06/graph-c/summary.json), [result](../benchmarks/renderer-denoiser-2026-09-06/graph-c/result.json) |

C's four paired runtime reductions were **3.197%, 0.566%, 0.332%, and −0.270%**.
Bucket reductions were 1.765% for 128 and 0.148% for 256. Three small wins and one loss
support neither the frozen speed threshold nor a production promotion. Runtime includes
conditioning, denoising, VAE, depth and JPEG encoding; planning, transport and display are
excluded. The supervisor total also includes deployment and cleanup, not per-image latency.
All three attempts have retained shutdown receipts showing stopped apps and zero containers.

Initial commands were blocked before dispatch by automatic approval review. The user then
explicitly approved the Modal upload; [that approval](../benchmarks/renderer-denoiser-2026-09-06/approved-dispatch.json)
and each attempt's separate manifest, funding, dispatch and shutdown receipts remain retained.
There were no automatic retries or image rerolls. Historical failures were not overwritten.

## Trace repair and measured bottleneck

C retained both raw traces even though its online analysis failed. Kineto emits each
`record_function` name as both a CPU `user_annotation` and an asynchronous
`gpu_user_annotation`. The old analyzer selected names alone, found eight ranges, and refused
because it expected four. The repaired offline analyzer selects the four CPU annotations,
then correlates CUDA runtime/driver launches with kernel events. A synthetic regression now
includes both annotation types. This establishes C's exact failure; it does not recover A's
missing trace.

Both repaired traces attribute **1,708 of 1,708 kernels**, with zero unmatched or ambiguous
launch correlations. The original C result and summary still say diagnostics incomplete;
the separate [128-token analysis](../benchmarks/renderer-denoiser-2026-09-06/graph-c/offline-analysis-0.json)
and [256-token analysis](../benchmarks/renderer-denoiser-2026-09-06/graph-c/offline-analysis-1.json)
record the offline repair against the original raw hashes.

| Profiled baseline, four denoiser calls | 128 tokens | 256 tokens |
| --- | ---: | ---: |
| Kernel busy time | 1.324681 s | 1.396449 s |
| Gaps within kernel spans | 0.006064 s | 0.007613 s |
| Matrix multiplication kernel time | 1.062757 s (80.23%) | 1.107092 s (79.28%) |
| FlashAttention kernel time | 0.149766 s (11.31%) | 0.172262 s (12.34%) |

The traces contain 100 flash SDPA calls per image; native attention is already using the
flash path. Kernel families are classified from names, with Triton reduction/pointwise
prefixes taking precedence over fused-operation names containing “attention.” CPU launches
queue far ahead of later GPU work. Launch-to-kernel delays therefore indicate queued work,
not equivalent GPU idle time. The profiled measurements can perturb execution and must not
be substituted for the unprofiled speed comparison. CPU ranges, launch durations and GPU
intervals overlap and are not additive.

The next supported direction is a matched faster-BF16-device comparison or a bounded GEMM
selection experiment, with the same fidelity checks and separately reported compilation cost.
More graph launch optimization has little demonstrated headroom. This is consistent with
[NVIDIA's explanation of graph performance limits](https://docs.nvidia.com/dl-cuda-graph/latest/troubleshooting/performance-issues.html)
and [PyTorch's capture requirements](https://docs.pytorch.org/docs/2.8/notes/cuda.html#cuda-graphs);
it is not a prediction of a particular replacement GPU's speed or image equivalence.

## Retained trace review

The compressed traces total **1,410,507 bytes** (20,488,402 bytes expanded). Inspection found
kernel/operator names, numeric launch dimensions, timing/correlation identifiers, L4 device
properties, trace IDs and generated `/tmp/bookforge-denoiser-…/trace.json` names. No prompt
text, private story source, tensor values, user home paths, email addresses, URLs or credential
patterns were found. Shapes/stack/memory profiling were disabled; kernel launch dimensions
remain present. These two synthetic-run traces can be retained for reproducibility, with their
runtime metadata understood. This review is specific to these artifacts, not a blanket claim
that arbitrary profiler exports contain no sensitive data.
