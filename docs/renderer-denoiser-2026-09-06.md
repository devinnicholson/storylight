# Denoiser optimization

Status: implemented and independently reviewed; cloud execution blocked before process start.
All 941 Python tests and four JavaScript suites pass. There is no GPU timing, capture-success
or speed-gain result from this candidate yet.

The conditioning trial located the dominant warm cost in the transformer: about 1.35 seconds
across four calls. This experiment tests whole-transformer CUDA Graph replay around the existing
regional compiler kernels. It changes neither weights nor the watercolor prompt treatment.

## Why this candidate

The earlier full-transformer compiler trial already improved on eager execution. Repeating that
comparison would not establish a gain over today's regional compiler baseline. Regional
`reduce-overhead` failed with overwritten output storage; regional `default` is already used.
The new adapter owns two independent graph pools, updates every input tensor before replay,
and clones each returned output so later replays cannot overwrite it. Capture forbids compiler
recompilation and measurements cannot silently capture or fall back to ordinary execution.

This follows [PyTorch 2.8's capture and static-storage requirements](https://docs.pytorch.org/docs/2.8/notes/cuda.html#cuda-graphs)
and [NVIDIA's guidance on graph scope and profiling](https://docs.nvidia.com/dl-cuda-graph/latest/troubleshooting/performance-issues.html).
Graphs reduce launch overhead; they do not accelerate an already saturated GPU's arithmetic.
The [Meta/Hugging Face Flux Fast work](https://pytorch.org/blog/presenting-flux-fast-making-flux-go-brrr-on-h100s/)
combines compilation, graph execution and attention changes. Its Flux.1/H100 results are not
performance predictions for Klein/L4.

Pinned upstream inspection found that Klein's twenty single-stream blocks already combine
their QKV and MLP projections. Only five dual-stream blocks remain candidates for separate
QKV fusion. Native SDPA already chooses optimized attention kernels; a kernel profile must
identify the selected implementation before replacing it. No approximate denoising cache,
quantization, smaller resolution or fewer steps is included here.

## Frozen screen

One isolated AWS us-east-1 L4, the existing baked weights and compiler cache, BF16, 1024×576,
four steps, guidance 1.0, and the frozen 128/256-token watercolor cases:

1. Two ordinary warmups, one per token bucket.
2. Two separately profiled baseline renders, with CPU and CUDA activities, no tensor shapes,
   values, stack traces or memory profiling.
3. Two preparation renders, capturing one transformer graph per bucket.
4. Eight unprofiled measured renders in paired alternating order: 128 B/C, 256 C/B, 128 C/B,
   256 B/C. Both measured variants run after the profiler has stopped.

All fourteen master/depth pairs must match historical JPEG hashes exactly. The speed screen
requires at least 5% median paired runtime reduction, positive median reduction in both buckets,
no more than 2% regression in the observed maximum, and peak reserved memory below 23 GiB.
This is a small engineering screen, not a production tail-latency guarantee or reference-image
qualification. Full runtime includes conditioning, denoising, VAE, depth and JPEG encoding;
planning, transport and presentation are outside that boundary.

The profiler correlates launch APIs with kernels in four transformer ranges. Kernel unions,
spans, launch durations and delayed starts overlap; they must not be added or used to label
all gaps as CPU overhead. Incomplete correlations remain explicit. Profiles run before capture
so a failed optimization still returns useful diagnostics. Failed or partial evidence cannot
produce a passing speed summary.

## Cost and execution bounds

The existing approved paid allowance stays $8, with the existing $2 reserve and $36 workspace
stop. A proposed $0.27 hold covers two resource slots for 150 seconds at $0.00082054 per second,
plus $0.02 extra margin: $0.266162. The slots include the regional pricing margin; this is a
conservative local bound, not a provider-enforced cap. No image build, new weights, platform
snapshot, retained GPU or automatic retry is configured.

The parent starts its 120-second work timer before deployment and allows 30 seconds for cleanup.
The one-shot deployment has minimum zero, maximum one container, one concurrent input, fixed
CPU/RAM limits and a durable claim. Existing closed-run receipts and fresh attributed charges
are reconciled before reserving; they are not described as settled refunds. Shutdown must show
zero tasks and containers. Exact funding and result receipts accompany any executed run.

## Execution status

Automatic approval review refused the deployment twice, including after the upload scope was
verified. Its stated reason was that explicit permission to export experiment source and
benchmark data to Modal was required. The payload consists of the four experiment/runtime
Python modules and a manifest whose two synthetic prompts match commit `26df35b` exactly;
it contains no private story passages, audio, user images or credentials. Model weights are
already present in the existing remote image and are not newly uploaded or downloaded.

Neither command started a process, deployment or GPU call. The $0.27 reservation was released
using the existing no-dispatch release path. The execution manifest is back in draft state;
the blocked authorization and source hashes are retained separately. The completed-run billing
reconciliation remains intact. Further cloud execution needs explicit upload approval; it must
not be attempted through another command or provider to bypass the rejection.

The [execution receipt](../benchmarks/renderer-denoiser-2026-09-06/execution-status.json) and
[frozen draft](../benchmarks/renderer-denoiser-2026-09-06/manifest-draft.json) preserve the ready
experiment. No live renderer or Jetson settings changed.
