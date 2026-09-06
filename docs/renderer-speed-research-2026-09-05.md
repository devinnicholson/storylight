# Renderer speed: evidence and next experiments

Preserve the restored watercolor paper-theater treatment, Klein 4B, BF16, L4, 1024×576,
four steps and guidance 1.0. Speed work must preserve factual content, continuity and legibility.
The flat illustration trial did not qualify the restored style. No optimization below is promoted.

## Measured boundaries

The [completed simple-scene trial](simple-scenes-2026-09-05.md) returned all twelve requests and
verified all 24 master/depth JPEGs on one GCP `us-east1` L4. Human correctness and continuity
remain ungraded. These are Mac SDK measurements, excluding planning and projector onset.

| Boundary | Observed time | Interpretation |
| --- | ---: | --- |
| Cold initialization | 31.669 s | Includes model loading and compiler-cache setup. |
| Model loading within initialization | 19.189 s | A substantial cold-start target; not the whole startup. |
| Compiler-cache setup within initialization | 0.890 s | Loading cached wrappers does not execute the first forward pass. |
| Remaining initialization time | 11.589 s | Imports and other setup were not individually profiled in this run. |
| First text pipeline image computation | 10.858 s | Additional first-execution work after initialization. |
| First complete client artifact | 49.287 s | Includes waiting, generation, download, verification and storage. |
| Later text pipeline image computation | 1.610–1.625 s | Text encoding, denoising and VAE decode are not separately timed yet. |
| Four text targets, complete client | 2.488 s median; 2.574 s maximum | One exceeded the fixed 2.5-second maximum. |
| Three later reference targets, pipeline image computation | 3.224–3.231 s | Already exceeds 2.5 seconds before delivery. |
| Three later reference targets, complete client | 4.768 s median; 4.983 s maximum | All failed the delivery gate; first reference execution is excluded here. |

Depth and JPEG encoding together added roughly 0.08 seconds. The remaining client/server
difference also includes SDK handling and transfer; it is not a measurement of network latency
alone. These small populations do not establish service tail latency. The initialization detail
comes from the [platform audit](../benchmarks/simple-scenes-2026-09-05/platform-audit.json);
request timings are retained in the [summary](../benchmarks/simple-scenes-2026-09-05/results/summary.json).

## Evidence-ranked work

| Priority | Candidate and primary evidence | Compatibility, prerequisites and limits |
| --- | --- | --- |
| Strongest systems lever: session preparation | [Modal's cold-start guide](https://modal.com/docs/guide/cold-start) supports preparing a worker ahead of demand and retaining it with a bounded idle window. | The existing renderer uses a 90-second idle window. Verify required shapes before reporting ready; preserve expiry and failure invalidation. Idle resources are billed. This can avoid cold work during reading, but does not eliminate preparation or guarantee survival. Automatic preparation and longer retention remain unqualified. |
| Tested — parallel loading not promoted | Parallel checkpoint loading is supported by the pinned [Transformers 4.57.1 environment options](https://huggingface.co/docs/transformers/v4.57.1/en/reference/environment_variables). | The four-worker comparison below preserved every reference image but showed inconsistent gains. The final sequential worker beat both parallel workers. Only the text encoder has multiple shards. |
| 2 — exact conditioning and stage profiling | Klein's [Diffusers 0.39 pipeline](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/flux2/pipeline_flux2_klein.py) accepts prepared prompt embeddings. Its Qwen call discards logits, while [Transformers 4.57.1 Qwen3](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen3/modeling_qwen3.py) still computes the vocabulary projection. | Measure tokenization, Qwen encoding, reference encoding, denoising and VAE decode. A conditioning path using the underlying Qwen model could avoid unused logits even for first-seen prompts. Require identical selected hidden states and final image hashes; no speed magnitude is established. |
| 3 — attention implementation | [Torch 2.8 SDPA](https://docs.pytorch.org/docs/2.8/generated/torch.nn.functional.scaled_dot_product_attention.html) already selects optimized kernels. The [Diffusers 0.39 dispatcher](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/models/attention_dispatch.py) provides native and optional FlashAttention backends. | Identify the actual kernel on L4 before changing it. Test both text and reference shapes; an optional backend needs a compatible binary and separate compiler artifacts. Different kernels can change floating-point results. Regional compilation is already enabled. |
| 4 — smaller VAE decoder | BFL's [FLUX.2 small decoder](https://huggingface.co/black-forest-labs/FLUX.2-small-decoder) supports Klein 4B and advertises about 1.4× faster decoding. | This applies only to decoding, not complete generation. Diffusers 0.39 has the required [separate decoder-channel configuration](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/models/autoencoders/autoencoder_kl_flux2.py). Checkpoint loading and visual quality still need qualification; changed decoder weights are not byte-preserving. |
| 5 — quantization or another serving engine | BFL publishes [FP8 and NVFP4 results](https://bfl.ai/blog/flux2-klein-towards-interactive-visual-intelligence); [NVIDIA's L4 specifications](https://images.nvidia.com/aem-dam/Solutions/Data-Center/l4/nvidia-ada-gpu-architecture-whitepaper-v2.1.pdf) include FP8 Tensor Cores. | BFL's up-to-1.6× FP8 and 2.7× NVFP4 figures use RTX 5080/5090 at 1024². NVFP4/Blackwell results do not transfer to L4. FP8 storage alone does not prove faster arithmetic. An exact loader/kernel combination and blinded fidelity comparison are prerequisites. |
| Defer approximate denoising shortcuts | [SGLang's optimization guide](https://docs.sglang.io/docs/sglang-diffusion/performance-optimization) separates kernel/residency tuning from TeaCache, Cache-DiT, progressive resolution and quantization. | These latter methods can change outputs. Four-step Klein leaves little computation to skip without changing its learned trajectory. NVIDIA's [TensorRT-LLM FLUX.2 speedup](https://developer.nvidia.com/blog/scaling-nvfp4-inference-for-flux-2-on-nvidia-blackwell-data-center-gpus/) concerns the dev model on Blackwell with several combined interventions; it does not establish compatibility or gains for our pinned Klein/L4 stack. |

Prepared embeddings must preserve the exact prompt, tokenizer/model revisions, chat template,
128/256-token bucket, BF16 and selected layers. Reusing them helps only when the prompt is already
known or repeated; report preparation, misses and hits separately. Moving work before the timer
is not a first-request speedup. Do not label semantic reuse of a similar prompt as an exact hit.

Parallel loading must be configured before framework imports. The pinned
[Diffusers loader](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/models/modeling_utils.py)
also requires its low-memory loading path; inspect that effective path rather than assuming an
environment variable proves concurrent I/O. First execution needs separate profiling:
[CUDA 12.8 lazy loading](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-c-programming-guide/index.html#lazy-loading)
can defer kernel work until first use. The observed 10.858 seconds is not all proven compilation.

Reference VAE conditioning can likewise be prepared for identical source-image bytes, but it
does not remove the larger attention sequence used during each denoising step. BFL's
[reference KV-cache design](https://github.com/black-forest-labs/flux2/blob/main/docs/flux2_klein_kv_cache.md)
uses a separate 9B-KV model. It is not a validated switch for the existing 4B checkpoint.

## Existing fixes and rejected directions

The workbench's Generate path no longer awaits background planner warmup before submitting
the scene request. A focused real-flow regression fails against the old code and passes with
the change; background readiness reporting remains. This removes a proven blocking dependency,
not a measured number of milliseconds. Server completed-scene lookup and planner cache lookup
already precede the work they can avoid. Explicit renderer prewarm reuse also already shares
overlapping preparation without extending its original deadline.

The [cold-start history](renderer-cold-start-2026-09-05.md) bounds further hypotheses:

- Reducing requested RAM from 64 to 16 GiB retained all 24 reference images but improved median
  four-render cycle time only 6.07%, failing the frozen 25% gate.
- Imports-only GPU snapshots restored twice with identical images, but later four-render client
  cycles took 76.608 and 35.767 seconds. Different client hardware and lifecycle prevent a causal
  comparison with the earlier baseline.
- The [warmed snapshot](renderer-warmed-snapshot-2026-09-05.md) internally verified four warmups,
  then platform checkpointing failed. No images returned. The separate serial-compiler trial
  also returned no images before its 180-second deadline. Neither qualifies restored latency;
  the timeout does not prove eventual incompatibility.

Modal's [snapshot limitations](https://modal.com/docs/guide/memory-snapshots) also identify storage
bandwidth and Torch Compiler compatibility constraints. Another snapshot attempt is not the
next experiment; the failed qualifications do not authorize automatic retries.

The loading experiment preserves the baked weights, compiler cache, watercolor references,
seeds and image/depth hashes. It measures complete four-render cycles and first-artifact worker
offsets separately; returning four images together does not measure first-image client delivery.
Fresh containers do not guarantee fresh physical hosts or uncached storage.

## Completed loading comparison

Parallel loading is **not promoted**. Four fresh AWS `us-east-1` L4 processes completed in ABBA
order, generating sixteen watercolor images. All 32 master/depth JPEGs matched the frozen
references exactly. The [journal](../benchmarks/renderer-loading-2026-09-05/results/journal.jsonl)
and [summary](../benchmarks/renderer-loading-2026-09-05/results/summary.json) retain the results.

| Order | Loading | Model load | First finished artifact inside worker | Complete four-render client cycle |
| --- | --- | ---: | ---: | ---: |
| 1 | Sequential | 14.562 s | 37.595 s | 77.997 s |
| 2 | Parallel | 4.972 s | 24.348 s | 41.645 s |
| 3 | Parallel | 5.272 s | 25.539 s | 43.154 s |
| 4 | Sequential | 4.643 s | 23.602 s | 41.038 s |

The two-sample median first-artifact offset improved 18.48%, but the final sequential worker
outperformed both parallel workers. The first baseline also encountered a documented capacity
wait. Host, filesystem-cache and order effects remain possible explanations; this experiment
does not identify their individual contributions. The 28.76% median client-cycle reduction is
not a demonstrated parallel-loading gain or first-image delivery improvement.

Actual inventory found two text-encoder shards totaling 8,044,981,992 bytes. The transformer,
VAE and depth model each have one safetensors file. Four configured threads therefore do not
provide four-way loading across the pipeline. Peak process RSS stayed between 19.57 and
19.89 GiB; repeat image computation stayed between 1.534 and 1.638 seconds.

Before the corrected run, the harness stalled during metadata lookup without submitting a GPU
input. Modal 1.5.5's raw RPC client had been created on one event loop and reused by the public
SDK on another. Metadata inspection now runs in a separate process; hydration has a 15-second
deadline. A behavioral regression fails against the archived client and passes with the fix.
The original attempt and source are retained under
[predispatch](../benchmarks/renderer-loading-2026-09-05/predispatch/manifest.json).

Both apps were stopped and verified at zero containers. The first attempt lasted 98.545 seconds;
the corrected run lasted 215.307 seconds including shutdown. The corrected 260-second work
deadline plus 60-second cleanup allowance and the entire first attempt fit the same $1.20
reservation under the conservative combined bound. No additional funding or refunds were used.
Reported workspace usage remained $14.56522654 afterward; this is a delayed billing floor, not
the settled experiment charge. The full reservation remains held.

The [conditioning comparison completed September 6](renderer-conditioning-2026-09-06.md):
all tensors and images matched, but median paired runtime changed only 0.077%. It is not
promoted. Stage timing identifies the transformer as the dominant warm computation; investigate
its kernel and launch costs next. Keep bounded session preparation as the primary defense
against cold starts; another snapshot or parallel-loading rollout is not supported here.

Verification: 929 Python tests, all four JavaScript suites, scoped Ruff and diff checks passed.
The public summary reproduces byte-for-byte, and independent review verified all 32 artifacts.
The browser submission fix is included; the accepted cloud renderer and Jetson service
were not redeployed.
