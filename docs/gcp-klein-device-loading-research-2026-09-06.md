# Native GCP Klein loading: next experiment

The strongest next loading candidate is `device_map="cuda"` during pipeline loading,
tested with four synchronized loading stages in both arms. This is a supported path in the pinned
libraries, not a demonstrated improvement on our Cloud Run GPU. This document proposes
offline preparation only; it does not authorize a deployment or paid comparison.

The current [runtime](../deploy/klein_scene_runtime.py) loads the complete Klein pipeline
on CPU and then calls `.to("cuda")`. It preserves BF16, four steps, 1024×576 output,
the original watercolor prompts and pinned model weights. Depth loading is separate
and uses FP16. Compiler setup happens after model loading; imported compiler artifacts
cannot remove checkpoint reads or parameter placement.

The [eager-loading ABBA comparison](../benchmarks/gcp-klein-loading-2026-09-06/summary.json)
reduced first-client median by 8.837%, but failed its gates. Factory times were 59.463,
43.941, 39.709 and 38.149 seconds in mmap/eager/eager/mmap order. The last mmap worker
outperformed both eager workers. The later
[L cache producer](../benchmarks/gcp-klein-cache-2026-09-06/l/renders/summary.json)
reported 72.394 seconds for its factory. L is a different worker experiment, not another
matched loading sample. This variability strengthens the case for stage measurements;
the existing factory timer does not identify storage, Qwen loading, allocation or
host-to-device copies as the bottleneck.

## Ranked options

1. **Direct GPU placement during loading. Low implementation difficulty.**
   Add `device_map="cuda"` to the pipeline's existing `from_pretrained` call, leaving
   checkpoint files, dtype, depth loading and compiler policy unchanged. Diffusers
   0.39.0 accepts the CUDA device string and forwards it to both Diffusers and
   Transformers components. Its single-device mapping also permits the existing
   trailing `.to("cuda")`; no map reset or CPU offloading is needed. This makes a
   one-keyword candidate possible.
   [Pinned pipeline validation and placement](https://raw.githubusercontent.com/huggingface/diffusers/v0.39.0/src/diffusers/pipelines/pipeline_utils.py),
   [component forwarding](https://raw.githubusercontent.com/huggingface/diffusers/v0.39.0/src/diffusers/pipelines/pipeline_loading_utils.py).

   With a device map, Diffusers and Transformers preallocate device allocator capacity
   before loading parameters. The current CPU-first path does not activate this GPU
   loading preparation. This is the specific mechanism to test; it does not eliminate
   reading weights or transferring their contents to the GPU. Verify resulting
   parameter devices, dtypes and values, and retain image/depth comparisons.
   [Diffusers 0.39.0 allocator path](https://raw.githubusercontent.com/huggingface/diffusers/v0.39.0/src/diffusers/models/modeling_utils.py),
   [Transformers 4.57.1 allocator path](https://raw.githubusercontent.com/huggingface/transformers/v4.57.1/src/transformers/modeling_utils.py).

2. **Reshard the transformer, then compare sequential and parallel loading. Medium
   difficulty; conditional on component measurements.** Qwen currently has two shards
   totaling 8,044,981,992 bytes, while the transformer has one 7,751,109,744-byte file.
   Four loading threads therefore cannot provide four-way transformer shard loading.
   Diffusers 0.39.0 supports parallel shard loading with `low_cpu_mem_usage`; Transformers
   4.57.1 separately reads the parallel-loading environment flag. If transformer loading
   dominates, smaller safetensors shards could expose more concurrent reads. Preserve
   every tensor's name, shape, dtype and byte hash, and compare both loaders against the
   same reshared layout. Do not combine resharing, parallelism and device placement in
   the first causal comparison.
   [Pinned Diffusers loader](https://raw.githubusercontent.com/huggingface/diffusers/v0.39.0/src/diffusers/models/modeling_utils.py),
   [pinned Transformers loader](https://raw.githubusercontent.com/huggingface/transformers/v4.57.1/src/transformers/modeling_utils.py).

   The [prior Modal comparison](renderer-speed-research-2026-09-05.md#completed-loading-comparison)
   did not establish a parallel-loading gain: its final sequential worker beat both
   parallel workers. Native image streaming is a different environment, but that alone
   is insufficient reason to repeat the same toggle without component evidence.

3. **Change storage delivery only after identifying read stalls. Higher difficulty.**
   Google documents optimized container image streaming and GCS buffered reads/file
   caching. Its service guidance describes GCS as more predictable but potentially
   slower for very large models. Optimized GCS access requires Direct VPC with
   `all-traffic` and Private Google Access; file caching consumes container memory.
   This would be a separate storage experiment with unchanged model files, not an
   established repair for the measured factory time.
   [Cloud Run GPU best practices](https://docs.cloud.google.com/run/docs/configuring/services/gpu-best-practices).

   A bounded sequential read of the existing checkpoint files could first test
   sensitivity to prefetch without changing loaders. Count all prefetch time and memory
   in cold-start results: it may merely move work earlier. Do not assume default mmap
   reads are universally random; safetensors 0.8.0 bulk tensor loading follows metadata
   offset order. The rejected eager experiment does not isolate page-fault behavior.
   [Pinned safetensors implementation](https://raw.githubusercontent.com/huggingface/safetensors/v0.8.0/bindings/python/src/lib.rs),
   [retained eager-source proof](../benchmarks/gcp-klein-loading-2026-09-06/loading-source-proof.json).

## Minimal matched instrumentation

The prepared [baseline](../experiments/renderer-device-loading/baseline/klein_scene_runtime.py)
and [candidate](../experiments/renderer-device-loading/candidate/klein_scene_runtime.py)
use identical constructor instrumentation. The
[AST proof](../experiments/renderer-device-loading/runtime-diff-proof.json) verifies that
the candidate differs only by the device-map keyword and that code outside the
constructor remains unchanged. Preserve identical resource settings, image base, weights,
prompts, seeds, compiler-cache policy and request schedule. Fresh instances do not
establish uncached hosts or storage.

Each stage ends with `torch.cuda.synchronize()` before logging its host wall duration
and Linux process peak RSS in KiB:

- `imports_and_cuda_initialization`: starts before the constructor's framework import
  statements and includes its first CUDA synchronization. The current worker imports
  and checks frameworks and CUDA before constructing this runtime, so this label does
  not measure total process imports or all CUDA initialization.
- `pipeline_from_pretrained`: loads the complete Klein pipeline. Text encoder,
  transformer, VAE, tokenizer and scheduler are not individually timed.
- `pipeline_to_cuda`: measures the existing trailing `.to("cuda")`. Candidate parameter
  placement occurs during pipeline loading, so this interval alone cannot compare total
  transfer cost between arms.
- `depth_construction`: combines depth model loading, image-processor loading and depth
  pipeline placement, preserving their existing sequence.

These synchronized boundaries and logs add overhead and can change overlap relative to
production. Compare the equally instrumented arms; do not treat an uninstrumented
factory as their control. Stage durations mix reads, allocation and completed device
work; they are not isolated GPU-copy measurements. RSS is a cumulative process
high-water mark, not stage allocation or container peak memory.

The external worker factory timer must enclose its earlier framework/device checks and
all constructor imports, synchronization and logging.
The runtime's own `load_seconds` still starts after the import stage and is a separate
measurement. Keep complete first-client artifact-ready timing, compiler-wrapper time
and first-use rendering of each bucket separate. Do not substitute the sum of stage
durations for the inclusive factory timer. No PCIe bandwidth, disk bandwidth or
model-loading speedup follows from checkpoint bytes divided by that mixed timer.

Finer timing of Qwen, transformer, VAE or individual depth components is future,
unimplemented work, not a prerequisite for this bounded screen. Add such wrappers or
CUDA profiling only if the four-stage result warrants them, with the same instrumentation
in both comparison arms.

## Ruled-out head payload and next import diagnostic

The exact Klein revision's Qwen configuration sets `tie_word_embeddings=true`.
Its checkpoint index has 398 tensor keys, including the required input embedding,
and no `lm_head` tensor. Transformers 4.57.1 declares the output head as tied.
A backbone-only loader therefore cannot remove separately stored head weights from
this checkpoint. This rules out that payload-saving hypothesis, without claiming
zero Python-object overhead. The [retained proof](../experiments/renderer-device-loading/research/qwen-head-proof.json)
binds the exact public configuration and index bytes; no model blobs were downloaded.
[Pinned Qwen implementation](https://raw.githubusercontent.com/huggingface/transformers/v4.57.1/src/transformers/models/qwen3/modeling_qwen3.py).

The retained G image configuration confirms `PYTHONDONTWRITEBYTECODE=1`; its build
history uses `pip install --no-cache-dir`, without `--no-compile`. Python's flag stops
writing bytecode during imports; it does not demonstrate that dependency bytecode is
absent. Pip's download cache and bytecode compilation are separate controls. The base
image history removes Python bytecode before the later dependency installation, so a
targeted standard-library/app precompile is conceivable, but first inventory the actual
image's usable `.pyc` files. There is no evidence yet for a blanket dependency
`compileall` repair or a speed estimate.
[Retained image configuration](../benchmarks/gcp-klein-2026-09-06/retry-g/image-config.json),
[Python bytecode and import timing options](https://docs.python.org/3.12/using/cmdline.html),
[pip compilation options](https://pip.pypa.io/en/stable/cli/pip_install/).

A more specific diagnostic is the constructor's `from transformers import pipeline`.
The pinned package maps this lazy export to `pipelines`, whose initializer imports
all task pipeline modules. Importing `pipelines.depth_estimation` directly still
executes that parent initializer. This gives a concrete dependency path to measure,
not an attribution of P's 12.47-second constructor import/synchronization stage.
Use `-X importtime` in a fresh process in the actual immutable image, following the
worker's earlier imports, and retain the enclosing wall timer. This profiling is
proposed, not run here.
[Pinned task imports](https://raw.githubusercontent.com/huggingface/transformers/v4.57.1/src/transformers/pipelines/__init__.py).

If that path dominates, a later narrow depth adapter could use the existing model
and image processor directly. It must reproduce image conversion, FP16 input casting,
device placement, no-grad execution, the pipeline's CPU output transfer, processor
postprocessing at the original image size, and min/max uint8 normalization before
the existing JPEG encoding. Compare intermediate depth tensors and exact depth JPEGs;
simply calling the model is not equivalent. No adapter is implemented, and moving
depth preparation later would shift work rather than remove it from artifact-ready
latency.
[Pinned depth pipeline](https://raw.githubusercontent.com/huggingface/transformers/v4.57.1/src/transformers/pipelines/depth_estimation.py),
[pinned pipeline execution](https://raw.githubusercontent.com/huggingface/transformers/v4.57.1/src/transformers/pipelines/base.py).
