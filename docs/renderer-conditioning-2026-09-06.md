# Exact conditioning: verified, no useful speed gain

Skipping Qwen's unused vocabulary projection preserved every checked tensor and image, but
did not materially accelerate generation. Keep the adapter isolated from the live renderer.

## What changed

The opt-in adapter passes the existing `Qwen3Model` backbone to Klein's original conditioning
helper. Tokenization, masks, selected hidden layers, BF16 and position IDs stay upstream.
It leaves registered models and weights intact and restores the helper after each call.
This follows the pinned [Diffusers 0.39 helper](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/flux2/pipeline_flux2_klein.py)
and [Transformers 4.57.1 Qwen implementation](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen3/modeling_qwen3.py).

One AWS `us-east-1` L4 used the existing baked checkpoints and regional compiler cache.
Two baseline warmups covered both token buckets, followed by eight measured renders in
balanced paired order. All calls recomputed conditioning. Resolution remained 1024×576,
with four denoising steps, guidance 1.0 and the frozen watercolor prompts and seeds.

## Measured result

| Prompt bucket | Baseline runtime median | Candidate runtime median |
| --- | ---: | ---: |
| 128 tokens | 1.6961 s | 1.6970 s |
| 256 tokens | 1.7892 s | 1.7877 s |

The four paired reductions were −0.455%, +0.353%, +0.343% and −0.190%: two wins and two
losses, with a median reduction of **0.077%**. This does not establish a useful speed gain.
All twenty master/depth JPEGs matched the historical references exactly. All ten prompt
embeddings and position-ID tensors matched their baseline values, shapes and dtypes.

Baseline medians across the four warm measurements:

| Boundary | Time |
| --- | ---: |
| Text conditioning, CUDA interval | 0.0716 s |
| Four transformer calls, summed CUDA intervals | 1.3512 s |
| VAE decoding, CUDA interval | 0.2432 s |
| Complete image pipeline, wall time | 1.6836 s |

The transformer is the dominant warm computation. Even eliminating all text conditioning
would remove only roughly 4% of the measured pipeline time. The candidate did not measurably
reduce conditioning time in this small sample.

These are instrumented server measurements. CUDA event intervals can include host launch gaps;
they are not sums of isolated kernel execution times. Host execution/enqueueing intervals are
recorded separately and must not be added to CUDA intervals. Wrappers stayed outside compiled
blocks, and tensor copying, equality checks and hashing ran after the existing render timer.
See [PyTorch CUDA timing](https://docs.pytorch.org/docs/2.8/notes/cuda.html#asynchronous-execution).
The experiment does not measure per-image transport, local planning or projector onset.

## Cold start and session preparation

Worker initialization took 26.250 seconds, including 12.477 seconds of model loading.
First pipeline executions took 9.388 seconds for the 128-token shape and 5.450 seconds for
256 tokens. The worker's complete comparison took 55.804 seconds; supervision, deployment,
queueing, transport and shutdown together took 133.265 seconds. Platform logs explicitly
reported waiting for L4 capacity. The difference is not an isolated network or queue measurement.

Existing explicit preparation already warms both shapes and coalesces overlapping requests.
Successful generation already reuses the worker. Adding a separate recent-generation label
would remove no current wait, and one warmed shape must not imply both are prepared. This
increment therefore leaves the existing 90-second preparation and expiry semantics intact.

The next inference investigation should separate transformer kernel work from launch overhead
before choosing another compiler configuration or attention backend. First inventory previous
compiler experiments to avoid repeating them. Session preparation remains the practical way
to move cold initialization ahead of reading; neither this adapter nor a new status label
solves capacity waiting.

## Evidence and cost

The [public evidence](../benchmarks/renderer-conditioning-2026-09-06/README.md) contains all twenty
JPEGs, tensor fingerprints, stage timings, source pins, metadata, and shutdown proof.
Aggregation reproduces the saved summary byte-for-byte. Independent review passed; 935 Python
tests, all four JavaScript suites, scoped Ruff and diff checks passed.

The isolated app stopped with zero tasks and containers. Both one-shot claims remain consumed.
The $0.50 reservation fit the existing $8 paid allowance after two completed experiments'
pending gross holds were tightened using their recorded lifetimes. This was not a settled
refund: their remaining holds and the new $0.50 remain pending. Fresh reported usage was
$14.70244036 before and after the run; provider billing is delayed.
The conservative 180-second, two-resource-slot bound plus $0.20 preparation allowance is
$0.4953944. It is not a provider-enforced billing cap.
