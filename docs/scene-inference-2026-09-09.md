# Scene extraction: training and serving results

The V5 experiment improved learned scene extraction, but failed its refusal gate. A separate serving experiment reduced median paired warm latency by 59.49% across 128 descriptions, preserving every output token. Independent replay verified all 260 calls, including warmups. A controlled restart experiment also reduced first compiled preparation by 79.41% through reproducible compiler-cache reuse. These are GCP L4 extraction measurements; the live voice demo and image generator are unchanged.

## What was tested

Training used the pinned Gemma 4 E2B model, NF4 QLoRA, rank 16, and 4,800 synthetic examples. Two 1,200-step pilots compared Q/V adapters with adapters across the text decoder's linear layers. An independently authored 256-example development set selected Q/V. A fresh 4,800-step run selected checkpoint 2,400 by development loss. The 128-case test was authored separately; no test answers went to the GPU.

Inference used a resident BF16 model on one NVIDIA L4 in GCP, PyTorch 2.10, Transformers 5.13 and XGrammar 0.2.6. The original comparison ran both adapters twice on each case. All 512 outputs passed offline token and grammar replay.

| Independent synthetic screen | V4 | V5 |
| --- | ---: | ---: |
| Strict positive matches, both repetitions | 64/96 | 81/96 |
| Literal refusals, both repetitions | 28/32 | 25/32 |

V5 produced five schema-valid admissions on refusal cases. Existing grounding/privacy validation blocked all five, but that does not turn them into correct model refusals. The frozen promotion gate failed. The full results and caveats are in [the V5 report](../experiments/scene-adapter-v5-2026-09-09/RESULTS.md).

## Finding a faster serving path

Merging the adapter removed its separate inference operations. Opposite phase orders reproduced 12.88% and 13.20% matched-output latency reductions on four training probes. BF16 logits changed, so limited token parity is not general numerical equivalence.

Static caching plus compiled decoding initially saved about 63% on matching outputs. It also incorrectly refused a valid scene: “One blue penguin walks without running.” The same failure occurred with static caching without compilation, and repeated when phase order was reversed.

A first-token probe found different logits before decoding began. The allowed grammar masks were identical. CPU reconstruction reproduced the actual static attention masks and confirmed their causal/padding structure; it did not establish a mask bug or the precise numerical cause.

The candidate therefore keeps dynamic prompt processing, selects the first token with the existing grammar, transfers the resulting cache state into static buffers, and compiles subsequent decoding. Transfer tests cover full and sliding caches, sequence counters, reset, and buffer reuse. This retains grammar enforcement throughout.

| Repeated eight-probe test | Dynamic | Compiled transfer |
| --- | ---: | ---: |
| Median measured latency | 3,660 ms | 1,428 ms |
| Matched measured pairs | 16/16 | 16/16 |

The median paired reduction was **60.95%**. All 54 outputs, including warmups, matched across the three modes. The trace recorded CUDA graph execution. The first compiled preparation request took **111.35 seconds**, including generation; it is not an isolated compilation timer. Model loading and merging are separate costs. This derivative run did not record memory usage.

A new-process repeat reused the saved compiler directories and preserved all 54 outputs. Its first compiled preparation still took **103.45 seconds**, versus 111.35 seconds originally—only 7.09% less. The compiler reported two AOTAutograd and two FX graph cache misses again; existing Triton files remained unchanged while new higher-level graph artifacts appeared. Reusing these directories therefore did not solve preparation latency. That run did not isolate why the higher-level cache keys changed. This repeat warmed the model and GPU in preceding modes, so it does not measure full service startup.

The follow-up found a concrete cause to test: the model iterates a Python set of attention-layer types. Changing Python's hash seed reverses the rotary-input order. CPU subprocesses reproduced that behavior, and the actual GPU model recorded the same reversal. A three-process experiment held the wrapper, imports, weights and complete package set constant:

| Process | First compiled preparation | AOT / FX graph cache |
| --- | ---: | --- |
| Seed 0, new compiler cache | 109.25 s | 2 misses / 2 misses |
| Seed 0, reused compiler cache | 22.49 s | 2 hits / 2 hits |
| Seed 1, reused compiler cache | 102.61 s | 2 misses / 2 misses |

The same-seed restart reduced preparation by **79.41%**. All 162 generated streams passed independent token/grammar replay, and each restart preserved all 54 prior outputs. Generated decoder partitions from the changed-seed run matched the originals after swapping only the two rotary-argument names; independent static analysis reproduced that comparison without executing downloaded code. This supports seed-sensitive graph caching and the rotary-order explanation in this runtime. The intervention also changes other hash-sensitive behavior, so it does not isolate rotary ordering as the only possible cause.

These are eight training probes and one three-process cycle. Every compiled preparation follows dynamic/eager modes that already warm the model and GPU; **22.49 seconds is not whole-service startup**. The measured fix is a reproducible launch configuration with saved compiler artifacts. See [the restart report](../experiments/scene-compiler-restart-2026-09-09/RESULTS.md) for receipts and timing scope.

Removing grammar enforcement was also tested separately. It saved only 4.73% on matching outputs and introduced one malformed graph. That experiment supports retaining the grammar.

## Broader confirmation and deployment limits

The follow-up completed all 128 previously exposed descriptions in two arms, plus four retained training warmups: 260 calls. All 128 measured pairs were token-identical. Median warm extraction fell from **2,851.86 ms to 1,089.81 ms**, with a **59.49% median paired reduction**. Independent verification replayed every token through the actual tokenizer and grammar, checked model/cache provenance, and confirmed CUDA graph execution. The first compiled preparation request took 110.42 seconds. Fresh local scoring gave both merged arms 81/96 strict positives and 27/32 literal refusals, with all five unexpected admissions blocked by the unchanged validator. Those scores do not replace the original unmerged training comparison or reverse its failed gate. The confirmation prompt range is 638–654 tokens; it is not a long-context test. See [the confirmation protocol](../experiments/scene-cache-confirmation-2026-09-09/README.md) for the fixed ordering and limits.

Neither warm L4 latency nor cache-transfer correctness establishes Jetson latency. The resident BF16 model exceeds the device's memory budget, and the existing TensorRT Edge runtime lacks the required grammar interface. Quantized export, grammar support, fresh refusal-quality confirmation and physical-device measurements remain prerequisites for deployment. [The integration audit](../experiments/scene-adapter-v5-2026-09-09/integration-feasibility.md) records those constraints.

The best live-demo checkpoint remains `checkpoints/best-demo-2026-09-08`. Experiment code, raw receipts and independent reviews are versioned; model and adapter binaries remain local and ignored. A Git clone alone is insufficient for weight replay.

Both completed allocations were cleaned up: their VMs, boot disks and dedicated firewall rules were verified absent. Their VM/disk/IPv4 list-rate estimates are **$3.63** for the training/serving campaign and **$0.46** for the controlled restart, excluding transfer and earlier allocations. These estimates are not an account spending total.
