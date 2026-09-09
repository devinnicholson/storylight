# Dynamic prefill and compiled decoding

The first static-cache experiment sped up matching outputs, but incorrectly refused
“One blue penguin walks without running.” Both eager and compiled static caches
produced the same refusal. Compilation alone therefore did not explain the change.
These are fixed training probes, not a general accuracy evaluation.

## Where the outputs diverged

`results/gpu/cache-prefill-01` isolates the first forward pass with a fresh cache for
each call. The model and merged adapter are identical; all eight inputs use both
cache types. Full vocabulary logits and the initial grammar masks are retained.

Seven first-token decisions agree. The penguin input changes at the first token,
with a maximum absolute logit difference of 22.5625. Other inputs differ by
0.4375–0.8125 without changing their first token. The grammar allows exactly the
same tokens in both modes, so different grammar masks do not explain this result.

The static full-attention mask has shape `[1, 1, 641, 1024]`; its dynamic counterpart
uses the SDPA causal shortcut. Both sliding-attention masks have shape
`[1, 1, 641, 641]`. `check-prefill-masks.py` reproduces the actual GPU mask hashes on
CPU, using identical Transformers masking/cache source files. The full mask is
correctly causal and excludes padding, and both sliding masks are identical.
This does **not** establish a logically incorrect attention mask. Attention shape,
kernel selection and numerical propagation remain possible explanations that have
not been isolated.

## The intervention

`cache-bridge-profile.py` preserves dynamic prefill and selects its first token
with the same grammar processor used for the remaining generation. It transfers
the cache into static storage and resumes generation with one uncached token.
The processor remains active throughout; completed outputs still require EOS,
complete grammar acceptance and the same size limits.

Gemma's sliding cache needs special handling: dynamic storage retains 511 tokens
for a 512-token window. Static storage holds 512 and drops its oldest slot before
appending the next token. After a saturated prefill, the bridge places the 511
retained entries after a disposable first slot. It preserves the absolute sequence
counters. Full-attention layers copy their complete prefix. The model's shared-KV
layers do not add separate cache entries.

`results/gpu/cache-bridge-eager-01` contains sixteen completed generations. All
eight bridged outputs match their dynamic baseline token for token, including the
previously refused penguin input. This is evidence for the intervention on these
probes, not broad accuracy or production acceptance. Eager bridging does not
establish a speed gain: excluding the first cold baseline pair, its median paired
latency is approximately 2.6% slower.

## Compiled experiment

`cache-bridge-compiled.py` tests dynamic generation, bridged eager decoding and
bridged compiled decoding on the same eight training probes. Each mode has two
preparation calls and sixteen measured calls. It starts with new local Inductor
and Triton directories, retains compilation counters and a decoder trace, and
records cache reset, transfer and pointer-stability evidence. A fresh local cache
directory does not imply cold driver or filesystem caches.

`cache-bridge-reuse.py` is a bounded helper for this pinned model. Its CPU test
checks long-to-short and sliding-boundary requests while preserving storage
addresses and matching the next three dynamic-cache K/V updates. It is not a
validated general cache conversion API. GPU token parity, observed compilation,
actual CUDA graph launches and cold preparation cost must be checked separately.
The verified results below establish the bounded serving gain; they do not authorize production promotion.

## Compiled bridge result

The 54-call run completed in 305.24 seconds. Independent verification found every
preparation and measured output identical across all three modes. The sixteen
measured compiled outputs reduce paired latency by a median **60.95%** relative
to dynamic generation. Median request times are 3,660 ms dynamic, 3,816 ms bridged
eager and 1,428 ms bridged compiled; the compiled maximum is 1,836 ms.

The first compiled preparation request takes **111.35 seconds**; the second takes
1,223 ms. Measured compiled requests spend a median 90.35 ms in dynamic prefill
and 3.74 ms transferring/resetting the cache. Pointer receipts remain stable.
The run records two compiled graphs and an actual CUDA graph launch in the decoder
trace. These measurements include the intervention's prefill and transfer costs.
They exclude model loading, adapter merging and tokenization/output decoding.
This derivative runner did not record GPU memory peaks.

This is a useful speed result with token parity on eight training probes. It is
not sufficient to promote the parser or claim general correctness. The separate
`../scene-cache-confirmation-2026-09-09` experiment expands confirmation to 128
already-exposed inputs, with answer-free inputs and a preregistered schedule.

## Broader confirmation

The separate 260-call run completed on the 128 already-exposed descriptions plus
four retained training warmups. Independent verification confirms **128/128 measured
outputs token-identical** between dynamic and compiled-transfer generation.
Median measured latency is **2,851.856 ms dynamic versus 1,089.807 ms compiled**;
the median paired reduction is **59.49%**. Independent verification replayed all
260 token streams and checked model/cache provenance and CUDA graph execution.
Both merged arms score 81/96 strict positives and 27/32 literal refusals; the
unchanged local validator blocks all five unexpected admissions. These are
confirmation scores on exposed cases, not replacements for the original unmerged
comparison. The previously failed V5 refusal gate remains unchanged.

## Fresh process with existing compiler artifacts

The warm-artifact run completed and passed independent verification. All 54
outputs match the prior run token for token. Its first compiled preparation
request takes **103.45 seconds**, versus 111.35 seconds with new local compiler
directories: only **7.09% less**. It retains a 61.11% median paired warm-latency
reduction on the repeated eight-probe screen, but does not solve startup latency.

Before execution, the compiler directory contained 992 files totaling 42.22 MB.
Every existing file remained unchanged; six new Inductor files were added. The
counters report two AOTAutograd cache misses and two FX graph cache misses, as in
the first process. All 878 existing Triton files remained unchanged, and the
benchmarking counters observed in the first run are absent. This shows retained
lower-level artifacts alongside missed higher-level graph caches. It does not
identify why their cache keys differed or attribute the full time to one stage.

This measures a fresh Python process with warm compiler artifacts, not a cold
service, cold GPU or fresh machine. The first preparation timer includes generation
and does not isolate compilation or whole-process startup. Further work should
explain the graph-cache misses and measure startup separately before choosing a
persistent-worker or packaged-graph deployment strategy.
