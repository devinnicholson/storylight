# Compiled scene extraction: 2.85 seconds fell to 1.09 seconds

Dynamic prefill followed by compiled static-cache decoding reduced median resident scene-extraction latency from 2,851.856 to 1,089.807 ms on an NVIDIA L4. Every one of the 128 measured output-token pairs matched exactly. The median paired reduction was 59.49 percent, measured after four retained warmup calls.

| Measurement | Dynamic | Compiled bridge |
| --- | ---: | ---: |
| Measured descriptions | 128 | 128 |
| Median resident extraction | 2,851.856 ms | 1,089.807 ms |
| Token-identical measured pairs | 128/128 | 128/128 |
| Strict positive score | 81/96 | 81/96 |
| Literal refusals | 27/32 | 27/32 |

The experiment ran 260 calls in one process: two warmups per arm and 128 measured descriptions per arm. Dynamic decoding used source order; the compiled bridge used reverse order. Prompt lengths ranged from 638 to 654 tokens. Static capacity was 1,024 tokens, with generation capped at 256 tokens.

## Why a cache bridge was necessary

The first attempt used a static key/value cache for both prompt prefill and token decoding. It was fast on matching outputs, though one valid prompt, "One blue penguin walks without running," changed into a refusal. The divergence appeared on the first generated token. Grammar masks were identical, while the maximum full-vocabulary logit difference reached 22.5625 for that prompt; the other seven probes stayed between 0.4375 and 0.8125.

The working intervention keeps prompt prefill on the dynamic path, where the reference output was stable, then transfers the populated attention state into fixed static buffers before decoding the remaining tokens. Full-attention layers copy the entire prefix. Gemma's sliding-window layers require a narrower transfer because dynamic storage keeps 511 tokens for a 512-token window; the bridge writes those entries after one disposable slot, preserves absolute sequence counters, then appends the next token without moving the buffers.

Stable buffer addresses are important because `torch.compile` and CUDA graphs benefit when tensor shapes and storage do not change between decode steps. The bounded runtime resets the reusable cache for each request, verifies transfer lengths, and applies the same grammar processor when it selects the first token and every token afterward. EOS handling and the final schema checks remain unchanged.

An earlier 54-call run on eight probes recorded two compiled graphs and an actual CUDA graph launch. Its measured compiled requests spent a median 90.35 ms in dynamic prefill and 3.74 ms on cache transfer and reset, then gained most of their time during repeated decoding. The broader 128-case run confirmed token parity on the exposed screen and retained the same 59-percent class of reduction.

## Reading the result correctly

This is resident extraction time for a merged BF16 Gemma adapter on one L4. The request timer excludes CPU tokenization and output decoding. It also excludes model loading and adapter merging. The first compiled preparation call took 110.416 seconds, so the steady-state gain arrives with a large preparation cost.

The cache bridge preserved observed token behavior across all 128 inputs while cutting median latency by more than half. It does not repair the adapter's five unexpected refusal admissions, which local validators still block, and it does not measure image generation or Jetson execution.

## What we learned and what is next

Compiling the repetitive decode step gave the speedup; preserving the reference prefill path recovered parity. Cache layout is part of model behavior, especially when attention kernels and sliding windows change with cache type. A fast decoder needs token-level comparison before aggregate quality scoring can be trusted.

The next run should randomize arm order and add repetitions. Fresh descriptions need a wider prompt-length distribution and longer continuations. Device work should then measure compilation time, peak memory, grammar support, and steady-state latency on the Jetson target. The preparation cost must be amortized over a known request volume before this path belongs in a live service.

Evidence: [`research/results.json`](../../research/results.json) and [`cache confirmation`](../../experiments/scene-cache-confirmation-2026-09-09/RESULTS.md). Design notes: [`CACHE-BRIDGE.md`](../../experiments/scene-adapter-v5-2026-09-09/CACHE-BRIDGE.md).
