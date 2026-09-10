# Experiments

Start with the [research results](../docs/research-results.md) and [numeric summary](../research/results.json) for measured outcomes and their limits.

This directory retains research source ported to the Storylight package name. Historical receipts, deployment identifiers, compiler caches, weights and redundant reports are preserved privately. References marked “archived” are not included here. These scripts document the methods; they are not a complete, ready-to-run reproduction bundle. Renamed source does not retain the original experiment source hashes.

| Area | Source |
| --- | --- |
| Scene extraction and adapter training | [V5 training](scene-adapter-v5-2026-09-09/) |
| Cache transfer and compiled decoding | [Broader confirmation](scene-cache-confirmation-2026-09-09/) |
| Compiler reuse across processes | [Controlled restart experiment](scene-compiler-restart-2026-09-09/) |
| Prepared image generation | [Serving worker](prepared-klein-serving/), [renderer session](renderer-session/) |
| Voice recognition startup | [ASR experiments](voice-asr-startup/) |

Before running an experiment, review its dependencies, model licenses, required private inputs and resource lifecycle. Cloud launch scripts can allocate paid resources; use an explicit budget and verify teardown. Runtime candidates still require the quality and integration checks described in the research summary before use in the demo.
