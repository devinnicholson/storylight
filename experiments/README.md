# Experiments

Start with the [research results](../docs/research-results.md) and [numeric summary](../research/results.json) for measured outcomes and their limits.

This directory retains the final public method for each research area. Historical candidates, operational receipts, deployment identifiers, compiler caches, generated corpora, model weights, and redundant reports are intentionally excluded. These scripts document the methods; model and cloud dependencies still require their own setup and licenses.

| Area | Source |
| --- | --- |
| Scene extraction and adapter training | [Selected V5 experiment](scene-adapter-v5-2026-09-09/) |
| Cache transfer and compiled decoding | [Broader confirmation](scene-cache-confirmation-2026-09-09/) |
| Compiler reuse across processes | [Controlled restart experiment](scene-compiler-restart-2026-09-09/) |
| Prepared image generation | [Serving worker](prepared-klein-serving/) |
| Voice recognition startup | [ASR experiments](voice-asr-startup/) |

Before running an experiment, review its dependencies, model licenses, required private inputs and resource lifecycle. Cloud launch scripts can allocate paid resources; use an explicit budget and verify teardown. Runtime candidates still require the quality and integration checks described in the research summary before use in the demo.
