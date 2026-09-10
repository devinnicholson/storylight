# Research results

These experiments measure learned scene extraction and its serving runtime on a Google Cloud NVIDIA L4. They do not measure image generation or microphone-to-projector latency, and they do not change the live voice demo.

| Experiment | Result | Scope |
| --- | --- | --- |
| Gemma adapter training | Strict positives improved from 64/96 to 81/96; literal refusals fell from 28/32 to 25/32 | Independent synthetic screen, two repetitions; promotion gate failed |
| Dynamic prefill with compiled static decoding | Median resident extraction fell from 2,851.86 to 1,089.81 ms; median paired reduction 59.49% | All 128 measured token pairs matched; previously exposed descriptions |
| Compiler reuse across fresh processes | First compiled preparation fell from 109.25 to 22.49 seconds, a 79.41% reduction | Same wrapper and hash seed, saved compiler artifacts; eight training probes |

The training campaign used Gemma 4 E2B with NF4 QLoRA. Independently authored development examples selected the recipe and checkpoint before test inference. Five schema-valid unexpected admissions remained; the existing grounding/privacy checks blocked them, but that does not make them correct model refusals.

The runtime candidate keeps dynamic prompt processing, transfers its cache state into static buffers and compiles subsequent decoding. All 260 calls in the broader confirmation, including warmups, passed independent token and grammar replay. Its fresh merged-model scores do not replace the original training comparison.

The restart experiment used three processes. Stable seed 0 produced two AOT and two FX graph cache hits; changing to seed 1 reversed recorded model ordering, restored cache misses and raised preparation to 102.61 seconds. All 162 streams remained valid and each restart preserved all 54 prior outputs. This supports a seed-sensitive caching explanation, without isolating rotary ordering from every other hash-sensitive operation.

Preparation follows modes that already warm the model and GPU and includes generation. **22.49 seconds is not whole-service startup.** The screens are small, fixed experiments, not general accuracy, long-context or Jetson benchmarks. Quantized export, device-specific grammar support and fresh refusal-quality confirmation remain necessary before deployment.

The [curated numeric summary](../research/results.json) preserves exact values and sample sizes. Original raw receipts and large artifacts are retained in a private archive; this public summary does not provide complete experiment reproduction.

## Models and licensing

The experiments use [Gemma 4 under Apache 2.0](https://ai.google.dev/gemma/apache_2). Historical Gemma 3 integrations use [Gemma's separate terms](https://ai.google.dev/gemma/terms). [Whisper code and weights](https://github.com/openai/whisper) use MIT. Optional image backends include [FLUX.2 Klein 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) and [SANA-Sprint 1.6B](https://huggingface.co/Efficient-Large-Model/Sana_Sprint_1.6B_1024px), whose model cards identify Apache 2.0 licenses. Other model variants can have different terms. Managed Gemini image generation is a cloud API, not an open-weight model bundled with this project.

Model weights, compiled engines and cloud credentials are not included in a repository checkout. Publishing the application source does not grant additional rights to third-party models or assets.
