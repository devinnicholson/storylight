# Local Whisper latency: 717 ms warm in the voice smoke

The Storylight voice smoke transcribed the synthetic sentence "A cat chasing a mouse." in 717.41 ms after the MLX Whisper model had already been loaded. The first call in that process took 4,362.48 ms. Both calls returned the expected English text, and the audio stayed on the local workbench.

| Measurement | Result |
| --- | ---: |
| First process-local transcription | 4,362.48 ms |
| Warm transcription | 717.41 ms |
| Model | Whisper `small.en` through MLX Whisper |
| Input | 41,954-byte synthetic audio clip |
| Output | `A cat chasing a mouse.` |

This single smoke test exposed the main latency term immediately: loading and preparing the model cost several seconds, while reuse brought the same path below one second. A later controlled A-B-B-A startup experiment reached the same conclusion with fresh Python processes. Its median target time was 3,461.835 ms without preparation and 498.42 ms after a different synthetic primer had loaded the model. Every target transcript matched exactly.

## How the local path works

The browser records Opus/WebM through `MediaRecorder` and sends the bytes to `/v1/audio:transcribe`. Storylight validates the content type and configured size limit, writes the clip into a temporary directory, then calls `mlx_whisper.transcribe` against a local model path. The temporary directory disappears when the call finishes.

`condition_on_previous_text=False` prevents a preceding reader phrase from becoming decoder context for the next recording. Language can be fixed or auto-detected; the measured smoke used English and `small.en`. Storylight also leaves `logprob_threshold` unset so a confident decoded token cannot override Whisper's no-speech decision during a silent recording window.

MLX matters because Apple Silicon can keep the model and inference path on the workbench without sending reader audio to a browser speech API or cloud transcription service. The blocking decoder runs in a worker thread. An async lock serializes access, and request cancellation remains shielded until the worker drains, preventing the API from releasing the lock while MLX is still using shared model state.

Preparation is explicit. If startup audio is configured, application startup performs one bounded local transcription and reports the measured preparation time in readiness. A completed warmup is therefore distinguishable from a model file merely existing on disk. The system does not add a primer after reader speech has already entered the queue.

## Follow-up startup experiment

The later four-process screen used an A-B-B-A order. Arm A transcribed only the retained target clip. Arm B transcribed a different fox sentence first, then the identical target bytes. Median target latency fell from 3.462 seconds to 0.498 seconds, while median total ASR work remained close: 3.462 seconds for A and 3.352 seconds for B. Peak process RSS in the retained receipts ranged from about 729 MB to 907 MB.

The result supports moving one-time model work into service preparation. It does not show that preparation reduces total compute, and the four-process screen is too small for a tail-latency claim. Host file caches, Metal state, and unrelated Mac activity were uncontrolled.

## What we learned and what is next

Storylight now has a local ASR lifecycle that can prepare once, report readiness, reuse the loaded model, and keep audio inside the operator's network boundary. Warm transcription below one second made the language stage small enough to sit beside a multi-second image request without dominating the interaction.

The next experiment is startup-to-ready timing from a fresh service launch, followed by an idle-gap study. A larger audio set should vary clip duration, accent, silence, and background noise. Recognition quality and latency need separate reports; one exact synthetic sentence establishes integration and model reuse, not general word-error rate.

Raw smoke: [`voice-smoke-2026-09-07.json`](voice-smoke-2026-09-07.json). Controlled startup screen: [`voice-asr-startup/README.md`](../../experiments/voice-asr-startup/README.md). Implementation: [`asr.py`](../../src/storylight/asr.py).
