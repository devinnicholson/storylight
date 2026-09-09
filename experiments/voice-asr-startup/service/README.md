# ASR preparation through the HTTP service

Four sequential fresh services ran A–B–B–A with preparation off/on/on/off.
Each transcribed the same retained synthetic cat/mouse WebM once; prepared
services first decoded the unrelated retained synthetic fox primer during lifespan
startup. All four target transcripts matched exactly. No microphone, image
request, or paid provider call was involved. The live demo was not restarted
for this benchmark. Model, decoder options and image quality were unchanged.

| Median | Unprepared | Prepared |
| --- | ---: | ---: |
| Process launch to HTTP readiness | 0.793 s | 3.091 s |
| First transcription HTTP request | 2.638 s | 0.298 s |

This moves work into startup; it does not establish lower total work, sustained
warm-request acceleration, or an end-to-end image-generation speedup. Two
observations per arm and one short synthetic target cannot establish p95 or
recognition quality. Host caches and competing activity were uncontrolled.

`measured/results.json` retains source/input hashes, request times and the exact
readiness response. Logs show each distinct child successfully bound port 18769,
handled its own GET/POST, and completed shutdown. They also retain a Python
multiprocessing resource-tracker warning about one semaphore on each exit in
both arms; these are not clean-stderr runs. No child was left serving.

The executed harness has a fixed port and no port reservation. Its actual logs
establish ownership for these runs; before reusing it, verify 18769 is free and
remove ambient HTTP proxy settings. Offline model flags were set, and isolated
working directories prevented loading the repository .env. This harness does
not install a network-denial hook. It enables only fake language models and
local ASR, with image backends disabled.

## Operation

Set `BOOKFORGE_ASR_STARTUP_AUDIO` to an existing short synthetic speech file
alongside `BOOKFORGE_ASR_BACKEND=mlx_whisper`. The API transcribes it once before
creating service clients or accepting requests. Its transcript is discarded.
`/v1/runtime:status` reports `prepared in … ms` only after successful decoding.
Missing, nonregular, empty, oversized, unsupported or speechless input fails
startup. Unset the option to restore the prior startup behavior.

The option is off by default and is not supported on other ASR backends.
Preparation is never triggered by microphone start or each recording. Repeated
sequential prepare calls do no work; the lifecycle is its sole production caller.
A same-process MLX worker cannot be hard-cancelled safely: cancellation joins it
before releasing its lock. There is no claim of a hard startup execution bound;
use a process supervisor if a hard deadline is required. The benchmark supervisor
allows 30 seconds to become ready, then terminates/kills and reaps its child.

The earlier `../abba` experiment remains untouched and pins the old ASR source;
its harness intentionally refuses the changed implementation.
