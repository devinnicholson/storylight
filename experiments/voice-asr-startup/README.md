# Local ASR startup preparation screen

A synthetic call before the first real target moved most of the first-process
delay into preparation. This supports testing one-time preparation at ASR service
startup. It does **not** support queuing extra work when the microphone starts,
nor explain the latest 1.207-second prefix from an already running service.
Production ASR and its configuration were unchanged.

Four fresh Python processes ran sequentially in the fixed A–B–B–A order. A called
only the target; B transcribed a different synthetic fox phrase, then the exact
same target bytes. The target was a retained real Chrome MediaRecorder Opus/WebM
prefix of synthetic speech, with expected text `A cat chasing a mouse.` The
primer was an existing macOS TTS recording converted locally to Opus/WebM.

| Run | Primer ASR | Target ASR | All ASR calls | Entire child process |
|---|---:|---:|---:|---:|
| A1, first target | — | 3.690 s | 3.690 s | 4.580 s |
| B1, prepared | 2.446 s | 0.354 s | 2.800 s | 3.555 s |
| B2, prepared | 3.262 s | 0.643 s | 3.905 s | 4.669 s |
| A2, first target | — | 3.234 s | 3.234 s | 3.940 s |

Median target time was 3.462 seconds without preparation and 0.498 seconds after
preparation. Median primer time was 2.854 seconds. Median combined ASR work was
3.462 seconds for A and 3.352 seconds for B; this tiny screen does not establish
a reduction in total work. All four target transcripts were exactly equal to the
reference, and both primer transcripts exactly matched the fox reference. This
is one target and one primer, not a recognition-quality evaluation.

All six calls used the actual `LocalTranscriber` with the current small.en model,
English, `condition_on_previous_text=False`, `logprob_threshold=None` and all
other production defaults. No prompts, output correction, decoder changes or
audio truncation were introduced. Installed mlx-whisper's `ModelHolder` was empty
before each process's first call and populated before both B targets. This
establishes model reuse; it does not isolate import, weight loading, FFmpeg,
Metal scheduling, encoder or decoder costs. A synthetic execution could also
exercise different decoder paths from later speech.

ASR times include temporary files, audio decoding and inference, plus first-call
imports and model load. There were no competing calls to each isolated lock;
queue contention was excluded by construction. Process times also include Python
startup, Bookforge imports and teardown. The four-child loop took 16.752 seconds;
input hash verification and primer conversion happened beforehand. Each child
had a 15-second deadline within a 60-second loop, with at most two seconds for
termination and two for kill/join on failure. All four exited normally and were
reaped. All retained stderr files are empty.

Host/file/Metal caches and unrelated Mac activity were uncontrolled; model-file
hash verification itself reads the weights before the experiment. No idle-gap
screen, fresh-boot measurement, user audio, live service restart, HTTP request,
microphone capture or cloud call occurred. `HF_HUB_OFFLINE` and related offline
flags were set, Python network connection/name lookup was denied, and the model
path and weight digest had to match the existing local files. Execution required
local Metal access outside the filesystem sandbox.

The next implementation candidate is one bounded, coalesced startup preparation
whose completed state is reported separately from mere model availability. It
should never add a primer behind already queued speech or repeat per recording.
Before adopting it, preserve the existing speech/noise controls and test the
real service's startup-to-ready behavior. Current fallback can use stochastic
decoding, so matching options alone cannot guarantee all future transcripts.

`abba/protocol.json` pins the executed harness, ASR source, weights, configuration,
audio inputs, FFmpeg executable, installed decoder sources and package versions.
`abba/*-process.json` records lifecycle and raw stdout/stderr digests;
`abba/[1-4]-[AB].json` contains each call. `abba/summary.json` derives the medians.
No executed source or receipt was modified after the run.

To repeat this four-process screen into a new directory from the repository root:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python \
  experiments/voice-asr-startup/run.py --output /tmp/bookforge-asr-startup-new
```

Input pins intentionally refuse a changed production ASR implementation. A future
experiment needs its own reviewed protocol; do not rewrite this retained result.
