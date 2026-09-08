# Voice to a new scene

Open [the voice interface](http://127.0.0.1:18767/workbench?voice=1&session=voice-demo), press **Describe scene**, and speak. Generation starts automatically from a usable partial transcript while you continue talking. The finished artwork appears once it matches the latest stable description. **Finish recording** stops capture and immediately processes the final transcript; it is not a separate generation approval. You can also edit the text and use **Generate scene** to retry.

The Mac captures and transcribes audio locally. A loopback-only gateway sends scene requests through SSH to the Jetson's existing API. The bounded description compiler remains available. An optional isolated CPU language service uses spaCy syntax predictions to prepare additional scene drafts, checked against the original description. Its configured managed GCP image route creates the artwork. The Jetson kiosk follows the same `voice-demo` session directly on port 8080. Raw audio never crosses the SSH tunnel or reaches GCP. This is separate from the fixed-passage reading mode.

Use one or two sentences with explicit subjects and actions, such as “A quick brown fox jumps over a lazy dog.” Counts, colors, action targets, spatial relationships, and supported negative constraints remain attached to their subjects. Unresolved descriptions stop before renderer preparation or image generation. The configured language service can offer a draft for additional wording; it cannot admit rendering itself. Other story-generation paths still use the local model.

Every candidate still receives local fact and privacy checks. The automatic voice flow passes the resulting digest with the matching request; the API and renderer adapter independently recheck it. There is no extra confirmation click. Names or places excluded by the privacy policy remain local and are disclosed in the interface; they are not replaced with invented scenery. Completed-scene reuse also checks the digest and stored renderer prompt. Learned drafts report model provenance. With the service configured, bounded facts also receive a syntax audit and report that model dependency; without it, the bounded path remains deterministic.

Two subjects can share an action and location: “The white golden retriever and the Merle Aussie are playing in the field.” The compiler preserves compound breeds and the shared location. The current bound is two subject-action clauses total, including clauses expanded from coordinated subjects.

The prior scene stays visible while generation runs. Voice jobs withhold drafts and incomplete artwork from both the workbench and the physical projector. Completed artwork and depth are presented only when the job still matches the latest description. A changed transcript supersedes a speculative result; one image job runs at a time and intermediate queued descriptions are replaced by the newest one. Already-started image calls may still incur cost even when superseded.

Transcription and submission have timeouts. If a submission response is lost, the interface checks the existing session rather than automatically submitting another billable request. An unresolved result offers **Check generation status**. Rejected descriptions can be edited and retried.

### Speech scheduling and timing

The voice frontend starts its first partial transcription check after 1.2 seconds.
Subsequent checks wait for the previous request to finish. While audio is active,
the next check has at least a 350 ms completion gap and 750 ms start spacing;
the gap grows with transcription duration, up to two seconds. Quiet periods use
a two-second gap. The audio meter affects scheduling only: every request still
contains the cumulative recording, and Finish sends the complete recording.
The fixed-passage reader retains its previous two-second interval.

Before a partial voice request, the frontend calls MediaRecorder's `requestData`
and waits for a data event, bounded to 250 ms. This makes newly recorded audio
available sooner than waiting for the next periodic chunk. The existing chunk
collector retains every blob; Stop still drains the pending partial operation
before sending the full final recording. A timeout uses the chunks already
available. The [recording specification](https://www.w3.org/TR/mediastream-recording/)
defines the asynchronous data event; exact timing depends on the browser.

The existing 350 ms submission debounce, local scene-fact checks, one-image-job
limit, latest-description queue and completed-scene reuse remain in place.
Punctuation changes are checked through the existing fact comparison; the
frontend does not guess that two different descriptions mean the same thing.

For a rehearsal, `window.bookforgeVoiceTiming()` returns the latest recording's
bounded, memory-only event trace. It includes recording start, first detected
audio activity, ASR completion, scene checks, image submission/completion,
presentation acknowledgement and preview activation. No transcript or audio is
stored in this diagnostic trace. It resets on a new recording or page reload.
Audio activity is an amplitude estimate, not verified speech onset.

`preview_activated` comes from the same-origin projector iframe after artwork
activation. It is separate from the server's presentation acknowledgement and
does not measure a remote physical monitor or completion of the visual crossfade.
Job and recording identity checks exclude unrelated older generation events.

The [scheduling benchmark](../benchmarks/voice-scheduling-2026-09-08/README.md)
retains synthetic recordings, real local Whisper responses, scheduling policies
and their limitations. It makes no image-inference or human-microphone accuracy
claim. The image model and watercolor settings are unchanged.

In the [actual Chrome/WebM screen](../benchmarks/voice-scheduling-2026-09-08/browser/README.md),
the short target phrase arrived at 1.578 rather than 2.511 seconds; the paused
phrase arrived at 3.916 rather than 4.315 seconds. Final transcripts matched,
with three ASR requests per arm and clip. The headless meter did not advance,
so these runs used quiet-period backoff. The separate chunked-WAV screen covers
the active scheduling approximation and used about 40% more ASR request time.
These small screens support a rehearsal, not a general latency or accuracy claim.
Reload the workbench to receive the new frontend; no Jetson package or image
provider change is needed.

## Running setup

The current processes are:

- Mac `127.0.0.1:18766`: local Whisper, now using `.models/whisper-small.en`.
- Mac `127.0.0.1:18768`: SSH forward to Jetson `127.0.0.1:8080`.
- Mac `127.0.0.1:18767`: voice gateway, serving the current frontend.
- Jetson kiosk: `http://127.0.0.1:8080/projector?pack=latest&session=voice-demo&present=1&reader=0&live=1`.

Keep the Mac API, gateway and tunnel running for voice input. The Jetson retains and displays the accepted scene independently. Its kiosk uses a temporary runtime override; the saved kiosk configuration is unchanged. The previous reverse tunnel used by the fixed-scene rehearsal is not needed for this voice flow.

Start local Whisper in its own terminal, after stopping the existing listener on 18766. This server has image generation disabled:

```sh
BOOKFORGE_MODEL_BACKEND=fake \
BOOKFORGE_ASSET_BACKEND=disabled \
BOOKFORGE_LIVE_SCENE_BACKEND=disabled \
BOOKFORGE_ASR_BACKEND=mlx_whisper \
BOOKFORGE_ASR_MODEL=.models/whisper-small.en \
BOOKFORGE_DATA_DIR=.bookforge/microphone-demo \
.venv/bin/uvicorn bookforge.api:app --host 127.0.0.1 --port 18766
```

Both model directories are already downloaded. To roll back speech recognition,
stop this process and repeat the command with
`BOOKFORGE_ASR_MODEL=.models/whisper-base.en`. This changes no Jetson renderer.
The [fixed-passage rehearsal](microphone-demo.md) has separate instructions.

Start the forward and gateway in separate terminals:

```sh
ssh -N -o BatchMode=yes -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
  -o HostKeyAlias=jetson.local -i ~/.ssh/bookforge_jetson \
  -L 127.0.0.1:18768:127.0.0.1:8080 operator@192.0.2.10
```

```sh
.venv/bin/uvicorn bookforge.voice_gateway:create_app_from_env --factory \
  --host 127.0.0.1 --port 18767 --no-proxy-headers --timeout-graceful-shutdown 3
```

The address above was verified September 7; check Tailscale if it changes. The gateway refuses non-loopback clients, foreign Host/Origin headers, unexposed routes and oversized uploads. It forwards no browser credentials and does not retry image requests. It deliberately blocks GPU prewarm; generation uses the Jetson's existing provider configuration and budget checks.

The optional language service has its own Python environment and an owner-only
Unix socket. See [language-service deployment and rollback](voice-language-service.md).
API integration is a separate reviewed deployment; starting the service alone
does not enable it or change the active planner.

## Measured check

The [automatic browser check](../benchmarks/automatic-voice-2026-09-07/README.md)
used actual MediaRecorder audio, local Whisper, Jetson planning, and the managed
GCP renderer. Image submission began **521 ms** after the first usable transcript;
the new image and depth finished in **4.177 seconds**. It appeared while recording
continued, without Finish or Generate. These are one synthetic-speech smoke run's
measurements, not a human speech accuracy or latency guarantee.

The [voice fidelity repair evidence](../benchmarks/voice-fidelity-2026-09-07/README.md) records the earlier deployed bounded path. Five fixed descriptions passed through the voice gateway and Jetson preparation endpoint with **22–66 ms** of local planning; three unsupported descriptions refused. The corrected fox-and-dog image reached `master_ready` in **3.778 seconds**, including **3.656 seconds** in the existing managed image provider. Local planning took **39.9 ms**. The projected artwork was visually checked: one brown fox jumping over one resting dog. These are engineering smoke measurements, not a general latency or accuracy guarantee.

The API now identifies this path as `bounded-description-v2` with deterministic provenance. The [coordinated-description repair](../benchmarks/voice-coordination-2026-09-07/README.md) records the exact previously rejected two-dog transcript generating in **4.349 seconds**, with **67.6 ms** of local planning. Its detailed watercolor artwork was visually checked in the live projector iframe.

Those earlier checks used the manual review flow. The automatic voice flow now starts image work from partial or final transcripts and defers presentation until the complete result matches the current description. Completed-scene reuse still checks the planning mode, reviewed compiler revision, and render contract. Historical artwork remains available.

The [silence screen](../benchmarks/local-asr-2026-09-07/silence/README.md) explains
the local Whisper setting that prevents confident tokens from overriding its
no-speech decision. It removed a false suffix on long silent WAV/WebM recordings
and preserved 19 tested speech controls. It also disables low-confidence
temperature fallback; the screen does not prove general microphone accuracy.

The Jetson service imports the installed package under `/opt/bookforge/.venv/lib/python3.12/site-packages/bookforge`, not its older `/opt/bookforge/src` tree. The deployment receipt retains before/after hashes and the backup location. Changes were import-tested on the Jetson, installed into that actual package, and activated with the restricted `bookforge-admin restart-api` helper.

### Earlier automatic-generation check

The [retained smoke result](../benchmarks/voice-to-scene-2026-09-07.json) used “The pink fox jumped over the river stream.” A synthetic WebM recording transcribed exactly through local Whisper in **0.299 seconds**. That recognized description was submitted through the browser, generating a new watercolor image of a pink fox jumping across a stream. The new image was visually inspected, and the Jetson kiosk was verified on the matching completed job.

Generation reached `master_ready` in **3.476 seconds**, including a **3.390-second** managed image call. This used a previously prepared local plan; preparing that plan initially took **10.028 seconds**. These separate measurements are not an uncached end-to-end latency guarantee. The image cost estimate was **$0.034**, not an invoice. The depth sidecar is a local projection gradient, not estimated scene geometry. The underlying renderer is the existing managed Vertex image model, not the experimental native Klein worker.

The earlier smoke check used synthetic speech and browser text submission; it does not substitute for a human microphone rehearsal on the localhost origin. Allow microphone access when prompted.

The later [chasing-phrase speech comparison](../benchmarks/local-asr-2026-09-07/chasing/README.md) motivates activating small.en. Across 36 fixed synthetic clips, repeated in base/small/small/base order, small retained the cat/chasing/mouse tokens in **17/18** relevant clips versus base's **11/18**. Warm medians were approximately **0.539 seconds** and **0.256 seconds**. Small was worse on some clean clips, and both models still misheard breed names. The earlier [clean-speech screen](../benchmarks/local-asr-2026-09-07/README.md) remains historical evidence. Neither screen proves accuracy on the user's microphone; the original inaccurate recording was discarded.

The [20-description language screen](../benchmarks/voice-language-2026-09-07/README.md) produced drafts for 15 descriptions when combining the bounded and learned paths. This is **draft coverage, not 75% accuracy** or proof of general English understanding. The automatic voice flow retains fact validation; no image quality claim follows from that benchmark.
