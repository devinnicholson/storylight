# Voice to a new scene

Open [the voice interface](http://127.0.0.1:18767/workbench?voice=1&session=voice-demo). Press **Describe scene**, speak a short description, then **Finish recording**. Review and correct the transcript, then press **Generate scene**. Finishing a recording never starts image generation. You can edit the text without recording another clip.

The Mac captures and transcribes audio locally. A loopback-only gateway sends scene requests through SSH to the Jetson's existing API. For reviewed voice descriptions, the Jetson compiles supported short sentences into typed visual facts using local rules. Its configured managed GCP image route creates the artwork. The Jetson kiosk follows the same `voice-demo` session directly on port 8080. Raw audio never crosses the SSH tunnel or reaches GCP. This is separate from the fixed-passage reading mode.

Use one or two sentences with explicit subjects and actions, such as “A quick brown fox jumps over a lazy dog.” Counts, colors, action targets, spatial relationships, and supported negative constraints remain attached to their subjects. Ambiguous or unsupported descriptions stop before renderer preparation or image generation. They never trigger an automatic model fallback. This bounded parser is not general story understanding or a Gemma accuracy improvement; other story-generation paths still use the local model.

Two subjects can share an action and location: “The white golden retriever and the Merle Aussie are playing in the field.” The compiler preserves compound breeds and the shared location. The current bound is two subject-action clauses total, including clauses expanded from coordinated subjects.

The prior scene stays visible until a new animated draft is available. Verified artwork then replaces the draft. Recording stops immediately when Finish is pressed. Transcription and submission have timeouts; overlapping submissions are blocked. If a submission response is lost, the interface checks the existing session rather than automatically submitting another billable request. An unresolved result offers **Check generation status**. Explicitly rejected descriptions can be edited and retried.

## Running setup

The current processes are:

- Mac `127.0.0.1:18766`: local Whisper, using `.models/whisper-base.en`.
- Mac `127.0.0.1:18768`: SSH forward to Jetson `127.0.0.1:8080`.
- Mac `127.0.0.1:18767`: voice gateway, serving the current frontend.
- Jetson kiosk: `http://127.0.0.1:8080/projector?pack=latest&session=voice-demo&present=1&reader=0&live=1`.

Keep the Mac API, gateway and tunnel running for voice input. The Jetson retains and displays the accepted scene independently. Its kiosk uses a temporary runtime override; the saved kiosk configuration is unchanged. The previous reverse tunnel used by the fixed-scene rehearsal is not needed for this voice flow.

After starting local Whisper as described in [microphone rehearsal](microphone-demo.md), restart the forward and gateway in separate terminals:

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

## Measured check

The [voice fidelity repair evidence](../benchmarks/voice-fidelity-2026-09-07/README.md) records the current deployed path. Five fixed descriptions passed through the voice gateway and Jetson preparation endpoint with **22–66 ms** of local planning; three unsupported descriptions refused. The corrected fox-and-dog image reached `master_ready` in **3.778 seconds**, including **3.656 seconds** in the existing managed image provider. Local planning took **39.9 ms**. The projected artwork was visually checked: one brown fox jumping over one resting dog. These are engineering smoke measurements, not a general latency or accuracy guarantee.

The API now identifies this path as `bounded-description-v2` with deterministic provenance. The [coordinated-description repair](../benchmarks/voice-coordination-2026-09-07/README.md) records the exact previously rejected two-dog transcript generating in **4.349 seconds**, with **67.6 ms** of local planning. Its detailed watercolor artwork was visually checked in the live projector iframe.

Finishing a recording makes no image request; Generate submits `reviewed_description: true`. Completed-scene reuse checks the planning mode, reviewed compiler revision, and render contract, so older interpretations cannot silently replace a new reviewed generation. Historical artwork remains available.

The Jetson service imports the installed package under `/opt/bookforge/.venv/lib/python3.12/site-packages/bookforge`, not its older `/opt/bookforge/src` tree. The deployment receipt retains before/after hashes and the backup location. Changes were import-tested on the Jetson, installed into that actual package, and activated with the restricted `bookforge-admin restart-api` helper.

### Earlier automatic-generation check

The [retained smoke result](../benchmarks/voice-to-scene-2026-09-07.json) used “The pink fox jumped over the river stream.” A synthetic WebM recording transcribed exactly through local Whisper in **0.299 seconds**. That recognized description was submitted through the browser, generating a new watercolor image of a pink fox jumping across a stream. The new image was visually inspected, and the Jetson kiosk was verified on the matching completed job.

Generation reached `master_ready` in **3.476 seconds**, including a **3.390-second** managed image call. This used a previously prepared local plan; preparing that plan initially took **10.028 seconds**. These separate measurements are not an uncached end-to-end latency guarantee. The image cost estimate was **$0.034**, not an invoice. The depth sidecar is a local projection gradient, not estimated scene geometry. The underlying renderer is the existing managed Vertex image model, not the experimental native Klein worker.

Automated tests cover one recording producing one transcription and no submission until Generate is pressed, empty speech, capture cleanup, request timeouts, stale session results, lost-response reconciliation, gateway routing and streamed response cleanup. The smoke check used synthetic speech and browser text submission; it does not substitute for a human microphone rehearsal on the new localhost origin. Allow microphone access when prompted.

The [matched local speech comparison](../benchmarks/local-asr-2026-09-07/README.md) tested base.en and small.en on twelve synthetic clips. Both preserved their meanings; small.en corrected numeral formatting but added about 0.31 seconds to warm recognition. Base.en remains the demo default. These clips do not establish accuracy on the user's microphone, and the original inaccurate recording was discarded.
