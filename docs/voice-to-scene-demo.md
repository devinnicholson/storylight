# Voice to a new scene

Open [the voice interface](http://127.0.0.1:18767/workbench?voice=1&session=voice-demo). Press **Describe scene**, speak a short description, then **Finish & generate**. The Mac transcribes once and submits the recognized text automatically. You can edit that text and generate again without recording another clip.

The Mac captures and transcribes audio locally. A loopback-only gateway sends scene requests through SSH to the Jetson's existing API. Its local Gemma planner creates the visual direction, and its configured managed GCP image route creates the artwork. The Jetson kiosk follows the same `voice-demo` session directly on port 8080. Raw audio never crosses the SSH tunnel or reaches GCP. This is separate from the fixed-passage reading mode.

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

The [retained smoke result](../benchmarks/voice-to-scene-2026-09-07.json) used “The pink fox jumped over the river stream.” A synthetic WebM recording transcribed exactly through local Whisper in **0.299 seconds**. That recognized description was submitted through the browser, generating a new watercolor image of a pink fox jumping across a stream. The new image was visually inspected, and the Jetson kiosk was verified on the matching completed job.

Generation reached `master_ready` in **3.476 seconds**, including a **3.390-second** managed image call. This used a previously prepared local plan; preparing that plan initially took **10.028 seconds**. These separate measurements are not an uncached end-to-end latency guarantee. The image cost estimate was **$0.034**, not an invoice. The depth sidecar is a local projection gradient, not estimated scene geometry. The underlying renderer is the existing managed Vertex image model, not the experimental native Klein worker.

Automated tests cover one recording producing one transcription and one submission, empty speech, capture cleanup, request timeouts, stale session results, lost-response reconciliation, gateway routing and streamed response cleanup. The smoke check used synthetic speech and browser text submission; it does not substitute for a human microphone rehearsal on the new localhost origin. Allow microphone access when prompted.
