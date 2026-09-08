# Voice to a new scene

Open [the voice interface](http://127.0.0.1:18767/workbench?voice=1&session=voice-demo). Press **Describe scene**, speak a short description, then **Finish recording**. Review and correct the transcript, then press **Generate scene**. Finishing a recording never starts image generation. You can edit the text without recording another clip.

The Mac captures and transcribes audio locally. A loopback-only gateway sends scene requests through SSH to the Jetson's existing API. The bounded description compiler remains available. An optional isolated CPU language service uses spaCy syntax predictions to prepare additional scene drafts, checked against the original description. Its configured managed GCP image route creates the artwork. The Jetson kiosk follows the same `voice-demo` session directly on port 8080. Raw audio never crosses the SSH tunnel or reaches GCP. This is separate from the fixed-passage reading mode.

Use one or two sentences with explicit subjects and actions, such as “A quick brown fox jumps over a lazy dog.” Counts, colors, action targets, spatial relationships, and supported negative constraints remain attached to their subjects. Unresolved descriptions stop before renderer preparation or image generation. The configured language service can offer a draft for additional wording; it cannot admit rendering itself. Other story-generation paths still use the local model.

For a learned draft, the interface shows extracted facts and local omissions before asking for confirmation. Names or places excluded by the privacy policy stay in local review metadata; the user must accept their omission or edit the description. They are not replaced with invented scenery. Confirmation carries a digest of the original text, style, parser revision, facts and omissions. The API and renderer adapter independently recheck it; changed facts require another review. Completed-scene reuse also checks the confirmed digest and stored renderer prompt. Learned drafts report model provenance. With the service configured, bounded facts also receive a syntax audit and report that model dependency; without it, the bounded path remains deterministic.

Two subjects can share an action and location: “The white golden retriever and the Merle Aussie are playing in the field.” The compiler preserves compound breeds and the shared location. The current bound is two subject-action clauses total, including clauses expanded from coordinated subjects.

The prior scene stays visible until a new animated draft is available. Verified artwork then replaces the draft. Recording stops immediately when Finish is pressed. Transcription and submission have timeouts; overlapping submissions are blocked. If a submission response is lost, the interface checks the existing session rather than automatically submitting another billable request. An unresolved result offers **Check generation status**. Explicitly rejected descriptions can be edited and retried.

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

The [voice fidelity repair evidence](../benchmarks/voice-fidelity-2026-09-07/README.md) records the earlier deployed bounded path. Five fixed descriptions passed through the voice gateway and Jetson preparation endpoint with **22–66 ms** of local planning; three unsupported descriptions refused. The corrected fox-and-dog image reached `master_ready` in **3.778 seconds**, including **3.656 seconds** in the existing managed image provider. Local planning took **39.9 ms**. The projected artwork was visually checked: one brown fox jumping over one resting dog. These are engineering smoke measurements, not a general latency or accuracy guarantee.

The API now identifies this path as `bounded-description-v2` with deterministic provenance. The [coordinated-description repair](../benchmarks/voice-coordination-2026-09-07/README.md) records the exact previously rejected two-dog transcript generating in **4.349 seconds**, with **67.6 ms** of local planning. Its detailed watercolor artwork was visually checked in the live projector iframe.

Finishing a recording makes no image request; Generate submits `reviewed_description: true`. Completed-scene reuse checks the planning mode, reviewed compiler revision, and render contract, so older interpretations cannot silently replace a new reviewed generation. Historical artwork remains available.

The Jetson service imports the installed package under `/opt/bookforge/.venv/lib/python3.12/site-packages/bookforge`, not its older `/opt/bookforge/src` tree. The deployment receipt retains before/after hashes and the backup location. Changes were import-tested on the Jetson, installed into that actual package, and activated with the restricted `bookforge-admin restart-api` helper.

### Earlier automatic-generation check

The [retained smoke result](../benchmarks/voice-to-scene-2026-09-07.json) used “The pink fox jumped over the river stream.” A synthetic WebM recording transcribed exactly through local Whisper in **0.299 seconds**. That recognized description was submitted through the browser, generating a new watercolor image of a pink fox jumping across a stream. The new image was visually inspected, and the Jetson kiosk was verified on the matching completed job.

Generation reached `master_ready` in **3.476 seconds**, including a **3.390-second** managed image call. This used a previously prepared local plan; preparing that plan initially took **10.028 seconds**. These separate measurements are not an uncached end-to-end latency guarantee. The image cost estimate was **$0.034**, not an invoice. The depth sidecar is a local projection gradient, not estimated scene geometry. The underlying renderer is the existing managed Vertex image model, not the experimental native Klein worker.

Automated tests cover one recording producing one transcription and no submission until Generate is pressed, empty speech, capture cleanup, request timeouts, stale session results, lost-response reconciliation, gateway routing and streamed response cleanup. The smoke check used synthetic speech and browser text submission; it does not substitute for a human microphone rehearsal on the new localhost origin. Allow microphone access when prompted.

The later [chasing-phrase speech comparison](../benchmarks/local-asr-2026-09-07/chasing/README.md) motivates activating small.en. Across 36 fixed synthetic clips, repeated in base/small/small/base order, small retained the cat/chasing/mouse tokens in **17/18** relevant clips versus base's **11/18**. Warm medians were approximately **0.539 seconds** and **0.256 seconds**. Small was worse on some clean clips, and both models still misheard breed names. The earlier [clean-speech screen](../benchmarks/local-asr-2026-09-07/README.md) remains historical evidence. Neither screen proves accuracy on the user's microphone; the original inaccurate recording was discarded.

The [20-description language screen](../benchmarks/voice-language-2026-09-07/README.md) produced drafts for 15 descriptions when combining the bounded and learned paths. This is **draft coverage, not 75% accuracy** or proof of general English understanding. Every learned draft still requires fact review, and no image quality claim follows from that benchmark.
