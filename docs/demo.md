# Run the demo

The local simulation needs Python 3.11 or newer and [uv](https://docs.astral.sh/uv/getting-started/installation/). It uses deterministic fake providers: no model weights, microphone transcription or cloud image generation. Installing dependencies still requires network access.

From the repository root:

```sh
uv sync --dev
STORYLIGHT_MODEL_BACKEND=fake \
STORYLIGHT_ASSET_BACKEND=fake \
STORYLIGHT_LIVE_SCENE_BACKEND=fake \
STORYLIGHT_ASR_BACKEND=disabled \
STORYLIGHT_ANTICIPATORY_BACKEND=disabled \
STORYLIGHT_LIVE_SCENE_CRITIC_BACKEND=disabled \
uv run uvicorn storylight.api:app --host 127.0.0.1 --port 8080
```

Open [the workbench](http://127.0.0.1:8080/workbench), enter a passage and compile it. Open its Story Pack in the projector to exercise local playback. The [API documentation](http://127.0.0.1:8080/docs) lists the available endpoints. Keep the server running while using either page.

## Voice and real artwork

Real voice-to-scene generation additionally requires a configured local ASR backend and image provider. On Apple Silicon, the optional `mac-asr` dependency supplies MLX Whisper; its model files must be downloaded separately. Cloud providers require their own credentials and budget configuration. See [privacy boundaries](privacy.md) before enabling them.

With those services configured, open `/workbench?voice=1&session=voice-demo`. Press **Describe scene**, allow microphone access and describe a scene. A usable partial transcript can start generation while you speak. **Finish recording** sends the final recording immediately; short noun-only descriptions wait for this final input. You can also edit the text and press **Generate scene**.

The previous scene stays visible until complete artwork matches the latest description. Corrections can supersede an earlier generation; an already-started cloud request may still incur cost. Unsupported or unresolved descriptions stop before rendering and can be edited.

For a separate display, open `/projector?pack=latest&session=voice-demo&present=1&reader=0&live=1&complete_only=1` on the same API. A Jetson connected to a monitor can run that page in a fullscreen browser. Sharing the session connects the displays; it does not automatically configure HDMI, a kiosk or a network tunnel.

The simulated setup above intentionally cannot transcribe your microphone or produce model-generated artwork. It is the quickest way to inspect the interface and local request flow.
