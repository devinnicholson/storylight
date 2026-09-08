# Microphone rehearsal

This page describes reading a fixed passage. To speak arbitrary descriptions and generate new artwork, use the [voice-to-scene demo](voice-to-scene-demo.md).

For a short demo, prepare the artwork before presenting. Open `/workbench?demo=1`, press **Start reading**, allow the microphone, and read the displayed passage. The embedded projector follows aligned words; **Open projection view** opens the same reading on a second screen. Stop releases the microphone while the last transcript finishes. Start again to reset the reading.

The demo uses stored artwork, not live image generation. Its provenance remains in the projection details. Keep generation and cold-start benchmarking outside the audience-facing reading. Microphone audio goes only to the local API, where Whisper transcribes it and removes the temporary recording. This requires the browser and API on the same computer.

On this Mac, the prepared rehearsal is running at [the read-aloud demo](http://127.0.0.1:18766/workbench?demo=1). Its separate data directory is `.bookforge/microphone-demo`; the original story library is unchanged. Cloud generation is disabled for this server.

To restart after the initial setup:

```sh
uv pip install --python .venv/bin/python 'mlx-whisper>=0.4.3'
BOOKFORGE_MODEL_BACKEND=fake \
BOOKFORGE_ASSET_BACKEND=disabled \
BOOKFORGE_LIVE_SCENE_BACKEND=disabled \
BOOKFORGE_ASR_BACKEND=mlx_whisper \
BOOKFORGE_DATA_DIR=.bookforge/microphone-demo \
.venv/bin/uvicorn bookforge.api:app --host 127.0.0.1 --port 18766
```

The fake planner is unused during this prepared-artwork rehearsal; speech recognition is real. Local weights must exist at `.models/whisper-base.en`, and `ffmpeg` must be installed. The project's `make asr-model-pull` downloads missing weights. [MLX Whisper documentation](https://github.com/ml-explore/mlx-examples/blob/main/whisper/README.md) describes the local model path and installation.

Warm speech recognition with one short reading before presenting. On September 7, a synthetic spoken version of the fox passage transcribed exactly through the real local endpoint: 20.309 seconds on first use, then 0.183 seconds warm. WebM/Opus, the browser recording format, also transcribed exactly in 0.345 seconds. The recognized sentence produced all fourteen word events and the projector visibly reported the fox reveal trigger. This is a single-machine smoke check, not a live microphone latency guarantee. The browser sends a partial recording approximately every two seconds; that interval adds to perceived response time.

If permission is denied, allow microphone access for this localhost page and restart reading. If audio hardware is unavailable, the projector's Space/Right Arrow controls remain a manual fallback; describe that fallback honestly rather than presenting it as voice input.

Verification: ten reader/ASR Python tests passed, alongside microphone lifecycle, projector navigation, workbench readiness and next-page JavaScript checks. Browser inspection verified prepared artwork and a connected reader. Live human speech still requires the microphone permission prompt to be accepted; synthetic audio checks do not verify the physical microphone.
