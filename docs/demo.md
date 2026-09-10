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

## Four prepared sentences

The sentence demo is inspired by *Alice's Adventures in Wonderland*: [English text](https://www.gutenberg.org/ebooks/11) and [Henri Bué's French translation](https://www.gutenberg.org/ebooks/55456). Project Gutenberg lists both editions as public domain in the USA. The descriptions in [`examples/alice-demo.json`](../examples/alice-demo.json) are demo adaptations, not quotations.

| Read this sentence | Prepared illustration |
| --- | --- |
| “A white rabbit checks a golden pocket watch in a meadow.” | Rabbit with a watch |
| “A grinning cat sits on a tree branch in a moonlit forest.” | Cat in a moonlit tree |
| “A hare sits beside a teapot in a garden.” | Hare beside a teapot |
| “La reine tient un flamant rose dans le jardin.” | Queen holding a pink flamingo |

Generate the artwork once against a configured rendering API. This command can make four billable generation requests. Disable optional motion for this preparation; playback requires a master image and depth map.

```sh
uv run python scripts/prepare_demo_cues.py \
  --api http://127.0.0.1:8080 \
  --output .storylight/alice-cues.json
```

Keep the catalog with the API's existing asset cache and set `STORYLIGHT_DEMO_CUES_PATH` to its absolute path before restarting that API. If preparation runs through a gateway, copy the catalog to the API host; the asset cache is already on that host.

For English and French speech, use a local multilingual MLX Whisper model and set `STORYLIGHT_ASR_LANGUAGE=auto` on the transcription service. English-only `.en` models cannot support the French sentence. `fr` selects French explicitly; the default remains `en` for existing demos. This setting applies to the MLX backend.

Open `/workbench?voice=1&cues=alice&session=alice-demo`, then the projector at `/projector?session=alice-demo&present=1&reader=0&live=1&complete_only=1`. Press **Describe scene** and read one complete sentence per recording. Matching ignores punctuation and capitalization, but keywords or incomplete sentences do not trigger playback. Unmatched speech does not start generation in this mode.

This is prepared-scene playback. The reported request time excludes speech recognition and physical display latency; it is not a fresh-generation benchmark. Sentence recognition supports the listed French sentence, not general French scene understanding. Raw audio remains on the configured local transcription service.

This is a fixed-sentence demonstration, separate from a Gutenberg reading pack. The English rendering description for the French sentence is prepared in advance.
