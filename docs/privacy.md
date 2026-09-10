# Privacy boundaries

Storylight's voice demo captures microphone audio in the browser and transcribes it with a local Whisper service on the Mac. The browser uses `MediaRecorder`, not a browser vendor's speech-recognition API. In the split Mac/Jetson setup, the gateway sends audio to the Mac's loopback ASR endpoint and scene descriptions through an SSH connection to the Jetson.

| Data | Configured voice-demo path |
| --- | --- |
| Microphone recording | Browser → local Mac transcription service |
| Transcript and scene checks | Local API or Jetson over SSH |
| Validated visual direction | Configured cloud image provider |
| Generated artwork and depth assets | Returned for local display and storage |

The cloud image request contains a validated visual prompt, not the raw recording. Local source-grounding and privacy checks reject unsupported content and exclude recognized private names or contact details before rendering. The interface reports omissions. These checks have a bounded vocabulary and are not a guarantee that arbitrary sensitive text will always be recognized.

The Mac ASR implementation writes a temporary local audio file while transcribing and removes its temporary directory afterward. Story Packs and generated assets can persist in the configured local data directory. The recording-timing trace is memory-only and excludes transcript and audio content; that does not mean the entire application stores no text.

These boundaries describe the configured voice path. Other model and provider settings can send text to remote endpoints. Review configuration before using private material; enabling a cloud model is not equivalent to local inference. Model downloads and cloud image generation require network access. The [simulated demo](demo.md) disables speech recognition and uses fake providers.

The gateway is intended for loopback use and rejects foreign origins and unexposed routes. It is not a hosted multi-user authentication service. The prototype has not established a general child-data compliance claim or undergone a comprehensive security audit.
