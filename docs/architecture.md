# Architecture

Storylight turns a description into artwork for a local browser or projector. It also supports reading a known passage against a prepared Story Pack. These are separate flows: voice-to-scene creates new artwork, while known-text reading advances existing visuals.

## Voice to scene

```text
Browser microphone → local Whisper → transcript
                                       ↓
                            local scene and privacy checks
                                       ↓
                              cloud image generation
                                       ↓
                         completed artwork → local projector
```

The FastAPI service serves the workbench, projector, scene jobs and reader sessions. In the demonstrated split setup, a Mac handles microphone capture and local transcription; a loopback gateway routes scene requests through SSH to the Jetson. A single-machine installation can serve the same interface without that gateway.

The voice compiler combines bounded description parsing with an optional local spaCy language service. Typed scene facts retain supported subjects, actions, relationships and negative constraints. Source-grounding and privacy checks run before a visual prompt reaches the configured renderer. Other authoring and intervention paths can use a configured language model; the voice path is not simply an unrestricted language-model prompt.

The frontend can prepare a scene from a partial transcript, then supersede it when the description changes. One image job runs at a time, and queued intermediate descriptions are replaced by the latest one. Job identity and validated scene digests prevent a stale result from replacing the current scene. Voice playback withholds incomplete artwork and keeps the previous scene visible.

## Reading and projection

Story Packs contain validated scene descriptions, trigger information and asset references. The reader aligns cumulative transcript text with a known passage and publishes local session events. The projector follows those events and displays cached artwork with browser animation or depth-based parallax when suitable assets exist. Playback does not require a fresh model call for each word.

Depth assets depend on the provider: a depth estimate and an authored projection gradient are not interchangeable geometric measurements. Automatic camera page tracking and demonstrated improvements in reading outcomes are not established features.

## Deployment and research

Image providers include managed Google Cloud and GPU-worker integrations. Cloud credentials remain server-side. The exact data boundary depends on configuration; see [privacy](privacy.md).

The Gemma adapter and compiled-cache experiments are isolated research work. Their improved extraction timings are not deployed microphone-to-image performance. The learned adapter failed its refusal gate and has not replaced the working voice compiler. See [research results](research-results.md) for measurements and limits.
