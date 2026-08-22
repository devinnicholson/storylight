# Architecture

Bookforge has two deliberately separate planes.

## Private live plane

The Jetson owns the microphone, camera, reader profile, forced alignment, latency-critical visual
triggers, and intervention decisions. Raw audio and video do not leave the device.

```text
microphone -> VAD -> streaming ASR -> forced alignment -> event bus -> projection renderer
                                                        -> Gemma intervention policy (as needed)
camera     -> page/calibration tracking -------------------------------^
```

Gemma is not in the ordinary word-to-animation path. Known words trigger cached assets
deterministically. Gemma receives a small structured reading event only when the policy needs a
judgment, and must answer using a constrained schema and allowlist.

## Cloud authoring plane

The cloud sees publisher-supplied book content, not a child's live session. The Story Compiler
turns pages into a versioned Story Pack containing scene layers, generation prompts, deterministic
word triggers, literacy scaffolds, and comprehension prompts.

```text
book input -> Bookforge API -> Gemma on GKE -> validated Story Pack -> Cloud Storage
                                                              -> downloaded to edge SSD
```

The API uses the same structured model client locally and in GCP:

- `ollama`: Mac development and the first Gemma smoke tests.
- `openai`: vLLM or NVIDIA NIM-compatible serving on GKE.
- `fake`: deterministic tests with no model process.

## Trust boundary

| Local only | Cloud-authoring input |
| --- | --- |
| Raw audio and video | Book text and publisher artwork |
| Voice and face features | Reading-level target |
| Mistakes, pauses, and reader profile | Visual style and curriculum constraints |
| Live decisions | Optional anonymous aggregate measurements |

