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

The current laptop vertical slice uses rolling two-second cumulative ASR clips and a deterministic
monotonic aligner. The projector subscribes to
`/v1/reader-sessions/{session_id}/events`; each ordered envelope contains a sequence number, event
type, timestamp, reading generation, and strict payload. Reset advances the generation, so delayed
ASR from an earlier reading is rejected. Reconnects recover the authoritative page identity,
generation, and aligned position before rendering. The same projector also accepts typed cumulative transcripts,
so an ASR or microphone failure does not end a live demonstration. On Jetson, a true streaming
backend can replace the clip transcriber without changing alignment or projector contracts.

On Jetson, `WhisperTrtBackend` lazy-loads one persistent engine and serializes GPU access. Compiled
Story Packs are written atomically with private permissions under the systemd-managed state
directory; the projector retrieves the latest validated pack from the local API, so playback does
not depend on a particular Chromium profile or a network connection. The browser-local copy remains
a recovery fallback.

Prepared book packages cross into the live plane through `bookforge.pack_installer`. It validates
the Story Pack, verifies every ready media checksum, restricts playable file types, prevents
package-root escape, copies media atomically into the private cache, and rewrites asset locations
to loopback API URLs. The renderer never receives arbitrary device filesystem paths.

## Cloud authoring plane

The cloud sees publisher-supplied book content, not a child's live session. The Story Compiler
turns pages into SceneSpec v2: a validated 16:9 master prompt, spatial composition, depth ordering,
camera motion, ambience, deterministic word triggers, literacy scaffolds, and comprehension prompts.
The provider-neutral Scene Foundry turns that specification into a finished master plus depth
sidecar, caches both by checksum, and only then promotes the complete Story Pack.

```text
book input -> Bookforge API -> Gemma -> SceneSpec v2 -> AssetGenerator -> master + depth
                                                                  -> checksummed Story Pack
                                                                  -> downloaded to edge SSD
```

The API uses the same structured model client locally and in GCP:

- `ollama`: Mac development and the first Gemma smoke tests.
- `openai`: vLLM or NVIDIA NIM-compatible serving on GKE.
- `fake`: deterministic tests with no model process.

Asset generation currently supports:

- `modal`: temporary NVIDIA T4 authoring with SDXL-Turbo and Depth Anything V2 in one call.
- `mflux`: local Apple Silicon generation and Depth Pro fallback.
- `fake`: deterministic contract tests.
- GCP/Cosmos: planned providers behind the same `AssetGenerator` boundary.

Playback never calls any of these authoring providers. The projector loads only loopback cache URLs.
Its `offline=1` mode rejects non-origin fetches, and its response CSP restricts images, media,
scripts, styles, and WebSocket traffic to the local application boundary.

## Trust boundary

| Local only | Cloud-authoring input |
| --- | --- |
| Raw audio and video | Book text and publisher artwork |
| Voice and face features | Reading-level target |
| Mistakes, pauses, and reader profile | Visual style and curriculum constraints |
| Live decisions | Optional anonymous aggregate measurements |

Reader-session and sensitive audio/model endpoints reject non-loopback connections and forwarded
client headers. Their
in-memory transcripts and word state disappear with the process; the browser never sends raw audio
to the projector or cloud compiler.

The supervised edge service binds only to loopback and refuses a remote model endpoint in Jetson
mode. Hardware acceptance maps the running process's socket descriptors through `/proc` and fails
closed when visibility is incomplete, a listener is exposed, or active TCP/UDP traffic is not
loopback. That is process-level evidence; the final whole-device offline claim additionally
requires a network-disabled rehearsal or independent packet capture.
