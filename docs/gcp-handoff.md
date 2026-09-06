# Bookforge GCP handoff

This records an earlier offline-package milestone. For current cloud providers, deployed
configuration and privacy boundaries, use the [September 5 system audit](system-audit-2026-09-05.md).
The live renderer receives a sanitized visual contract, not the raw source passage.

The Modal milestone produces a cloud-neutral artifact, not a Modal-dependent runtime. The accepted
deliverable is the self-contained `silver-fox-lost-words` handoff directory. It contains one
validated Story Pack, six still masters, six motion loops, and a checksum manifest. A clean machine
can install all 12 assets without Modal, GCP, or network access.

## Proven boundary

Modal was used only for finite offline SANA image generation, SigLIP scoring, and LTX-Video motion
generation. There is no deployed Modal endpoint. Playback does not call a generative model: the
Jetson loads checksum-addressed assets, performs private speech recognition and word alignment
locally, and emits tiny reader events to the local projector.

The accepted package is created with:

```bash
python -m bookforge.handoff_bundle \
  artifacts/visual-lab/silver-fox-lost-words.story-pack.json \
  --asset-root . \
  --output-dir artifacts/visual-lab/handoff/silver-fox-lost-words
```

Install it from inside the resulting directory:

```bash
python -m bookforge.pack_installer silver-fox-lost-words.story-pack.json \
  --asset-root .
```

The installer fails closed if an asset is missing, traverses outside the bundle, uses an unsupported
format, or differs from its recorded SHA-256 checksum.

## GCP mapping

Keep the existing contracts and replace only the provider adapter:

| Bookforge responsibility | Current milestone | GCP destination |
| --- | --- | --- |
| Bounded generation job | finite Modal run | finite GPU-backed batch job |
| Model and prompt provenance | pinned revision in manifests | same pinned revision and manifest |
| Candidate artifacts | local ignored artifact directory | private object storage with checksum keys |
| Quality report | local JSON plus human review | immutable job output plus approval record |
| Final Story Pack | portable handoff directory | versioned downloadable release object |
| Live reading, microphone, alignment | local API / Jetson | remains local; never uploaded |

The cloud should receive story text, art direction, deterministic seeds, and selected model
configuration. It should return candidate assets and immutable manifests. Raw microphone audio,
camera frames, live transcripts, and reading telemetry stay on the Jetson. This is the privacy claim
the contest demo can show clearly: powerful cloud creation before story time, private edge inference
during story time.

## Promotion gate

Do not mark a GCP-generated pack ready until all of these are true:

1. Every model name, immutable revision, prompt, seed, runtime, duration, and cost is recorded.
2. Every candidate passes the existing automated projection and motion measurements.
3. A human explicitly approves identity, child safety, typography, and midpoint motion frames.
4. The final bundle reinstalls into an empty cache with every checksum verified.
5. The projector plays every page and every read-aloud trigger through the production event path.
6. A billing guard rejects new jobs before the configured credit reserve can be crossed.

The exact accepted evidence for the pre-GCP milestone is in
`benchmarks/lost-words-modal-acceptance-2026-08-23.json`. The next independent gate is physical
Jetson microphone and projector acceptance; typed browser simulation is not represented as hardware
ASR evidence.
