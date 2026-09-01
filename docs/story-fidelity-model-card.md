# Bookforge Story Fidelity planner model card

Status: template; no tuned candidate has been accepted

## Intended use

The candidate converts a short, privacy-sanitized story passage into four bounded visual-planning
slots for Bookforge:

```text
SETTING:
ACTOR:
ACTION:
MAGIC:
```

It is not a general chat model, reading assessor, safety classifier, or image generator. Raw audio,
camera frames, learner profiles, and live reading telemetry are outside its input contract.

## Model lineage

Complete this table from the immutable release manifest. Do not fill it from memory.

| Field | Accepted value |
| --- | --- |
| Base model and revision | Pending |
| Story Fidelity dataset manifest | Pending |
| JAX and MaxText revisions | Pending |
| Training container digest | Pending |
| LoRA configuration | Pending |
| Merged Hugging Face checkpoint | Pending |
| TensorRT Edge-LLM revision | Pending |
| INT4-AWQ export manifest | Pending |
| Jetson engine SHA-256 | Pending |

## Training data

Bookforge Story Fidelity data is generated from repository-owned grammars and lexicons. The
dataset must contain no copied books, scraped passages, user-submitted stories, raw learner data,
named fictional properties, or prompts requesting imitation of living authors. Semantic families
and counterfactual pairs remain in one split. The hidden evaluation passages are stored privately;
public evidence contains only their immutable manifest, category totals, IDs, and passage hashes.

## Evaluation surfaces

Results are reported separately for:

1. raw four-slot model output;
2. output after deterministic Bookforge repairs and privacy separation; and
3. the final renderer-safe visual contract.

This separation prevents deterministic application logic from being reported as learned model
ability.

## Acceptance evidence

| Gate | Required | Candidate result |
| --- | ---: | ---: |
| Four-slot schema | 100% | Pending |
| Privacy | 100% | Pending |
| Existing contest suite | 20/20 | Pending |
| Hidden semantic atom recall | at least 98% | Pending |
| Hidden exact-example pass | at least 95% | Pending |
| Critical adversarial categories | 100% | Pending |
| Counterfactual sensitivity | at least 98% | Pending |
| Unsupported-concept rate | at most 1% | Pending |
| Jetson p95 | at most 2.00 s | Pending |
| Jetson maximum | at most 2.25 s | Pending |
| Unified-memory peak | at most 4.0 GB | Pending |
| Available appliance memory | at least 768 MiB | Pending |

No candidate should be described as improved or production-ready until this table is populated
from a checksum-bound acceptance report and every non-negotiable gate passes.

## Privacy and deployment

The trained checkpoint is merged and quantized offline. Runtime inference remains local on an
NVIDIA Jetson behind a loopback-only TensorRT service. Only the bounded, privacy-validated visual
contract may reach a cloud image renderer. The cloud training plane never receives live audio,
camera frames, learner identity, or reading telemetry.

## Known limitations

- Synthetic grammar coverage does not establish performance on every narrative form.
- Lexical and structured evaluators can miss visual ambiguity; locked human adversarial review is
  required for promotion.
- INT4 quantization may change behavior relative to the merged checkpoint, so the physical engine
  is evaluated independently.
- The model proposes a compact visual scene; deterministic application logic still validates,
  repairs, caches, and renders that proposal.
- English is the only accepted language for the first release.

## Rejection and rollback

A rejected candidate remains an immutable research artifact and is never silently promoted. The
accepted production engine, configuration, and cache-contract revision remain available for exact
rollback. Failed cloud or conversion experiments do not alter Jetson routing.
