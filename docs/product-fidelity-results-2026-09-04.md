# Product fidelity increment

Decision: **stop before the development gate**. The bounded parser repair is implemented,
but the seven-request resident-model smoke produced no accepted graphs. The full 512-case
benchmark, paid image comparison and projector acceptance rehearsal did not run.

The [six-page story](../examples/lantern-bridge-fidelity-story.json) is frozen with a visible-fact
checklist and separate temporal display requirements. Its SHA-256 is
`a1fb181fa5e217a9b259b622c7340b64fc3788311ea50bdf3e973eb5265469ed`.
It is an authored engineering demonstration, not a holdout. Pages will not be rewritten to
hide failures.

## Local construction gate

The opt-in adapter now understands a complete clause of the form “one blue lantern is carried
by two orange foxes.” The noun after “by” becomes the actor and the carried item becomes its object.
Counts and colors survive graph construction, integration and exact renderer-prompt compilation.

The shared graph validator proves the complete original passive sentence before converting it
to the active order used by action and relationship checks. Only an exact setting prefix may
precede the clause. A detached fragment after “an owl says,” cannot supply the proof. Temporal
event grounding is not extended by this conversion. Original source privacy checks still run.
Only graph cache identities advance to revision 3; accepted four-slot caching is unchanged.

The frozen synthetic positive changes from refusal to a compiler-verified graph. Eight retained
adversarial cases cover negation, reporting, conditional scope, swapped agents/objects, borrowed
objects and ambiguous agents. Independent probes additionally cover past/plural forms, printed
payloads, names, relative clauses and attached spatial tails.

On the same first 32 public training controls, both versions produce two integrated graphs and
one exact graph. Two passive cases move past grounding to their next unsupported grammar clause;
the full-contract totals do not improve. Paired reports are in
[`benchmarks/product-fidelity-2026-09-04/`](../benchmarks/product-fidelity-2026-09-04/).
This repair establishes a bounded capability, not improved live-model accuracy.

## Measurement gates

The accepted-first benchmark now records an explicit advance/reject decision and freezes its
criteria in the journal. It requires a complete 512-case matched run, nonzero accepted graphs,
strict exactness and required-fact gains, no category-level loss, zero final privacy failures,
median planning at most 1.5 seconds and p95 at most 1.10 times the matched accepted path.

Graph renderer scores now require the actual prompt, graph and visual style to pass the existing
compiler-proof evaluator. Accepted renderer text remains a lexical screen; the two representations
are labeled in the journal. Output equality still hashes the actual prompt, so metadata alone
cannot change the output hash. Text scores do not establish image fidelity. New journals bind
the revised scoring surface and implementation; historical reports retain their original code
and scores.

The seven-request smoke consists of the six frozen pages and one fixed passive control. It
compares accepted and candidate construction from the same resident model response. Compiler
proof is distinct from the story's visual and temporal acceptance. Only a complete smoke with
a useful graph permits the full development run; a full semantic gate is required before the
bounded image comparison.

## Resident Jetson results

Implementation revision: `e64c634d229884e164a0ef1c42cfc762dd907ff7`.
The isolated candidate used the installed Python environment and existing loopback TensorRT
service. All seven requests completed once, with valid raw envelopes and no transport failures.

| Inputs | Requests | Valid accepted contracts | Compiler-proved graphs |
| --- | ---: | ---: | ---: |
| Frozen story pages | 6 | 3 | 0 |
| Fixed passive control | 1 | 1 | 0 |
| Total | 7 | 4 | 0 |

Story pages 3, 5 and 6 and the passive control produce valid accepted fallback contracts.
Those four candidate prompts have exactly the same hashes as their accepted counterparts.
Pages 1, 2 and 4 fail construction on both paths. Four paths explicitly pass privacy checks;
the other three have unassessed privacy results, not seven successful privacy checks.

Median HTTP inference is 887.6 ms; p95 and maximum are 1,130.8 ms across these seven requests.
These are inference timings, not projector latency or a passed 512-case performance gate.
Sampled process RSS peaks at 3,999,436,800 bytes, system used memory at 7,013,941,248 bytes,
and temperature samples span 45.031–51.625°C. GPU allocator peak remains unavailable.

Before/after hashes match for the engine, installed modules, accepted prompt and environment.
Server revision, PID/start time, planner configuration and power mode also match. The accepted
backend remains `tensorrt_slots` and the API reports ready. The candidate was not installed or
promoted. No cloud inference or image generation ran.

The journal contains only hashes, flags and measurements. Detailed refusal causes and raw
model outputs were not retained, so these results cannot attribute failures to a particular
phrase or distinguish every unsupported construction from a wrong model binding. Validated
renderer contracts remain in a private file on the device and are excluded from repository evidence.

## Reproduction and verification

The smoke summary reproduces byte-for-byte on both Jetson and the Mac without inference:

```bash
.venv/bin/python scripts/benchmark_story_fidelity_smoke.py \
  --manifest examples/lantern-bridge-fidelity-story.json --model llm \
  --provenance benchmarks/product-fidelity-2026-09-04/before/provenance.json \
  --memory-pid 1504944 \
  --evidence benchmarks/product-fidelity-2026-09-04/story-smoke.jsonl \
  --output /tmp/story-smoke-reproduced.json --aggregate-only
cmp /tmp/story-smoke-reproduced.json \
  benchmarks/product-fidelity-2026-09-04/story-smoke-summary.json
```

Use the frozen implementation revision for reproduction. A future inference experiment requires
fresh provenance and a new journal. The measured engine directory ends in
`tensorrt-edgellm-v0.10.0/models/gemma4-e2b-it-int4-awq-v010/engines/llm`; the server Git repository
is its installation's `src` subdirectory. Capture installed provenance without candidate
`PYTHONPATH`, then use `PYTHONPATH=src` only for the isolated candidate commands.

All 760 tests pass, including both JavaScript suites, with the existing Starlette/httpx warning.
Scoped Ruff and whitespace checks pass. Independent reviews checked parser attacks, same-response
fallback measurement, evidence privacy, aggregate reproduction and the gate decision. The test
count increased by 19 for the new behavior and measurement boundaries, without restoring the
removed exhaustive matrices.

## Remaining limits

This grammar accepts explicit “carried by” clauses, not general passive voice, pronoun resolution,
reporting, arbitrary event chains or all secondary story facts. The training examples' separate
result wording remains unsupported. Pre-existing active reporting/conditional grounding gaps
were also observed during review; this change does not claim to repair them.

The transformation and ordered-action pages require visible before/after states. Neither a
single image nor depth parallax can satisfy those criteria. Physical projection, blind visual
review and offline rehearsal remain separate gates in the [plan](product-fidelity-plan-2026-09-04.md).

The next bounded investigation should distinguish accepted-plan construction failures from
adapter refusals using value-free stage reasons and synthetic/public training fixtures before
another model run. Preserve this story and failed result rather than weakening its checklist.
