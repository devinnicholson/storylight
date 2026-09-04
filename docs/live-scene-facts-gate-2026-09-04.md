# Live SceneFacts adapter and development gate

The candidate is opt-in through `BOOKFORGE_LIVE_SCENE_PLANNER_BACKEND=tensorrt_graph`.
The installed appliance continues using `tensorrt_slots`. No renderer comparison or model
promotion is authorized by target-adapter coverage alone.

## Adapter contract

`bookforge.live_scene_facts.adapt_live_scene_facts(slots, source_text=...)` accepts the four
parsed `SETTING`, `ACTOR`, `ACTION`, and `MAGIC` fields. It returns an immutable exclusive
`LiveSceneFactsResult`: either `facts: SceneFactsV2` or a value-free `LiveSceneFactsRefusal`.
The refusal values are `invalid_input`, `invalid_hybrid`, `unsupported_syntax`,
`ambiguous_binding`, `ungrounded`, `privacy`, and `invalid_graph`.

The source is limited to 4,000 characters and each slot to 512 characters. The raw runtime
envelope is limited to 2,048 characters and exactly four ordered nonempty lines. Hybrid
clauses require three pipe fields and pass the existing ID validator before binding.

The adapter uses a finite noun/predicate grammar. Slots select entities and actions; local
source clauses recover directly bound counts, colors, states, action objects, spatial anchors,
ownership, motion, salience, explicit negatives, two-event order, and transformations.
Unsupported syntax, repeated indefinite entities, incompatible descriptors, ambiguous references,
and hypothetical clauses refuse. This is deliberately incomplete English coverage. It neither
imports the public target adapter nor reads target labels during construction. Graph validation,
source grounding, and privacy compilation run before success is returned.

`LiveSceneGraphWirePlan` and `LiveSceneGraphPlan` extend the accepted schemas only in the new
path. Facts survive the local persistent cache and replace the master image prompt through the
existing scene-job interface. The accepted schemas and protocol prompts remain unchanged.
Completed but refused graph responses trigger one accepted-protocol request; its latency and
tokens are added to the initial hybrid request. Ambiguous transport failures are not retried.
Graph validation errors expose a fixed message rather than model/source values.

## Adversarial review

Independent review covered malformed/rebound/undefined IDs, wrong actor/object associations,
source distractors, count/color conflicts, repeated entities, negated states and salience,
direction/destination binding, temporal inversion, private names and printed payload grammar.
Shared fixes close wrong-actor motion destinations, cross-clause event objects, negated object
states and salience, and preposed `before` order inversion. Privacy fixes cover lowercase and
Unicode names after locative clauses and beside roles, reading gerunds, adjectival printed
payloads, perfect passive printing, and credential-like noun phrases.

The prior train/development target-coverage artifact must still reproduce byte-for-byte.
These are deterministic parser and validator changes, not a learned model improvement.

## Measurement procedure

The Jetson uses an installed wheel under `/opt/bookforge/.venv`; `/opt/bookforge` is an exported
release without Git metadata. Read-only inspection verified the resident accepted prompt hash,
64-token cap, loopback endpoints, engine digest, TensorRT source revision, JetPack, power mode,
clocks, temperatures, and active environment hash. `capture_jetson_graph_provenance.py` stores
only allowlisted non-secret configuration and file hashes. Device credentials are never copied.

Candidate source is staged in an isolated temporary directory and run with `PYTHONPATH=src`
using the installed Python environment. This deliberately synchronizes the benchmark code
without replacing the running appliance, changing power mode, rebuilding an engine, or restarting
services. The installed wheel manifest and candidate implementation manifest are separate.

`scripts/benchmark_live_scene_facts.py` generates only the frozen 512-record development split.
Each record receives one accepted and one hybrid request on the same resident engine, with the
existing prompts, temperature 0, top-p 1, and a 64-token cap. It separately scores accepted raw,
hybrid raw, deterministic graph, accepted renderer text, and final renderer text. Raw output,
source passages, target values, and private tokens are not written to evidence; only hashes,
value-free scores/refusals, timings, token counts, and resource measurements are retained.

The accepted raw request matches the deployed prompt. The accepted renderer comparison uses the
candidate checkout's common privacy/postprocessing code, not a claim of byte-identical deployed
wheel postprocessing. Graph exactness is typed closed-world scoring; renderer text remains a
lexical prescreen. Those surfaces are not interchangeable accuracy measures or image judgments.
Fallback renderer measurements reuse the matched accepted response and sum both measured
request durations; they are reconstructed fallback costs, not a separate production replay.

The evidence journal writes a start marker before any request. An interrupted case is never
automatically repeated. Resume checks bind the dataset, all implementation bytes, prompt hashes,
request options, and environment/engine provenance. `--aggregate-only` reproduces the report
from retained scores without inference. Semantic re-evaluation requires a new local inference
run because unsafe raw model outputs are intentionally not retained.

Example commands, run inside the isolated Jetson checkout:

```bash
PYTHONPATH=src /opt/bookforge/.venv/bin/python scripts/benchmark_live_scene_facts.py \
  --endpoint http://127.0.0.1:11435 --model llm --max-output-tokens 64 \
  --provenance /tmp/bookforge-graph-gate-20260904/provenance.json \
  --evidence /tmp/bookforge-graph-gate-20260904/development.jsonl \
  --output /tmp/bookforge-graph-gate-20260904/summary.json --memory-pid 1504944
```

Discover the actual resident PID for each new run. For aggregation, use identical options plus
`--aggregate-only`. `--limit` is only for an explicitly incomplete smoke, using separate evidence.
Reported process RSS and unified system used-memory peaks are sampled at 100 ms; GPU allocator
peak is unavailable and must not be inferred from RSS. Median, nearest-rank p95, and maximum
latency are all reported.

## Decision

The matched run and resulting promotion decision are recorded below after evidence capture.
The acceptance gate requires a semantic gain without material category regressions, zero
outbound privacy failures, and median local planning at or below 1.5 seconds. Image A/B is gated
on those results and is not part of a failed local candidate's next step.
