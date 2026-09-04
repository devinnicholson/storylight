# Live SceneFacts adapter and development gate

**Decision: reject this candidate for promotion.** The complete matched development run accepted
zero graphs and reconstructed a 2.449-second median fallback path, above the 1.5-second gate.
The code remains available for research through `BOOKFORGE_LIVE_SCENE_PLANNER_BACKEND=tensorrt_graph`;
the installed appliance continues using `tensorrt_slots`. No cloud renderer comparison was run.

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
Raw hybrid scoring retains literal ID syntax: the existing raw evaluator may count ID tokens
as novel output. Its exact score is a raw-contract diagnostic, not a losslessly decoded measure
of learned semantic accuracy. No model-improvement claim is based on that cross-protocol score.
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
  --provenance /tmp/bookforge-graph-gate-30c71fa/provenance.json \
  --evidence /tmp/bookforge-graph-gate-30c71fa/development.jsonl \
  --output /tmp/bookforge-graph-gate-30c71fa/summary.json --memory-pid 1504944
```

Discover the actual resident PID for each new run. For aggregation, use identical options plus
`--aggregate-only`. `--limit` is only for an explicitly incomplete smoke, using separate evidence.
Reported process RSS and unified system used-memory peaks are sampled at 100 ms; GPU allocator
peak is unavailable and must not be inferred from RSS. Median, nearest-rank p95, and maximum
latency are all reported. Available thermal sensors are sampled at the same cadence, retaining
per-case extrema and a whole-run range. Sampling can miss shorter memory or thermal peaks.

## Decision

The frozen candidate revision is `30c71fad024a4d90168ebbfcd8e99cdbf096091b`.
All 512 matched pairs completed with no interrupted cases or request failures. Of the 512 graph
attempts, 505 refused as `invalid_hybrid`, two as `ungrounded`, and five failed the strict
envelope/completion gate. No graph passed; deterministic recovery delivered no measured gain.

| Surface | Valid schema/contract | Exact cases | Privacy failures | p50 ms | p95 ms | Max ms | Output tokens p50/p95/max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Accepted raw | 512/512 | 0/512 | 0 | 1,037.0 | 1,222.3 | 1,359.5 | 28 / 35 / 40 |
| Hybrid raw | 507/512 | 0/512 | 10 | 1,370.5 | 1,754.8 | 2,008.0 | 41 / 55 / 64 |
| Graph candidate | 0/512 | 0/512 | 0 | 1,370.7 | 1,755.0 | 2,008.0 | 41 / 55 / 64 |
| Accepted renderer | 346/512 | 0/512 | 0 | 1,050.4 | 1,230.7 | 1,368.6 | 28 / 35 / 40 |
| Final renderer with fallback | 346/512 | 0/512 | 0 | 2,449.4 | 2,868.0 | 3,171.8 | 70 / 86 / 97 |

Graph-row output tokens are the upstream hybrid model's measured tokens, not a graph wire-token
estimate. No graph wire was emitted. Zero graph privacy failures therefore provides no positive
coverage evidence. The final path produced 346 nonempty contracts and 166 empty/refused results;
it did not produce 512 successful render jobs. All 512 final surface hashes match their accepted
renderer counterparts, including empty results. Every per-category exact score is zero on both
renderer surfaces; there is no category-level gain hidden by the aggregate.

Exactness is all-or-nothing. The accepted raw response satisfied 1,399 of 2,560 required atoms,
while the accepted/final renderer text satisfied 741 of 2,560. These partial scores do not rescue
the exact gate, and the lexical renderer screen is not an image-fidelity measurement. The ten
raw hybrid privacy failures did not appear in final contracts. These measured refusals and
screens do not establish general-purpose name recognition for arbitrary English.

The resident TensorRT process peaked at 4,037,586,944 sampled RSS bytes; total system used memory
peaked at 6,853,017,600 bytes. There were 12,988 sampling attempts, with available thermal sensors
ranging from 48.406°C to 69.625°C. Dedicated GPU allocator peak is unavailable. Before/after
engine/tokenizer files, installed Python modules, active environment hash, accepted prompt,
planner configuration, power mode, and resident service identity match. Both loopback services
remain active and the API reports ready. The full read-only device check found two camera
failures: no `/dev/video*` node and no V4L2 enumeration. Physical camera acceptance remains open;
no camera, power, or service configuration was changed to repair it during this experiment.

Evidence is retained in
[`benchmarks/live-scene-facts-2026-09-04/`](../benchmarks/live-scene-facts-2026-09-04/):
`summary.json`, the 512-case `development.jsonl`, before/after hardware and provenance snapshots,
and `verification.json`. The aggregate reproduces byte-for-byte on the Mac using the frozen
implementation and this command:

```bash
.venv/bin/python scripts/benchmark_live_scene_facts.py \
  --endpoint http://127.0.0.1:11435 --model llm --memory-pid 1504944 \
  --provenance benchmarks/live-scene-facts-2026-09-04/provenance.json \
  --evidence benchmarks/live-scene-facts-2026-09-04/development.jsonl \
  --output /tmp/live-scene-facts-reproduced.json --aggregate-only
cmp /tmp/live-scene-facts-reproduced.json benchmarks/live-scene-facts-2026-09-04/summary.json
```

Verification: 1,509 Python tests and both JavaScript suites pass; scoped Ruff, diff checks,
known-credential-pattern scanning, independent adversarial review, and the previous public
target artifact reproduction pass. The only Python warning is the external Starlette/httpx
deprecation. No hidden development data or paid service was used.

The next bounded hypothesis is an **accepted-prompt-first, single-request adapter**: attempt
binding on the already captured plain four-slot output and preserve that same validated output
on refusal. This avoids depending on invalid hybrid IDs and avoids a second inference merely
to recover the baseline. It needs a separate frozen development comparison; this run supplies
no evidence that it will improve semantics. Do not begin a paid image A/B, change the accepted
environment, or claim a model improvement from this rejected candidate.
