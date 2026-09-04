# Story Fidelity V2 runtime increment

Status: implemented as an opt-in research candidate; not promoted to the kiosk default.

## Outcome

This increment adds a strict, private-edge fact graph and fixes a false-positive class in the
existing fidelity evaluator. It also adds a backwards-compatible TensorRT hybrid protocol for
measured comparison. The accepted four-slot planner remains the default and rollback path.

The implementation reuses the existing Story Fidelity V2 corpus: 4,096 train, 512 development,
and 512 HMAC-derived hidden records. No parallel small corpus was created, and the hidden split
was not opened during development.

## SceneFacts V2

`bookforge.scene_facts` defines bounded nodes for settings, subjects, objects, relationships,
motion, salience, ordered events, negatives, and transformations. It includes:

- immutable Pydantic validation with stable entity references;
- order-preserving 64/96/128-budget wire serialization;
- local source grounding for entity-bound count, color, state, action, direction, destination,
  event order, spatial relation, and transformation;
- fail-closed rejection for negated positive facts, mismatched entity associations, contact data,
  printed source payloads, and conservative proper-name candidates;
- a renderer prompt compiler that resolves internal references back to visible labels and never
  includes the source passage.

The fidelity evaluator recognizes SceneFacts V2 directly or nested under `semantic_facts` or
`scene_facts`. Relation, role, attribute, and count checks now require one matching node or edge.
For example, `dog on mat` plus `cat under table` can no longer satisfy `dog under table` merely
because all words occur somewhere in the output. Legacy four-slot evaluation remains supported;
its shared semantic normalization now also handles common irregular and doubled-consonant
inflections consistently with the graph evaluator.

For public-corpus graph evidence, exactness is closed-world: entity references may differ, but the
normalized nodes, edges, events, negatives, and transformation must match the deterministic typed
contract without source-mentioned distractors. The schema also rejects opposing directional
edges, incompatible object states, and a positive fact negated by the same graph.

The deterministic public target adapter covers 4,605 of 4,608 public records under the strict
64-token estimate, and every eligible graph passes the graph-aware evaluator. Train is 4,096 of
4,096; development is 509 of 512. The three development refusals are intentionally fail-closed
same-label containment cases whose source contracts do not identify which repeated basket owns
the relation. Median wire size is 35 estimated tokens and the maximum is 54. Counts by category
and split are recorded in
[`benchmarks/story-fidelity-v2-graph-coverage-2026-09-04.json`](../benchmarks/story-fidelity-v2-graph-coverage-2026-09-04.json),
and the report can be reproduced with `scripts/benchmark_fidelity_graph_targets.py`.

This adapter is public-corpus evaluation and training-target tooling, not a general story parser
and not the kiosk's live path. The benchmark cannot select the hidden split, and the adapter exits
on a hidden split before it inspects or derives semantic fields. The
privacy filter handles marked names, Unicode names, and unknown lowercase clause-head actors
conservatively; ambiguous unmarked names outside that grammar remain a documented reason to add
a measured local NER stage before making broader privacy claims.

## Jetson protocol experiment

All probes used the resident loopback-only Gemma/TensorRT endpoint. They used public development
examples only and made no Modal or Google Cloud calls.

| Protocol | Cases | Schema adherence | Median planning | Result |
| --- | ---: | ---: | ---: | --- |
| Full six-line fact graph | 12 | 5/12 | about 2.24 s | Rejected: truncation and privacy failure |
| Compact five-line fact graph | 12 | 0/12 | about 1.64 s | Rejected: omitted fields and invalid IDs |
| Hybrid four-line envelope | 12 | 12/12 | 1.375 s | Retained as opt-in candidate |

The hybrid experiment produced 41 completion tokens at p50 and no more than 60, so the accepted
64-token output cap remains sufficient. Nine examples were visibly correct, two were mixed, and
one lost a magical result. These are engineering observations, not an independent blind score.

A separate five-case repository benchmark compared the integrated protocols on the same engine:

| Protocol | Automatic semantic cases | Median | Maximum | Mean output tokens |
| --- | ---: | ---: | ---: | ---: |
| Accepted slots | 2/5 | 1.035 s | 1.228 s | 27.0 |
| Hybrid slots | 3/5 | 1.399 s | 1.703 s | 41.8 |

The hybrid candidate meets the provisional 1.5-second median planner target and recovered a
carried lantern that the accepted prompt dropped. It costs roughly 0.36 seconds at the median and
still misses secondary details, so it is not promoted. The benchmark's lexical screen is a
prescreen, not proof of image fidelity. The sanitized machine-readable engineering note is
[`benchmarks/jetson-gemma4-tensorrt-hybrid-2026-09-04.json`](../benchmarks/jetson-gemma4-tensorrt-hybrid-2026-09-04.json).
It is not promoted as reproducible benchmark evidence because the exploratory run did not retain
exact JetPack/TensorRT, power-mode, clock, thermal, or sanitized per-case provenance.

## Activation and rollback

Use `BOOKFORGE_LIVE_SCENE_PLANNER_BACKEND=tensorrt_hybrid` only for controlled comparisons.
`tensorrt_slots` remains the accepted Jetson protocol, and `configured` retains the general model
client. Both TensorRT protocols require the standard outer wire contract and a loopback endpoint.

The next promotion gate is deterministic construction of a complete SceneFacts V2 graph from the
reliable four-line envelope, followed by the existing 512-record development evaluation. The
graph must then beat the accepted slots protocol on semantic accuracy without breaking the
planner latency budget. A paid renderer comparison should not run until that local gate passes.
Training remains a later option if prompt and postprocessing changes cannot reach the accuracy
target.
