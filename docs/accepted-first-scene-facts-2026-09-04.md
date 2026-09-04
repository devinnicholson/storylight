# Accepted-first SceneFacts development gate

The `tensorrt_accepted_graph` research backend makes one request with the unchanged accepted
four-slot prompt and 64-token cap. It attempts finite graph binding on that response and the
local source. A refused graph retains the same accepted wire fields without another model call.
The installed appliance remains on `tensorrt_slots`.

## Frozen hypothesis and gates

This experiment follows the rejected [hybrid-first comparison](live-scene-facts-gate-2026-09-04.md).
Removing dependence on generated hybrid IDs should allow useful deterministic recovery without
the cost of a second inference. This is a postprocessing hypothesis, not a model improvement.
Before measurement, eight synthetic plain-slot cases improved from six accepted graphs to eight
after bounded handling of explicit `as`/`while` clauses and directly stated result motion.
Simultaneity does not establish causality. The live adapter never imports corpus target derivation.

Promotion requires semantic improvement over the matched accepted renderer, zero final privacy
failures, no material regression in previously passing categories, and median local planning at
or below 1.5 seconds. Report p95, maximum, refusals, token counts, sampled memory and thermals.
Paid image comparison remains conditional on this local gate. Neither the hidden split nor cloud
inference is used for this experiment.

## Runtime behavior

Set `BOOKFORGE_LIVE_SCENE_PLANNER_BACKEND=tensorrt_accepted_graph` only for an explicit research
deployment. The accepted backend, prompt, wire schema, and cache identity remain unchanged;
this candidate has its own cache identity. Graphs survive persistent cache reads. The graph is
grounded and privacy checked through default-style page compilation before attachment. An unsafe
or invalid accepted response still refuses. The candidate cannot rescue an invalid accepted plan.

Safe visual styles longer than the graph compiler's 120-character bound reuse the accepted page
after graph, source, and style privacy validation. Unsafe styles and invalid graph facts still
reject. Incomplete or malformed model responses fail with value-free errors and no inference retry.

## Measurement and interpretation

The `--mode accepted_first` benchmark obtains exactly one accepted response per public development
record. It separately scores accepted raw slots, accepted renderer text, typed graph, and final
renderer text. Fallback hashes and token counts must match the accepted output. Per-case required
atom booleans expose losses and gains even when total recall is unchanged. Grading runs outside
all construction timers; graph construction includes accepted normalization and the graph helper.
Latency is HTTP inference plus measured local construction, excluding queueing, rendering, and
network transport to an image provider.

Code, dataset, prompts, request options, engine, installed wheel, environment, and hardware
snapshots are bound to the journal. Evidence retains hashes, scores, timings, and resource samples;
it does not retain source passages, raw model outputs, or target strings. Interrupted requests
are not retried. Aggregate reproduction does not rerun inference or regrade unavailable raw text.

The scorer is unchanged. `scripts/diagnose_live_scene_facts_scoring.py` runs positive controls
using only public development targets. Exact raw target envelopes can pass the raw evaluator.
Typed exactness also compares representation: removed articles or a modifier stored as an
attribute instead of inside an entity label can fail. The lexical renderer screen can penalize
facts expressed in separate clauses. These controls describe scoring limits; they neither certify
candidate semantics nor replace the frozen gate with a more favorable score. No image fidelity
claim follows from a text score.

GPU allocator peak is unavailable. The 100 ms sampler reports resident process RSS, system used
memory, and available thermal-sensor extrema; short peaks can be missed. Candidate source runs
from an isolated Jetson directory using the installed Python environment, without replacing the
wheel or changing service, engine, power mode, or environment settings.
