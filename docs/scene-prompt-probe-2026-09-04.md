# Isolated scene prompt experiment

The accepted prompt requests a magical result. The initial complete-scene story also contained
ordinary secondary motion, and its last page received an ungrounded model result. The candidate
prompt describes the same four fields using explicit visible entities and transformations,
without assuming magic or causation. The final source-selected version is available in opt-in
scene scope; the accepted appliance configuration is unchanged. The
[latest run](story-display-results-2026-09-04.md) passed construction and awaits image generation.

`scripts/benchmark_scene_prompt_probe.py` freezes eight independent synthetic controls and one
candidate prompt. It makes exactly 16 matched control requests, alternating which prompt runs
first. Both use the same resident engine, decoding settings and complete-scene constructor.
Positive controls require exact typed nodes, bindings, setting and all graph semantics. Negative,
missing-result and printed-payload controls require refusal. Advancement requires all eight
candidate outcomes, complete matched timings, median planning at most 1.5 seconds and p95 within
10% of the accepted prompt.

Only an explicit separate invocation can run the original seven frozen story inputs after controls
advance. That stage must produce seven proved graphs. Raw responses remain in an exclusive
owner-only archive on the Jetson. Journals retain hashes, boolean checks, refusals and measurements.
Interrupted requests are not retried, and repeated invocation cannot issue another completed call.

Preflight also exposed a separate motion-grounding defect: “balloons rise” failed where “balloon
rises” passed. The narrow repair accepts the base and inflected forms of the two existing motion
directions. Wrong actors and opposite directions still refuse. Global normalization and evaluator
criteria are unchanged, and the public target coverage artifact remains identical. This repair is
separate from any measured effect of the prompt.

The fixture and prompt are frozen before live inference. Supplied-slot and mocked-network checks
validate the harness; they are not model accuracy measurements.

## Frozen live result

Revision `1d298d12f02a7f17822a12575cbcfbe68351f929` completed all 16 requests with no
request failures, truncation or missing timings. The candidate passed 3/8 control outcomes;
the accepted prompt passed 6/8. The decision is **reject**. Candidate median/p95 planning was
1,015.6/1,193.9 ms, versus 1,005.1/1,289.8 ms accepted. No story-stage or image calls followed.

Four candidate positive controls proved every checked node, binding, action, motion and
transformation, but failed exact setting representation. Their setting slots began with an
article. The remaining positive control refused as ungrounded; its value-free slot shape alone
does not establish the cause. All three candidate refusal controls passed. The original result
and oracle remain frozen while independent synthetic examples test constructor normalization.

Those independent examples exposed two constructor defects. Setting labels retained a leading
article, so equivalent locations produced different graphs. Also, an explicit model count of
one could not match a source noun introduced with “a” or “an.” The repair removes one leading
setting article and retains noun-local singular evidence across bound references. It preserves
the source graph's existing count representation; definite mentions, unsupported plurals,
different counts and borrowed identities still refuse. The prompt, frozen controls and exact
oracle are unchanged. A fresh matched run will measure this constructor revision separately.

At revision `b3de71b5fc1eb36707ded941870207b6cf986b83`, the repeated 16-request control run
passed all eight candidate outcomes, versus six accepted. All 16 raw-response hashes matched the
first run: the gain came from construction, not changed model output. Candidate median/p95 was
1,079.9/1,199.0 ms, versus 1,004.7/1,295.4 ms accepted. The control gate advanced.

The separately gated seven-input story stage proved all six story pages, including both ordered
actions on the last page. Its extra passive-voice control refused, leaving the overall decision
**reject** at 6/7. Median/max planning was 1,236.5/1,422.5 ms. A private on-device grammatical check
identified a passive predicate fragment in the refused ACTION field; no raw response left the
Jetson. The next repair must bind that fragment to a unique explicit source passive clause.

The bounded repair now recognizes “carried by” followed by an agent only when the source contains
exactly one matching complete carried-by clause. Both the selected actor and fragment agent must
match that source actor's identity, count and color before the patient is recovered. Active source
clauses, multiple possible patients and unasserted clauses refuse. Other passive verbs are outside
this repair. Fresh synthetic controls verify these boundaries; the original story and prompt stay
frozen for the next measurement.

Revision `33f336e2c8a5487cb5a8100a752df281768bc08b` retained 8/8 candidate controls,
with 1,068.8 ms median and 1,197.1 ms p95 planning. The story stage remained **reject** at
6/7: all six pages passed, while the extra control's model ACTION named a by-agent whose
entity and color contradicted the source. The new fragment grammar correctly refused it.
Story median/max was 1,256.7/1,427.0 ms. The
[reports](../benchmarks/scene-passive-fragment-2026-09-04/story-summary.json) reproduce under
the pinned revision; before/after runtime identity is unchanged. No images were generated.

Both [control](../benchmarks/scene-prompt-grounding-repair-2026-09-04/controls-summary.json) and
[story](../benchmarks/scene-prompt-grounding-repair-2026-09-04/story-summary.json) aggregates
reproduce byte-for-byte under the pinned revision. Before/after metadata confirms the resident
process, engine, deployed files, prompt, planner settings and power mode stayed unchanged.

The [sanitized evidence](../benchmarks/scene-prompt-probe-2026-09-04/controls-summary.json)
includes all per-request checks and before/after provenance. The resident process, engine,
deployed files, accepted prompt, planner configuration and power mode stayed unchanged.

## Display assembly

`scripts/assemble_fidelity_display.py` reconstructs seven proved model captures locally before
building the frozen story's eight display pages. It checks source/response hashes, request and
implementation identity, graph proof and exact renderer prompts. The source-bearing pack stays
in an owner-only file outside the checkout; only validated visual requests may be exported.

The original accepted baseline produced valid contracts only for story pages 3, 5 and 6.
The finite comparison therefore contains eight candidate requests and three accepted requests,
11 images total. Missing accepted pages remain failures. The single accepted still for page 5
or 6 is reused for comparison with the corresponding two candidate states; it cannot demonstrate
event order. Assembly remains gated on all seven candidate story constructions passing.

`scripts/render_fidelity_display.py` preflights the entire proved batch, exact frozen seeds,
renderer receipt and locally cached tokenizer before any paid call. Its explicit execution mode
uses the existing overnight plan and ledger with a $2.75 reservation ceiling: 11 calls at $0.25
each, including conservative startup/idle allowances. It has no prewarm or paid retry. A durable
batch-hash marker beside the ledger prevents re-execution even with a new output directory.
Ambiguous failures retain that marker and reservation. Negative prompts are recorded but this
Klein route does not execute them. Returned model/runtime identity, seeds, dimensions, tokens,
master/depth checksums and timings must match; visual acceptance still requires human review.

`scripts/install_fidelity_display.py` verifies the complete render journal and all 11 bundles,
then binds only the eight candidate pages to 16 master/depth assets. It requires an independently
supplied checksum of the entire private pack and a fresh data directory outside Git checkouts.
Existing asset-cache and story-store APIs verify checksums again during local replay. No running
appliance is changed by this offline command.

`scripts/build_fidelity_review.py` prepares a static local gallery with neutral randomized A/B
labels, frozen fact checklists and separate correctness, legibility and continuity ratings.
Ordered still states remain in sequence; the original baseline still repeats for temporal pages.
The mapping key stays outside the gallery in an owner-only file. This hides model labels, but
different temporal behavior and missing baseline options remain distinguishable. Exported unrated
fields remain unrated; the tool cannot grant visual acceptance.

## One-request source routing

The next opt-in candidate selects the accepted prompt for exactly one supported, explicit
carried-by source clause and otherwise uses the unchanged scene prompt. Selection happens before
generation, from source syntax alone. It never retries a failed response or repairs a conflicting
model agent. Both full prompts and the routing revision enter the scene cache identity; the focal
planner keeps its existing request and cache behavior.

The selector checks the original clause against shared assertion filtering before parsing it,
so removing quoted text cannot create an eligible passive. Negated, reported, hypothetical,
coordinated, incomplete and pronoun-based clauses do not qualify. Multiple complete passives use
the scene prompt; downstream grounding still decides whether a particular response is valid.

The `routed-v1` probe profile keeps the original eight controls and adds eight independent
synthetic routing controls. Its 32 paired requests record route, source hash and exact request
hash against a frozen schedule. Journal validation also enforces that schedule's execution order.
All 16 candidate outcomes and the existing latency gates must
pass before a separate invocation can revisit the seven original story inputs. The fresh controls
are engineering tests, not a new holdout.

Adversarial review tightened the passive-fragment repair: all source passives referring to the
resolved actor are counted before count/color compatibility. A later bare reference cannot hide a
second possible patient. This preserves refusal for ambiguous selection even when the complete
scene could otherwise retain both carried objects.

At revision `ba15b484c3a91ff65ce6f7949fc851669c137298`, all 32 control requests completed.
The candidate passed 16/16 outcomes (ten positive graphs and six required refusals), versus
12/16 accepted. Candidate median/p95 was 1,064.1/1,196.3 ms, versus 1,018.6/1,282.8 ms accepted.
The separately gated story run proved all seven inputs at 1,205.2 ms median and 1,420.9 ms maximum.
There were no request failures, incomplete generations or missing timings in either stage.
The frozen story and criteria are unchanged. These results advance to image comparison; they
do not establish visual fidelity or qualify the routed prompt on the full 512-case split.
