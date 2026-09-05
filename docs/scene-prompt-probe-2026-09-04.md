# Isolated scene prompt experiment

The accepted prompt requests a magical result. The complete-scene story also contains ordinary
secondary motion, and its last page still receives an ungrounded model result. The candidate
prompt describes the same four fields using explicit visible entities and transformations,
without assuming magic or causation. It is confined to an experimental script; the production
prompt and appliance configuration are unchanged.

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
