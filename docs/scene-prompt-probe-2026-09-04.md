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
