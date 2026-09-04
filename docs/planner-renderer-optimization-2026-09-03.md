# Planner and renderer optimization — September 3, second pass

## Deployed fixes

The Jetson now rejects explicitly unfinished TensorRT responses even when four fields parse;
there is no ambiguous fallback retry. Written counts are no longer mistaken for capitalized
names, while `named Two` remains blocked. Privacy rewriting preserves an **open** book as a
state instead of changing it into an **opening** book, and preserves `no other people` as
`no additional people`.

Leading absence clauses are now scene constraints, not required objects to place at upper right.
A real object with a negative attribute (`two moths without wings`) remains a supporting visual.
This introduces no model call and uploads no extra source text.

TensorRT semantic-cache identity now includes exact messages, output-token bound, and
postprocessor revision. Previously, changed instructions could reuse an old plan if the engine
digest was unchanged. Stable instructions still reuse the private disk cache across restarts.
Compiler contract `subject-counts-constraints-v3` prevents old completed packs from masking the
new assembly. Old packs are retained and remain readable.

Final installed wheel SHA-256:
`be128205c2f5df79e0d4fbd7b1c224baad94de79c383440a4666b4402eec2e46`.
Model weights, 25 W power mode, resident TensorRT, and cloud routing are unchanged. Only the API
was restarted. The private Mac workbench tunnel was restored after finding no listener on 18081;
the physical `bookforge-live` projection session was not switched.

## Warm renderer: 20.2% lower median latency

One finite Modal L4 run compared the pinned FLUX.2 Klein 4B transformer before/after
`torch.compile(mode="reduce-overhead", fullgraph=True)`, following
[Diffusers compilation guidance](https://huggingface.co/docs/diffusers/optimization/fp16).
Both variants used BF16, 1024×576, four diffusion steps, guidance 1, and text length 128. All
actual prompts fit without truncation. Depth Anything V2 Small ran in FP16. Each variant was
warmed separately, then six fixed synthetic cases alternated variant order with paired seeds.
No output-image cache was used.

| Boundary, six warm samples each | Eager | Compiled |
| --- | ---: | ---: |
| Median image generation | 2.014 s | 1.596 s |
| Median image + depth + JPEG encoding | 2.070 s | 1.651 s |
| Maximum image + depth + JPEG encoding | 2.076 s | 1.658 s |
| Peak reserved GPU memory | 17.84 GiB | 17.84 GiB |

This excludes edge planning, network delivery, and presentation; it is not sub-two-second
end-to-end generation. All six pairs retained the primary diagnostic detail on visual review:
boat/moon, fox left of lantern, lighthouse left of owl, lantern in fox's mouth, exactly two boats,
and child holding an open green book. Images are not byte-identical; handles, branches, and other
details change. The book scene has unrequested curtains in **both** variants. This is six
inspected examples, not general accuracy or a full-fidelity pass.

Compilation's first warmup took **45.57 s**, versus 3.45 s eager. Download/loading took 61.56 s;
the successful remote call took 146.78 s. The extra warmup needs roughly 100 subsequent warm
renders to amortize at this saving. Compilation remains **experimental**: compiled GPU-snapshot
restore/cold-start testing is the next gate. A 90-second scale-down policy cannot be assumed to
preserve compilation.

The first attempt failed before model inference because a helper module was not mounted. Its
app was explicitly stopped; the corrected bounded run succeeded. Both apps are stopped with
zero tasks. No deployment or minimum GPU instance was added. Modal billing still returned no
rows for these apps at the end of the pass: actual cost is pending, not zero.

[Harness](../experiments/renderer-fidelity/klein_compile.py),
[timings/hashes](../benchmarks/planner-optimization-2026-09-03/klein-compile/results.json),
[compiled example](../benchmarks/planner-optimization-2026-09-03/klein-compile/compiled-3.jpg).

## Corrected planner baseline; experimental prompts rejected

The historical 20/20 TensorRT evidence used the benchmark's **repair** prompt, not the instruction
currently sent by `TensorRTSlotModelClient`. Replaying old outputs through today's parser does
not establish live-model accuracy. The new baseline calls the actual production message builder
on the resident Jetson engine, with 64 output tokens and temperature 0.

For 34 synthetic diagnostics (20 existing contest cases, six static scenes, eight additional
negation/privacy/relationship cases):

| Variant | Median request | Existing 20-case lexical screen | Schema/privacy gates |
| --- | ---: | ---: | ---: |
| Current production | 1,457 ms | 16/20 | 31/34 |
| More context in existing instruction | 1,437 ms | 15/20 | 30/34 |
| Repair-style instruction + focus example | 1,505 ms | 18/20 | 31/34 |

The latter recovered the floating school, carried lantern, and lighthouse, but still selected an
untouched background actor and lost other relationships. No experimental prompt was promoted.
Lexical presence can pass with a wrong action or relationship; 18/20 is not 90% real-world accuracy.

These counts are **before** the fixes above. One counted-turtle output passes after correcting
false name detection; the reserved personal-name output remains blocked. The initial screen
stopped on an uncaught privacy exception; the harness now records such failures and continues.
144 outputs are retained across three files. They are development diagnostics, not the sealed
training holdout; no tuned-model acceptance is claimed.

[Harness](../experiments/renderer-fidelity/slot_scene_compare.py),
[initial screen](../benchmarks/planner-optimization-2026-09-03/prompt-screen.json),
[baseline/context](../benchmarks/planner-optimization-2026-09-03/prompt-context.json),
[repair/focus](../benchmarks/planner-optimization-2026-09-03/prompt-repair-focused.json).

## Deployed flow and remaining bottleneck

Local-only preparation of the synthetic open-book passage took **1,539 ms**, with 435 input / 26
output tokens. Repetition took **0.79 ms** with zero input/output tokens. This is semantic-cache
reuse, not faster uncached inference. Neither preparation call invoked a cloud renderer.

The real browser → Jetson → Vertex → projector preview completed without browser errors.
Before the final constraint-assembly fix, job `scene_7cc6d4f9a17b435bbae6622d` took **18,157 ms**,
including an 18,068 ms managed Vertex request. The child was in front of the bridge instead of
on it, and an unrequested owl appeared at upper right. The planner had already omitted the
on-bridge relation; the image model cannot reliably recover a requirement it never received.

After the fix, the **same passage, style, and seed 3918819362** produced job
`scene_edc7f73281044865bbe6cba5` in **4,248 ms** (4,164 ms managed request). Its contract contains
`Scene constraint: no additional people` and no placement for an extra supporting object. The
owl is absent in this pair, but the child remains in front of the bridge. This verifies the
assembly correction, **not** complete relationship accuracy or a proven 4× speedup: there is only
one remote sample per version, and Vertex does not expose queue/kernel timing. Both used cached
local plans and recorded $0.034 estimated image cost each, not settled billing.

[Before](../benchmarks/planner-optimization-2026-09-03/workbench-before-constraint.jpg),
[after](../benchmarks/planner-optimization-2026-09-03/workbench-after-constraint.jpg),
[final job/contract](../benchmarks/planner-optimization-2026-09-03/workbench-after-constraint.json).

The earlier job registry entry did not survive API restart; its stored Story Pack and
checksum-verified image were retained instead. Its timing is from the pre-restart API/UI.
The full suite passes **1,112 tests**; Ruff passes on changed code. The pre-existing Starlette
deprecation warning remains. A targeted invocation also exposed existing collection-order
sensitivity when importing the API before its fake-backend fixture; the normal full suite passes.

Next: compiled snapshot restoration, grounded relation retention on fresh development cases, and
actual provider p50/p95 comparison before changing live routing. The warm experimental renderer
does not yet replace Vertex.
