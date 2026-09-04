# Performance and fidelity pass — September 3

## The user-visible bottleneck was a seed bug

A real workbench request sent seed `2810313968` to Vertex. Vertex rejected it with HTTP 400:
`Invalid value at 'generation_config.seed' (TYPE_INT32)`. The safe fallback then cold-started
Modal. The draft appeared in 8 ms, but the completed master/depth took **60,423 ms**.

Bookforge uses uint32 seeds; Vertex's [GenerationConfig uses int32](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/reference/rpc/google.cloud.aiplatform.v1).
The adapter now maps to a nonnegative 31-bit provider seed, preserving the original seed in
cache identity and recording both values in the manifest. No retry or extra model request is
introduced. Boundary tests cover zero, both sides of 2^31, the observed failing seed, and uint32 max.

After deploying the fix to the Jetson, a new browser request with seed `2742810865` sent provider
seed `595327217`. Vertex succeeded directly: **5,267 ms** total, **5,169 ms** managed generation,
17 ms packaging, 13 ms cache promotion. The local semantic plan was cached; this is not an
uncached-planner benchmark. The browser displayed the actual master/depth without errors or
warnings. The original projection session was not switched.

The two browser requests use the same synthetic boat passage but different art styles. This is
evidence that the broken routing boundary is fixed, not a controlled 11× model-speed claim.
The default workbench route is **Vertex → safe Modal fallback**; the separate anticipatory GKE
path uses Cloud Run SANA/depth. Optimizing that Cloud Run worker does not automatically optimize
every regular workbench request.

Visual inspection still found a fidelity failure in the fixed Vertex result: an unrequested
human performer and an inset scene. Its cached local plan also omitted the requested moon.
Successful delivery must not be called a successful semantic match.

### Duplicate-detail fix, tested end to end

A subsequent real UI request for exactly two red paper boats completed in 3,526 ms but produced
four boats: two in the foreground and two in an inset. Inspection showed the same boat description
in both foreground and supporting-detail slots. Even with a “no inset” instruction, the prompt
explicitly requested a smaller second copy at upper right.

The assembly now omits a supporting clause and its placement when its ordered content tokens are
already present in the foreground or background. It does not erase a different count, color,
action, or relationship. The unconditional “exactly one main actor” instruction is removed, and
story-specific positions/contact take precedence over default composition. No extra model call,
source-text upload, or privacy relaxation is involved.

With **identical passage, style, and seed 168560962**, a fresh image completed in **3,494 ms**:
exactly two boats, no inset, no added person. This is one visual regression pair, not a universal
fidelity claim. [Before](../benchmarks/performance-fidelity-2026-09-03/workbench-two-boats.jpg)
and [after](../benchmarks/performance-fidelity-2026-09-03/workbench-two-boats-fixed.jpg) are retained.
The browser reported 60 ms client activation and a 320 ms blend; those are not GPU generation time.

An optional Story Pack compiler-contract revision prevents older completed scenes from shadowing
the fixed assembly. Existing packs remain readable and are not deleted. Repeating the corrected
request reused the exact master/depth hashes in **54.269 ms**, with zero provider inference and zero
estimated additional image cost. That was the provider's artifact cache, not the completed-pack
shortcut (`scene_cache_hit` remained false).

A separate 22-call, local-only TensorRT comparison tried retaining non-magical secondary details.
Both prompts still omitted the moon and lighthouse; the candidate also lost some contact/result
details, though it recovered jellyfish in another case. It was **not promoted**. The original
planner instruction and privacy boundary remain unchanged. These missing-detail cases now have
[raw regression evidence](../benchmarks/performance-fidelity-2026-09-03/slot-static-v1.json).

## Cloud Run loading and cancellation

One identical image was deployed with each loading strategy. Six requests per variant used three
fixed prompts twice, BF16 SANA, FP16 depth, 1024×576, two steps, CFG 4.5, and fixed seeds.

| Measured boundary | CPU then CUDA | Direct CUDA |
| --- | ---: | ---: |
| Cold first request | 75.88 s | 65.13 s |
| Library imports | 27.96 s | 24.80 s |
| Image-model loading | 35.55 s | 26.48 s |
| Depth-model loading | 1.14 s | 1.10 s |
| Warm HTTP median, five requests | 314.9 ms | 323.5 ms |
| Warm HTTP maximum | 332.3 ms | 437.8 ms |

Every corresponding master **and** depth file matched byte-for-byte, including the first request.
Direct placement is supported by the [pinned Diffusers 0.39.0 loader](https://huggingface.co/docs/diffusers/v0.39.0/en/using-diffusers/loading).
It is enabled on the tested RTX worker; the portable source default remains CPU-then-CUDA.
There is only one cold startup per option. Import variability and first-kernel work are substantial;
no stable percentage, startup p95, or warm-inference speedup is claimed.

The worker now publishes model readiness only after both models load. Cancelling an HTTP request
does not release its GPU lock while the backing thread is still running. Tests cover repeated
cancellation, partial initialization, and serial inference. Normal generation now reports warm
state correctly even when no explicit prewarm endpoint was called. Load telemetry includes imports
and records separate loading stages; older aggregate load times excluded imports.

The first direct-loader attempt was a pre-inference HTTP 503 because the temporary baseline
revision occupied the project's only RTX slot. Only that temporary revision (`00013-dor`) was
removed, retaining its immutable image and measurements. The subsequent finite comparison passed.
Production now uses `00014-ceb`; the former production revision `00007-4fr` remains available for
rollback. No minimum instance, GPU quota, billing budget, or disconnect threshold was increased.

## Review integrity, not a false accuracy claim

Accepted fixes:

- Repair guidance supplements the original requirements; it cannot replace the contract used
  to judge the repaired image.
- An `accept` verdict cannot simultaneously report identity inconsistency or unintended text.
- Explicitly truncated or otherwise unfinished responses are rejected even if their JSON parses.

The alternative prompts were **not promoted**:

| Diagnostic set | Existing critic | Image-blind observation + text comparison |
| --- | --- | --- |
| Six fox/lantern contracts, v3 | 3/6 correct; three false accepts | 4/6 correct; rejected both valid cases |
| Eight frozen holdout contracts across three images | 3/8 correct; five false accepts | 5/8 correct; rejected all three valid cases |

The candidate rejected everything. Its higher aggregate count therefore is not a useful quality
improvement. On the holdout, its median was 6.55 s versus 5.08 s for the baseline. A shorter
question-style prompt produced eight truncated responses out of fourteen; the six completed
responses all rejected. These small diagnostic sets expose failures; they are not general accuracy
estimates. The current NIM verdict must not be represented as dependable proof of action, count,
or spatial fidelity.

The cached TensorRT engines were reused on a newly provisioned GKE node. Pod creation to readiness
was **652 seconds**: 432 seconds before the NIM container started, then 220 seconds of container
initialization. This separates clean-node provisioning/image loading from the previously measured
234-second same-node restart. The NIM Deployment was scaled back to zero after testing.

## Stronger open image model experiment

[FLUX.2 Klein 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) is Apache-2.0 licensed.
The experiment pins revision `e7b7dc27f91deacad38e78976d1f2b499d76a294` and compares it with the
existing pinned SANA on the **same NVIDIA L4**, same dimensions, prompts, seeds, and BF16 precision.
Native settings differ: SANA uses two steps/CFG 4.5; Klein uses four steps/guidance 1.0.

On six fixed scenes, visual inspection found:

| Requirement | SANA | Klein |
| --- | --- | --- |
| One golden boat below a crescent moon | Pass | Pass |
| One silver fox left of a lantern | Two foxes | Pass |
| Owl with lighthouse to its left | Lighthouse on right | Pass |
| Walking fox carrying lantern in its mouth | Lantern below neck, no mouth contact | Pass |
| Exactly two red boats | Three boats | Pass |
| One child holding an open green book on a wooden bridge | Pass | Pass |

This is **2/6 versus 6/6 on these examples**, not a benchmark-wide accuracy estimate. Klein's warm
image-only latency was approximately 2.37 s versus SANA's 0.72 s on the L4. Download/load,
network delivery, depth estimation, and projection are excluded. Klein is promising for a
quality-first route, but it has not replaced production or passed the full edge-to-cloud handoff.

The next finite L4 test alternated 512- and 128-token text-encoding lengths for each of six prompts
at two seeds. Every actual chat-template input was 44–55 tokens; the harness aborts instead of
truncating. Excluding the first inference at each length, **11 samples per option** gave medians
of **2.3723 s → 2.0198 s**, a **14.9% reduction** in warm image-only latency. Outputs are not
byte-identical, so they were inspected separately: both variants passed the 12 tested core scene
requirements. Literal curtains/staging still sometimes appeared with the “paper theater” style.
This supports the bounded candidate setting, not an unconditional 128-token production limit.
Longer prompts require a larger encoding length or an explicit local validation error.

[Raw same-GPU comparison](../benchmarks/performance-fidelity-2026-09-03/model-comparison/results.json),
[text-length timings](../benchmarks/performance-fidelity-2026-09-03/klein-text-length/summary.json),
and [unblinded visual review](../benchmarks/performance-fidelity-2026-09-03/visual-review.json) include
all tested images/checksums. The image-only benchmark does not establish video generation or
temporal consistency; Bookforge's current fast delivery is an image/depth-driven animated scene.

The initial L40S job was refused before inference because this Modal account requires a payment
method for that GPU. An L4 run completed inference but its result included PyTorch's version-object
type, which the lightweight Mac client could not deserialize. CPU-only recovery found that the
consumed output was no longer available. Casting the version to a plain string fixed export;
one explicit bounded rerun produced the images. Both L4 runs generated identical checksums.
These failures are retained as experimental limitations, not hidden as successful deliveries.

## Evidence and release

- [Raw loader and critic evidence](../benchmarks/performance-fidelity-2026-09-03/summary.json)
- [Successful browser delivery](../benchmarks/performance-fidelity-2026-09-03/workbench-vertex-fixed.json)
- Reproducible finite model test: `experiments/renderer-fidelity/modal_compare.py`
- Renderer image: `sha256:d3f38cdc192b9ee283a01ea60cc93531e762a1ef4c3f8a6a3b775d05d26ce768`
- CPU coordinator image: `sha256:ce5d10b541509061f432732fa51dfaae62a3c827289719691ce7aa331985c105`
- CPU build: `9926c31f-b7e7-4b87-a6e0-2ab6672b6bba`; no GPU replica change.
- Jetson wheel: `sha256:ea7d0cf88384cde302af7ca8e05a75d621d68cec942b7258aac6e8cfd6629f55`
- Full regression suite: **1,097 passed**; Ruff and whitespace checks passed. One pre-existing
  Starlette/httpx deprecation warning remains.

The round's declared incremental compute ceiling is $5, not a provider-enforced spending cap.
Existing $150 gross budget alerts and the $175 disconnect threshold remain unchanged. All new
Modal tests are finite, one GPU maximum, 900-second timeout, zero retries, no public deployment.
The five new Modal apps were confirmed stopped with zero tasks. The metered monthly report still
showed $12.31848146 and had no new experiment rows at reconciliation time; billing can lag, so that
is **not** a claim the new work was free. No payment method was added and no budget was raised.

## Next acceptance gate

Promote no new image model or planner on these six examples alone. First preserve the missing
secondary objects/relations in a larger held-out edge-planner set, with privacy and negation tests;
then evaluate Klein through the actual sanitized handoff, depth estimation, and projector path.
Include fresh prompts and multiple seeds, cold/warm end-to-end latency, and blinded visual review.
The current Nemotron critic is not sufficiently reliable to serve as the sole quality gate.
