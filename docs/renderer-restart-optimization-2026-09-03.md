# Restartable renderer and contract fidelity

This pass adds an isolated, provider-independent Klein runtime, validates compiler-cache reuse
across actual containers, and deploys two scene-validation fixes. Vertex and the existing Modal
fallback remain the live routes. No GCP resources, budgets, minimum instances, or billing settings
were changed. The candidate generates artwork plus depth for the existing animated projector;
these results are not generated-video latency measurements.

## What is implemented

- Pinned Klein 4B and Depth Anything V2 Small weights baked on CPU, before GPU allocation.
- Offline-only model loading in the inference process; no Hugging Face downloads at startup.
- Regional compilation in standard mode, without GPU snapshots or CUDA-graph mode.
- Portable compiler artifacts validated against runtime source, model revisions, libraries,
  CUDA/GPU identity, inference settings, and a checksum before deserialization. These artifacts
  are executable material and must stay in our controlled private cache, not user uploads.
- Token buckets of 128/256/512 selected using the tokenizer's actual chat-template length.
  Oversize prompts fail explicitly instead of silently losing story requirements. Only 128 and
  256 were exercised on GPU in this pass; 512 has boundary tests but still needs GPU qualification.
- One GPU, single-use containers, 300-second function limits, no automatic retries or endpoint.

Runtime: [klein_scene_runtime.py](../deploy/klein_scene_runtime.py).
CPU build: [klein_weights.py](../experiments/renderer-fidelity/klein_weights.py).
The approach uses [Modal's supported weight-image build](https://modal.com/docs/guide/model-weights)
and [PyTorch's save/load compiler artifacts](https://docs.pytorch.org/tutorials/recipes/torch_compile_caching_tutorial.html).
The latter APIs were also verified in the pinned PyTorch 2.8 source, not assumed from newer docs.

## Three genuinely separate containers

The test makes 24 renders per container: six short briefs and six generic-wire full contracts,
then an identical second repetition. Different Modal task IDs establish fresh containers; no
image-result cache is used. The private artifact is 11,607,463 bytes.

| Measured boundary | Build | Restore 1 | Restore 2 |
| --- | ---: | ---: | ---: |
| Offline model loading | 20.446 s | 6.804 s | 6.759 s |
| Cache loading / compilation setup | 0.812 s | 1.291 s | 0.775 s |
| First 128-token render, including depth/encoding | 23.072 s | 8.624 s | 8.355 s |
| First 256-token render, including depth/encoding | 18.607 s | 4.733 s | 4.815 s |
| Warm short-scene median, six second-repetition samples | 1.652 s | 1.739 s | 1.766 s |
| Warm full-contract median, six second-repetition samples | 1.735 s | 1.836 s | 1.845 s |
| Entire 24-render call, client boundary | 120.522 s | 80.167 s | 80.497 s |

The previous download-plus-load run took 84.643 seconds. This implementation removes downloads
from that path, but these are separate runs: host disk/page caching and scheduling can affect
the observed load difference. Fresh container does not prove fresh physical host or cold disk.
Imports, scheduling, transfer, planning, and display are not part of the per-render numbers.
Neither an eight-second first click nor a reliable p95 is established here.

All 144 saved image/depth files match recorded SHA-256. Within each container, repeated seeds
were identical. Both restored containers produced identical outputs for all twelve cases, but
build-versus-restored images were not byte-identical: median RGB absolute pixel difference was
1.763 on the 0–255 scale; depth difference was 1.284. Cache restoration therefore passes these
repeatability checks, not a claim of bitwise equivalence to uncached compilation.

[Harness](../experiments/renderer-fidelity/klein_restart.py),
[complete results](../benchmarks/renderer-restart-2026-09-03-v2/results.json).
The executed harness SHA is `89227213b3b21e18e736c1bd3904a22410e51e17553d2e6791690f237cdec6d9`;
runtime SHA is `f87ab40c8b6a1ed457a10f1bce6bf92e07075eed40c0fdb224ba3ed18c5c3441`.

## Accuracy mattered more than another small kernel gain

The generic-wire fixtures exposed failures before the renderer: red was dropped from boats,
lantern details changed, and `no other people` became a positive request for `other people`.
This is the older generic wire sanitation path, **not** evidence that the current TensorRT
client used that path. The TensorRT client already handles this specific negative phrase.
Those bad fixtures are retained as diagnostic evidence, not counted as successful scenes.

A separate paired test used authored four-slot answers through the **production TensorRT
parser**, then compared the full assembled prompt with a concise rendering instruction. Both
versions used identical slots, source passages, style, weights, seeds, dimensions and steps.
Concise composition retains the wire action before the full formatter's ten-word action limit;
the change therefore tests the contract as a whole, not token count alone. Complete concise
prompts pass the local privacy gate, including clause-boundary checks. No Gemma inference or
raw private stories were used, and this does not measure the planner's accuracy.

| Six warm pairs | Full contract | Concise contract |
| --- | ---: | ---: |
| Token count | 185–199 | 55–59 |
| Median artwork + depth + encoding | 1.728 s | 1.644 s |
| Maximum | 1.740 s | 1.652 s |

The 4.9% median reduction is modest; the visual difference is more useful. Full prompts produced
a duplicate owl, a duplicate fox, and a third boat. Those duplicates disappeared in their concise
counterparts. The child stands on the bridge holding an open green book, with no other people.
The lantern is attached at the fox's mouth and the lighthouse stays left of the owl. Limitations:
the silver canid looks somewhat wolf-like in both variants; the concise boat scene adds curtains;
watercolor edges still occur despite full-bleed instructions. This is visual inspection of six
development pairs, not independent human evaluation, a sealed holdout, or general accuracy.

This is the more promising candidate contract, but it is **not enabled in live routing**.
The next gate is fresh passages and seeds through actual Gemma, plus physical playback and
click-to-first-scene measurements before promotion.

[Paired harness](../experiments/renderer-fidelity/klein_contracts.py),
[results and prompts](../benchmarks/renderer-contract-2026-09-03/results.json),
[two boats](../benchmarks/renderer-contract-2026-09-03/concise-4-r1.jpg),
[child on bridge](../benchmarks/renderer-contract-2026-09-03/concise-5-r1.jpg).
All 48 image/depth hashes verified. Harness SHA:
`5bf24aeeda9b589eb38ae297523105ca1839c252348d2cc1322f3c474fe6892f`.

## Deployed reliability fixes

1. `Exactly` followed by a count is no longer mistaken for a source person's name.
   Explicit names such as `named Exactly` remain blocked, including `named Exactly 2`.
2. Generic privacy rewriting preserves `no other people` as `no additional people`.
   Other rewrites fail closed if their chosen deletion would remove a negation.

The semantic cache is versioned `semantic-v19-preserve-negation`; completed scene contracts use
`subject-counts-constraints-v4`, so older successful-but-wrong packs cannot mask the correction.
Existing files are retained. Other generic paraphrase losses and the planner's missed relations
remain open problems, not solved by these two fixes.

Final wheel: `31777d3fcfcb87199ff44896b2c5d153be273b7a92c702a69dd93055a03da55d`.
Only the Jetson API needs restarting; the resident TensorRT engine, physical projection session,
25 W mode, and provider selection are unchanged. Previous wheels remain for rollback.
Verification and deployment status: [audit](../benchmarks/renderer-restart-2026-09-03-v2/audit.json).

## Reproduce and spend boundaries

```sh
.venv/bin/modal run experiments/renderer-fidelity/klein_restart.py --output-dir NEW_DIRECTORY
.venv/bin/modal run experiments/renderer-fidelity/klein_contracts.py::main \
  --output-dir ANOTHER_NEW_DIRECTORY --cache-run-id RUN_ID_FROM_FIRST_REPORT
```

The CPU image-builder's first attempt failed because a helper module was not packaged; the next
attempt built the weights successfully, then stopped on a local fixture privacy failure before
any GPU call. Both failures were corrected before the successful comparisons. Every experimental
app was subsequently reported stopped with zero tasks; the final container inventory was empty.

Modal reported $0.12761670 for the restart test, $0.04413812 for the contract comparison, and
$0.00553042 for the CPU-build/preflight attempt: **$0.17728524 reported total**. The earliest failed
packaging attempt had no billing row yet; absence is not proof of zero cost. Billing can lag.
The private compiler cache and reusable weight image are intentionally retained, not running GPUs.

The cleanup pass kept the candidate isolated instead of adding unused production routing flags.
