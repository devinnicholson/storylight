# Native GCP cold-loading comparison — September 6, 2026

The four-service comparison completed all **40 requests and 80 verified JPEGs**, but **rejected eager loading** under the frozen screen. Median first-request time improved by 8.84%, below the required 15%; only one paired load comparison improved, and 38 of 40 paired image comparisons differed. The candidate is not promoted. The prospective protocol and separate $5 maximum allowance below remain unchanged; measured results follow it.

G's first client request took 72.769 seconds, including 47.371 seconds of model loading; later client requests had a 0.492-second median. That motivated this loading experiment. G still failed one of eight assistant-reviewed scene cases, and human review remains pending. This experiment does not waive those quality requirements. See the [completed qualification record](gcp-klein-qualification-2026-09-06.md).

## One candidate, four fresh services

Use the fixed order **A, B, B, A**, with a new private service and worker for every run. A uses the exact G image and original loading behavior. B uses a frozen experiment-only runtime copy whose sole loading change is `disable_mmap=True` for the Diffusers transformer and VAE. Preserve the approximately 8 GB Transformers text encoder's existing loading path, all model weights, packages, precision, compilation settings, prompts, seeds, and rendering parameters. Production runtime files and live provider routing remain unchanged.

The pinned Diffusers 0.39.0 loader implements this option by reading a safetensors file into bytes instead of using its file-backed loading path. That may change startup time and peak host memory; it is not evidence of a speed improvement. Before building, verify the option actually reaches the transformer and VAE loaders and is not silently ignored by pipeline argument handling. Bind the runtime file and expected identity to each manifest using explicit source hashes. [Pinned Diffusers loader](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/models/model_loading_utils.py).

Each service receives the **same ten-request schedule**: retained cases 0 and 1, the six original watercolor cases, then retained cases 0 and 1 again. Four completed services therefore yield **40 requests and 80 JPEGs**, not 40 independent scene examples. Preserve the first request and all priming work. There is no health request, prewarm, automatic POST retry, or replacement sample after a failure.

Use `us-central1`, one RTX PRO 6000 Blackwell Server Edition, 20 vCPUs, 80 GiB RAM, concurrency one, minimum zero, and both service and revision maximum one. Keep CPU boost off, private IAM, one revision with all traffic, and no fallback provider. Verify the deployed metadata and exact unconditional invoker binding. Wait 120 seconds for IAM propagation inside each service's work deadline; this wait is not proof of propagation. Obtain credentials without contacting the renderer.

Each service has **600 seconds from deployment start for all work, plus 60 seconds for cleanup**. Delete and verify closure before starting the next service. A failed or ambiguous deployment retains its evidence and requires explicit closure; an absent service alone does not resolve a pending create operation. Stop on workload, integrity, or unresolved cleanup failure. Preserve missing runs as missing, never as a balanced comparison. A request timeout does not terminate server work. [Cloud Run timeout semantics](https://docs.cloud.google.com/run/docs/configuring/request-timeout).

The manifests must remain valid for their complete scheduled runs. Preserve the existing expiry checks; do not extend an expired baked manifest implicitly. Record the actual baseline and candidate image digests, manifest hashes, service/revision identities, source commit, and worker identity for every run.

## Measurements and decision

The primary measurements are **first-request client artifact-ready time** and **worker model-load time**, reported per fresh service. Keep build, deployment, IAM wait, credential acquisition, first-bucket priming, and later rendering separate. The client timer ends only after receiving, verifying, and saving master and depth. Use the existing inclusive model-factory timer; this protocol adds no component-level loading timer or host-RSS measurement. The 80 GiB host-memory limit remains enforced.

Compare A1 with B1 and B2 with A2, preserving the reversed order of the second pair. Report both pairs, variant medians, and every failure. A causal interpretation requires all four valid runs and evidence that only the loading option changed. Even then, two runs per variant are a small screen: fresh processes do not imply cold host file caches, equivalent image-streaming caches, or identical underlying hosts. ABBA reduces a simple order effect but does not control those factors. Do not claim a p95, population-wide improvement, or physical end-to-end speedup.

Output preservation is a separate gate. Compare matched first occurrences with first occurrences for each case and matched final repeats with final repeats; do not compare A's cold first rendering against B's later rendering. Require exact master and depth hashes for a strict behavior-preserving result. Any mismatch remains a failed equivalence check with both artifacts retained; do not lower the threshold after seeing it. G's existing first-versus-repeat differences are a reason to preserve these comparisons, not permission to ignore new differences.

The frozen speed screen requires **at least a 15% reduction in median first-request client time**, improved model-factory load times in **both** adjacent pairs, and no more than a **5% regression in later-request worker median time**. All 40 requests and 80 images must verify, and every matched master/depth pair must be byte-equal. Preserve separate timing and output verdicts; neither can compensate for the other. The exact service pairings and gates are recorded in the [comparison plan](../benchmarks/gcp-klein-loading-2026-09-06/comparison-plan.json).

Retain the frozen scene and watercolor checks, including the prohibition on added foxes in the single-fox case. No speed result repairs a semantic failure. Human visual acceptance, an expanded scene set, repeated cold reliability, and the Jetson-to-projector demonstration remain necessary before product promotion.

## Separate funding and image scope

The original qualification phase retains its **$4.8584904 allowance** and unresolved billing reconciliation. This phase adds its own allowance; it does not reset or release the previous phase's holds. The loading preflight retained a project notification of **$21.59 at 21:16 UTC**; notifications can lag and do not establish the final cost of G or this phase. Refresh billing before dispatch and again at closeout.

| Component | New phase allowance |
| --- | ---: |
| One tiny overlay build: eight CPUs, 100 GB disk, 600-second timeout | $0.17 |
| Four GPU service lifecycles: two resource slots × 660 seconds each | $4.6739616 |
| Incremental small-overlay storage | $0.02 |
| Remaining margin | $0.1360384 |
| **New gross phase cap** | **$5.00** |

The GPU allowance uses the reviewed $0.00088522/second for one RTX GPU, 20 vCPUs, and 80 GiB, without free-tier credits. Configured maximum instances are not an absolute platform billing cap; the two-slot allowance and external service deadline remain necessary. [Cloud Run pricing](https://cloud.google.com/run/pricing), [maximum-instance limits](https://docs.cloud.google.com/run/docs/configuring/max-instances).

Build only the small candidate runtime/application/manifest overlay on G's immutable image, using the pinned BuildKit approach. Reuse all existing model-layer digests and verify the layer prefix and runtime configuration before dispatch. No model rebuild, new model download, or duplicate full model layer is included in this allowance. The original phase retains the shared base image; the new $0.02 covers only small incremental overlays for at most seven days. The build's ten-minute CPU rate is $0.156, rounded to $0.17; the first 100 GB disk is included. [Cloud Build pricing](https://cloud.google.com/build/pricing), [Artifact Registry storage](https://cloud.google.com/artifact-registry/pricing).

Closeout must retain all four attempted service identities, terminal operations and absence checks, request journals, successful artifacts, image/source proofs, and attributable cost metrics. Report rate-based estimates separately from settled charges, and carry unresolved liabilities forward. No further build or service is automatically authorized by unused margin.

## Completed comparison

The [recomputed summary](../benchmarks/gcp-klein-loading-2026-09-06/summary.json) verifies ten requests per service, four distinct workers, eight nonpriming requests per worker, all 80 JPEGs, and clean supervisor closure for every service. It reports no verification errors. Each row below is one fresh service, in execution order.

| Service | Loading | First client artifact-ready | Model-factory load | Later worker median | Supervised lifetime, including cleanup |
| --- | --- | ---: | ---: | ---: | ---: |
| H | Default mmap | 88.248 s | 59.463 s | 0.346633 s | 237.023 s |
| I | Eager | 69.139 s | 43.941 s | 0.346636 s | 220.448 s |
| J | Eager | 63.269 s | 39.709 s | 0.349132 s | 212.791 s |
| K | Default mmap | 56.995 s | 38.149 s | 0.346515 s | 204.928 s |

Default-loading median first-request time was **72.622 seconds**, versus **66.204 seconds** for eager loading: an **8.837% reduction**, which fails the 15% threshold. I loaded faster than H, but J loaded slower than K, so the requirement that both paired model-load times improve also failed. The candidate's median of later-worker medians regressed by **0.378%**, within the permitted 5%.

The strict output-preservation check also failed: **38 of 40** paired master/depth comparisons differed. H versus I differed at every ordinal for both roles. K versus J matched only ordinal zero's master and depth. All files passed their own response-hash and JPEG validation; that integrity result does not establish equality between variants or visual correctness. The earlier G/H baseline variability remains a separate observation, and the frozen paired equality gate has not been relaxed.

The default-loading first request varied from 88.248 to 56.995 seconds across its two runs. Host and storage cache state were not controlled, so these four measurements do not establish that eager loading causes a reliable improvement. Neither an aggregate timing reduction nor the successful transport of all images repairs the failed image-preservation or outstanding semantic and human-review gates.

The candidate image is `sha256:930b37035208b3dd2c670794bcdbca9d960af3a09b5bcc55562097fe346b0270`. Its overlay build succeeded, and independent inspection confirmed all 13 G base layers and the full runtime configuration were preserved. The new 7,235-byte compressed layer contains only the exact frozen application, manifest, and candidate runtime files, owned by UID/GID 65532, with no links or special entries. See the [image verification](../benchmarks/gcp-klein-loading-2026-09-06/image-verification.json).

All four supervisor receipts report terminal creation, service deletion and observed absence, with no cleanup error or unresolved late creation. The combined supervised lifetime was **875.190 seconds**; these local receipts are not settled provider billing. Preserve the phase allowances and billing uncertainty until attributable closeout evidence is reconciled.

The final regional service inventory contains only the existing renderer and billing guard; all four experiment services are absent. The latest project notification reports **$21.92 gross at 21:43 UTC**, up from $21.59 before this phase. That delayed project-wide difference is not an attributable experiment charge. The [closeout receipt](../benchmarks/gcp-klein-loading-2026-09-06/closeout.json) retains both phase allowances without releasing liabilities and separates a two-slot rate estimate from settled billing.

To reproduce the result offline, run `python benchmarks/gcp-klein-loading-2026-09-06/summarize.py` from a checkout containing the retained artifacts and the two execution-source files at their pinned `bd1e3096304dc85f058ece995290dcd31093abee` bytes. The reporter checks those source pins, both plans, manifests, deployed metadata, journals, all images, and closure receipts before calculating the gates. Independent recomputation matched the saved summary, whose SHA-256 is `76ce41f81f68a0021ffdafdaa2439736f45f5f2ad0e67b9a0b7386ed07880cba`.
