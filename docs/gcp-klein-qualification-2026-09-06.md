# Native GCP Klein qualification — September 6, 2026

Klein 4B completed **10 native Cloud Run requests with 20 verified JPEGs** in attempt G. The eight later client requests had a 0.492-second median; the first request took 72.769 seconds. Assistant inspection passed seven of eight unique scenes but found an extra fox and lantern in the remaining scene. **Human review is pending and the candidate is not promoted.** This record preserves the original protocol and repairs within an isolated **$5 gross-cost phase**, targeting the restored illustrated watercolor appearance and the GCP path required for submission. The existing SANA service, live application routing, GKE workloads, and Modal rollback remain unchanged.

The existing GCP adapter's automatic readiness probe now prepares credentials without requesting `/health`, which could activate a billed GPU before generation. Readiness means credentials are available; paid operations still verify runtime identity. This correction does not select the Klein candidate or change deployed services.

The September 6 audit reported linked billing and **$21.18 gross project usage at 18:38:44 UTC**. The $150 alert and $175 emergency disconnect remain unchanged. Cloud Run has quota for one RTX PRO 6000 and no L4; the separate regional Compute Engine L4 quota is not Cloud Run capacity. GKE reports four nodes, with coordinator replicas at one and Nemotron at zero. An empty Compute Engine instance-list response does not establish that the project has no running compute. Budget alerts are notifications, not spending caps. [Google Cloud budgets](https://docs.cloud.google.com/billing/docs/how-to/budgets).

## Fixed first phase

Build one image with `E2_HIGHCPU_8`, a 100 GB disk, and a server-side **1,200-second build timeout including push**. Upload only the approved worker, requirements, Dockerfile, pinned runtime and weight-download code, synthetic manifest, and build configuration. No private passage, credentials, local model cache, or repository checkout belongs in the build context. Record the context hashes, build ID, image digest, and build outcome; cancel a build that exceeds its bound rather than submitting it again.

Deploy one new IAM-private service in `us-central1`, using one RTX PRO 6000, 20 vCPUs, 80 GiB RAM, concurrency one, minimum zero, and both service and revision maximum one. Use one immutable revision, no traffic tags, no zonal redundancy, and no fallback to another provider. Verify the deployed resource limits, image digest, revision, and IAM policy before inference. Cloud Run requires at least 20 vCPUs and 80 GiB for this GPU. Its advertised infrastructure startup time is not model readiness. [Cloud Run GPU configuration](https://docs.cloud.google.com/run/docs/configuring/services/gpu).

Start the external service-lifetime deadline before deployment: at most **600 seconds of work plus 60 seconds for process cleanup, deletion, and verification**. Delete this exact candidate on completion, failure, ambiguity, or deadline. A Cloud Run request timeout can leave code running, and configured maximum instances can briefly be exceeded; neither is a hard cost cutoff. Preserve failed attempts and never automatically retry a generation. [Request timeout](https://docs.cloud.google.com/run/docs/configuring/request-timeout), [maximum instances](https://docs.cloud.google.com/run/docs/configuring/max-instances).

If deployment times out locally, even an empty service inventory cannot prove that creation has stopped. Record closure as unresolved and inspect the exact service's pending operation, then finish cleanup before any further attempt. The ordinary 660-second allowance is not a claim that an unresolved control-plane incident has ended.

Send exactly **10 sequential POST requests covering eight unique public cases**: the two retained benchmark prompts, the six original watercolor cases, then the two retained prompts again. Freeze prompts, seeds, order, source hashes, and expected model identity before dispatch. There is no health request or prewarm before the first generation. Cloud Run's infrastructure startup probe and deployment itself remain separate events; the first POST is not necessarily a scale-from-zero measurement.

Retain all successful master and depth JPEGs, including poor images: up to 20 files. Stop on a request or integrity failure. Bind every result to its manifest, case, seed, image/runtime identity, service revision, instance, and artifact hashes. The process request limit is supplementary; the client enforces the finite schedule and the supervisor owns cleanup.

## What the result can establish

Measure client request start through fully received, verified, saved artifacts with one monotonic clock. Report model loading, compilation, worker rendering, depth, and JPEG durations separately. Token acquisition, image build, deployment, browser display, and projection are outside that client timing. Do not attribute their difference to network time alone.

The first model load on an instance is **worker-cold**. The first occurrence of each sequence bucket on each instance is **bucket priming**. Report the later samples by bucket with their counts, medians, and maxima; this small screen does not establish a p95 or cold-start reliability. Report replacements and failures rather than dropping them from the result.

Judge the new GCP images against the frozen scene facts and the restored watercolor appearance. Required actors, counts, colors, actions, object bindings, legibility, and continuity need explicit review; missing or unsure judgments remain unresolved. Still images do not prove transformations or event order. Human review is pending until someone actually grades the images.

Historical byte matches are diagnostics, not a substitute for this review. The earlier same-noise L40S replay failed its prospective 40 dB equivalence gate: the retained images reached only 22.41–28.64 dB. This new semantic and style qualification does not turn that failure into a pass or claim pixel equivalence across GPUs. See the [retained replay evidence](../benchmarks/renderer-latents-2026-09-06/replay/).

Before any product promotion, separately qualify at least five controlled cold starts, an expanded scene set, user visual review, and the physical Jetson → GCP → displayed-artifact path. Those are later phases, not additional calls silently included in this $5 screen. Record actual NVIDIA/GCP execution and a reproducible end-to-end demonstration for the submission; service deployment alone is insufficient.

## Cost and closure

The initial allowance uses published on-demand rates without free-tier credits. At the reviewed `us-central1` rates, RTX compute plus 20 vCPUs and 80 GiB costs `$0.00088522/second`. Two full 660-second resource slots allow **$1.1684904** for the candidate lifecycle. This is a conservative planning allowance, not a promise that service configuration imposes a perfect billing cap. Startup CPU boost is disabled and checked in deployed metadata. [Cloud Run pricing](https://cloud.google.com/run/pricing).

| Component | Planned allowance |
| --- | ---: |
| One 20-minute, eight-vCPU image build at $0.0156/minute | $0.312 |
| Two full candidate resource slots for 600 + 60 seconds | $1.1684904 |
| At most 50 GiB of candidate images retained for seven days | $1.20 |
| Remaining margin for transfer, logs, scanning, and cleanup uncertainty | $2.3195096 |
| **Gross phase cap** | **$5.00** |

The build rate comes from [Cloud Build pricing](https://cloud.google.com/build/pricing). Artifact Registry storage is $0.000136986 per GiB-hour: 50 GiB for 168 hours is about $1.151, rounded up to $1.20. Keep image storage in the runtime region and record any scanning or transfer charges. Delete the candidate image after this phase unless a specific reuse justifies retaining it within the seven-day limit; do not delete shared or accepted assets. [Artifact Registry pricing](https://cloud.google.com/artifact-registry/pricing).

Closeout must retain the build and service identities, exact request journal, deployment settings, deletion result, and post-run service inventory. Refresh project billing and collect attributable build/runtime/storage metrics. Preserve provisional cost labels while reporting lags; a deleted service does not prove zero outstanding charges. Existing GKE, Vertex, storage, and other project usage remain separate from this phase's attributed cost and continue to count toward project guardrails.

## Build A failure and bounded Build B repair

Build A (`58b9484a-d19f-4a7f-ae93-fc0a13a9b5d3`) reached terminal `TIMEOUT` at **19:29:39 UTC**. Its build stage completed in about 8 minutes 47 seconds; the subsequent image push ran for about 11 minutes 13 seconds before timeout. Start-to-finish elapsed time was 20 minutes 23 seconds. No candidate service or GPU inference was created. Preserve its original [build context and scope](../benchmarks/gcp-klein-2026-09-06/) and this document's initial prospective configuration; do not present the failed push as a successful image deployment.

Build B changes only the build configuration to `E2_HIGHCPU_32`, a 500 GB disk, and an explicit Docker push step. Google recommends this configuration for model images whose builds can be limited by network throughput. This supports testing the repair; it does not establish the exact cause of A's timeout. [GPU image-build guidance](https://docs.cloud.google.com/run/docs/configuring/services/gpu-best-practices). The native Artifact Registry path is not subject to the external-registry layer-size restriction, so that restriction is not a diagnosed cause. [Supported container registries](https://docs.cloud.google.com/run/docs/deploying).

Keep the same Dockerfile, package versions, runtime, model revisions, and eight cases. Use a new manifest identity, `bookforge-klein-qualification-20260906-b`, with a fresh expiry, and retain both manifest hashes. The server build timeout remains 1,200 seconds including push. A is terminal before B starts; this is one explicit build repair, with no generation reroll, bucket migration, or new IAM scope.

The **same $5 gross phase cap** now allocates:

| Component | Revised allowance |
| --- | ---: |
| Failed Build A, including observed timeout overrun | $0.33 |
| Build B: 21 minutes of 32-vCPU billing plus 400 GiB additional SSD | $1.35 |
| Candidate GPU lifecycle, unchanged | $1.1684904 |
| Aggregate candidate image storage, unchanged | $1.20 |
| Remaining margin | $0.9515096 |
| **Gross phase cap** | **$5.00** |

At $0.0624/minute, 21 minutes costs $1.3104. The additional SSD allowance is `400 × 21/60 × $0.000232877 = $0.03260278`, giving $1.34300278 before rounding to $1.35. These are conservative allowances, not settled charges. Recheck actual build duration and billing after completion. The 50 GiB/seven-day storage limit covers both A's retained partial uploads and B's image together; do not assume rebuilt model layers deduplicate. [Cloud Build machine and SSD pricing](https://cloud.google.com/build/pricing).

## Build B quota rejection and prospective Build C

Build B was [rejected before build creation](../benchmarks/gcp-klein-2026-09-06/retry-b/build-rejected.json). Source upload occurred, but no build operation or GPU call followed. The regional default-pool CPU quota is ten; the separate private-pool quota is four. The proposed 32-vCPU builder does not fit. Retain B's source context and a **$0.02 preparation allowance**, without claiming zero charge or requesting a quota increase.

Build C uses `E2_HIGHCPU_8`, the 500 GB disk and explicit push, and a server-side **2,400-second timeout including push**. Its new manifest identity is `bookforge-klein-qualification-20260906-c`, with an expiry that covers the longer build and subsequent supervised trial. Runtime, model weights, packages, cases, GPU limits, and inference schedule remain unchanged. This tests whether the larger disk and longer push window suffice; it does not assume the eight-CPU builder has gained the networking capacity of the rejected builder. Model delivery stays in the image, with no new Cloud Storage model bucket or IAM changes.

The current prospective allocation supersedes B's unused build allowance:

| Component | Current allowance |
| --- | ---: |
| Failed Build A | $0.33 |
| Rejected Build B preparation | $0.02 |
| Build C, including 41 minutes of CPU and extra SSD allowance | $0.71 |
| Candidate GPU lifecycle | $1.1684904 |
| Aggregate candidate image storage | $1.20 |
| Remaining margin | $1.5715096 |
| **Gross phase cap** | **$5.00** |

Build C's allowance is `41 × $0.0156 + 400 × 41/60 × $0.000232877 = $0.70325305`, rounded to $0.71. The extra minute covers limited timeout overrun; it is not an extension of the 40-minute server timeout. Inspect the terminal build record and attributable charges before any further repair. The aggregate 50 GiB/seven-day storage bound includes retained uploads from all three attempts. [Cloud Build pricing](https://cloud.google.com/build/pricing).

## Build C cancellation and precise Build D exclusion

Build C reached its explicit image-push step at 19:56:29 UTC and was deliberately cancelled after inspection found an unused checkpoint in the baked model directory. Its [terminal record](../benchmarks/gcp-klein-2026-09-06/retry-c/build-final.json) is `CANCELLED` at **19:57:52 UTC**; no GPU request was made. Preserve the full **$0.71** C allowance pending billing reconciliation.

The [pinned model inventory](../benchmarks/gcp-klein-2026-09-06/retry-c/public-weight-inventory.json) contains a 7,751,105,712-byte root file, `flux-2-klein-4b.safetensors`, alongside the Diffusers component directories. The unchanged runtime uses `Flux2KleinPipeline.from_pretrained` on those component directories, not the single-file loader. The pinned Diffusers loader resolves each component's subdirectory. [Diffusers 0.39.0 component loading](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/pipeline_loading_utils.py).

Build D introduces an isolated GCP weight-download helper that excludes **only that exact root filename, only for the exact pinned Klein model and revision**. An offline check against the inventory confirms that all 18 other files remain selected, including the transformer checkpoint, text-encoder shards, tokenizer, scheduler, and VAE. Depth-model downloads are unchanged. The selected Klein files shrink from 23,731,237,457 to 15,980,131,745 bytes; these are repository file sizes, not measured compressed image size or startup improvement.

The historical weight helper, GPU runtime, Dockerfile, packages, and loaded model revisions remain unchanged. D's [seven-file build context and hashes](../benchmarks/gcp-klein-2026-09-06/retry-d/build-scope.json) bind the new helper and fresh manifest identity `bookforge-klein-qualification-20260906-d`. The builder remains eight CPUs, 500 GB disk, explicit push, and a 2,400-second server timeout. A successful build still requires the same subsequent GPU integrity and human-quality checks.

Retaining A at $0.33, B at $0.02, C at $0.71, D at $0.71, GPU lifecycle at $1.1684904, and aggregate storage at $1.20 allocates **$4.1384904** of the same $5 cap, leaving **$0.8615096**. The 50 GiB/seven-day storage limit covers all attempts together. No build cancellation or excluded file is treated as a billing refund.

## Completed image and prospective deployment E

Build D [completed successfully](../benchmarks/gcp-klein-2026-09-06/retry-d/build-final.json) at **20:26:14 UTC**, producing immutable image digest `sha256:bc6d050e2aa869e5d97b90146a74e7e79ea59633d9068bb7525869af200de0c9`. Construction took 5 minutes 33.20 seconds; the explicit push took **16 minutes 31.90 seconds**. The separate final `timing.PUSH` value of 1.18 seconds records artifact publication after that upload. Start-to-finish elapsed time was 22 minutes 12.53 seconds. Published CPU and additional-SSD rates imply approximately $0.380937 for that interval; this is not settled billing, and the $0.71 build hold remains.

D's first service deployment then exceeded its 120-second local deployment wait before the benchmark client started. The supervisor attempted deletion and correctly [marked closure unresolved](../benchmarks/gcp-klein-2026-09-06/retry-d/trial/closure.json). An [independent control-plane audit](../benchmarks/gcp-klein-2026-09-06/retry-d/trial/manual-closure.json) subsequently found both matching operations terminal and the exact service absent. No generation request or client journal was created. Available deployment logs show service creation and revision waiting, without an application startup failure; they do not identify a more specific cause.

Deployment E reuses **the same immutable D image and D manifest**, with fresh service name `bookforge-klein-qualification-20260906-e`. The supervisor records the manifest experiment identity and service identity separately, refuses an existing service, and verifies the expected E revision and image. Its deployment wait increases to 240 seconds **inside the unchanged 600-second work deadline and 60-second cleanup reserve**. There is no image rebuild, altered prompt, or generation reroll. Pin the reviewed supervisor commit in E's execution receipt before dispatch; preserve D's original source receipt and failed closure.

Add a **$0.24 potential-resource hold** for D's closed deployment attempt. This exceeds two full RTX resource slots over its observed 125.871952-second supervisor interval, approximately $0.222849, without assuming no allocation occurred. All earlier allowances remain: the phase now allocates **$4.3784904**, leaving **$0.6215096** of the same $5 cap. Native GPU generation, visual quality, and latency remain unqualified until E produces and verifies actual results.

## E authorization rejection and prospective F propagation wait

E deployed and its observed IAM policy granted the intended service account `roles/run.invoker`. Its first POST nevertheless received a front-end **403** after about 0.216 seconds; the request log reported missing `run.routes.invoke` permission and zero service latency. There were no worker results or images. The [benchmark summary](../benchmarks/gcp-klein-2026-09-06/retry-e/trial/renders/summary.json) rejects the screen, and the [supervisor closure](../benchmarks/gcp-klein-2026-09-06/retry-e/trial/closure.json) verifies deletion after 17.856150 seconds. The original rejected request is retained without retry.

A [fresh token check using the same source](../benchmarks/gcp-klein-2026-09-06/retry-e/token-source-check.json) found the expected principal and audience and an unexpired token. It did not retain the token or contact the renderer. This supports checking the token source; it does **not** authenticate the historical token sent during E. IAM propagation is a plausible explanation, not an established diagnosis.

F reuses D's immutable image and manifest under fresh service name `bookforge-klein-qualification-20260906-f`. Before dispatch, the supervisor requires an explicit, unconditional invoker grant for the intended service account and rejects public invoker membership. It then waits **120 seconds inside the unchanged 600-second work deadline**, followed by the existing 60-second cleanup reserve. The delay adds no health request, prewarm, POST retry, public access, or project-level role change. Insufficient remaining time refuses dispatch and proceeds to cleanup.

Google documents policy propagation as typically two minutes, potentially seven minutes or longer. The fixed wait is a prospective mitigation, not proof that authorization has propagated; a further rejection still stops this attempt. [IAM access-change propagation](https://docs.cloud.google.com/iam/docs/access-change-propagation).

Retain **$0.04** for E's closed deployment and rejected invocation, above its two-slot elapsed-rate estimate of approximately $0.031613. The phase now allocates **$4.4184904**, leaving **$0.5815096** of the same $5 cap. Build and service closure evidence does not establish native Klein performance or visual quality; both remain pending actual verified output and review.

## F load failure and prospective G diagnostics

F's authenticated request reached the worker after the IAM wait, then returned **503**. The server logged 13.073 seconds of request latency; the failed client request took 13.307 seconds. The source-free application diagnostic reports `qualification.failed`, stage `load`, and exception type `ValueError`. There were no images; the [benchmark result](../benchmarks/gcp-klein-2026-09-06/retry-f/trial/renders/summary.json) remains rejected. [Verified deletion](../benchmarks/gcp-klein-2026-09-06/retry-f/trial/closure.json) closed the service after 151.752676 seconds. Reaching the worker establishes authorization for this request; it does not establish successful model loading or image generation.

An earlier [GCP hardware report](../benchmarks/bookforge-gemma4-g4-export-and-jetson-build-2026-08-30.json) records the device name `NVIDIA RTX PRO 6000 Blackwell Server Edition`. F's qualification worker expected the name without `Server Edition`. This makes the exact device-name check a plausible failure point, but F's broad `load` stage does not identify which validation or initialization step raised. Do not treat the historical report as a measurement of F's device.

G will add bounded configuration diagnostics and support that explicit device name while retaining exact manifest-to-observed identity checks. It uses a fresh G manifest and service, the same public cases, model revisions, GPU limits, and finite request schedule. Diagnostic output must exclude tokens, prompts, environment dumps, and exception messages. A mismatch still refuses model use; there is no broad GPU-name bypass or automatic request retry.

The reviewed image repair is a tiny application/manifest overlay on immutable D, using pinned builder and Dockerfile-frontend images and `COPY --link`. It has no `RUN` step or local image export; BuildKit pushes the resulting image directly. This configuration permits reuse of base layers without downloading them, but the actual build log and image provenance must verify that reuse before GPU dispatch. The CPU build uses eight CPUs, the default 100 GB disk, and a **600-second server timeout**. This remains a prospective build and runtime test. [Docker's `COPY --link` behavior](https://docs.docker.com/reference/dockerfile/#copy---link).

Retain **$0.27** for F, above its two-slot elapsed-rate estimate of $0.268669. Allow **$0.17** for G's CPU overlay build: ten minutes at $0.0156/minute is $0.156, with the remainder covering limited preparation or timeout overhead. The first 100 GB build disk remains included in the published rate. The previously reserved $1.1684904 GPU lifecycle allowance now covers the prospective G trial; it is not added twice. Total allocation becomes **$4.8584904**, leaving **$0.1415096** within the same $5 cap. Any failure or overrun must be reconciled against that remaining margin before further work.

G's [overlay build completed successfully](../benchmarks/gcp-klein-2026-09-06/retry-g/build-final.json), producing image `sha256:e18009ca1785394b84340be7f462bd6cb4ee2b4455484a9f3d7769fbc26ffaa8`. The terminal record reports 15.330 seconds from build start to finish, after 61.110 seconds between submission and build start. The recorded build phase lasted 10.827 seconds; its log reports 0.4 seconds pushing layers and 0.2 seconds publishing the manifest. These are CPU image-build timings, not model or generation performance.

Independent [image verification](../benchmarks/gcp-klein-2026-09-06/retry-g/layer-reuse-verification.json) confirms all 12 of D's compressed layer digests and sizes, plus their uncompressed `diff_ids`, remain an exact prefix. The only added layer is **5,212 compressed bytes**. Runtime user, command, environment, working directory, ports, and other execution fields are preserved. Docker layer media-type labels become their OCI equivalents; omitted empty legacy fields and the intermediate build-image label are recorded separately. The [new layer's content check](../benchmarks/gcp-klein-2026-09-06/retry-g/overlay-content-verification.json) binds the two regular application and manifest files to the frozen context with owner `65532:65532`. No model-layer blobs were downloaded for this verification. G's GPU qualification remains prospective.

## G native GPU result: complete, visually ungraded

G completed **10 of 10 requests on one worker**, with all **20 master/depth JPEGs** passing hash, format, and dimension checks. The [saved summary](../benchmarks/gcp-klein-2026-09-06/retry-g/trial/renders/summary.json) reproduces byte for byte from the journal and artifacts. Observed runtime identity matches the manifest, including `NVIDIA RTX PRO 6000 Blackwell Server Edition`; deployed image, revision, resource limits, and private IAM checks passed. This establishes native GCP rendering and artifact integrity. The recorded decision remains **ungraded**, not a visual-quality acceptance.

| Measurement | Result |
| --- | ---: |
| First client request through verified, saved artifacts | 72.768514 s |
| First worker request | 66.045609 s |
| Model loading | 47.370640 s |
| Compile-wrapper setup | 1.784706 s |
| First image generation, including bucket priming work | 16.243296 s |
| First 256-token-bucket client request | 6.532509 s |
| Eight later client requests, mixed buckets: median / maximum | 0.491930 / 0.665431 s |
| Later 128-token bucket: count, median / maximum | 7; 0.489415 / 0.665431 s |
| Later 256-token bucket: count, single observation | 1; 0.522899 s |

The 1.784706-second compile-wrapper value is not the full first-use compilation cost: the initial bucket renders still include priming work. The mixed-bucket median is dominated by seven 128-token observations. These results establish neither a p95 nor repeated cold-start reliability. They also exclude token acquisition, the 120-second IAM wait, deployment, browser display, and physical projection. The first request is worker-cold; the evidence does not establish a controlled scale-from-zero latency distribution or explain the client/worker difference as network time alone.

Same-worker repetition was not universally byte-identical. Retained case 0's first and final master/depth images differ despite matching prompt and seed; retained case 1's pair matches exactly. None of the retained outputs matches the historical L4 artifact hashes. Report these findings separately from semantic quality; do not claim universal determinism or cross-GPU pixel equivalence.

The [supervisor closure](../benchmarks/gcp-klein-2026-09-06/retry-g/trial/closure.json) verifies terminal creation, service deletion, and absence after **225.234813 seconds**, with no unresolved late creation. Billing reconciliation remains required. Keep the existing phase allowances until attributable costs are reconciled; the completed render does not itself settle charges. User visual review, five controlled cold starts, and physical end-to-end qualification remain outstanding.

For offline reproduction, use the source commit recorded in [G's execution receipt](../benchmarks/gcp-klein-2026-09-06/retry-g/execution-source.json), `3aadaa8d1ebc830a0d96e75f8bcab4649773dc24`, with the retained G manifest, journal, and all 20 JPEGs copied into the same relative evidence directory. Run the following without a live endpoint; it writes a separate recomputed summary:

```sh
python scripts/benchmark_gcp_klein.py --aggregate-only \
  --manifest benchmarks/gcp-klein-2026-09-06/retry-g/manifest.json \
  --proof-manifest-sha256 d1b37a6842f393a30ce05d6d884881a0dc5a3d80f7f44d8d4285f6d2e78d3a07 \
  --output benchmarks/gcp-klein-2026-09-06/retry-g/trial/renders \
  --service bookforge-klein-qualification-20260906-g \
  --revision bookforge-klein-qualification-20260906-g-trial
```

The expected summary SHA-256 is `2bc57f313665597c05cadc080e5d2f3756b66c29d1c4e96f71f23cd922412cff`; the journal SHA-256 is `8e97ca7b73973b3f4c6e2299810f9d92c117036da09c0f7baa9da18409e9de44`. Historical source selection matters because aggregation checks its supporting code hashes.

## G visual closeout and unsettled costs

The [assistant's inspection](../benchmarks/gcp-klein-2026-09-06/retry-g/visual-review-agent.json) covered all 20 original JPEGs with prompts and case labels visible; it was neither blind nor a human acceptance review. **Seven of eight unique scenes passed the frozen rubric.** The single-silver-fox case (`retained-1`) failed because an additional upper-right fox carries a second blue lantern. The same defect appears in its repeat. All six original watercolor cases passed their frozen fact checks, and all ten rendered scenes passed the assistant's watercolor-style check.

The two-boat scene and child-with-book scene also contain unrequested theater curtains. The inspection records those additions separately because the frozen rubric did not prohibit curtains; it does not quietly redefine that rubric after seeing the images. The duplicated fox is already a required-fact failure. Human review remains pending, and neither style success nor depth alignment repairs incorrect scene content. **No promotion is recommended.**

The [same-worker repeat comparison](../benchmarks/gcp-klein-2026-09-06/retry-g/same_gpu_repeat-comparison.json) reports 29.679 dB master PSNR and 33.865 dB depth PSNR between retained case 0's first and final images. Retained case 1's pair is byte-identical. These diagnostics leave every prior equivalence gate unchanged.

The phase retains **$4.8584904 in allowances**. The latest saved [project-cost notification](../benchmarks/gcp-klein-2026-09-06/retry-g/billing-after.json) reports **$21.55 at 20:51 UTC**, before G's successful GPU run. It is lagging project-wide data, not G's settled charge. Keep the separate build estimates, closed-service holds, image-retention obligation, and outstanding billing reconciliation visible.
