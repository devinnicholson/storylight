# Native GCP compiler-cache comparison — September 6, 2026

This is a **separate $6 maximum phase** to test whether a compiler cache produced on the native RTX worker reduces first-request and first-bucket latency. It has not run. Default model loading remains in place: the [eager-loading comparison](gcp-klein-loading-2026-09-06.md) failed its frozen speed and strict image-equivalence gates. Production routing, model weights, precision, compiler settings, prompts, seeds, and the watercolor target remain unchanged.

PyTorch provides APIs to save compilation artifacts and populate caches in another process. The pinned 2.8.0 implementation deserializes the supplied artifact and populates its component caches; a non-null return reports cache population, not proof that a subsequent graph matched or compilation was avoided. The official tutorial describes the intended cross-process use and the importance of matching PyTorch, Triton, and GPU. Measure the actual first executions rather than inferring a speedup from successful restoration. [PyTorch 2.8 API implementation](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/compiler/__init__.py#L440-L469), [cache implementation](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/compiler/_cache.py#L258-L284), [official caching tutorial](https://docs.pytorch.org/tutorials/recipes/torch_compile_caching_tutorial.html).

## Four services and one trusted producer

Use the fixed order **L, M, N, O**: baseline without imported cache, cache consumer, cache consumer, baseline without imported cache. L and O use the same producer-capable image, but only L exports a cache. Each service is fresh, private, and receives the same ten POSTs: retained cases 0 and 1, six original watercolor cases, then retained cases 0 and 1 again. The completed comparison contains **40 requests, 80 JPEGs, and four distinct workers**. No health request, prewarm, automatic POST retry, or replacement sample is included.

All services use `us-central1`, one RTX PRO 6000 Blackwell Server Edition, 20 vCPUs, 80 GiB RAM, concurrency one, minimum zero, and both service and revision maximum one. Keep CPU boost off, no GPU zonal redundancy, and one revision receiving all traffic. Verify resource metadata and the exact unconditional private invoker grant. The 120-second IAM propagation wait is inside the work deadline and does not guarantee propagation.

Allow **600 seconds for all work from deployment start, plus 60 seconds for cleanup**, for each service. Delete and verify closure before starting the next. An ambiguous create operation requires separate terminal-operation evidence; service absence by itself is insufficient. Stop on request, integrity, export, or unresolved cleanup failure, and retain incomplete runs. Request timeout alone does not stop the server's work. [Cloud Run timeout behavior](https://docs.cloud.google.com/run/docs/configuring/request-timeout).

After L's ten responses and all 20 JPEGs verify, the exporter makes **one authenticated cache GET** within L's original deadline. It has a 30-second local deadline covering token acquisition and transport, no redirects or retries, and a fresh retained attempt record. The worker permits one serialization attempt only after ten successful receipts and completed 128- and 256-token buckets; instance ID and manifest SHA must match. No extra rendering occurs for export.

The transfer is bounded to **32 MiB of compiler artifacts and 8 KiB of canonical metadata**, with a fixed envelope. The host independently reconstructs the ten ordered receipts from the verified journal and checks their digest, producer instance/service/revision, manifest, runtime identity, buckets, artifact length, and SHA. Cache bytes are executable compiler material from this specific trusted producer, not arbitrary user-uploaded input. Preserve them and their exact provenance for the consumer build; do not publish credentials or accept an unverified replacement artifact.

The consumer image adds only that verified artifact and metadata, plus the two literal digest pins in the isolated worker copy. Preserve the default runtime source and every other worker statement. Verify the exact source difference, immutable image layer prefix, runtime configuration, file contents, and ownership before M starts. Both consumers use this same image. They validate the pinned metadata, artifact, and runtime identity before model construction, then restore the cache after loading and before the existing `compile(None)` call. A producer must reject a mounted cache; a consumer must reject missing, changed, or incompatible bytes.

Build the producer overlay before L and the consumer overlay only after L's export verifies. Each uses the existing immutable model image and the pinned BuildKit method, with no model rebuild or new model download. Refresh manifest expiry through an explicit frozen build context; do not silently extend a baked manifest. The producer manifest must cover both L and the final O run, including the intervening consumer build.

## Frozen measurements and gates

Report all four first-client artifact-ready times, inclusive model-load and compile-wrapper times, first image-generation time for each token bucket, later-request worker medians, cache-export duration, and cache-restore duration. Preserve deployment, IAM wait, transport, cleanup, and build timings separately. The compile-wrapper timer excludes deferred work during first execution, and cache-population logs do not prove graph hits. No new component-level model-load or host-RSS measurement is claimed.

Pair **L with M** and **O with N**. The speed screen requires all of the following:

- All 40 requests and 80 images verify, each service uses one distinct worker, and all four closures verify.
- Cache consumers reduce median first-client artifact-ready time by **at least 15%** across the two variants.
- In **each pair**, both the first 128-token and first 256-token `image_seconds` improve by **at least 30%**: four separate bucket comparisons must pass.
- The candidate's median of the two later-worker medians regresses by **no more than 5%**. Use the same eight nonpriming rows per worker as the prior comparison.

These thresholds test whether reduced first-execution work reaches the user-visible request time. They are prospective screening criteria, not a statistical reliability claim. Two workers per variant cannot establish a p95 or general cold-start guarantee. Fresh services do not control underlying host, storage, image-streaming, or driver cache state; the intervening consumer build adds an order gap that must remain visible.

Keep **speed, strict image equivalence, and visual quality as separate verdicts**. Report all 40 same-ordinal master/depth comparisons between paired services, matching first occurrences to first occurrences and final repeats to final repeats. Any nonexact pair is a failed strict-equivalence check, with both original files retained. Known baseline nondeterminism does not turn a mismatch into a pass or justify dropping a case. A speed-screen pass with nonexact images is only a candidate for further investigation.

No result automatically promotes the renderer. The existing single-fox duplication failure, frozen scene facts, watercolor appearance, human visual acceptance, repeated cold reliability, and physical Jetson-to-projector demonstration remain separate requirements. Human quality is pending, even if every transport and timing gate passes.

## Separate funding

Prior qualification and loading allowances remain unchanged. This phase has its own **$6 gross cap**, shared across both builds and all four services, not a separate allowance for each service.

| Component | Allowance |
| --- | ---: |
| Four GPU lifecycles: two resource slots × 660 seconds each | $4.6739616 |
| Two overlay builds: eight CPUs, 100 GB disk, 600 seconds each | $0.34 |
| Incremental overlay and cache storage, at most seven days | $0.02 |
| Remaining margin | $0.9660384 |
| **New gross phase cap** | **$6.00** |

The GPU calculation retains $0.00088522 per second for one RTX GPU, 20 vCPUs, and 80 GiB. Two slots provide allowance for transient platform overlap; configured maximum instances are not an absolute billing cap. Each ten-minute CPU build costs $0.156 at the published rate, rounded to $0.17; the first 100 GB disk is included. Existing shared model layers remain attributed to prior retention allowances, while this phase stores only small overlays and the bounded cache. [Cloud Run pricing](https://cloud.google.com/run/pricing), [maximum-instance limits](https://docs.cloud.google.com/run/docs/configuring/max-instances), [Cloud Build pricing](https://cloud.google.com/build/pricing), [Artifact Registry storage](https://cloud.google.com/artifact-registry/pricing).

Refresh billing before paid dispatch and at closeout. Retain every attempted build and service, source/image/manifest pins, export attempt, artifact hashes, request journal, result image, closure, and attributable cost observation. Report estimates and unsettled charges separately. Unused margin does not automatically authorize another build, service, export, or rendering attempt.
