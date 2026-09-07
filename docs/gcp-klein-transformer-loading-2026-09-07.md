# Native GCP transformer loading triage — September 7, 2026

At the execution receipt frozen at **02:58:51 UTC**, the common packed image and both application overlays were verified; GPU qualification had not begun. There is no measured loading improvement or CUDA equivalence result for this phase yet. The [frozen plan](../benchmarks/gcp-klein-transformer-loading-2026-09-07/plan.json) and [execution receipt](../benchmarks/gcp-klein-transformer-loading-2026-09-07/execution-receipt.json) bind the prospective runs, source files, manifests, builds and actual image digests.

The comparison is one baseline service followed by one candidate service in `your-gcp-project`, `us-central1`. Both use RTX PRO 6000 Blackwell, 20 CPUs and 80 GiB RAM, with minimum zero and maximum one instance. Each receives the unchanged ten-request public watercolor schedule: two retained references, six broader synthetic scenes, then the retained pair again. There is no health check or prewarm generation before the first request, and no retry or fallback.

The baseline uses the original runtime. The candidate constructs only the transformer on the meta device, assigns the verified FlashPack tensors directly to CUDA without casting, explicitly sets evaluation mode, and supplies that object to the original pipeline loader. Qwen, VAE, depth, BF16 image generation, four steps, dimensions, render and compile methods remain unchanged. All candidate inspection, imports, construction and assignment remain inside the enclosing worker factory timer. The [runtime source proof](../experiments/renderer-transformer-flashpack/runtime/source-proof.json) records the limited constructor change.

The [CPU producer proof](../benchmarks/gcp-klein-transformer-pack-2026-09-07/proof.json) established exact source → pack → assigned CPU bytes for all 169 transformer tensors. That is preparation evidence, not proof of CUDA assignment or rendered-image equivalence. The candidate performs bounded metadata checks at startup; it deliberately avoids reading and hashing the entire 7.75 GB pack before loading it. Deployment integrity relies on the verified immutable image and producer proof.

Both arms inherit the same packed base, including the original checkpoints and the extra pack. Baseline does not load the pack before its ten renders. This controls image contents and size more closely than placing the pack only in the candidate image; it does not control physical host caches or image streaming state.

| Verified artifact | Immutable SHA-256 digest | New compressed layer |
| --- | --- | ---: |
| Common packed base | `42acfcdbb13dde3af9864afe7c5b1a0b8c1740b6eb39adea7abae4dbc4321eda` | 6,087,237,044 bytes |
| Baseline overlay | `ced765e732c00a4843e34f84a7db400ec7e6da6e99fb9b772e5f036302857033` | 11,799 bytes |
| Candidate overlay | `3a8c47aa8ceba871521c37a033392bd0c3431ef4a07d8516512cee08684d2028` | 11,827 bytes |

The [common verification](../benchmarks/gcp-klein-packed-verify-v2-2026-09-07/results/verification.json) checked compressed digest, decompressed diffID and all four added file hashes while preserving the original 13 layers. The [baseline](../benchmarks/gcp-klein-transformer-loading-2026-09-07/baseline-image-verification/verification.json) and [candidate](../benchmarks/gcp-klein-transformer-loading-2026-09-07/candidate-image-verification/verification.json) checks preserved all 14 common layers and config and verified the six exact overlay files. The full pack stayed in GCP. The overlay checks downloaded no inherited layer bytes; that does not describe what Cloud Run may later transfer to start a worker. Both overlay builds reported `SUCCESS` in their retained build JSON.

After each service completes ten verified renders, one finite oracle hashes all 169 actual resident CUDA parameters in bounded CPU chunks. It binds the instance, manifest, producer proof and ordered ten-render receipt digest. It generates no additional images. Both oracles must pass before tensor correctness is reported. Master/depth JPEG equality counts and human visual review remain separate; equal model tensors do not guarantee equal generated images.

The offline reporter retains first-client artifact-ready time, whole factory time, compile-wrapper time, first 128/256-bucket timings and the median of the later eight client times. Prospective screening thresholds are at least 15% lower factory time, at least 10% lower first-client time and no more than 5% regression in the later-eight median. Meeting them would justify a separately funded counterbalanced repeat, not promotion. Two sequential fresh services provide neither a balanced ABBA comparison nor a cold-start reliability distribution, p95, host-cold proof or physical-display latency.

The separate gross allowance is **$6.50**: $5.5237728 for conservatively reserved lifecycle/release capacity, $0.34 for two bounded builds, $0.05 for small storage/network operations and $0.5862272 margin. Prior phase holds of **$28.5099592 remain unreleased**. These are prospective allowances, not invoices or a platform-enforced spending cap. Each service has a 600-second work deadline including the oracle, 60 seconds for cleanup and up to 900 seconds for release observation; the budget allows two GPU slots per service despite the configured maximum of one.

Fresh historical-R bootstrap evidence must replay before baseline creation. Successful baseline deletion/release evidence must replay before candidate creation. Each closure must follow that service's exact terminal deletion audit event. Missing or uncertain admission stops the phase. Historical zero samples and current service absence do not prove current GPU count, reserve quota or settle billing. The reporter requires pinned deployment metadata and replayable bootstrap plus both release directories before reporting closure evidence; absent receipts remain incomplete.

The retained [bootstrap summary](../benchmarks/gcp-klein-transformer-loading-2026-09-07/bootstrap/summary.json) subsequently reported `historical_zero_and_current_absence` at **03:00:11 UTC**; this is first-service admission evidence with the limitations above. The [pre-GPU billing refresh](../benchmarks/gcp-klein-transformer-loading-2026-09-07/preflight/billing-before-gpu.json) contains a **$23.58 gross project** report timestamped **02:43:31 UTC**. It is lagging project-wide information, not this phase's invoice or a release of prior holds. GPU results remain pending at this preparation checkpoint.

## Baseline result

The baseline service was created at **03:01:44.912652 UTC**, after bootstrap completion, and reported the pinned `bookforge-klein-qualification-20260907-a-trial` revision. Its [retained summary](../benchmarks/gcp-klein-transformer-loading-2026-09-07/baseline-trial/renders/summary.json) contains **10 successful renders, 20 verified JPEG artifacts and one worker**. The post-render CUDA oracle matched **169/169** parameter hashes to the producer proof in **4.274 seconds**, binding the same worker, manifest and ordered render receipts. The client exited successfully. These results establish baseline tensor integrity; candidate assignment remains untested.

| Baseline measurement | Seconds |
| --- | ---: |
| First client artifact ready | 90.880 |
| Whole worker factory load | 61.100 |
| Compile-wrapper setup | 2.385 |
| First 128-bucket image generation | 19.569 |
| First 256-bucket image generation | 6.584 |
| First 256-bucket client artifact ready | 6.886 |
| Later eight client requests, median | 0.478 |

Factory, compile setup and first generation are distinct stages; first-client timing includes their work and response handling but excludes prior deployment and token acquisition. The 256-bucket request primes that bucket despite reusing the process. The first two retained images are not byte-identical to the historical references, so successful tensor checks must not be presented as image equivalence or visual acceptance.

The [supervisor closure](../benchmarks/gcp-klein-transformer-loading-2026-09-07/baseline-trial/closure.json) reports terminal creation observed, service deleted and absent, no cleanup error, and **392.115 seconds** for its full deployment/work/cleanup run. Service absence does not prove zero remaining GPU charges. Prospective release evidence is still required before candidate admission. Candidate timing, paired image comparison, both-arm correctness and final phase closure remain pending; no speed gain is claimed.
