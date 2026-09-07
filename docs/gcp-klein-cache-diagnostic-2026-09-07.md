# GCP compiler diagnostic: connection failure before rendering

The approved $3.50 diagnostic built successfully but produced no image timings or compiler observations. Its service deployed through `us-central1-run.googleapis.com`; the next IAM operation stalled while connecting to `run.googleapis.com` and hit the supervisor's 30-second command deadline. The supervisor deleted the service and observed it absent after 60.68763 seconds. No collection request ran. This does not establish that no GPU was allocated or billed.

The build took 30.97 seconds. Independent source and OCI replays verified all six uploaded files, all 16 inherited image layers, and a 9,848-byte diagnostic overlay containing only the worker, observer, and expiring manifest. The unchanged parent supplies the model, compiler cache, overlap loader, BF16 precision, 1024×576 output, four sampling steps, and restored watercolor profile. Production was not changed.

The first build submission stalled before upload; a subsequent Cloud Build query confirmed no build existed before recovering that submission. Registry verification and later monitoring reads encountered the same address-selection problem. Process-local routes obtained through Google DNS restored connectivity with original-hostname certificate verification. Failed connection attempts remain in the evidence directory; none is presented as model latency.

## Release evidence

The terminal Service deletion audit is `2026-09-07T16:53:28.021764Z`. The original release observation had a 900-second bound. Its recovered monitoring reads returned `{"unit":"1"}` without instance-count series. Missing samples are not zero samples, so the frozen historical-zero release qualification cannot pass on this evidence. Deletion, service absence, GPU quota availability, and settled billing remain separate claims.

The observation stopped at its original deadline. All eleven recovered reads replay as unknown. The final exact-service audit returned HTTP 200 with no post-deletion mutation, and the service GET returned HTTP 404/`NOT_FOUND` at `17:10:27.839401Z`. The closeout is `failed_phase_deleted_release_unproven`; no admission, quota, billing, or promotion flag is set. Failed connection reads and recovered raw reads are preserved separately, with no synthetic release summary written into either directory.

Google documents instance count as a gauge sampled every 60 seconds, with up to 120 seconds of visibility delay; it does not define missing series as zero. [Cloud Monitoring metrics](https://docs.cloud.google.com/monitoring/api/metrics_gcp_p_z). Deleting a Cloud Run service deletes its revisions, and the service stays listed until deletion completes. That does not promise immediate GPU quota return or reconciled charges. [Service deletion](https://docs.cloud.google.com/run/docs/managing/services#delete).

## Connection fix and next measurement

The new [transport preflight](../experiments/renderer-gcp-transport/README.md) checks all ten required Google endpoints, including global and regional Run, token minting, monitoring, and cleanup. Its first live run verified all ten in 12.23 seconds; OAuth, registry, global Run, IAM credentials, and monitoring required Google DNS alternatives. Tests cover certificate hostname preservation, nested child inheritance, refusal of invalid or stale routes, and bounded local termination. This fixes an experiment-path gap; it is not an image-generation speed improvement.

The next diagnostic can reuse the verified image while its embedded manifest remains valid. It needs a separately reviewed finite plan because the approved attempt allowed one service and no retry. That plan must state any admission based on service absence despite unavailable instance metrics, preserve previous unreconciled holds, and retain the same rendering workload. Compiler decisions still await actual counter and timer observations. [Conditional optimization research](../experiments/renderer-cache-diagnostics/followup-research.md).

Evidence: `benchmarks/gcp-klein-cache-diagnostics-2026-09-07/`. The approved gross allowance was $3.50; the reserved hold was $2.9518864, bringing unreconciled holds to $60.8407096. These are experimental reservations, not an invoice or a platform spending cap.
