# Compiler-pool trial: setup failure before inference

The September 7 l trial produced no images and no compiler observations. It did not test whether `TORCHINDUCTOR_COMPILE_THREADS=1` improves startup. The retained k baseline remains 76.011 seconds for the first client request and 0.483 seconds median for the later eight requests. Production remains unchanged.

## Cause and correction

The trial manifest retained experiment ID j, which the successful k baseline had already used. The frozen qualification client exclusively creates a permanent local claim keyed by **experiment ID**, before creating its output directory or acquiring a token. A fresh Cloud Run service name did not create a fresh client attempt.

This was an orchestration error in our plan and review. The image verifier correctly checked the planned worker-and-expiry-only changes, but that plan omitted the persistent claim constraint. The [retained claim](../benchmarks/gcp-klein-compiler-pool-2026-09-07/prior-attempt-claim.json) identifies k, and an [offline reproduction](../benchmarks/gcp-klein-compiler-pool-2026-09-07/claim-reproduction.json) exercised the unchanged client with the same manifest and a copied claim: `FileExistsError`, zero token/client calls, no qualification directory, and unchanged claim bytes. Real claim files were not modified.

The [prospective admission helper](../experiments/renderer-compiler-pool/attempt.py) requires the manifest ID to match the new service and checks the downstream claim paths before paid preparation. Existing exclusive claims remain authoritative; they must not be removed to force a retry. The next manifest must use a fresh ID, and these checks must run before both build and deployment. The l image and evidence remain frozen.

## Actual execution

One CPU build, `fb51bb66-a615-413a-8d49-e6383dbf851a`, succeeded. The actual source archive contained six files totaling 37,469 bytes. Independent replay verified the inherited seventeen layers and the three-file overlay in image `32023196118a0faaf7097839437f6fe3c9983f4f6fa32bb8d23fa22a8e3e3a44`.

Service l was created at 19:34:41.561181 UTC. The collector rejected before qualification; its journal records a completed failure, with qualification, noise and cache results absent. The supervisor finished after 137.781 seconds with service deletion and absence verified. The terminal Service deletion was 19:36:53.960196 UTC. The retained runtime query returned lifecycle audit entries and no image-request or compiler records. That alone does not prove zero GPU allocation or zero charges.

The [closeout report](../benchmarks/gcp-klein-compiler-pool-2026-09-07/summary.json) binds the actual build, execution, failed collection, cleanup and release evidence. Release observation started at 19:40:01 UTC, about 187 seconds after deletion, using the unchanged 900-second guard. All fifteen metric responses omitted instance-count series, so the result is unknown. A fresh scoped audit and service GET at 19:55:46 UTC confirmed no later activity and a 404. The final status is `failed_phase_deleted_release_unproven`; missing metrics are not treated as zeros, available capacity or settled billing.

The approved gross allowance was $3.50. The conservative hold is $2.9518864, bringing cumulative unreconciled holds to $66.7444824. These are holds, not measured spend or an invoice. The one-build, one-service phase allowed no retry; no second paid attempt was launched. The [recovery proposal](../benchmarks/gcp-klein-compiler-pool-2026-09-07/recovery-plan.json) remains unfunded and requires a fresh ID plus both local and cloud admission before paid work.

## Verification

All 960 project Python tests and both JavaScript suites passed before execution. Focused checks cover image provenance, malformed pool evidence, rejected collections, cleanup bindings and the claim collision. Independent review covers the actual build and the failed-run report. The original renderer, compiler candidate and claim files remain unchanged.
