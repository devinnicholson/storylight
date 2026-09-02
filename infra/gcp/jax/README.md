# Guarded JAX CustomJob

`job_plan.py` is a pure planner. It emits one `ct6e-standard-1t` worker, a 2,700-second timeout,
disabled retries, separate scratch and release locations, and an exact approval token. Planning
does not authenticate or contact Google Cloud.

The listed gross ceiling is a planning estimate and admission guard, not a provider-enforced
spend cap. The job timeout and no-retry policy bound resource duration; the final provider billing
report must still be reconciled before another paid attempt.

`stage_inputs.py` creates the only accepted full-training input population. Before upload it
requires a checksum-bound MaxText Orbax receipt with role `base-maxtext` and step `0`, resolves
that exact `items` leaf, and verifies the leaf against its artifact manifest. It binds the
receipt, manifest, configuration, dataset manifest, prepared examples, base-checkpoint bytes,
and tokenizer manifest and bytes under one run-specific checksum, then uploads
`inputs.manifest.json` last with generation-match zero. The Vertex and Modal workers independently
repeat the receipt identity, leaf-manifest, and byte checks before passing the resolved leaf to
MaxText; an unverified checkpoint root is never used for paid full training. The worker rejects
undeclared objects, missing objects, or any hash drift.
For a `-v2` experiment, staging additionally requires `--preparation-manifest` and
`--prepared-validation`. Both files are uploaded under `prepared/`, and the input manifest binds
their hashes, the preparation policy, record count, prompt contract, source split, prepared data,
and tokenizer. The staging verifier repeats those checks immediately before any provider upload.
`stage_modal_inputs.py` builds the identical population for the fixed private Modal input volume.
It first rejects an existing run prefix, uploads each declared input without `--force`, uploads the
manifest last, and reads that manifest back before recording success.

Before submission, the integration owner must separately verify the active project is
`your-gcp-project`, billing is enabled, promotional credits are present, the display name has
never been submitted, the container URI is digest-pinned, and the exact printed token was
approved. A submission intent must be recorded before the single create request; an ambiguous
request is terminal and cannot fall back to Modal under the same run ID.
An ordinary preflight refusal is written once as `*.preflight-rejection.json`; Modal fallback
accepts that checksum-bound evidence only when no submission intent and no CustomJob were created.
The refusal includes the original Vertex spec hash and the exact staged-input binding hash, so a
refusal for one plan cannot authorize a different Modal input population.
A refusal caused by an already-existing CustomJob is explicitly marked fallback-ineligible, even
after billing reconciliation, so a completed Vertex attempt can never authorize a second Modal
spend for the same run ID.

If billing is disabled before a runnable training image exists,
`record_unavailable_fallback.py` provides a separate, truthful pre-build path. It consumes the
content-addressed image *build plan*, not a fabricated image digest or incomplete CustomJob plan.
The recorder verifies the active project, explicit `billingEnabled: false`, a run date no more
than seven days old, and an empty exact-name `CreateCustomJob` Admin Activity query over the full
400-day retention window. It also refuses any local build intent or receipt for the planned image
source. The write-once rejection binds all seven staged-input hashes, the unbuilt image source and
build-plan hashes, and the intended Vertex machine, account, storage prefixes, timeout, and retry
policy. Its tagged image target is explicitly non-runnable: `runnable_digest_uri` remains null and
no `spec_sha256` is claimed because no complete Vertex submission specification exists.
Modal accepts this distinct producer only while the evidence is fresh and every billing, audit,
image, input, and intended-resource binding is internally exact. This path records why GCP could
not be used; it does not submit an image build, create a CustomJob, or assert that billing-disabled
API errors prove the absence of historical resources.

The worker service account receives the custom role in `least-privilege-role.yaml` only on the
two private buckets. The launcher receives Vertex job creation plus
`iam.serviceAccounts.actAs`; neither account receives Owner, Editor, or Secret Manager access.
Buckets must enable uniform access and public-access prevention. Scratch expires after seven days;
releases have no automatic deletion. The worker receives only checksum-bound checkpoint and
tokenizer bytes and sets the Hugging Face, Transformers, and Datasets runtimes to offline mode;
no model token or secret resource is accepted by the job contract.

`vertex_entrypoint.py` uploads artifacts with generation-match zero and writes `completion.json`
last. The release includes portable training run/completion evidence, the adapter manifest and
bytes, and the exact runtime lock. No model endpoint, autoscaler, persistent accelerator, or
retrying service is created.
Success additionally requires the training run's nonempty terminal evidence and a nonempty output
artifact manifest; process exit zero alone is insufficient.

## Image and admission chain

`build_image_plan.py` hashes every tracked regular file in the Docker build context and derives a
content-addressed tag. It refuses an unpinned Docker builder and the training Dockerfile refuses an
unpinned base image. `image-cloudbuild.yaml` is a build recipe, not a deployment: after the single
approved build, resolve the Artifact Registry digest and pass only
`.../trainer:<source-prefix>@sha256:<digest>` to `job_plan.py`. A mutable tag is never accepted by
the Vertex plan. Before upload, `--materialize-context` copies only the plan's checksum-bound files;
the emitted build command's `{MATERIALIZED_CONTEXT}` placeholder must be replaced with that new
directory. This prevents ignored or untracked local output from entering Docker's `COPY .`. The
Artifact Registry repository and a verified, digest-pinned Cloud Build Docker
builder are external prerequisites; this directory does not create either one.
`submit_image_build.py` is the only repository execution path for that paid command. It re-verifies
the external context, requires the exact one-purpose approval token, records a write-once intent
before invoking Cloud Build, disables retries, and refuses any source hash that already has intent
or receipt state. An ambiguous build therefore remains terminal instead of silently spending twice.

`cloud_preflight.py` performs only `gcloud` describe/list operations. It checks the active account
and project, billing link, required APIs, absent CustomJob ID, private bucket policies and distinct
lifecycle behavior, and regional TPU quota. Promotional credit balances are not exposed by a
reliable public quota API, so the script requires a run-bound manual attestation instead of
pretending billing enabled means credits exist. The resulting evidence binds the exact job-spec
and input-binding hashes. `submit_vertex_job.py` refuses evidence older than 15 minutes or evidence
for any other plan before it records submission intent.
Every recorded submission intent globally blocks another Vertex attempt until
`reconcile_vertex_attempt.py` validates a terminal/not-found job observation and internally
consistent final provider-cost evidence. The reconciliation is write-once and never authorizes a
retry of the same run. An ambiguous create response therefore cannot lead to a second spend.

`fetch_gcs_release.py` requires the expected SHA-256 of `completion.json`, generation-matches every
GCS download, rejects extra objects, verifies every declared byte, and validates the same portable
training package contract as the Modal bridge. It performs no deletion. The trusted checksum must
come from terminal job evidence or a separately authenticated operator channel, never from the
bucket listing being verified.

Modal training and TensorRT export append pre/post billing-report observations to the local
reconciliation ledger. These observations do not claim that the declared ceiling is provider
enforced, and neither workflow automatically deletes remote release, scratch, export, or intent
state. `fetch_modal_jax_release.py` retrieves either a portable training release or a checksummed
ONNX export laid out for `build-trained-planner-candidate.sh` without changing remote state.
When its export mode also receives `--candidate-output`, it runs that Jetson builder and then the
candidate installer's `--verify-only` path, yielding an installable bundle plus its manifest hash.
The Modal ledger atomically reserves the attempt before the remote call; duplicate or unresolved
attempts are rejected before another GPU can start, rather than being discovered during post-run
reconciliation.
Exact raw Modal billing reports are retained locally in a ledger-adjacent ignored directory and
revalidated before another paid call. They are deliberately not committed because a workspace
report can include unrelated application names and costs; a fresh checkout therefore fails closed
until an authorized operator restores that private evidence.
