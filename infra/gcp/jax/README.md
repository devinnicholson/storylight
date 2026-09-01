# Guarded JAX CustomJob

`job_plan.py` is a pure planner. It emits one `ct6e-standard-1t` worker, a 2,700-second timeout,
disabled retries, separate scratch and release locations, and an exact approval token. Planning
does not authenticate or contact Google Cloud.

The listed gross ceiling is a planning estimate and admission guard, not a provider-enforced
spend cap. The job timeout and no-retry policy bound resource duration; the final provider billing
report must still be reconciled before another paid attempt.

`stage_inputs.py` creates the only accepted input population. It binds the configuration,
dataset manifest, prepared examples, base-checkpoint manifest and bytes, and tokenizer manifest
and bytes under one run-specific checksum, then uploads `inputs.manifest.json` last with
generation-match zero. The worker rejects undeclared objects, missing objects, or any hash drift.
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

The worker service account receives the custom role in `least-privilege-role.yaml` only on the
two private buckets and the one read-only Hugging Face secret. The launcher receives Vertex job
creation plus `iam.serviceAccounts.actAs`; neither account receives Owner or Editor. Buckets must
enable uniform access and public-access prevention. Scratch expires after seven days; releases
have no automatic deletion.
The worker receives a numeric Secret Manager version such as `/versions/1`; `latest` is rejected.

`vertex_entrypoint.py` uploads artifacts with generation-match zero and writes `completion.json`
last. The release includes portable training run/completion evidence, the adapter manifest and
bytes, and the exact runtime lock. No model endpoint, autoscaler, persistent accelerator, or
retrying service is created.
Success additionally requires the training run's nonempty terminal evidence and a nonempty output
artifact manifest; process exit zero alone is insufficient.

Modal training and TensorRT export append pre/post billing-report observations to the local
reconciliation ledger. These observations do not claim that the declared ceiling is provider
enforced, and neither workflow automatically deletes remote release, scratch, export, or intent
state. `fetch_modal_jax_release.py` retrieves either a portable training release or a checksummed
ONNX export laid out for `build-trained-planner-candidate.sh` without changing remote state.
When its export mode also receives `--candidate-output`, it runs that Jetson builder and then the
candidate installer's `--verify-only` path, yielding an installable bundle plus its manifest hash.
