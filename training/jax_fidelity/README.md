# Bookforge JAX Story Fidelity package

This directory packages the offline Gemma 4 E2B LoRA experiment. It is not part
of the Jetson runtime. Importing it does not import JAX, download a model, start
cloud resources, or execute a MaxText command.

## Frozen behavior

- The base is `google/gemma-4-E2B-it` at revision
  `3e22461f65e89153144f8adb70e3b8c2cc9845a7`.
- MaxText is release 0.2.4 at Git revision
  `538fe7a3f3376d94cf3f04e77741aa6d7e8efa45`.
- The formatter calls Bookforge's deployed `_slot_messages` function directly.
- Targets are exactly four non-empty lines in `SETTING`, `ACTOR`, `ACTION`,
  `MAGIC` order.
- SFT uses ordinary LoRA with MaxText's completion-only prompt masking. Contrast
  examples live inside the masked system prompt, leaving exactly one supervised
  assistant turn: the record's canonical target. QLoRA is disabled.
- Gemma 4 E2B is text-only with `scan_layers=False` and
  `use_multimodal=False` through conversion, training, and validation.
- Conversion must preserve EOS IDs `[1, 106, 50]` and produce forward KL
  divergence no greater than 0.03.

## Local, no-spend checks

```bash
python -m training.jax_fidelity \
  --config experiments/jax-fidelity-lab/config.json

python -m training.jax_fidelity.roundtrip_smoke \
  --config experiments/jax-fidelity-lab/config.json \
  --print-contract
```

Prepare training data only after independently approving the dataset manifest
hash:

```bash
python -m training.jax_fidelity.prepare \
  --dataset-manifest datasets/story-fidelity-v1/manifest.json \
  --dataset-manifest-sha256 APPROVED_SHA256 \
  --output artifacts/jax-fidelity/prepared-train.jsonl
```

Download the base model either through an injected read-only Hugging Face token
or, when the pinned revision is still verified public and ungated, through the
explicit anonymous mode. The access mode is bound into the one-purpose approval
token. The command is plan-only until `--execute` and the printed exact
`BOOKFORGE_JAX_EXECUTION_APPROVAL` value are both supplied:

```bash
python -m training.jax_fidelity.hf_snapshot \
  --config experiments/jax-fidelity-lab/config.json \
  --snapshot artifacts/jax-fidelity/hf-base \
  --tokenizer artifacts/jax-fidelity/tokenizer \
  --snapshot-manifest artifacts/jax-fidelity/hf-base.manifest.json \
  --tokenizer-manifest artifacts/jax-fidelity/tokenizer.manifest.json \
  --completion artifacts/jax-fidelity/hf-snapshot.completion.json \
  --access-mode public-anonymous
```

`artifact_contract.py` builds the named, checksum-bound inputs for conversion.
`checkpoint_evidence.py` and `roundtrip_evidence.py` derive architecture,
tokenizer, PLE, KV-sharing, EOS, and measured MaxText KL evidence from the real
artifacts and terminal completion receipts; a converter exit code is never
treated as compatibility proof by itself.

The finite Modal compatibility worker performs HF-to-MaxText conversion, an
exact five-step LoRA smoke, merged-HF export, and the forward-KL gate in one
no-retry, two-L4 call. Its staging command includes the public train and
development files so the remote worker can independently validate the dataset
manifest; private hidden records are never staged.

```bash
python infra/gcp/jax/stage_roundtrip_inputs.py \
  --run-id RUN_ID \
  --config experiments/jax-fidelity-lab/config.json \
  --dataset-manifest datasets/story-fidelity-v1/manifest.json \
  --prepared-train PREPARED_TRAIN_JSONL \
  --hf-snapshot HF_SNAPSHOT \
  --hf-snapshot-manifest HF_SNAPSHOT_MANIFEST \
  --tokenizer TOKENIZER_DIRECTORY \
  --tokenizer-manifest TOKENIZER_MANIFEST \
  --output ROUNDTRIP_STAGING_PLAN

modal run deploy/modal_jax_roundtrip.py --help
```

Both commands are non-executing until their printed, checksum-bound approval
values are supplied to the documented environment/CLI boundary. A successful
worker records the only valid step-0 base Orbax `items` leaf and step-4 LoRA
`items` leaf, publishes their complete byte manifests, and writes
`completion.json` last. Fetching is separately checksum-gated:

```bash
python scripts/fetch_modal_jax_roundtrip.py \
  --run-id RUN_ID \
  --expected-completion-sha256 TRUSTED_COMPLETION_SHA256 \
  --destination ROUNDTRIP_RELEASE
```

`train`, `convert`, and `evaluate` print their complete command and a
one-purpose approval token by default. They execute only with `--execute` and
an exact `BOOKFORGE_JAX_EXECUTION_APPROVAL` environment value. MaxText commands
also require a clean checkout at the pinned commit. Commands are passed as
argument arrays; they are never evaluated by a shell.

Conversion commands also require `--run-directory`. Their stable run ID binds
the config and input-manifest hashes, output directories are never overwritten,
and `completion.json` is written only after the converted bytes or logit check
finish successfully.

Cloud workers package training output as `adapter/`, `adapter.manifest.json`,
`training/run.json`, `training/completion.json`, and `runtime.lock.json` before
publishing the provider completion receipt. Artifact paths in the packaged
training completion are relative to `adapter/`, so a checksummed fetch remains
valid on a different machine. The campaign `record-training` boundary accepts
the successful roundtrip's step-0 `base-maxtext` Orbax root, receipt, and leaf
manifest—not the pre-conversion Hugging Face snapshot. It requires explicit
trusted SHA-256 values for that receipt, manifest, and the remote completion;
re-verifies the leaf bytes; retains the tokenizer-manifest binding; and records
the base content digest in the typed training-stage evidence.

The Modal full trainer reserves the final 600 seconds of its one-hour function
deadline for durability and publication. Immediately after successful training,
it commits the complete scratch checkpoint and terminal training receipt. It
then publishes release data in one commit and `completion.json` in a second
commit. If inline publication cannot finish, the same deployment exposes a
CPU-only recovery mode; it validates the exact input, attempt, two-GPU FSDP, and
training-completion hashes and cannot invoke training or overwrite conflicting
release bytes:

When the active Google Cloud project has billing disabled before its
digest-pinned JAX image can be built, produce the fallback rejection with
`infra/gcp/jax/record_unavailable_fallback.py`. The recorder consumes the exact
image build plan and intended Vertex resource without inventing an image digest.
It performs only project, billing, and exact-name `CreateCustomJob` Admin
Activity reads, then writes the evidence once. The Modal worker accepts this
producer only when billing is explicitly disabled, the dated run ID and 400-day
audit absence are fresh and exact, no image build or job intent exists, and all
input and intended-resource hashes match the Modal request. The image plan,
rejection, intended resource, approval token, and running Modal image must all
name the same packaged Bookforge source-manifest hash; legacy generic preflight
rejections do not authorize this fallback.

```bash
modal run deploy/modal_jax_fidelity.py --help
# Reuse the original arguments and add both options printed by the approved plan:
#   --finalize-only
#   --finalize-approval-token-value \
#   APPROVE_MODAL_JAX_FINALIZE:{run_id}:{input_manifest_sha256}:{bookforge_source_manifest_sha256}:{gcp_rejection_sha256}
```

Completion-less derivative staging is safe to rebuild from the retained
checkpoint. Completion-bearing staging is immutable. A recovery interrupted
after its first release commit resumes by verifying every existing byte and
adding only missing files before publishing completion last.

After the full adapter succeeds, `modal_jax_merge.py` is the only remote merge
boundary. The roundtrip run and its input manifest supply only the immutable
source-HF and step-0 base provenance. A separate `training_input_run_id` and
`training_input_manifest_sha256` identify the exact v2 config, dataset,
tokenizer, prepared population, and cloned base used by training. One exact
approval binds both input populations, the successful roundtrip completion,
the portable full-training completion and adapter manifest, and the training
run hashes. Merge re-verifies that the v2 base matches the roundtrip base and
that the training release names the v2 input-manifest hash before selecting the
single configured terminal LoRA `items` leaf. The finite L4 worker invokes
MaxText-to-HF once and writes a prediction-only `candidate.manifest.json`. It
has no retries or web endpoint, and `completion.json` is published last.

```bash
modal run deploy/modal_jax_merge.py --help

python scripts/fetch_modal_jax_merge.py \
  --merge-run-id MERGE_RUN_ID \
  --completion-sha256 TRUSTED_COMPLETION_SHA256 \
  --config experiments/jax-fidelity-lab/config-v2.json \
  --destination artifacts/jax-fidelity/merged-candidate \
  --execute
```

Prediction does not require downloading and re-uploading the merged checkpoint.
Fetch only the four small receipts needed to plan the handoff (these commands are
read-only), then build an exact local plan:

```bash
modal volume get bookforge-jax-fidelity-inputs \
  /ROUNDTRIP_RUN_ID/inputs.manifest.json /tmp/roundtrip-inputs.manifest.json
modal volume get bookforge-jax-fidelity-release \
  /merged/MERGE_RUN_ID/completion.json /tmp/merge-completion.json
modal volume get bookforge-jax-fidelity-release \
  /merged/MERGE_RUN_ID/candidate.manifest.json /tmp/candidate.manifest.json
modal volume get bookforge-jax-fidelity-release \
  /merged/MERGE_RUN_ID/merged-hf.manifest.json /tmp/merged-hf.manifest.json

python scripts/plan_modal_jax_prediction_handoff.py \
  --source-input-manifest /tmp/roundtrip-inputs.manifest.json \
  --merge-completion /tmp/merge-completion.json \
  --candidate-manifest /tmp/candidate.manifest.json \
  --checkpoint-manifest /tmp/merged-hf.manifest.json \
  --dataset-manifest datasets/story-fidelity-v1/manifest.json \
  --development-records datasets/story-fidelity-v1/development.jsonl \
  --prediction-run-id PREDICTION_RUN_ID \
  --output /tmp/prediction-handoff-plan.json
```

The planner is local and non-mutating. Inspect its hashes, references, and exact
approval values. The following is the only handoff mutation; it runs in one
CPU-only, no-retry container and publishes only
`prediction/PREDICTION_RUN_ID/inputs.manifest.json`. It copies no checkpoint
bytes:

```bash
modal run deploy/modal_jax_prediction_handoff.py \
  --source-run-id "$(jq -r .source_run_id /tmp/prediction-handoff-plan.json)" \
  --source-input-manifest-sha256 "$(jq -r .source_input_manifest_sha256 /tmp/prediction-handoff-plan.json)" \
  --merge-run-id "$(jq -r .merge_run_id /tmp/prediction-handoff-plan.json)" \
  --merge-completion-sha256 "$(jq -r .merge_completion_sha256 /tmp/prediction-handoff-plan.json)" \
  --prediction-run-id "$(jq -r .prediction_run_id /tmp/prediction-handoff-plan.json)" \
  --candidate-id "$(jq -r .candidate_id /tmp/prediction-handoff-plan.json)" \
  --candidate-manifest-sha256 "$(jq -r .candidate_manifest_sha256 /tmp/prediction-handoff-plan.json)" \
  --checkpoint-manifest-sha256 "$(jq -r .checkpoint_manifest_sha256 /tmp/prediction-handoff-plan.json)" \
  --checkpoint-content-sha256 "$(jq -r .checkpoint_content_sha256 /tmp/prediction-handoff-plan.json)" \
  --target-manifest-sha256 "$(jq -r .target_manifest_sha256 /tmp/prediction-handoff-plan.json)" \
  --approval-token-value "$(jq -r .approval_token /tmp/prediction-handoff-plan.json)"
```

The reference stager re-verifies the source input manifest, public 512-record
development population, L4 merge completion, candidate manifest, and every
merged checkpoint byte. It rejects hidden paths, symbolic links, legacy
backends, changed hashes, and existing output prefixes. The prediction worker
reloads both Modal volumes, rebuilds the same reference manifest, and reads the
checkpoint directly from `/releases/merged/MERGE_RUN_ID` while retaining the
original prediction approval contract:

```bash
BOOKFORGE_NVIDIA_PYTORCH_IMAGE='nvcr.io/nvidia/pytorch:TAG@sha256:DIGEST' \
modal run deploy/modal_jax_prediction.py \
  --run-id "$(jq -r .prediction_run_id /tmp/prediction-handoff-plan.json)" \
  --candidate-id "$(jq -r .target_manifest.bindings.candidate_id /tmp/prediction-handoff-plan.json)" \
  --config-sha256 "$(jq -r .target_manifest.bindings.config_sha256 /tmp/prediction-handoff-plan.json)" \
  --dataset-manifest-sha256 "$(jq -r .target_manifest.bindings.dataset_manifest_sha256 /tmp/prediction-handoff-plan.json)" \
  --development-records-sha256 "$(jq -r .target_manifest.bindings.development_records_sha256 /tmp/prediction-handoff-plan.json)" \
  --candidate-manifest-sha256 "$(jq -r .target_manifest.bindings.candidate_manifest_sha256 /tmp/prediction-handoff-plan.json)" \
  --checkpoint-manifest-sha256 "$(jq -r .target_manifest.bindings.checkpoint_manifest_sha256 /tmp/prediction-handoff-plan.json)" \
  --checkpoint-content-sha256 "$(jq -r .target_manifest.bindings.checkpoint_content_sha256 /tmp/prediction-handoff-plan.json)" \
  --input-manifest-sha256 "$(jq -r .target_manifest_sha256 /tmp/prediction-handoff-plan.json)" \
  --batch-size "$(jq -r .prediction_batch_size /tmp/prediction-handoff-plan.json)" \
  --approval-token-value "$(jq -r .prediction_approval_token /tmp/prediction-handoff-plan.json)"
```

The older fetched `merged-hf/` plus `candidate.manifest.json` staging path
remains accepted for offline recovery. Either path leaves the candidate
explicitly ineligible for release until development evaluation succeeds.

After development eligibility passes, `release.py` does not trust the fetched
directory by location alone. Its mandatory inputs include the trusted merge
completion SHA, the complete merge file table, the candidate and source-binding
receipts, all three MaxText-to-HF conversion receipts, the original HF
manifest, the step-0 base Orbax receipt/manifest, and the terminal full-adapter
receipt/manifest. It replays the conversion input contract against those exact
three source leaves, validates the merged-HF manifest against the produced
bytes, and copies every custody hash into `terminal_evidence` before writing the
immutable release manifest last.

The accepted-engine development baseline is recoverable without cloud compute.
`baseline_development.py` accepts either the original report and its trusted
SHA-256 or regenerates the report against the loopback-only accepted TensorRT
endpoint. Both paths re-check the accepted engine bytes, accepted identity
manifest, repository dataset manifest, complete 512-record development file,
privacy-only report schema, and the four headline metrics frozen before
training. It writes a read-only report and publishes `completion.json` last.
Headline signatures are keyed by the exact dataset-manifest SHA-256 in
`baseline_development.py` rather than shared across dataset versions. An
unregistered manifest fails closed even when its population shape matches a
registered dataset. The v1 signature remains frozen exactly as measured. The
accepted Jetson engine's v2 signature was measured before candidate prediction
on September 2, 2026 and is registered under manifest
`fd3317ef440a04c9adc41825dda2b58c03a52ae0829bd422750522b4e11d428e`:
schema-valid `1.0`, semantic recall `0.537109375`, exact pass `0.0`, and privacy
pass `1.0`.

On Jetson, first capture the accepted engine identity if a durable copy does not
already exist. This copies the engine locally; it does not upload it or change
the active route:

```bash
cd /opt/bookforge
deploy/jetson/emit-accepted-planner-identity.sh \
  --source-dataset-manifest-sha256 \
  e717eb38c44fceeeae3a2bc88981767c316ca1339198ce1077b893252afeb1de \
  --output "$HOME/bookforge-evidence/accepted-baseline-identity"
```

Then start the accepted TensorRT service on `127.0.0.1:11435`, compute the
printed identity-manifest SHA-256, and plan the free local regeneration. The
regenerator verifies that this service's main process names the accepted engine
directory before and after evaluation:

```bash
systemctl --user start bookforge-tensorrt-planner.service
/opt/bookforge/.venv/bin/python -m training.jax_fidelity.baseline_development \
  --dataset-manifest datasets/story-fidelity-v1/manifest.json \
  --dataset-manifest-sha256 \
  e717eb38c44fceeeae3a2bc88981767c316ca1339198ce1077b893252afeb1de \
  --development-records datasets/story-fidelity-v1/development.jsonl \
  --development-records-sha256 \
  5de3cfe3532b28e907816a6a77dfc45143de696f2b715890aef1460f60474533 \
  --accepted-manifest \
  "$HOME/bookforge-evidence/accepted-baseline-identity/candidate.manifest.json" \
  --accepted-manifest-sha256 ACCEPTED_MANIFEST_SHA256 \
  --accepted-engine \
  "$HOME/.local/share/bookforge/tensorrt-edgellm-v0.10.0/models/gemma4-e2b-it-int4-awq-v010/engines/llm/llm.engine" \
  --accepted-engine-sha256 \
  95b69991b68c57a2d2d4bfa4116feb9ec57295588551d109353a42a9c16c4fdf \
  --base-url http://127.0.0.1:11435 \
  --output-directory "$HOME/bookforge-evidence/baseline-development-e717eb38"
```

Set `BOOKFORGE_BASELINE_DEVELOPMENT_APPROVAL` to the exact token printed by the
plan and repeat the same command with `--execute`. If the original report is
recovered instead, add `--source-report PATH` and
`--source-report-sha256 SHA256`; those options preserve and verify its exact
bytes rather than calling the endpoint. Preserve the printed completion
SHA-256. `development_eligibility.py` requires the report and completion from
this same directory, their two trusted hashes, and the accepted manifest and
engine hashes. It rejects a loose report even when that report's checksum is
known. `completion.json` therefore binds the accepted engine, dataset,
development population, source, and report into the eligibility lineage.

The evaluator consumes predictions already generated by a checkpoint runner:

```bash
python -m training.jax_fidelity.evaluate \
  --config experiments/jax-fidelity-lab/config.json \
  --records authorized-development.jsonl \
  --records-sha256 APPROVED_RECORDS_SHA256 \
  --predictions candidate-predictions.jsonl \
  --predictions-sha256 APPROVED_PREDICTIONS_SHA256 \
  --prediction-completion prediction-completion.json \
  --prediction-completion-sha256 TRUSTED_PREDICTION_COMPLETION_SHA256 \
  --surface raw \
  --output candidate-report.json \
  --completion evaluation-completion.json
```

`predict.py` generates all 512 development predictions from one immutable
merged checkpoint on CUDA. `development_eligibility.py` compares that raw
report against the checksum-bound accepted baseline. Its terminal evidence
must join the candidate ID, config, dataset, development records, prediction
bytes, prediction completion, and checkpoint manifests into one chain before
the development improvement and non-regression gates can pass. The evaluator's
approval token covers the record, prediction, and prediction-completion hashes.
Hidden evaluation remains unavailable until that chain passes.

The eligibility invocation must include these baseline arguments in addition
to the candidate prediction and evaluation arguments:

```bash
python -m training.jax_fidelity.development_eligibility \
  ... \
  --baseline-report BASELINE_DIRECTORY/baseline-development.json \
  --baseline-report-sha256 BASELINE_REPORT_SHA256 \
  --baseline-completion BASELINE_DIRECTORY/completion.json \
  --baseline-completion-sha256 TRUSTED_BASELINE_COMPLETION_SHA256 \
  --baseline-candidate-manifest-sha256 ACCEPTED_MANIFEST_SHA256 \
  --baseline-engine-sha256 \
  95b69991b68c57a2d2d4bfa4116feb9ec57295588551d109353a42a9c16c4fdf
```

The private endpoint evaluator accepts an explicit absolute
`--hidden-state-root`, which must be a mode-0700 durable directory, so the
one-shot claim works on an encrypted workstation as well as `/var/lib`.

Hidden records are deliberately not materialized by this package. An authorized
runner may pass them to the endpoint evaluator once; public evidence retains
only population hashes and aggregate metrics, never passages or raw generations.
