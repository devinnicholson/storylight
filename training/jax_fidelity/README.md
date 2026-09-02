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
- SFT uses ordinary rank-8 LoRA with MaxText's completion-only prompt masking. Every
  assistant turn is supervised, including the fixed demonstration; this is not a
  custom final-answer-only mask. QLoRA is disabled.
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
no-retry L40S call. Its staging command includes the public train and
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
worker records the only valid step-0 base Orbax `items` leaf and step-5 LoRA
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

After the full adapter succeeds, `modal_jax_merge.py` is the only remote merge
boundary. One exact approval binds the original staged HF snapshot and input
manifest, the successful roundtrip completion and base-Orbax receipt, the
portable full-training completion and adapter manifest, and the config,
dataset, and training run hashes. The finite L40S worker selects the single
configured terminal LoRA `items` leaf, invokes MaxText-to-HF once, and writes a
prediction-only `candidate.manifest.json`. It has no retries or web endpoint,
and `completion.json` is published last.

```bash
modal run deploy/modal_jax_merge.py --help

python scripts/fetch_modal_jax_merge.py \
  --merge-run-id MERGE_RUN_ID \
  --completion-sha256 TRUSTED_COMPLETION_SHA256 \
  --config experiments/jax-fidelity-lab/config.json \
  --destination artifacts/jax-fidelity/merged-candidate \
  --execute
```

The fetched directory contains `merged-hf/` and
`candidate.manifest.json`, the exact layout accepted by
`stage_modal_prediction_inputs.py`. The candidate remains explicitly
ineligible for release until development evaluation succeeds.

After development eligibility passes, `release.py` does not trust the fetched
directory by location alone. Its mandatory inputs include the trusted merge
completion SHA, the complete merge file table, the candidate and source-binding
receipts, all three MaxText-to-HF conversion receipts, the original HF
manifest, the step-0 base Orbax receipt/manifest, and the terminal full-adapter
receipt/manifest. It replays the conversion input contract against those exact
three source leaves, validates the merged-HF manifest against the produced
bytes, and copies every custody hash into `terminal_evidence` before writing the
immutable release manifest last.

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
The private endpoint evaluator accepts an explicit absolute
`--hidden-state-root`, which must be a mode-0700 durable directory, so the
one-shot claim works on an encrypted workstation as well as `/var/lib`.

Hidden records are deliberately not materialized by this package. An authorized
runner may pass them to the endpoint evaluator once; public evidence retains
only population hashes and aggregate metrics, never passages or raw generations.
