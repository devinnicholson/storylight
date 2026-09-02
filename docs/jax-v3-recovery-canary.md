# JAX v3 recovery inputs

The v3 recovery experiment is a diagnostic, not a release candidate. Its
configuration keeps full-development evaluation and checkpoint merging disabled.
It exists to prove that MaxText creates trainable LoRA leaves, propagates a
nonzero gradient, changes an adapter, and can learn a small exact-output task
before another full campaign is considered.

`training.jax_fidelity.recovery_inputs` deterministically derives two public
populations from Story Fidelity v2:

- `overfit-canary`: one complete counterfactual pair from each of the 20
  categories, or 40 train records;
- `public-probe`: two complete counterfactual pairs from each category, or 80
  development records.

The selector deduplicates pair content before selection, preserves A/B
adjacency, and hashes the policy, seed, purpose, category, and pair content.
The v3 config pins the resulting record IDs, pair IDs, source JSONL, and exact
production-formatted teacher bytes. The manifest also proves that record, pair,
family, passage, and template-family identities are disjoint. Hidden bytes are
neither accepted nor written.

The builder requires the sealed v2 config, training completion, development
rejection, and TensorBoard event file. Their exact hashes and rejected training
run ID are part of the v3 config, so a newly assembled population cannot lose
the provenance of the failed run it is intended to diagnose. Outputs use
exclusive creation and `inputs.manifest.json` is written last.

No command below starts a GPU or submits cloud work:

```bash
python -m training.jax_fidelity.recovery_inputs \
  --config experiments/jax-fidelity-lab/config-v3-canary.json \
  --dataset-manifest datasets/story-fidelity-v2/manifest.json \
  --rejected-config experiments/jax-fidelity-lab/config-v2.json \
  --rejected-training-completion /path/to/v2/training-completion.json \
  --rejected-development-rejection /path/to/v2/candidate-v2-rejection.json \
  --rejected-training-events /path/to/v2/training.tfevents \
  --output-directory /new/path/jax-v3-recovery-inputs
```

The four JSONL outputs separate evaluation records, which retain semantic and
privacy expectations, from MaxText message records, which use the exact
production prompt and supervised assistant target. A later training worker must
consume only `overfit-canary.train.jsonl`; both source files remain evaluation
evidence. The probe must stay held out from hyperparameter selection.

## Cached Modal input path

The v3 path is separate from the v2 staging contract. A plan created by
`infra.gcp.jax.stage_modal_v3_inputs` contains exactly nine host-uploaded files:
the v3 config, the public v2 manifest/train/development files, and the sealed
recovery manifest plus its four declared JSONL files. It rejects hidden paths,
tokenizer files, checkpoints, and any additional input. Planning is read-only;
staging requires the exact approval token printed by the plan.

```bash
python -m infra.gcp.jax.stage_modal_v3_inputs \
  --target-run-id "$RUN_ID" \
  --config experiments/jax-fidelity-lab/config-v3-canary.json \
  --dataset-manifest datasets/story-fidelity-v2/manifest.json \
  --train datasets/story-fidelity-v2/train.jsonl \
  --development datasets/story-fidelity-v2/development.jsonl \
  --recovery-directory /path/to/sealed/jax-v3-recovery-inputs \
  --source-tokenizer-manifest-sha256 "$TOKENIZER_MANIFEST_SHA256"
```

After the overlay is staged, `scripts/plan_modal_jax_full_input_v3.py` derives
the full target manifest from the cached roundtrip release. The separate
`deploy/modal_jax_full_input_v3.py` worker then copies the cached tokenizer and
base checkpoint inside Modal, so those large bytes are never re-uploaded by the
host. It maps `recovery/overfit-canary.train.jsonl` byte-for-byte to
`prepared/train.jsonl`, retains the public probe only as held-out evidence, and
writes the target manifest last.

Before either training or finalize-only recovery, the training worker runs
`training.jax_fidelity.recovery_staging`. That verifier independently checks the
config, dataset, rejected-v2 lineage, four public population hashes, all five
disjointness claims, exact canary mapping, untouched probe, cached tokenizer
binding, and absence of hidden or unapproved files. A v3 input cannot fall back
to the v2 preparation verifier.

## Learnability acceptance

The v3 config fixes the acceptance thresholds; operators cannot relax them at
launch time. A one-step smoke is accepted only when the event stream covers the
single optimizer step and proves raw and clipped gradients above `1e-12`, a
nonzero FP32 adapter update, at least one changed trainable leaf, a positive
learning rate, and supervised completion tokens. Before compilation, the native
state must also contain exactly 410 trainable LoRA tensors and optimizer moment
slots for every tensor. Its terminal Orbax checkpoint must contain the exact 205
paired rank-16 LoRA modules at step zero and bind the approved MaxText patch.

The 100-step canary adds three requirements:

- at least 90% of raw and clipped gradient samples exceed `1e-12`;
- mean loss across the final 20 steps is at least 10% below mean loss across
  the first 20 steps;
- the LoRA-filtered `learning/param_norm` changes by at least `1e-6` relative
  to its first recorded value.

The first and final loss windows are disjoint. Constant loss, a single isolated
nonzero gradient, or an unchanged selected-parameter norm therefore cannot pass.
The training command writes a successful completion only after this gate and
the terminal checkpoint proof pass. The Modal finalizer recomputes the same
terminal acceptance receipt before publishing, including finalize-only recovery.
