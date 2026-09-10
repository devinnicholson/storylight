# Gemma 4 scene adapter

This is the retained public version of the final scene-extraction adapter experiment. The model used NF4 QLoRA over Gemma 4 E2B and a grammar-constrained output contract. The candidate improved strict positive extraction from 64/96 to 81/96, while literal refusals fell from 28/32 to 25/32. It therefore failed the promotion gate and is not used by the default voice compiler.

[`RESULTS.md`](RESULTS.md) defines the measured outcome and its limits. [`TRAINING.md`](TRAINING.md) documents the selected training protocol. [`CACHE-BRIDGE.md`](CACHE-BRIDGE.md) covers the dynamic-prefill/static-decode work that followed.

The repository retains the generator, trainer, evaluator, runtime, protocol, selected public development data, and verification code. Generated message corpora, cloud allocation scripts, private authoring attestations, reviewer workflow records, and superseded diagnostics are excluded. Recreate generated training inputs with `generate_training.py`; model weights and private evaluation material are not distributed.
