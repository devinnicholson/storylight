# Scene parser reliability and acceleration campaign

Start: 2026-09-09, from `98993657ae3baa4368b00f79f2592eba4ab4d97a`.
User authorizes up to $100 of GCP spend for this campaign, including experiment code, adapter weights and synthetic inputs. This is an authorization ceiling, not a spending target or a verified credit balance.

## Outcome

Make learned scene extraction useful in the existing experience: better coverage and correct bindings without slowing descriptions the current parser already handles. V4 improved strict extraction from 21/48 to 33/48 and refusal from 6/16 to 15/16, but still invented facts, missed temporal structure and took about three seconds to parse. It remains an experimental adapter, not the demo parser.

## Work in parallel

1. **Grounding repair — Franklin.** Fix relation target binding across clauses in `scene_facts.py`. Start with the demonstrated invented `looks_at` relation, then attack negation, distractors, conjunctions and directional/symmetric relationships. Preserve valid near-neighbors. Replay old raw outputs separately to attribute deterministic validator changes honestly.
2. **Training data — Mill.** Build 4,800 synthetic rows: 3,600 positives and 1,200 refusals, grouped into 1,800 linked examples rather than claiming independent samples. Cover temporal event references, repeated actors retaining earlier actions, explicit negatives, static descriptions, target binding and valid/invalid identity contrasts. Vary construction and word order; freeze before the independent development labels are released.
3. **Evaluation and demo baseline — Fermat.** Independently author development256 (192 positive/64 refusal) and test128 (96 positive/32 refusal). Keep test inputs private until source/data freeze and labels off the GPU. Attest the current demo path read-only and measure its actual output and prepare-only latency on synthetic inputs. HTTP rejection is not automatically model abstention.
4. **Integration and GPU execution — root.** Build bounded training and inference runners, verify every artifact, profile the selected candidate, run full repository checks for implementation changes, reconcile cloud costs, and push independently reviewed fast-forward commits.

## Experiment sequence

### A. Fix and measure the deterministic boundary

Implement the general relation-binding repair with adversarial counterexamples. Run focused tests, the full Python suite and both JavaScript suites. Capture before/after acceptance of unchanged V4 outputs under explicitly identified validator versions; never overwrite the original V4 scores or call validator gains learned improvements.

### B. Compare two training recipes

Keep the pinned Gemma model, completion-only objective, NF4 training, rank16/alpha32, learning rate and prompt fixed initially. Compare current Q/V targets with all eligible linear projections in the **text decoder only**. Do not accidentally train the vision/audio towers, language head or token embeddings.

Use a fixed 1,200-step pilot for each recipe on the same seeded ordering of the expanded corpus. Select the recipe using independently authored development loss, retaining both outcomes. Train the selected recipe from its documented initial state for a complete 4,800-example pass, with checkpoint selection using development only. This changes supervision, training budget and potentially adapter targets; it is a recipe comparison, not proof of a single isolated cause.

Audit actual module names, trainable counts, initialization, completion masks, context limits, finite gradients and saved/reloaded tensors. No output-dependent test retries, fabricated refusal labels or test-based checkpoint selection. If a pilot exposes a setup defect, retain it and repair before releasing fresh test outputs.

### C. Confirm quality before serving optimization

Compare the selected candidate, frozen V4 and the actual demo parser on the fresh synthetic set. Keep strict contract exactness, correct refusal, schema rejection, grounding rejection and infrastructure failures separate. Report positive/refusal denominators, per-family errors, false refusals and repeated-call stability. Semantic equivalence remains diagnostic; do not relax the frozen primary score to erase known failures.

A candidate must improve useful coverage against the measured demo baseline, preserve currently correct behavior, reject every refusal control correctly and introduce no unexpected admissions on the confirmation set before integration consideration. These authored tests are not population accuracy or a guarantee of safety.

### D. Improve serving where the profile points

Measure model loading, grammar compilation, prefill, token generation, output length and validation separately. Benchmark merged versus unmerged weights and resident execution on matched inputs. Try smaller precision only as a separately verified arm; require semantic and numerical checks. Do not treat tiny timing differences or shorter refusal outputs as kernel speedups.

Preserve the existing low-latency deterministic path. First integrate a learned candidate in synthetic shadow evaluation; consider it as a fallback for unsupported descriptions rather than adding a three-second model call to every utterance. Physical deployment must preserve local source/audio privacy, validation, cancellation, stale-result suppression and the accepted rollback path. No live private speech is sent to a cloud training or evaluation job.

## Budget and execution

Current regional quota permits one L4; larger accelerator quota is unavailable in the inspected region. Use one finite GPU job at a time while agents work in parallel locally. Reserve up to $20 for pilots/data checks, $40 for full training/confirmation, $20 for serving experiments and $20 for remaining repairs. Reallocate within the $100 authorization when evidence warrants it.

Each allocation has a provider deletion deadline and process timeouts, a specific owned resource name, authenticated maintenance access and no public serving endpoint. Keep job-specific ceilings and actual lifecycle estimates. Retain successful output before deleting the VM, then verify its disk and owned firewall rules are absent. Billing alerts and list-rate estimates are not hard invoice caps.

## Evidence and rollback

Retain frozen source/data hashes, raw synthetic predictions and token IDs, environment versions, checkpoint provenance, timing and cleanup receipts. Keep model weights out of Git. Preserve all earlier experiments and `checkpoints/best-demo-2026-09-08`. The stopping point is a measured integration decision with documented remaining errors, not merely a completed training job.

Primary implementation references: [PEFT quantization](https://huggingface.co/docs/peft/developer_guides/quantization), [Gemma4 architecture](https://huggingface.co/docs/transformers/model_doc/gemma4), and [Transformers gradient checkpointing](https://huggingface.co/docs/transformers/grad_checkpointing). PEFT documents broader linear-layer targeting for QLoRA; actual architecture compatibility and throughput must be measured here.
