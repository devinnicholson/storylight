# Bookforge JAX Story Fidelity orchestration plan

Status: repository implementation and no-spend preflight complete, 2026-09-01. Model-heavy,
paid-cloud, hidden-evaluation, and physical-device stages remain deliberately unexecuted.

## Objective

Build a reproducible offline post-training system that improves Bookforge's scene understanding
without changing its privacy or live-performance architecture:

```text
copyright-safe synthetic stories
        -> Story Fidelity Lab
        -> JAX + MaxText LoRA on guarded cloud compute
        -> merged Gemma 4 E2B Hugging Face checkpoint
        -> TensorRT Edge-LLM INT4-AWQ export
        -> side-by-side Jetson engine
        -> semantic, privacy, latency, memory, and projector gates
        -> explicit promotion or exact rollback
```

The accepted TensorRT planner remains production throughout the program. JAX is an offline
training and evaluation technology, never a dependency of the Jetson runtime.

## Frozen baseline

- Repository baseline: `d857327`
- Model: `google/gemma-4-E2B-it`
- Hugging Face revision: `3e22461f65e89153144f8adb70e3b8c2cc9845a7`
- TensorRT Edge-LLM: `v0.10.0`, revision
  `71dd1bae032e70771265917ec74d3ff4cad07a10`
- Accepted engine SHA-256:
  `95b69991b68c57a2d2d4bfa4116feb9ec57295588551d109353a42a9c16c4fdf`
- Production protocol: exactly four bounded slots: `SETTING`, `ACTOR`, `ACTION`, `MAGIC`
- Accepted semantics: 20/20 cases, 80 required checks, five forbidden-term checks
- Accepted performance: 1,598.663 ms mean, 1,769.105 ms p95, 1,927.361 ms maximum
- Accepted output: 35.35 mean completion tokens, 44 maximum
- Accepted engine memory: approximately 3.84 GB unified memory with roughly 1 GB appliance
  headroom and no swap
- Repository validation baseline: 529 collected tests

Every result must identify whether it evaluates raw model output, Bookforge postprocessing, or the
final renderer-safe contract. Deterministic repairs must not be presented as learned improvement.

## Implemented agent campaign

The campaign uses four coordinated owners after the root agent freezes shared schemas and file
ownership. Agents work in parallel only where the dependency graph permits it.

### Agent A: dataset and provenance

Owns:

- `src/bookforge/fidelity_schema.py`
- `src/bookforge/fidelity_dataset.py`
- `src/bookforge/fidelity_manifest.py`
- `datasets/story-fidelity-v1/`
- `tests/fixtures/story-fidelity-smoke.jsonl`
- dataset-specific tests

Builds 5,120 original synthetic records: 4,096 train, 512 development, and 512 hidden test. All
members of a semantic family and counterfactual pair stay in one split. The hidden split uses
held-out template families and vocabulary; only its manifest and checksum are committed.

The generator must cover actor/object selection, action binding, attributes, counts, spatial and
containment relations, scale, reversed motion, passive voice, coreference, transformation,
temporal order, destination, salience, negation, hallucination, proper names, reserved contact
data, Unicode, and story-contained prompt injection. It may not ingest books, web passages, user
stories, named fictional properties, or living-author style prompts.

### Agent B: evaluator and regression system

Owns:

- `src/bookforge/fidelity_evaluation.py`
- `src/bookforge/fidelity_benchmark.py`
- compatibility-preserving refactors in `src/bookforge/planner_benchmark.py`
- evaluator and benchmark tests
- `.github/workflows/ci.yml`

Implements deterministic slot, relation, attribute, count, role, transformation, order,
allowed-concept, hallucination, counterfactual, and privacy evaluators. It reports three surfaces
separately: raw four-slot output, postprocessed plan, and renderer-safe contract.

CI runs project-only Ruff, the existing tests, new fidelity tests, dataset determinism and leakage
checks, manifest validation, and shell validation. CI never has cloud credentials and never starts
paid work.

### Agent C: JAX and MaxText compatibility

Owns:

- `training/jax_fidelity/`
- `training/jax_fidelity/Dockerfile`
- `training/jax_fidelity/roundtrip_smoke.py`
- `experiments/jax-fidelity-lab/config.json`
- JAX packaging and conversion tests

Pins Python 3.12, JAX, MaxText, Flax, Optax, Orbax, container digest, base-model revision, and every
training parameter. It reproduces the exact production system prompt, demonstration exchange,
`STORY:\n{passage}` input, and four-line target. It uses completion-only loss, text-only
`gemma4-e2b`, and `scan_layers=False`.

The first candidate is ordinary LoRA, rank 8 or 16. QLoRA and Tunix are later experiments only if
they produce a measured memory, cost, or quality improvement. Neither is added for branding.

### Agent D: guarded cloud, export, and Jetson candidate path

Owns:

- `infra/gcp/jax/`
- `infra/gcp/monitoring/jax-fidelity-dashboard.json`
- `deploy/modal_jax_fidelity.py`
- `experiments/jax-fidelity-lab/modal-plan-2026-09.json`
- tuned-checkpoint support in the existing Gemma 4 exporters
- versioned Jetson candidate install, shadow, promotion, and rollback scripts
- infrastructure and packaging tests

Builds a dry-run-first, restart-safe cloud path. The preferred backend is one bounded Vertex AI
CustomJob using one TPU v6e chip in `us-east1`, one replica, a 2,700-second timeout, and retries
disabled. Modal L40S is a fail-closed fallback only when GCP rejects the run before a paid job has
started. There is never an automatic fallback after an ambiguous billable boundary.

The candidate uses a new immutable ID and a separate engine directory. The accepted engine stays
on port 11435; a candidate uses port 11436 for shadow acceptance. Promotion changes an atomic
pointer or environment entry only after a separate explicit approval token.

### Root integration owner

The root agent owns shared contracts, merge conflict resolution, complete test execution, durable
run state, cloud launch decisions, evidence reconciliation, Git history, and user checkpoints. It
must not allow agents to modify the same shared file concurrently.

## Dependency graph

```text
M0 contract freeze
   |
   +----------------+------------------+------------------+
   v                v                  v                  v
M1 dataset       M2 evaluator      M3 JAX package     M4 infra/rollback
   |                |                  |                  |
   +----------------+------------------+------------------+
                            |
                            v
                 M5 local integration gate
                            |
                  explicit paid-work approval
                            |
                            v
                 M6 conversion round-trip gate
                            |
                   Jetson no-op engine parity
                            |
                            v
                 M7 bounded LoRA experiment
                            |
                  frozen hidden evaluation
                            |
                            v
                 M8 INT4 export + Jetson shadow
                            |
                  explicit promotion approval
                            |
                            v
                 M9 promote or exact rollback
```

Dataset, evaluator, JAX packaging, and infrastructure work run in parallel after M0. No model
training starts until both dataset validation and the checkpoint round-trip gate pass. No
production change happens until every physical-device gate passes.

## Milestones and stop conditions

### M0: freeze contracts and ownership

1. Record the baseline commit, engine digest, model revisions, prompt revision, cache-contract
   revision, and accepted evidence.
2. Freeze the fidelity record schema and exact production message formatter.
3. Allocate non-overlapping files to agents.
4. Add a durable run manifest and stage state machine.

Exit: every artifact can trace back to code, dataset, model, and configuration hashes.

### M1-M4: build the complete no-spend system

Add:

```text
src/bookforge/fidelity_schema.py
src/bookforge/fidelity_dataset.py
src/bookforge/fidelity_manifest.py
src/bookforge/fidelity_evaluation.py
src/bookforge/fidelity_benchmark.py
scripts/build_fidelity_dataset.py
scripts/build_fidelity_gate.py
scripts/orchestrate_jax_fidelity_lab.py
training/jax_fidelity/
datasets/story-fidelity-v1/
experiments/jax-fidelity-lab/
infra/gcp/jax/
deploy/modal_jax_fidelity.py
.github/workflows/ci.yml
```

The orchestrator state machine is:

```text
dataset -> cpu-smoke -> compatibility-package -> baseline -> roundtrip
        -> train -> candidate-eval -> hf-export -> int4-export
        -> jetson-shadow -> gate -> promotion-or-retain-baseline -> reconcile
```

Each stage is content-addressed, idempotent, and resumable by run ID. Completion manifests are
written last. A failed or ambiguous paid stage cannot be retried under the same run ID.

Exit: all repository tests and dry-run infrastructure checks pass with no cloud spend.

### M5: local integration and baseline

1. Freeze the 5,120-record dataset and verify family-level split isolation.
2. Generate 32 deterministic CI fixtures from development-only families.
3. Run the accepted planner across development and the private hidden set.
4. Store only IDs and passage hashes in public evidence.
5. Prove that training records match the exact deployed four-slot prompt and stay within a
   512-token input and 64-token output budget.

Stop on leakage, non-determinism, PII, copyright provenance failure, or prompt mismatch.

### M6: compatibility round trip before full training

1. Convert the exact pinned Hugging Face checkpoint to MaxText Orbax.
2. Perform a zero-effect or one-step near-zero LoRA canary.
3. Merge and export a complete Hugging Face SafeTensors checkpoint.
4. Verify tensor names and shapes, tokenizer, special tokens, PLE weights, KV sharing, Gemma 4
   metadata, and EOS set `[1, 106, 50]`.
5. Require forward-logit comparison with KL divergence no greater than 0.03.
6. Run that checkpoint through the existing TensorRT Edge-LLM INT4-AWQ export.
7. Build a separate no-op Jetson engine and reproduce 20/20 semantics and the accepted performance
   envelope.

This is the highest-risk seam. Any failure ends the training campaign while leaving production
untouched.

### M7: bounded post-training

1. Start with a 5-10-step LoRA smoke.
2. Train one bounded rank-8 or rank-16 candidate with a frozen seed and completion-only loss.
3. Select by semantic development score, schema validity, and privacy—not training loss.
4. Lock the candidate before running the hidden evaluation once.
5. Permit a larger learning-rate/rank sweep only after a separate evidence and cost decision.

Every run manifest records the Bookforge commit, MaxText commit, JAX stack, container digest,
dataset hashes, model revision, hardware, hyperparameters, checkpoints, elapsed time, and declared
and reconciled gross cost.

### M8: export and physical shadow acceptance

1. Merge the selected adapter into the pinned base model.
2. Export a versioned, checksummed Hugging Face checkpoint.
3. Quantize to INT4-AWQ and export the text-only thinker using a verified Bookforge calibration
   corpus if the pinned quantizer supports it; otherwise preserve the accepted calibration path.
4. Install without overwriting any accepted model or engine path.
5. Stop the active engine only inside a trap-protected, rollback-guaranteed acceptance window if
   both engines cannot fit simultaneously.
6. Run counterbalanced base/candidate measurements with the projector and kiosk active.

Exit only when all promotion gates pass.

### M9: explicit promotion, evidence, and teardown

Promotion requires an exact one-purpose approval token. The promotion script backs up the active
configuration, switches atomically, restarts and probes the service, exercises one live scene, and
automatically restores the previous TensorRT engine on any failure.

The finalizer reconciles actual cost; verifies zero active GCP jobs, Modal tasks, and paid minimum
instances; archives immutable manifests and reports; and lists any proposed scratch-resource
deletion or billing unlink for separate approval.

## Promotion gates

Semantic and privacy gates:

- Four-slot parse and schema validity: 100%
- Privacy gate: 100%
- PII, proper-name, source-echo, and injection leaks: zero
- Existing contest suite: 20/20, all 80 requirements and all five forbidden checks
- Hidden semantic atom recall: at least 98%
- Hidden exact-example pass: at least 95%
- Each noncritical category: at least 90%
- Negation, prompt injection, transformation, and passive voice: 100%
- Counterfactual sensitivity: at least 98%
- Unsupported-concept rate: at most 1%
- At least five percentage points of hard-development improvement, or half as many baseline
  failures
- No category regression greater than one percentage point
- Human review of 25 locked adversarial examples

Jetson and projector gates:

- Completion maximum remains 64 tokens
- p50 no greater than 1.70 seconds
- p95 no greater than 2.00 seconds
- maximum no greater than 2.25 seconds
- p95 no more than 5% slower than the counterbalanced same-run base
- unified-memory peak no greater than 4.0 GB
- at least 768 MiB available with the kiosk running
- zero inference swap, OOM, external planner socket, or restart failure
- planner ready within the existing 90-second bound
- projector contention and live scene flow pass
- exact restoration of the accepted engine is demonstrated

A faster candidate that misses semantics fails. A more accurate candidate that destabilizes the
appliance also fails.

## Cost and cloud guardrails

Current execution state: the repository does not assume billing, credit, accelerator quota, or
service availability. Those account facts must be re-verified immediately before a paid action.
No paid GCP, Modal, or device stage was started during implementation.

Before a paid run, the user must explicitly approve:

1. relinking billing account `000000-000000-000000`;
2. verifying applicable promotional credits;
3. the exact run ID, backend, resources, timeout, and gross ceiling;
4. a read-only Gemma/Hugging Face token delivered through Secret Manager or a Modal Secret, never a
   repository or password file.

Default first-experiment controls:

- GCP JAX campaign ceiling: $3.50 gross
- one TPU v6e chip, one replica, 2,700-second timeout, retries disabled
- no endpoint, autoscaling service, or persistent accelerator
- at least $6.50 preserved below the existing $10 gross emergency disconnect
- Modal only as an explicitly authorized fallback before a GCP billable start
- Modal L40S: one container, one finite call, fixed timeout, no web endpoint, no automatic retry,
  and a new September ledger
- any separate Modal TensorRT export retains its own declared command ceiling

Budget notifications and billing disconnects are containment controls, not mathematical spend
caps. Resource count, timeout, retry disablement, and explicit teardown are authoritative.

## External actions that cannot be hidden inside the agent run

- Relinking or unlinking GCP billing
- Enabling TPU services or creating buckets, repositories, service accounts, builds, or paid jobs
- Starting a Modal GPU call or changing a workspace-wide Modal budget
- Changing a Jetson service, engine, power mode, or reboot state
- Promoting the candidate
- Deleting cloud objects, scratch buckets, Modal volumes, or other resources

The orchestrator performs read-only preflights first and prints the exact proposed action. Each
external mutation uses a narrow, stage-specific approval token. It never stores a sudo password,
cloud credential, or Hugging Face token in the repository.

## Rollback and recovery

- Cloud training never changes production routing.
- Every paid job has one run ID, one attempt, a timeout, disabled retry, and an explicit cancel
  command.
- Scratch and release artifacts use separate private buckets with public-access prevention,
  uniform bucket-level access, checksums, generation-match writes, and completion manifests last.
- Scratch may expire after seven days; accepted releases remain through the contest.
- Modal fallback has no endpoint or minimum container. Its volume is retained until checksums are
  verified outside Modal.
- The Jetson candidate is side-by-side and cannot overwrite the accepted checkpoint or engine.
- Failed promotion restores the exact prior TensorRT engine and cache-contract revision, not merely
  the slower Ollama fallback.
- If the $10 billing disconnect fires, the run treats billing-disabled as terminal and never
  automatically relinks it.

## Definition of done

The program is complete only when:

1. the dataset, evaluator, JAX package, guarded infrastructure, CI, orchestrator, and rollback tools
   are committed and reproducible;
2. the HF-to-MaxText-to-HF-to-TensorRT seam is proven on the pinned E2B architecture;
3. one locked candidate is evaluated once on the hidden set;
4. the physical Jetson and projector evidence passes every gate;
5. the accepted engine is either explicitly promoted or demonstrably retained after rejection;
6. actual gross spend and teardown state are reconciled; and
7. the model card clearly distinguishes learned behavior, deterministic repair, local privacy,
   cloud training, and cloud rendering.

This yields a defensible contest narrative: Bookforge post-trains an open Gemma model with JAX and
MaxText on Google Cloud, compiles it with NVIDIA TensorRT Edge-LLM, and runs private, real-time scene
understanding on an NVIDIA Jetson without sending a reader's raw audio or camera data to the cloud.

## Verified implementation checkpoint

The final no-spend integration run used:

- configuration SHA-256:
  `a009feaaaef4907f1ed41e82d7c0c5ef906b00adc574cd0e00fca8a986097f9c`;
- dataset manifest SHA-256:
  `e717eb38c44fceeeae3a2bc88981767c316ca1339198ce1077b893252afeb1de`;
- 4,096 public training records, 512 public development records, and 512 private hidden records;
- successful dataset, 32-record CPU smoke, and production-format compatibility stages; and
- 705 passing repository tests after trusted stage-producer, cloud-admission,
  immutable Jetson-tooling, and checkpoint-evidence integration.

The accepted engine's 512-case development baseline is now frozen separately
from model outputs: 100% schema validity, 58.7109375% semantic-atom recall,
0% exact-example pass, and 74.609375% privacy pass. This deliberately strict
raw-surface result is the comparison target for learned improvement; it is not
the previously accepted 20-case renderer-safe contest score.

The next permitted stage is `baseline`. It requires an accepted-engine identity bundle plus
checksum-bound development and one-shot private-hidden endpoint reports. Before that run, the
private hidden key, records, and custody receipt must be moved from volatile local storage to a
durable encrypted location outside the repository. Round-trip conversion, training, cloud export,
Jetson shadowing, and promotion remain behind separate approval tokens.
