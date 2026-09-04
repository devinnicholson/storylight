# Bookforge handoff for a new Astra session

Prepared: 2026-09-04  
Repository: `https://github.com/devinnicholson/bookforge.git`  
Authoritative branch: `main`  
Implementation baseline: `583be0716a2fc6b3d182904e8c1441ed9ba5e11b`  
Starting point: current `origin/main` containing this document  
Local checkout: `/Users/operator/Documents/ChatGPT/golden-ticket`  
Jetson checkout: `/opt/bookforge`

## 1. Mission

Continue building **The Book That Listens Back**: a portable reading experience that listens to a
story locally, understands the current story event on an NVIDIA Jetson Orin Nano, generates a
beautiful scene in the cloud, and projects a responsive moving world around the physical book.

The immediate engineering objective is narrower:

> Connect the reliable four-line Jetson Gemma/TensorRT response to a complete, source-grounded
> `SceneFactsV2` graph in the live planner path, prove the result on the frozen 512-record public
> development split, and only then test whether the richer graph materially improves generated
> imagery.

Do not begin with microphone work, a new renderer, a larger model, or another prompt-only accuracy
experiment. The current bottleneck is loss of entity bindings and secondary story facts between
local understanding and image generation.

## 2. Product and contest story

The contest-facing story is:

1. The child, book, audio, camera frames, names, and reading behavior remain on the Jetson.
2. A local open Gemma model converts the current story event into a small visual fact contract.
3. Only the privacy-validated visual contract may leave the device.
4. Google Cloud generates or coordinates the expensive visual work.
5. NVIDIA acceleration runs at both ends: TensorRT on the Jetson and NVIDIA GPU/NIM workloads in
   Google Cloud.
6. The projector displays an immediate local draft, then atomically upgrades to generated artwork
   with local depth motion. Cached scenes replay without cloud inference.
7. Optional GKE/Nemotron anticipation prepares the next known page while the current page is being
   read and validates only synthetic imagery plus sanitized intent.

This is not “an image generator attached to a book.” The differentiator is the measured,
privacy-preserving edge/cloud system: grounded story understanding, progressive low-latency
projection, anticipatory generation, verified assets, and explicit rollback.

## 3. Non-negotiable constraints

- Keep raw audio, webcam frames, full story passages, names, contact data, and reading telemetry on
  the Jetson.
- Never send the source passage to a remote image renderer or cloud critic.
- Keep the accepted four-slot TensorRT planner and accepted renderer available as rollback paths.
- Treat every candidate as opt-in until it passes its documented semantic, privacy, latency,
  memory, thermal, and end-to-end gates.
- Do not open or derive the hidden Story Fidelity split during development.
- Do not claim generated video when the output is a still image animated by depth/parallax.
- Do not claim a model improvement from deterministic postprocessing.
- Do not claim a benchmark is reproducible unless its environment and per-case provenance were
  actually retained.
- Do not store passwords, SSH private keys, Modal credentials, API keys, pairing tokens, or Google
  credentials in the repository, prompts, logs, or evidence artifacts.
- Do not enable a public GKE or Jetson endpoint. The Jetson API and TensorRT server remain loopback
  only; remote maintenance uses Tailscale/SSH.
- Cloud budgets are alerts and emergency controls, not guaranteed hard caps. Query current billing
  state before billable work and use one bounded experiment at a time.
- Preserve user-owned untracked directories such as `output/`, `post/`, `post-temp-*`, and `tmp/`.
- Use `apply_patch` for source edits, run the complete verification matrix, deslop the diff, and
  push only reviewed changes.

## 4. Verified repository state

At commit `583be07`:

- `1,389` Python tests pass.
- The two projector/workbench JavaScript suites pass.
- Scoped Ruff checks and `git diff --check` pass.
- The public SceneFacts coverage artifact reproduces byte-for-byte.
- `origin/main` and the working branch both point to `583be07` at handoff creation time.
- No paid Modal or Google Cloud service was called for the SceneFacts increment.
- The only test warning is an external Starlette/httpx deprecation warning.

The most recent verification commands were:

```bash
cd /Users/operator/Documents/ChatGPT/golden-ticket
.venv/bin/python -m pytest -q
source ~/.nvm/nvm.sh
nvm use 22.17.0 >/dev/null
node --test tests/anticipatory_workbench.test.js tests/hand_interaction.test.js
.venv/bin/ruff check src/bookforge scripts/benchmark_fidelity_graph_targets.py tests
git diff --check
```

The repository contains user-owned untracked presentation/poster output. Do not make whole-tree
lint success depend on those unrelated files; scope lint to repository implementation files.

## 5. Current architecture

```text
typed story text now / microphone later
                |
                v
Jetson private edge
  local alignment and session state
  Gemma 4 E2B through TensorRT Edge-LLM :11435
  accepted four-line SETTING/ACTOR/ACTION/MAGIC envelope
  deterministic privacy + grounding gates
                |
                | sanitized visual contract only
                v
provider-neutral scene job
  preferred Google Cloud renderer / Vertex route
  authenticated Modal fallback where explicitly configured
  optional GKE/Nemotron next-page critic and coordinator
                |
                v
checksum-verified master + depth assets
                |
                v
Jetson projector runtime
  immediate procedural draft
  atomic master/depth upgrade
  WebGL depth parallax + authored ambience
  optional local MediaPipe interaction
  exact local cache replay
```

Important distinctions:

- `tensorrt_slots` is the accepted appliance protocol.
- `tensorrt_hybrid` is an opt-in research protocol, not the default.
- `SceneFactsV2` is currently a strict schema, compiler, evaluator surface, and deterministic
  public-corpus target adapter. It is **not yet the live kiosk parser**.
- The current projector motion is primarily local depth animation. Cosmos/world-model video is
  intentionally outside the critical path.
- Nemotron is an asynchronous cloud critic/anticipation component, not another blocking model
  before the first image.

## 6. Hardware and remote access

Known device:

- NVIDIA Jetson Orin Nano developer kit, 8 GB unified memory.
- JetPack 7.2.1 / L4T 39.2.1.
- Samsung 9100 Pro NVMe.
- Projector attached directly to the Jetson display output.
- Webcam available; physical MediaPipe acceptance is still pending.
- Accepted TensorRT Edge-LLM service is loopback-only on port `11435`.
- Bookforge API is loopback-only on port `8080`.
- Paired phone controller may listen on `8081` only on a private operator-controlled WLAN.

Remote maintenance uses Tailscale. The last known Jetson Tailscale address was
`192.0.2.10`; discover the current address rather than assuming it is permanent.

From the Mac:

```bash
tailscale status
ssh -o HostKeyAlias=jetson.local \
  -i /Users/operator/.ssh/bookforge_jetson \
  operator@192.0.2.10
```

For a local tunnel to the TensorRT endpoint:

```bash
ssh -N -o BatchMode=yes -o HostKeyAlias=jetson.local \
  -i /Users/operator/.ssh/bookforge_jetson \
  -L 18435:127.0.0.1:11435 \
  operator@192.0.2.10
```

Do not ask the user to put a sudo password in a file. Restricted passwordless administration was
installed through `/usr/local/sbin/bookforge-admin`; inspect its supported fixed commands rather
than attempting arbitrary passwordless sudo.

Read-only first checks on the Jetson:

```bash
cd /opt/bookforge
git rev-parse HEAD
./deploy/jetson/check-device.sh --strict
systemctl status bookforge-tensorrt-planner.service --no-pager
systemctl status "bookforge@operator.service" --no-pager
curl -fsS http://127.0.0.1:11435/v1/models
curl -fsS http://127.0.0.1:8080/readyz
sudo -n /usr/local/sbin/bookforge-admin status
```

Do not change JetPack, rebuild the accepted engine, switch power mode, restart the kiosk, or
promote an environment setting merely to collect a baseline.

## 7. Cloud state and cost boundary

Google Cloud project: `your-gcp-project`.

Historically verified configuration includes:

- promotional credits present;
- a $150 gross monthly budget alert;
- an emergency billing-disconnect notification at $175;
- one regional NVIDIA L4 quota used by the bounded GKE experiment;
- GKE/Nemotron acceptance completed and the NIM workload scaled back to zero;
- private Cloud Run/Vertex/Modal provider adapters already implemented;
- Google CLI/application-default authentication configured on the Mac during prior work.

These are historical facts, not proof of current state. Before any GCP experiment, run read-only
checks for active account, project, billing, quota, clusters, workloads, Cloud Run revisions,
Artifact Registry, and current cost. The $150/$175 controls are not hard real-time spend caps.

The latest recorded Modal report in the September 4 overnight evidence attributed `$0.39271299`
to that pass and `$13.53057755` month-to-date. Modal reporting can lag and this is not a remaining
credit balance. Query current usage before another paid call.

Cloud rules:

1. Prefer local deterministic tests and the existing public corpus before cloud inference.
2. Scale GKE/NIM GPU replicas from zero only for a supervised, bounded run.
3. Use private `kubectl port-forward`/authenticated routes; do not create public ingress.
4. Disable automatic paid retries and fail closed after ambiguous responses.
5. Scale GPU workloads back to zero immediately after evidence capture.
6. Keep the CPU coordinator, storage, network, build, and cluster-management costs visible; zero
   GPU replicas does not mean the whole cloud footprint is free.

Primary references:

- `docs/anticipatory-story-engine.md`
- `docs/nemotron-performance.md`
- `docs/gcp-handoff.md`
- `infra/gcp/README.md`
- `infra/gcp/billing-kill-switch/README.md`

## 8. What was just implemented

Commit `583be07` added the Story Fidelity V2 graph foundation.

### SceneFacts V2

`src/bookforge/scene_facts.py` defines immutable, bounded facts for:

- setting;
- subjects and objects with stable internal references;
- count, color, state, and attributes bound to the correct entity;
- relationships and ownership;
- motion direction and destination;
- foreground/background salience;
- events and temporal order;
- explicit negatives;
- transformations.

It includes:

- an order-preserving `V2` wire format with deterministic 64/96/128-token estimates;
- source-grounding checks;
- graph integrity checks for invalid references, opposing relationships, contradictory states,
  and positive facts negated by the same graph;
- renderer prompt compilation that resolves internal references and never includes the source
  passage;
- privacy rejection for contact data, names, injection-like directives, distinctive source echo,
  and printed source payloads across direct, adjectival, and passive constructions.

Shared privacy and semantic normalization were extracted into:

- `src/bookforge/privacy_policy.py`
- `src/bookforge/semantic_text.py`

### Graph-aware fidelity evaluation

`src/bookforge/fidelity_evaluation.py` now evaluates typed nodes and edges rather than satisfying
relations through bag-of-words overlap. Public graph exactness is closed-world and independent of
internal reference names: source-mentioned distractors, wrong bindings, wrong typed salience,
wrong event order, extra nodes/edges, and contradictions invalidate an exact pass.

The deterministic target adapter is in `src/bookforge/fidelity_graph_targets.py`. It is for public
evaluation/training targets, not arbitrary live-story parsing.

Public coverage at a 64-token estimate:

| Split | Total | Eligible | Exact among eligible | Refused |
| --- | ---: | ---: | ---: | ---: |
| Train | 4,096 | 4,096 | 4,096 | 0 |
| Development | 512 | 509 | 509 | 3 |
| Combined | 4,608 | 4,605 | 4,605 | 3 |

The three refusals are deliberate ambiguous same-label containment cases. The target contract
cannot identify which repeated basket owns the relation, so the adapter refuses to invent a
binding. The hidden split was not opened.

Reproduce the aggregate artifact:

```bash
.venv/bin/python scripts/benchmark_fidelity_graph_targets.py \
  --token-budget 64 \
  --output /tmp/story-fidelity-v2-graph-coverage.json
cmp /tmp/story-fidelity-v2-graph-coverage.json \
  benchmarks/story-fidelity-v2-graph-coverage-2026-09-04.json
```

### TensorRT hybrid experiment

The existing four-line envelope was retained because direct graph decoding was unreliable:

| Protocol | Public probes | Outer schema | Approximate median | Decision |
| --- | ---: | ---: | ---: | --- |
| Six-line graph | 12 | 5/12 | 2.24 s | Reject |
| Five-line graph | 12 | 0/12 | 1.64 s | Reject |
| Four-line hybrid | 12 | 12/12 | 1.375 s | Keep opt-in |

In a separate five-case integrated screen, accepted slots passed `2/5` automatic semantic checks
at a `1.035 s` median; hybrid passed `3/5` at `1.399 s`. This is promising but far too small for
promotion. The machine-readable file is explicitly an exploratory engineering note because exact
JetPack/TensorRT/power/thermal and sanitized per-case provenance were not retained:

- `benchmarks/jetson-gemma4-tensorrt-hybrid-2026-09-04.json`

Activation is deliberately opt-in:

```text
BOOKFORGE_LIVE_SCENE_PLANNER_BACKEND=tensorrt_hybrid
```

Do not add that setting to the accepted Jetson environment yet.

## 9. Previous experiments that must not be repeated blindly

- More generic renderer prompt instructions did not solve missing counts, relationships, or
  transformations. The local planner had already discarded those facts.
- A full six-line graph prompt truncated and violated privacy on the Jetson.
- A compact five-line graph prompt had zero outer-schema adherence in the 12-case probe.
- The FLUX.2 Klein L4 candidate produced attractive images and warm artwork/depth inference around
  `1.7–1.9 s`, but strict visual correctness was low: `5/24` for the full contract and `10/24` for
  the concise contract in one unblinded screen. It remains opt-in.
- Depth animation does not prove a temporal transformation; a still image cannot visibly show a
  feather becoming a boat across time.
- Nemotron must not block first-image display. It belongs in asynchronous review/anticipation.
- MediaPipe passed deterministic lifecycle tests, but physical webcam FPS and projector mapping
  are still unmeasured.
- Nsight confirmed TensorRT/CUDA Graph/CUTLASS kernels dominate the local inference window. The
  captured trace was diagnostic evidence, not proof of a new engine speedup.
- Training is not automatically the next answer. The JAX/MaxText tooling exists, but a tuned model
  should be considered only after deterministic graph construction and the public development
  gate show where errors remain.

## 10. Exact next milestone

### Objective

Build a deterministic live adapter that converts the four-line TensorRT response plus local source
text into a validated `SceneFactsV2`, without exposing the source outside the edge.

### Required flow

```text
TensorRT four-line response
        |
        v
strict hybrid slot parser
        |
        v
deterministic entity/action/relation binding from local source
        |
        v
SceneFactsV2 model validation
        |
        v
source grounding + privacy validation
        |
        +---- reject ----> accepted four-slot fallback
        |
        v
renderer-safe graph prompt / existing scene job
```

### Definition of done

All of the following must be true:

1. A public, documented adapter accepts the existing parsed four-line response and source text and
   returns either a valid `SceneFactsV2` or a value-free refusal code.
2. No arbitrary story parser is faked from the deterministic corpus target adapter.
3. The adapter is deterministic, bounded, and has no network calls.
4. It never includes source text or private tokens in exceptions, logs, metrics, wire output, or
   renderer prompts.
5. It resolves entity references, counts, colors, states, actions, ownership, spatial relations,
   destinations, salience, temporal order, negatives, and transformations when explicitly
   supported by the local source and four-line response.
6. Ambiguous bindings fail closed to the accepted four-slot plan.
7. Every prior privacy and graph-spoof regression remains closed.
8. A matched 512-record public development A/B compares:
   - accepted `tensorrt_slots` raw response;
   - opt-in `tensorrt_hybrid` raw response;
   - graph-postprocessed candidate;
   - final renderer-safe contract.
9. The report separates learned output from deterministic recovery and reports schema adherence,
   exact semantic pass, per-category pass, privacy failures, refusals, p50/p95/max latency, output
   tokens, and peak memory.
10. The candidate beats the accepted semantic baseline on the frozen development set with zero
    privacy failures and no material regression in categories already passing.
11. Median local planning stays at or below `1.5 s`; p95 and maximum must be reported rather than
    hidden behind the median.
12. The accepted runtime remains unchanged unless a separate promotion decision is recorded.
13. Full tests, browser tests, Ruff, artifact reproduction, secret scan, and diff cleanup pass.

Do not set an accuracy percentage target until the accepted 512-record live baseline is captured
with the exact deployed prompt and engine. The existing `4,605/4,605` number measures deterministic
target/evaluator coverage, not live Gemma accuracy.

## 11. Recommended implementation sequence

### Phase A — establish a clean baseline

1. Fetch `origin/main` and verify commit `583be07` or a known descendant.
2. Confirm the Mac test suite and public artifact reproduction.
3. Connect to the Jetson read-only over Tailscale.
4. Record exact JetPack, TensorRT Edge-LLM, engine digest, power mode, clocks, thermal state, memory,
   service status, and current environment revision.
5. Run a small accepted-protocol smoke without changing the appliance environment.

Stop if repository and Jetson revisions differ. Synchronize code deliberately; do not overwrite
device-local credentials or environment files.

### Phase B — implement the local adapter

1. Define one narrow result type: valid facts or a stable refusal enum.
2. Reuse `privacy_policy.py`, `semantic_text.py`, and the existing SceneFacts validators.
3. Parse hybrid IDs only through the existing strict parser.
4. Add deterministic binding helpers by semantic category, with conservative ambiguity detection.
5. Keep corpus-specific target derivation out of the live adapter.
6. Integrate behind a new opt-in path; preserve `tensorrt_slots` behavior byte-for-byte.
7. Add adversarial tests before optimizing code size or latency.

Suggested new module: `src/bookforge/live_scene_facts.py`. Avoid growing
`scene_facts.py`, `fidelity_evaluation.py`, or `tensorrt_slot_client.py` into larger mixed-purpose
modules.

### Phase C — benchmark before rendering

1. Add a restartable public-development benchmark with per-case sanitized evidence.
2. Capture accepted slots and hybrid candidate from the same resident engine, power mode, prompt
   revision, and thermal envelope.
3. Evaluate raw, postprocessed, and renderer-safe surfaces separately.
4. Inspect failures by category; do not tune on the hidden split.
5. Reject prompt or deterministic changes that merely shift failures between counterfactual pairs.

### Phase D — finite image A/B

Only after Phase C passes:

1. Select a small, frozen, category-balanced public subset emphasizing counts, ownership, spatial
   relations, salience, and transformations.
2. Render accepted and graph-aware contracts with matched provider, model revision, seed,
   dimensions, steps, and warm/cold state.
3. Record exact prompt hashes, asset checksums, inference/packaging/transport times, and full
   provider cost.
4. Use automated screens only as prescreens. Conduct blind human pair review for final visual
   fidelity claims.
5. Keep beautiful-but-wrong images as failures.

### Phase E — decide, do not drift

- Promote only if accuracy and performance gates pass.
- Retain opt-in if promising but incomplete.
- Roll back cleanly if no material improvement.
- If remaining errors are model limitations rather than deterministic binding failures, prepare one
  bounded JAX LoRA candidate using the existing orchestration plan.

## 12. Suggested agent orchestration for Astra

Use parallel agents only after freezing interfaces and file ownership.

### Agent A — live adapter

Own only:

- proposed `src/bookforge/live_scene_facts.py`;
- its focused unit tests.

Deliver deterministic graph construction and refusal codes. No cloud calls and no changes to
shared evaluator/schema files without root coordination.

### Agent B — benchmark and evidence

Own only:

- a new public development benchmark script;
- benchmark tests;
- sanitized report schema.

Deliver restartable matched A/B execution and provenance capture. Do not open hidden data.

### Agent C — adversarial review

Read-only until findings are accepted. Attack:

- wrong-subject/wrong-object binding;
- source distractor inclusion;
- negated positive facts;
- contradictory relations/states;
- temporal-order spoofing;
- salience spoofing;
- marked, unmarked, Unicode, and lowercase names;
- printed payloads in active, adjectival, quoted, inverted, and passive forms;
- prompt injection and contact data;
- malformed/rebound/undefined hybrid IDs;
- source or secret leakage through logs and evidence.

### Root/Astra — integration owner

Own shared types, integration points, Jetson execution, full tests, cost authorization, evidence
reconciliation, deslop, Git history, and promotion decisions. Do not let two agents edit the same
shared file concurrently.

## 13. Key files

Start here:

- `README.md` — current product and local/cloud operating model.
- `docs/implementation-plan.md` — chronological living plan.
- `docs/story-fidelity-v2-runtime-2026-09-04.md` — newest graph increment and limitations.
- `src/bookforge/scene_facts.py` — strict typed graph, grounding, privacy, wire, prompt compiler.
- `src/bookforge/privacy_policy.py` — shared outbound privacy grammar.
- `src/bookforge/semantic_text.py` — shared semantic normalization.
- `src/bookforge/tensorrt_slot_client.py` — accepted and hybrid four-line protocols.
- `src/bookforge/fidelity_graph_targets.py` — public deterministic evaluation targets only.
- `src/bookforge/fidelity_evaluation.py` — graph-aware evaluator and public exactness logic.
- `scripts/benchmark_fidelity_graph_targets.py` — reproducible public coverage report.
- `deploy/jetson/README.md` — Jetson topology and guarded operations.
- `docs/overnight-candidate-results-2026-09-04.md` — latest renderer, Nsight, MediaPipe, cost, and
  rejection evidence.
- `docs/jax-story-fidelity-orchestration-plan.md` — later training path; do not start here.
- `docs/anticipatory-story-engine.md` — GKE/Nemotron design and accepted bounded experiment.

Relevant evidence:

- `benchmarks/story-fidelity-v2-graph-coverage-2026-09-04.json`
- `benchmarks/jetson-gemma4-tensorrt-hybrid-2026-09-04.json`
- `benchmarks/overnight-20260904/final-verification.json`
- `benchmarks/overnight-20260904/renderer-summary.json`
- `benchmarks/overnight-20260904/nsight-kernels.json`
- `benchmarks/anticipatory-gke-2026-09-03-v5.json`
- `benchmarks/anticipatory-jetson-playback-2026-09-03.json`

## 14. Git and completion protocol

Before editing:

```bash
cd /Users/operator/Documents/ChatGPT/golden-ticket
git fetch origin
git status --short
git rev-list --left-right --count HEAD...origin/main
```

Do not stage the unrelated `output/`, `post/`, `post-temp-*`, or `tmp/` directories.

Before each push:

1. Run focused tests during development.
2. Run the entire Python suite.
3. Run both JavaScript suites.
4. Reproduce every changed benchmark artifact.
5. Run scoped Ruff and `git diff --check`.
6. Scan the staged diff for credentials and private source data.
7. Review `git diff --cached --stat` and every staged filename.
8. Deslop: remove duplicate helpers, corpus-shaped logic from general modules, formatting-only
   churn, speculative comments, and claims unsupported by evidence.
9. Commit a coherent change.
10. Fetch again and push only a fast-forward to `main`; never force-push.

## 15. First message to give Astra

Paste this into the new session:

> Work from `/Users/operator/Documents/ChatGPT/golden-ticket` and read
> `docs/astra-handoff-2026-09-04.md` completely before taking action. Continue from commit
> `583be0716a2fc6b3d182904e8c1441ed9ba5e11b` or its verified descendant. Your immediate objective
> is the live four-line TensorRT-to-SceneFacts adapter and matched 512-record public development
> gate described in sections 10 and 11. Orchestrate non-overlapping agents as described in section
> 12, keep accepted runtime defaults unchanged, do not open hidden data, do not perform billable
> work until local gates pass, and do not stop at a plan: implement, adversarially test, benchmark,
> deslop, document, and push reviewed fast-forward changes to `main`. Lead with verified evidence,
> never overstate image/video/model results, and preserve all user-owned untracked files.

## 16. First-session success condition

The new session is successful if it ends with one of two evidence-backed outcomes:

1. **Candidate advances:** a real, opt-in live adapter exists; the 512-record matched public A/B is
   complete; privacy is perfect on the measured set; semantic accuracy materially improves; local
   latency remains acceptable; and the accepted appliance remains unchanged pending image A/B.
2. **Candidate is rejected cleanly:** the adapter or hybrid protocol fails a defined gate; the
   failure is documented with reproducible evidence; no production setting changes; no hidden data
   is opened; and the next bounded hypothesis is stated without inventing success.

Anything short of a real adapter plus matched evidence is progress, not completion.
