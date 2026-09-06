# Simple scenes and reference continuity

Status: implemented locally; the twelve-image trial has not run. No speed or image-correctness
improvement is established. The current funding envelope cannot cover its $1.75 ceiling.

Local verification: 923 Python tests and all three JavaScript suites pass, along with scoped
Ruff, diff checks and the staged credential scan. The draft manifest and offline tokenizer
preflight reproduce exactly. Independent review covered reference binding, incomplete ratings,
corrupt artifacts, deadlines and transient cleanup failures. GPU responses in tests are mocked;
these checks establish implementation behavior, not model quality or cloud timing. See the
[verification record](../benchmarks/simple-scenes-2026-09-05/verification.json).

The user prefers reliable, fast, legible artwork over fine detail. This experiment uses simple
flat illustrations to test essential counts, bindings, actions and continuity. It is a small
engineering comparison, not an accuracy estimate for arbitrary stories.

## Fixed comparison

The [four public synthetic controls](../experiments/renderer-fidelity/simple-scene-controls-v1.json)
cover carrying an object, two distinct actors, a spatial transition and opening a box. Each has
a before state and after state with explicit fact checks. Every control generates exactly three
images: the before state, a text-only after state, and an after state conditioned on that exact
before JPEG. Text and reference variants share the same target prompt and seed; their order
alternates between controls. There are twelve image calls, no extra warmups, retries or rerolls.

The first comparison fixes 1024×576, four steps and guidance 1.0. It reuses the pinned Klein
runtime and text compiler cache. Reference conditioning adds encoding and transformer work;
its new compilation shape and first execution must be measured. This is not eager execution.
The pinned Diffusers 0.39.0 implementation accepts a reference image but has no mask or edit
strength control. It does not guarantee unchanged pixels or objects. See its
[pipeline](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/flux2/pipeline_flux2_klein.py)
and [image processor](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/flux2/image_processor.py).

Lower resolution and fewer steps remain separate experiments. Earlier SANA tests saved only
18.8 ms from a resolution cut and 65.6 ms from one-step sampling; the latter duplicated a subject.
Those results do not predict Klein's behavior. Changing several settings now would obscure
whether reference conditioning helps.

## Measurements and decision

- Retain every call and failure, including the first text call and first reference call.
- Measure client start through reference read/upload, generation, download, verification and
  artifact storage. Server inference timing alone is insufficient.
- Report the first text call (ordinal 0) and first reference call (ordinal 2) separately.
  The four later text targets and three later reference targets each have a maximum 2.5-second
  artifact-ready gate. Report all-call distributions as well. This tiny sample is not a service SLO.
- Verify all twelve master/depth pairs, exact source and model identity, US placement and a
  single container. The offline tokenizer preflight checks all eight prompts against the exact
  cached tokenizer revision; all use the 128-token bucket.
- Build a blind A/B gallery showing both states and all required facts. The shared first image
  is never rerolled or preselected for correctness. A failed first-state fact fails both options; an unresolved
  first-state check prevents either option from passing. Unrated facts stay unrated.
- Score recognizable subjects, correct relationships, continuity and legibility. Fine detail and
  overall aesthetic ratings do not determine correctness. Browser review does not establish
  physical projector legibility.

A reference variant advances only if it corrects a text-only failure without losing controls
already passing, and its delivery/timing gates pass. Equal scores provide no evidence to pay
reference overhead. If text-only simplicity works better, carry that result forward. If both
fail, record the actual failure classes before choosing another experiment. No automatic
production promotion follows a benchmark or a review export.

## Implementation and local operation

- [Reference runtime](../deploy/klein_reference_runtime.py) adds one strictly bounded RGB JPEG
  input while retaining the old runtime and compiler identity unchanged.
- [Modal declaration](../deploy/modal_klein_simple_scenes.py) permits one L4 worker, eight CPU
  cores, 64 GiB RAM, one concurrent request, no snapshots, 120-second startup and 60-second input
  timeouts, and 90-second idle scale-down. Durable claims allow one heavy initialization and
  one attempt for each fixed request. The authenticated method accepts no arbitrary prompt.
- [Benchmark](../scripts/benchmark_simple_scenes.py) prepares source-pinned manifests, performs
  offline preflight, dispatches the fixed sequence and revalidates all evidence on aggregation.
- [Review tool](../scripts/review_simple_scenes.py) builds the gallery and scores its export against
  the original journal, manifest and images. Its private A/B key stays outside the served gallery.
- [Supervisor](../scripts/run_simple_scene_trial.py) checks authorization, ledger, source pins,
  tokenizer and live resource metadata; enforces a 600-second global deadline plus 60 seconds
  for cleanup; stops the exact trial app and records final inventory.

Use the existing tokenizer environment with downloads disabled:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_CACHE=/private/tmp/bookforge-visual-hf-cache \
  /private/tmp/bookforge-tokenizer-env/bin/python scripts/benchmark_simple_scenes.py \
  --preflight --manifest benchmarks/simple-scenes-2026-09-05/manifest.json \
  --output /private/tmp/bookforge-simple-scenes-fresh-preflight
```

The tracked manifest is a draft. Before a paid run, refresh billing/inventory, reserve the
complete ceiling under an authorized envelope, create the private SHA-bound authorization, and
set the manifest's bounded expiry. Use the supervisor's `--help` for exact required arguments.
Do not use direct deploy or benchmark `--run` as a substitute for supervision. A failed attempt
requires a new reviewed experiment identity; deleting claims to retry invalidates the protocol.

Once funded, with `SIMPLE_SCENE_AUTH_SHA256` set to the newly created authorization file's SHA-256:

```sh
HF_HUB_CACHE=/private/tmp/bookforge-visual-hf-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /private/tmp/bookforge-tokenizer-env/bin/python scripts/run_simple_scene_trial.py \
  --authorization /private/tmp/bookforge-simple-scenes-auth/authorization.json \
  --authorization-sha256 "$SIMPLE_SCENE_AUTH_SHA256" \
  --ledger /Users/operator/Documents/ChatGPT/golden-ticket/.bookforge/overnight-candidate/modal-ledger.json \
  --output /Users/operator/Documents/ChatGPT/golden-ticket/.bookforge/simple-scenes-20260905-a
```

That temporary Python 3.12 environment contains Modal 1.5.5 and Transformers 5.13.0. No GPU model
is loaded locally; preflight uses the cached tokenizer only. The output directory must be fresh.

After a successful supervised trial, run `review_simple_scenes.py` with the same manifest,
authorization and authorization SHA, `--output` pointing to the benchmark artifact directory,
and fresh `--gallery` and `--key` paths. The key's parent must be private (0700). Serve only the
gallery on loopback. Run the same command with `--review` pointing to the human export to score
it. Keep the raw export and sanitized summary; never replace missing ratings with inferred ones.

## Cost boundary

The [budget preflight](../benchmarks/simple-scenes-2026-09-05/budget-preflight.json) retains the
audited proof hashes and arithmetic. The latest retrieved workspace report is $14.51014746;
it is delayed billing, not the complete final bill. Closed-attempt resource bounds support
refining two prior gross holds, freeing $1.15. Unresolved holds and shared-app credits remain
untouched. Pending bills are not treated as settled.

After those refinements, conservative workspace exposure is $33.81088685 and cumulative phase
usage is $20.67302229. Adding the trial's $1.75 ceiling yields $35.56088685 workspace exposure
and $22.42302229 phase usage, above the authorized $35 and $21.85 limits. No trial reservation
has been created. One additional dollar of paid authorization would permit a $36 workspace
stop and $22.85 phase cap while retaining the $2 reserve, subject to fresh billing preflight.

The $1.75 trial allowance rounds up a $1.68157760 bound: two full 660-second resource slots,
four 30-second teardown allowances and $0.50 setup overhead at the retained maximum rate of
$0.00082054 per second. These are conservative operating controls, not a provider billing cap.
Any unexplained work or cleanup failure remains reserved and blocks another trial.

If a rendering method qualifies, the next engineering step is a bounded reading-session
preparation path using it, with cold preparation displayed separately from scene-ready
status. Its idle cost, expiry, failure handling and first usable scene time need their own
measurements; this implementation does not claim to have solved cold starts.
