# Prepared renderer: current voice quality screen

The next decision is whether the modern voice compiler fixes the extra-subject
failure while retaining watercolor detail. This screen changes prompts, not
weights, precision, resolution, steps, depth generation or compilation. It is
prepared locally and has not run on a GPU.

## Fixed workload

Retain the first two legacy cases verbatim as the actual 128- and 256-token
warmups. After the existing 30-second idle interval, render those two controls,
the six modern cases below, then repeat the controls. This is twelve renders:
two warmups and ten measured deliveries, including six new images.

The source fixture receipt is
[`modern-voice-cases.json`](../benchmarks/prepared-gcp-2026-09-08/support/modern-voice-cases.json).
Only each selected case's identifier, compiled prompt and seed belong in the
cloud workload. Typed facts, source sentences and omitted locations stay local.
These fixtures exercise the real compiler, but do not measure speech recognition
or learned fact extraction. The owl and golden-boat cases are deferred.

The [materialized workload](../benchmarks/prepared-voice-quality-2026-09-08/cloud/cases.json)
and [local profile](../benchmarks/prepared-voice-quality-2026-09-08/local/profile.json)
were generated and verified locally. The external profile pin is
`57d202f3fff9b6a25efe4e3aa29853576403c95a92ccd8f605e918178a2c9c61`.
Verification recompiles the fixtures, checks retained warmup/tokenizer evidence,
and rejects changed prompts, seeds, ordering, extra files and provenance drift.
All 59 session-experiment tests and scoped Ruff checks passed. Independent
correctness and deslop reviews found no remaining blocker for local preparation.
No new image, latency or cloud-release result is claimed.

| Modern case | Required visible facts | Clear failure examples |
| --- | --- | --- |
| Silver fox | One silver fox carrying one blue lantern in a forest; one golden ribbon beside it | Second fox or lantern; lantern detached from the carrying action |
| Two foxes | Two silver foxes carrying one blue lantern in a forest; one golden ribbon beside them | One or three foxes; duplicated lantern |
| Cat and mouse | One cat chasing one mouse in a dark alley | Merely facing each other; mouse missing; reversed pursuit |
| Pink fox | One pink fox visibly jumping over one stream | Standing beside the water; ordinary orange fox; extra fox |
| Two boats | Exactly two red paper boats floating side by side on a calm blue pond | Wrong count or color; ordinary full-size boats |
| Child and book | One child standing on one wooden bridge holding one open green book with both hands | Seated child; closed or wrong-color book; missing bridge; book floating separately |

London is deliberately omitted from the cat prompt. No reviewer should score
London landmarks as required or claim that this screen renders the city.

## Review before any demo promotion

Inspect every modern master image at full resolution and inspect its paired
depth image for broken dimensions, corrupt output or gross subject displacement.
Two independent reviews should record pass, fail or uncertain for each required
fact, plus a short description of what is visible. Resolve disagreement against
the actual image; an uncertain required fact does not pass.

Compare both control images with their retained r/s/t counterparts. Record hash
equality separately from visual judgment. If controls differ, investigate before
attributing a change in the modern cases to the compiler. The known legacy fox
failure is a control observation, not a reason to pass or fail a modern image.

For every modern image, also inspect pigment variation, paper texture, layered
depth and readable composition. A flat or simplified image does not satisfy the
user's restored-detail requirement even when its object count is correct.
Retain original JPEGs, a labeled local contact sheet, both reviews and any
disagreement. Automated hashes and timings cannot supply visual acceptance.

A candidate passes this small screen only if all six modern images satisfy their
required facts and detail review, both controls are accounted for, and all ten
measured deliveries succeed within the existing median ≤1 second / maximum ≤2
seconds screen. A miss produces a specific next hypothesis; it does not justify
loosening the rubric or silently selecting another seed. Six successful modern
images would still be a demo screen, not a general accuracy estimate.

## Finite execution and demo boundary

Use a new, explicitly admitted service/profile identity. Preserve the n–t
archives. Validate the exact selected prompts, order, seeds, compiler and
tokenizer provenance before building. Keep the reviewed immutable parent,
private IAM, single GPU, twelve-render limit, 300-second session, finite build
and supervisor deadlines, dispatch claims, deletion and release verification.
The current manager admits only the completed legacy program; preparing a local
profile does not authorize a new deployment.

The prior $25 program has $24.50 in internal reservations. A further $3.50
reservation needs $3 more allowance; it is not an invoice or a cloud hard cap.
No reservation is recycled on the assumption that a failed request was free.

After a passing quality screen, the serving wrapper still needs a separately
verified image and an external finite lifecycle owner before a demo can use it.
The owner must bind the deployed service, revision and deletion deadline to the
adapter, prepare the actual worker, and delete it even if the demo disconnects.
Closing the HTTP adapter alone does not delete the GPU service.

The demo should expose preparation, ready, expired and failed states. It must
retain the previous complete scene until both replacement assets validate,
refuse stale worker identities, and reconcile an ambiguous accepted submission
without another paid generation. Exercise that boundary with the existing local
voice planner before measuring microphone-to-projector latency. The subsecond
renderer measurements exclude those stages.

## Integration handoff after quality approval

The smallest demo integration is an explicit candidate backend in an isolated
local API process, leaving the production `gcp_resilient` route unchanged.
The following work is still required; none of these settings currently activates
`PreparedKleinProvider`.

- In [`config.py`](../src/bookforge/config.py), add an opt-in backend and validated
  service, revision, external-owner identity and persistent claim-directory
  settings. Reuse the GCP URL, impersonation and timeout settings, bounded to the
  adapter's 180-second maximum. The URL is also its identity-token audience.
- In [`live_scene.py`](../src/bookforge/live_scene.py), add a dedicated
  `build_live_scene_provider` branch wrapping
  [`PreparedKleinProvider`](../src/bookforge/prepared_klein_provider.py) in
  `FiniteModalLiveSceneProvider`. Fix the profile to 1024 × 576, four steps,
  guidance 1, deferred fidelity, and no preview, motion or automatic prewarm.
  Existing defaults are two steps, guidance 4.5 and inline fidelity; substituting
  the adapter without changing those would reject every request. Use the concise
  render contract to avoid appending the older focal prompt, and verify that
  reviewed-description requests still produce the exact modern compiled prompt.
- In [`api.py`](../src/bookforge/api.py), pass the candidate settings into that
  factory and align its completed-pack compiler revision with the chosen render
  contract. The existing local-only prewarm and warm-status endpoints already fit
  the adapter interface. Preparation must be an explicit owner action before
  microphone use; polling status must not provision or restart the service.
- Isolate the demo's story store, generated-output directory, asset cache and
  session identity. The current completed-story lookup checks planner provenance
  and input identity, but does not bind a renderer identity. Sharing that store
  could replay an older provider's image as a native demo result. If stores must
  be shared later, add exact provider/model/revision cache eligibility first.
- Keep [`finite_modal_provider.py`](../src/bookforge/finite_modal_provider.py)'s
  reviewed-description privacy checks, digest confirmation and asset promotion.
  Its promotion helper already reads model provenance from the returned bundle.
  For local recovery after packaging failure, explicitly call
  `load_prepared_klein_bundle` with the exact request, service, revision and owner;
  the generic finite loader is not the Klein loader. Recovery must make no HTTP
  request, preserve original cost provenance, and never renew a readiness lease.

Do not insert this candidate into `ResilientFastSceneProvider`: the finite demo
needs one explicit backend and terminal failure, without an alternate paid route.
Its strict saved manifest also must not acquire the router's extra fields.
Use a separately verified image of
[`prepared-klein-serving/app.py`](../experiments/prepared-klein-serving/app.py),
not the fixed-case benchmark endpoint. Bind its immutable image and deployment
resources in the external owner's receipt; an owner-ID string alone is not that
verification. The owner must retain its cleanup obligation after API exit,
lease expiry or request failure.

Before deployment, add mocked factory/API tests for exact profile and prompt
preservation, zero HTTP before preparation or after expiry, refusal without a
fallback, readiness during close, and bundle promotion/recovery without another
dispatch. Include a completed-cache test showing an old provider's image cannot
satisfy the demo request. Then exercise the reviewed-description-to-ASGI-worker
path with a fake renderer and both JPEGs. These are local implementation checks;
the existing $0.50 unallocated allowance does not fund another $3.50 trial.

## Following cold-start experiment

The supervisor currently waits a fixed 120 seconds after granting access. Google
documents policy propagation as typically two minutes but potentially seven
minutes or longer, so that sleep is neither a readiness proof nor a guaranteed
upper bound. A separate prospective experiment should replace it with bounded,
authenticated, non-rendering status checks within the same supervisor deadline.
An authenticated `NEW` response proves access, not model readiness; the following
single preparation request must still prove both warmups. Stop polling on access
success; retain cleanup time on failure. Never retry a generation
POST as a readiness check. This is a hypothesis for reducing the access wait before preparation,
not a measured gain. Keep it separate from the prompt-only quality screen.
[IAM propagation](https://docs.cloud.google.com/iam/docs/access-change-propagation).

Cloud Run's GPU guidance also calls for model-aware readiness. A listening port
is insufficient: this implementation must still validate its actual warmups,
worker identity and unexpired lease before accepting a scene. Preserve that
contract when changing startup behavior.
[GPU inference guidance](https://docs.cloud.google.com/run/docs/configuring/services/gpu-best-practices).
