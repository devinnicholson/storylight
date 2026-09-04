# Overnight candidate: measured results

September 4, 2026. Decision: keep Klein opt-in; do not replace the accepted live renderer.
The candidate is integrated through real jobs, verified artwork/depth assets, and the normal
projector. It creates new artwork animated with depth, **not generated video**.

## Try it

- Candidate workbench: <http://127.0.0.1:18087/workbench?session=klein-candidate>
- Local comparison gallery: <http://127.0.0.1:18088/?split=validation&start=0>
- Accepted Jetson route, engine, power mode, and projector defaults were not promoted or replaced.

The Mac must remain awake and the local processes running for these links. Click **Prepare full
path**, then **Generate moving scene**. Opening the page alone does not start cloud GPU work.
Preparation explicitly reserves $3 of the shared experimental allowance once per server process;
each prewarm/generation debits a conservative $0.25 against that reservation. These are spending
reservations, not measured charges. Reservations survive failure/restarts and are not automatically
refunded based on delayed billing.

To restart locally, from the repository, keep this private tunnel running:

```sh
ssh -N -o BatchMode=yes -o HostKeyAlias=jetson.local \
  -i /Users/operator/.ssh/bookforge_jetson \
  -L 18435:127.0.0.1:11435 operator@192.0.2.10
```

In another terminal:

```sh
.venv/bin/python scripts/run_klein_candidate.py
```

For the gallery, run `.venv/bin/python scripts/serve_renderer_review.py`. Generated images remain
in `.bookforge/overnight-20260904`; they are not downloaded by a fresh Git clone. The deployment
entry point is `deploy/modal_klein_scene.py`, using the existing pinned model image and private
compiler-cache volume. Rebuilding that cache is a separate explicitly budgeted experiment, not
an automatic fallback. The current candidate accepts only qualified 128/256 token buckets;
larger prompts fail rather than silently truncate or compile an unqualified shape.

Rollback is simply closing the candidate and using the accepted workbench. No production
configuration changed. Stop the local process with Ctrl-C; the private Modal deployment has
zero minimum containers and a 90-second idle scale-down window. No public inference endpoint.

## What changed

1. Added the opt-in pinned FLUX.2 Klein 4B L4 provider, reusing the existing job, asset, privacy,
   provenance, projector, and budget interfaces. Image and depth bytes, dimensions, model
   revisions, seed, and token bucket are checked before publication.
2. Moved conservative session authorization out of every render: one billing lookup/reservation
   at session start, then bounded local accounting. No automatic paid retries; cancellation
   explicitly terminates the remote call/container.
3. Added a short render contract over actual, locally sanitized Gemma output. Removed a second
   action-word truncation that discarded relationships, preserved “side by side” and negative
   coordination, and fixed a false positive on generic “Nothing glows.” Explicit proper names
   stay protected. Contract and semantic caches are versioned separately; full/concise completed
   packs cannot masquerade as one another.
4. Removed automatic paid cloud preparation on opening/rehearsing the workbench. Explicit
   preparation now uses a 90-second idle window, not a 600-second warm lease.
5. Hardened opt-in MediaPipe handling for changed camera/resolution/projection, duplicate results,
   malformed calibration, interrupted permissions, hidden pages, and cleanup.
6. Captured Nsight node-level kernel evidence with request timestamps, then restored and checked
   the live Jetson planner, kiosk, and API. No engine rebuild or power-mode change.

## Speed evidence

All cloud comparisons used the same Klein model, dimensions (1024×576), four steps, style,
and matched seeds. This is **not** a matched performance comparison against Vertex or SANA.

| Boundary | Observed result | Limitation |
|---|---|---|
| Validation warmed artwork + depth + JPEG, full contract | median 1.909 s | 22 second-repetition cases |
| Same, concise contract | median 1.838 s | About 3.7% faster; not a large kernel speedup |
| Three new real HTTP jobs after session authorization change | 4.200 / 4.015 / 4.119 s | Gemma and scene cache misses; warm L4 |
| Local planning in those jobs | 1.077–1.175 s | Includes tunnel/network and application work |
| Inference in those jobs | 1.711–1.719 s | Artwork + depth; packaging reported separately |
| Remaining provider overhead | 1.211–1.304 s | RPC, transport, orchestration; not GPU inference |
| Separate browser rabbit scene | backend 5.066 s, activation 24 ms | Activation follows asset delivery; 320 ms visual blend |
| Separate exact completed-scene replay | backend 8 ms, activation 22 ms | Local cache replay, **not new generation** |
| Fresh-container model loading | 12.85–13.02 s | Separate from compilation setup and first render |
| First render after compiler-cache restore | 22.32–22.54 s | Cold penalty remains; do not advertise warm latency cold |

The earlier same-fox browser sample before session accounting changed had 3.124 s provider
overhead; later samples were lower, but prompts/network conditions differ. This is evidence
of a removed repeated billing boundary, not a statistically controlled speedup estimate.
58 of 60 repeated same-seed image pairs were byte-identical; deterministic output is not guaranteed.

The HTTP smoke ran before the final completed-cache version marker was added. A final browser
scene verified `klein-concise-v1` end to end: 3.821 s backend, prepared local-plan cache hit,
new artwork, 25 ms activation, and no browser error logs. It reused a validation passage for
integration checking, not for another claimed accuracy measurement. Detailed identities and
test results are in [final-verification.json](../benchmarks/overnight-20260904/final-verification.json).
Pinned identities, samples, checksums, timings, and
reported costs are in [renderer-summary.json](../benchmarks/overnight-20260904/renderer-summary.json).

## Accuracy gate: not passed

Frozen original corpus: 8 development, 24 validation, 4 separate privacy/adversarial cases.
The sealed training holdout was not used. Actual Jetson Gemma produced accepted local contracts
for 22/24 validation cases; two stopped before paid rendering. The matched batches produced
120 artwork/depth pairs (240 verified files), including two repetitions.

The first repetition was visually inspected against the original requirements. This is a single,
unblinded assistant review, not an independent human evaluation or benchmark of general accuracy.
Under a conservative all-visible-requirements screen, full-contract images passed 5/24 cases,
concise images 10/24. Ambiguous results and absent evidence of a transformation were not passes.
Those low results are why production promotion was rejected, despite attractive individual images.

| Case | Full contract | Concise contract |
|---|---|---|
| v01 leafless tree / two kites | Tree absent | Trees leafy |
| v02 three ducks in row | Scattered arrangement | Pass |
| v03 red bicycle / yellow wall / no rider | Pass | Pass |
| v04 mouse / blue cup / both paws | Duplicate mouse | Pass |
| v05 kneeling child / closed purple book | Duplicate child, open book | Book still open |
| v06 owl above door / key below branch | Duplicate owl, wrong placement | Key/branch relation unclear |
| v07 hedgehog left of empty wagon | Pass | Pass |
| v08 dog under table / ball on table | Duplicate dog, wrong placement | Dog intersects table, ball on head |
| v09 fox / blue book / red book on shelf | Duplicate fox | Pass |
| v10 | Planner rejected | Planner rejected |
| v11 fish jumping from bucket | Pass | Pass |
| v12 feather becomes boat | Separate objects; no transformation | Separate objects; no transformation |
| v13 teapot water becomes flowers | Duplicate teapot | Transformation not established |
| v14 touched stone becomes lantern | Duplicate otter; no transformation | Duplicate otter; no transformation |
| v15 rabbit leaves rope / badger raises flag | Badger/flag absent, duplicate rabbit | Badger/flag absent, rabbit holds rope |
| v16 | Planner rejected | Planner rejected |
| v17 two-tower sandcastle | Pass | Pass |
| v18 sleeping cat inside suitcase | Duplicate cat and suitcase | Pass |
| v19 four flowers / one blue pot | Wrong count, pot absent | Two brown pots, wrong count |
| v20 frog / mushroom / rain / no umbrella | Pass | Pass |
| v21 child between differently colored trees | Required trees absent | Required trees absent |
| v22 one bear pulling empty sled uphill | Duplicate bear | Pulling/empty relationship unclear |
| v23 red cube in transparent bowl | Wrong color assignment, robot in bowl | Wrong color assignment, robot in bowl |
| v24 one snail / striped green shell / wall | Duplicate snail | Pass |

Development also exposed wrong boat counts, open instead of closed umbrellas, missing bowls,
and lost cave/direction constraints. Validation was not silently reused to tune further repairs.
The important next accuracy work is richer **private** actor/object/relation extraction and fresh
evaluation, not adding more generic cloud prompt instructions. A renderer cannot recover facts
the private planner already discarded. Temporal transformations require a real motion-generation
path; depth animation of a still frame does not satisfy that requirement.

Privacy checks: proper-name cases and a contact-data case failed closed; the separate injected
password instruction was ignored and did not appear in sanitized output. No adversarial case
was sent for image generation. This small screen is not a privacy certification.

## Nsight and hand tracking

The node-level trace covered three short synthetic direct-HTTP prompts, not production planning.
Unprofiled mean: 750 ms. Profiled mean: 874 ms; instrumentation overhead about 16.5%.
Kernel interval unions occupied 80.2–80.8% of those profiled HTTP windows. CUTLASS and TensorRT
GEMM kernels dominate; the previous trace already confirmed CUDA Graph use. Startup and warmup
were excluded from these totals. We did not claim a new engine optimization or a TensorRT speedup.
See [request-window kernel summary](../benchmarks/overnight-20260904/nsight-kernels.json).

MediaPipe regression coverage includes 30 start/stop cycles and 1,000 fake worker results with
bounded timers/one worker, plus missing-model errors, permission races, changed mapping, pauses,
and duplicate responses. These are deterministic lifecycle tests, **not physical FPS evidence**.
Tomorrow: connect webcam, grant camera access, calibrate the four corners, and measure actual
Jetson playback cadence with tracking off/on. Keep tracking disabled if it exceeds the existing
latency/cadence limits.

## Cost and deployment boundary

The captured Modal report attributed **$0.39271299** to the three apps from this pass; total
workspace month-to-date was $13.53057755. Reporting can lag: neither number is a final bill or
a remaining-credit balance. The additional allowance was $10. Experimental comparison apps
were stopped and the candidate had **zero tasks** in the captured state.

No new GCP infrastructure, IAM, billing configuration, public endpoint, or live-provider
promotion was performed. Existing budget alerts/disconnect settings were not changed.
The candidate's per-call reservation includes startup, compute, CPU/RAM, and idle allowance;
the UI's separately labeled estimated GPU-only cost is not the complete invoice cost.

The verification skill caught the unintended automatic cloud warmup and traced actual UI →
private planner → provider → verified assets → projector behavior. Deslop review kept the
candidate on existing interfaces and removed duplicate runtime metadata work.

Final verification: **1,167 Python tests passed**, both JavaScript suites passed, scoped Ruff
checks passed, and `git diff --check` was clean. The existing Starlette/httpx deprecation warning
remains; it did not fail the suite.
