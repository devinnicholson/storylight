# Story image review: two of six candidate pages correct

Revision `ba15b484c3a91ff65ce6f7949fc851669c137298` passed the source-selected scene probe
on the existing resident Jetson model. The story and acceptance criteria remain frozen.

| Stage | Candidate outcomes | Accepted outcomes | Candidate median / p95 / maximum |
| --- | --- | --- | --- |
| 16 matched controls | 16/16: ten graphs, six required refusals | 12/16 | 1,064.1 / 1,196.3 / 1,196.3 ms |
| Seven story inputs | 7/7 complete proved graphs | No requests in this stage | 1,205.2 / 1,420.9 / 1,420.9 ms |

All 39 requests completed without failures, incomplete generations or missing timings. Accepted
control median/p95 was 1,018.6/1,282.8 ms. This is a small engineering demonstration, not an
independent accuracy estimate or a 512-case qualification of the routed prompt. The earlier
[focal graph gate](product-fidelity-color-repair-2026-09-04.md) is a separate measurement.

Evidence: [controls](../benchmarks/scene-routing-2026-09-04/controls-summary.json),
[story](../benchmarks/scene-routing-2026-09-04/story-summary.json), and adjacent request journals
and before/after provenance. Raw model responses and the source-bearing Story Pack stay on the
Jetson. The accepted appliance remains on `tensorrt_slots`.

Both report aggregates reproduce under the pinned revision. Before/after hardware hashes verify
the retained snapshots; engine, resident process, deployed files, accepted prompt, planner settings
and power mode stayed unchanged. The implementation passed 870 Python tests, both JavaScript
suites, scoped lint and unchanged public-coverage reproduction before measurement. Independent
review cleared the sanitized journals and batch; local image preflight also reproduces byte-for-byte.

## Prepared image batch

The private pack contains eight display pages: four still pages, transformation before/after,
and ordered first/then actions. The original planned pack's whole-file SHA-256 is
`d09aca8f05f2222bc72b3158566520d0377ef00805e934ad7f7acf658d0dd45f`.
The installed pack binds those pages to 16 verified artwork/depth assets and has SHA-256
`6f833c893e5c8a1eb7234c9cbdff20d33e55dd102c471609b034de51d25c0ae7`.

The [exact sanitized batch](../benchmarks/scene-routing-2026-09-04/visual-batch.json) contains eight
candidate images and the three valid original accepted contracts for pages 3, 5 and 6. Missing
accepted pages remain failures; accepted temporal pages contribute one still each. Batch SHA-256:
`4601956b9a3a39a4f5b9b802b694db70aea737e6182d03d1411499326a1b1331`.

Local [preflight](../benchmarks/scene-routing-2026-09-04/render-preflight.jsonl) passed for all 11
requests using the pinned tokenizer: candidate prompts use 55–95 tokens, accepted prompts use
186–191, all within the qualified 256-token maximum. The budget ledger did not change.

The destination was the authenticated Modal app `bookforge-klein-candidate`, class
`KleinSceneStudio`, running FLUX.2 Klein 4B on an NVIDIA L4. Only the validated visual contracts
and generation parameters are sent. The batch uses fixed seeds, 1024×576 images, four steps and
guidance 1.0. Negative prompts are retained as evidence but are not executed by this renderer.

Execution reserved $2.75 in the existing shared ledger, with no prewarm, retries or rerolls.
All 11 calls succeeded. The first cold request took 41.58 seconds end to end; ten warm requests
took a median 3.66 seconds (range 3.51–7.50). Warm image inference alone had a 1.63-second median
(range 1.61–5.24). The first 256-token request was the slowest warm request; a warm container
does not establish equal preparation costs across token buckets.

The pre-execution monthly billing report was $13.62730577; the first post-run report was
$13.66017456; a later snapshot reached $13.71088685, a total reported increase of $0.08358108.
This is provisional usage, not final batch cost or a credit balance. The full reservation remains.
After the idle window, Modal reported zero
running renderer containers. See the [render results](../benchmarks/scene-routing-2026-09-04/render-summary.json)
and [complete journal](../benchmarks/scene-routing-2026-09-04/render-journal.jsonl).

Automatic approval review initially required specific payload, destination and spend approval.
The user supplied that approval before execution. Independent review verified all 11 requests,
22 JPEG checksums, runtime identities, fixed seeds and the single shared reservation.

The [installation receipt](../benchmarks/scene-routing-2026-09-04/install-receipt.json) verifies
eight candidate pages and 16 cached assets in a fresh private Jetson data directory. The accepted
appliance was not changed. The local gallery uses neutral A/B labels and a separately stored
mapping key. The completed page-level human review is recorded below.
Render bundles and the gallery are retained locally under the ignored
`.bookforge/fidelity-display-ba15b48/` directory; the private mapping key is outside the gallery.

## Cached delivery and restart

The [Jetson rehearsal](../benchmarks/scene-routing-2026-09-04/cache-rehearsal.json) verified the
eight-page pack and all 16 assets over three complete HTTP passes, restarted its isolated API,
then verified three more passes. Each pass checked pack identity, asset bytes, dimensions and
ETags. All six passed in 658–784 ms per complete pack; API startup took 1.80 and 1.64 seconds.
These are local HTTP measurements, not screen-transition latency.

The runner disables generation, uses a prebound loopback socket on port 18089, and terminates
only its own API processes. Source-bearing responses stay on the Jetson. The stored pack hash
remained unchanged. No cloud inference is used for this rehearsal.

Its first preflight caught an installer permissions defect: recursive directory creation left
the intermediate cache folder at the device's default mode 0775. The enclosing data directory
was already 0700. The installer now creates that intermediate directory explicitly as 0700;
the existing test verifies all installed directories under a permissive umask. The isolated
installation was restricted to 0700 before the successful rehearsal.

Final verification passed 872 Python tests, both JavaScript suites, scoped lint and unchanged
public-coverage reproduction. The isolated API port closed after rehearsal; the accepted API
remained ready and the original resident model process remained present.

To repeat on the Jetson with a fresh output receipt:

```bash
PYTHONPATH=src /opt/bookforge/.venv/bin/python scripts/rehearse_fidelity_cache.py \
  --data-dir /home/operator/.local/state/bookforge/fidelity-display-ba15b48 \
  --installation-receipt evidence/install-receipt.json \
  --proof-installation-sha256 4d7f16e0c04d394b4e9a60e84ab3f33b28d3b7c9fd6933f9c90cc589435f5ba4 \
  --output evidence/cache-rehearsal-repeat.json
```

## Human review

The [original export](../benchmarks/scene-routing-2026-09-04/human-review.json) was matched to the
gallery's review ID, frozen checklists, image hashes and saved A/B mapping. The
[validated summary](../benchmarks/scene-routing-2026-09-04/human-review-summary.json) records the
result. The candidate fails
the visual gate: only pages 1 and 2 were rated correct and consistent. All nine available options
were rated legible. Individual fact annotations were left blank and remain unrated; they are not
inferred from the page ratings.

| Page | Candidate correctness | Candidate rating | Accepted correctness | Accepted rating |
| --- | --- | ---: | --- | ---: |
| 1 | Correct | 5 | No image | — |
| 2 | Correct | 5 | No image | — |
| 3 | Incorrect | 3 | Incorrect | 2 |
| 4 | Incorrect | 2 | No image | — |
| 5 | Incorrect | 1 | Incorrect | 3 |
| 6 | Incorrect | 2 | Incorrect | 1 |

The candidate's overall rating wins two paired pages and loses one, but all three paired
correctness comparisons fail on both sides. Preference does not establish fidelity. These are
one reviewer's ratings of the frozen demonstration, not a general accuracy estimate.

The summary reproduces byte-for-byte. Its importer rejects mismatched galleries and preserves
blank fact ratings; independent review and the full 874-test suite passed.

## Diagnosis and next step

A separate [assistant inspection](../benchmarks/scene-routing-2026-09-04/assistant-image-screen.json)
of the saved candidate images found duplicated lanterns on page
3, extra foxes and lanterns on page 4, four birds after the page 5 transformation, and surplus
birds, baskets and lanterns in the page 6 sequence. Page 5's feather also lacks a clear lifting
action. These observations are machine prescreen findings, not annotations supplied by the user.

The exact outbound contracts already specify the required bindings and counts. The renderer
receives only text and a seed for each independent image; it has no reference image connecting
successive states. Even the two states sharing a seed change their objects and composition.
The failure is now in rendering fidelity and continuity, beyond the passed graph-construction
gate. Loosening source validation would not repair these pictures.

The next bounded hypothesis is to preserve a verified scene as an image reference while changing
only the next grounded action. First qualify reference-image input locally, then freeze an
independent two-state control covering character identity, object count and a spatial relation
before any new paid comparison. Count and binding checks must still pass; reference conditioning
alone does not guarantee them. Keep this run's images, story and ratings unchanged, with no rerolls
or appliance promotion. Physical projection remains unverified and cannot override this failed
visual gate.
