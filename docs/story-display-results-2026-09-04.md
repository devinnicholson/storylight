# Story construction passed; image review pending

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
and ordered first/then actions. Its whole-file SHA-256 is
`d09aca8f05f2222bc72b3158566520d0377ef00805e934ad7f7acf658d0dd45f`.
It has no generated assets yet.

The [exact sanitized batch](../benchmarks/scene-routing-2026-09-04/visual-batch.json) contains eight
candidate images and the three valid original accepted contracts for pages 3, 5 and 6. Missing
accepted pages remain failures; accepted temporal pages contribute one still each. Batch SHA-256:
`4601956b9a3a39a4f5b9b802b694db70aea737e6182d03d1411499326a1b1331`.

Local [preflight](../benchmarks/scene-routing-2026-09-04/render-preflight.jsonl) passed for all 11
requests using the pinned tokenizer: candidate prompts use 55–95 tokens, accepted prompts use
186–191, all within the qualified 256-token maximum. The budget ledger did not change.

The proposed destination is the authenticated Modal app `bookforge-klein-candidate`, class
`KleinSceneStudio`, running FLUX.2 Klein 4B on an NVIDIA L4. Only the validated visual contracts
and generation parameters are sent. The batch uses fixed seeds, 1024×576 images, four steps and
guidance 1.0. Negative prompts are retained as evidence but are not executed by this renderer.

Execution reserves at most $2.75 in the existing shared ledger, with no prewarm, retries or
rerolls. The latest pre-execution billing report was $13.62730577 month-to-date; billing may lag
and this is not a credit balance. Billing is checked again before reserving the batch.

Automatic approval review rejected execution because it requires explicit authorization for
this payload, external destination and spend. No image call ran, no attempt marker was created,
and no batch reservation was made. Generation, asset installation, human visual review and
physical playback remain pending.

## Completion path

After approval, run the frozen batch once, verify all returned identities and asset checksums,
and install the eight candidate display pages into a fresh private Jetson data directory. Build
the label-blinded gallery with its mapping key stored separately. Record human fact, legibility
and continuity ratings; an attractive but incorrect image still fails. Rehearse cached replay
three times and restart recovery, reporting browser and physical-projector evidence separately.
