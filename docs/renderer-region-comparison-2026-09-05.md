# Region-controlled renderer comparison

Status: the approved attempt failed during container import, before model initialization.
One SDK warmup was dispatched; no images completed and no HTTP generation operation was dispatched. Both applications
are stopped with zero containers. The packaging defect is fixed and tested locally; the repair
has not been deployed. The accepted provider and previous latency evidence remain unchanged.

## Attempt and repair

The deployed service imported `modal_klein_latency.py`, but Modal had not included that sibling
module in the container. Its log reported `ModuleNotFoundError` before `RegionRuntime` initialized.
The original local tests inherited the repository import path, so they missed the missing file.

The repair explicitly mounts that module. A new isolated test copies only declared runtime mounts
and the service entrypoint into a fresh directory, then imports them in a separate Python process
without repository paths or installed packages. Removing the new mount reproduces the original
failure. Model initialization and provider calls are excluded from this check.

The original [manifest](../benchmarks/renderer-region-2026-09-05/manifest.json),
[journal](../benchmarks/renderer-region-2026-09-05/journal.jsonl), and
[rejected summary](../benchmarks/renderer-region-2026-09-05/summary.json) are retained against
deployment revision `ba963b1e1a151e0f74aeb41138313d487e200560`. Use that revision to reproduce
the failed summary: the repaired source intentionally no longer matches the failed manifest.
The consumed attempt marker and all request IDs remain untouched.

The [repair draft](../benchmarks/renderer-region-2026-09-05/repair-draft-manifest.json) uses a new
experiment identity and the corrected deployment hash. It has no expiry or authorization and
cannot execute. A fresh run needs a funding-capacity decision before activation; the previous
reservation cannot be silently reused or released based on provisional billing.

Manual stop commands succeeded. The supervisor then attempted its own cleanup and received
"already stopped" for both apps; those return codes remain recorded as 1. Separate app-state and
container-list checks confirm shutdown in the [cleanup receipt](../benchmarks/renderer-region-2026-09-05/cleanup-cost.json).
The temporary proxy token was deleted and its local and Jetson credential files removed. The
reader stayed ready and the accepted planner process retained its original start time.

## Frozen comparison

The [first comparison](renderer-latency-results-2026-09-05.md) used different continents and clouds.
The new deployments request **AWS `us-west`** and require observed **AWS `us-west-2`**. Both routing
proxies explicitly use `us-east`, preserving the SDK spawn/get path. Placement is checked before
model initialization and in every successful client response, including warmup. Modal allocates
the GPU container before that startup check; a refused placement can still incur bounded cost.

The two new applications are `bookforge-klein-region-sdk` / `RegionStudio` and
`bookforge-klein-region-http` / `RegionServer`. They reuse the pinned runtime, cache, baked weights,
four steps, dimensions, seeds, depth model, JPEG encoding, authentication and cancellation controls.
Original comparison files are preserved so its summary and image proofs still reproduce.

The draft retains the same six sanitized prompts, paired alternating order, two prewarm calls,
and 12 measured-image calls: **14 operations and 16 images**. Request IDs are new. Qualification
still requires a 25% median improvement, no p95/maximum regression, identical paired image/depth
bytes, no failed or unresolved requests, stable warmed containers and verified shutdown. There is
no automatic retry, region fallback or extra image allocation. Six observations per transport
remain an engineering screen, not a production latency guarantee.

## Funding boundary

Modal documents a **1.75× multiplier on GPU, CPU and memory** for the `us-west` selector. The base
L4 + eight CPU cores + 64 GiB rate is $0.00046888 per second; including 120 seconds startup,
180 seconds execution, 90 seconds idle and 30 seconds shutdown gives $0.3446268 after the premium.
Rounding the previous operation margin upward gives **$0.39 × 14 + $1.50 setup = $6.96**.
Sources: [region pricing and selectors](https://modal.com/docs/guide/region-selection) and
[resource rates](https://modal.com/pricing).

Before approval, the canonical ledger held $13.83 in previous reservations. Reported workspace
usage was $13.83014709; accounting for that report left **$0.32771747** under the then-$14.85 phase
cap. Even setup alone did not fit. The proposed experiment brought conservative projected
workspace usage to **$34.62014709**, above the then-$28 stop.

The user approved the [proposal](../benchmarks/renderer-region-2026-09-05/budget-proposal.json):
$7 of separate paid funding, retaining $30 of credits and the $2 reserve. The canonical ledger
and shared plans now use a $35 workspace stop and $21.85 cumulative phase cap. The
[amendment](../benchmarks/renderer-region-2026-09-05/budget-amendment.json) preserves prior records,
estimates and reservations and records the old/new ledger hashes.

The new $6.96 reservation remains held, bringing total held reservations to $20.79. Actual reported
workspace usage stayed at **$13.83014709** immediately after shutdown; the new run had no attributed
charge yet. This is provisional, not proof of free GPU startup. Reservations are spending ceilings,
not actual charges, and can overlap already-reported usage until reconciled. The earlier $0.33
figure described internal unreserved capacity, not the remaining credit balance.

`BudgetEnvelope.authorized_paid_usd` defaults to zero. Legacy ledgers remain readable, and merely
changing a plan cannot expand an existing ledger: strict envelope equality still requires an
explicit amendment. Invalid paid values and an unapproved plan/ledger mismatch fail closed.

## Activation procedure for a separately funded attempt

1. Read fresh billing and recheck the bounds. Under the existing ledger lock, amend only the
   explicitly approved funding/phase fields, preserving all reservations, records and estimates.
   Update shared active plan fields to match; never reset the ledger or raise the credit amount.
2. Prepare the same manifest with `status="authorized"` and an expiry no more than two hours ahead.
   Reserve $6.96 centrally using the existing budget function and bind the private authorization
   receipt to those exact prospective bytes. Record its SHA separately in the operator invocation.
3. Use `prepare_klein_region_comparison.py --activate-until` with that receipt and SHA to write
   a fresh active manifest. Preserve the failed run's manifest; updating the deployment's manifest
   path changes its source hash, so freeze that path before issuing the new authorization. The
   checked-in repair draft cannot execute. No private authorization is checked in.
4. Create a temporary proxy token, deploy the two new applications and verify unauthenticated
   rejection plus metadata lookups. Run `benchmark_klein_region.py --execute` from an isolated
   Jetson checkout with the exact authorization SHA. Its permanent per-home experiment marker
   prevents restarting the run from another output or authorization directory.
5. Supervise the finite run, stop both applications on completion or failure, verify zero
   containers, revoke the token and retain provisional billing. Recompute the summary from saved
   artifacts before making a promotion decision.

Session prewarming remains gated on successful transport qualification. This increment does not
enable paid work from the reader UI or change the separate failed visual-correctness result.

## Offline verification

Create and check the repaired draft without credentials or provider calls:

```bash
.venv/bin/python scripts/prepare_klein_region_comparison.py \
  --experiment-id klein-region-20260905-b --output /tmp/region-draft.json
.venv/bin/python scripts/benchmark_klein_region.py \
  --manifest /tmp/region-draft.json --output /tmp/region-preflight
```

Both output paths must be fresh. Tests exercise the full 14-operation journal with a fake provider,
the real preparation CLI through server startup with a fake model, render and warmup placement
refusals, strict response identity, numeric type tampering and funding isolation. They do not
measure regional cloud latency. The [draft and funding check](../benchmarks/renderer-region-2026-09-05)
are retained for review. After the repair, the full suite passed 891 Python tests; both JavaScript
suites and scoped lint passed. The new draft reproduces exactly, and the original latency code
remains unchanged. The earlier public coverage and latency-summary checks remain in the
[preparation verification record](../benchmarks/renderer-region-2026-09-05/verification.json).
