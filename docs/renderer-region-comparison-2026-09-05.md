# Region-controlled renderer comparison

Status: implemented and locally tested; inactive pending funding approval. No new cloud deployment
or generation ran. The accepted provider and the previous latency evidence remain unchanged.

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

The unchanged canonical ledger holds $13.83 in previous reservations. Current reported workspace
usage is $13.83014709; accounting for that report leaves **$0.32771747** under the current phase
cap. Even setup alone does not fit. The proposed experiment would bring conservative projected
workspace usage to **$34.62014709**, above the existing $28 stop.

The [proposal](../benchmarks/renderer-region-2026-09-05/budget-proposal.json) requests $7 of separate
paid funding, retaining $30 of credits and the $2 reserve. Its workspace stop is $35 and cumulative
phase cap $21.85. The $6.96 is a conservative reservation, not an expected charge. This proposal
does not change the live ledger or grant credits. All existing reservations remain held.

`BudgetEnvelope.authorized_paid_usd` defaults to zero. Legacy ledgers remain readable, and merely
changing a plan cannot expand an existing ledger: strict envelope equality still requires an
explicit amendment. Invalid paid values and an unapproved plan/ledger mismatch fail closed.

## Activation after approval

1. Read fresh billing and recheck the bounds. Under the existing ledger lock, amend only the
   explicitly approved funding/phase fields, preserving all reservations, records and estimates.
   Update shared active plan fields to match; never reset the ledger or raise the credit amount.
2. Prepare the same manifest with `status="authorized"` and an expiry no more than two hours ahead.
   Reserve $6.96 centrally using the existing budget function and bind the private authorization
   receipt to those exact prospective bytes. Record its SHA separately in the operator invocation.
3. Use `prepare_klein_region_comparison.py --activate-until` with that receipt and SHA to write
   `benchmarks/renderer-region-2026-09-05/manifest.json`. The checked-in draft has no active expiry
   and cannot execute. No executable authorization is checked in.
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

Create and check a draft without credentials or provider calls:

```bash
.venv/bin/python scripts/prepare_klein_region_comparison.py \
  --experiment-id klein-region-20260905-a --output /tmp/region-draft.json
.venv/bin/python scripts/benchmark_klein_region.py \
  --manifest /tmp/region-draft.json --output /tmp/region-preflight
```

Both output paths must be fresh. Tests exercise the full 14-operation journal with a fake provider,
the real preparation CLI through server startup with a fake model, render and warmup placement
refusals, strict response identity, numeric type tampering and funding isolation. They do not
measure regional cloud latency. The [draft and funding check](../benchmarks/renderer-region-2026-09-05)
are retained for review. The full suite passed 890 Python tests and both JavaScript suites;
scoped lint and independent review passed. The public coverage benchmark and the previous
latency summary reproduced unchanged. [Verification record](../benchmarks/renderer-region-2026-09-05/verification.json).
