# Eastern-region renderer comparison

The frozen comparison completed all 14 operations in observed AWS `us-east-1`, with routing
through `us-east`. HTTP delivered verified artwork in a **2.200-second median**, versus SDK
**2.611 seconds**: **15.72% faster**, below the unchanged 25% gate. All six master/depth pairs
matched byte for byte, with zero request failures. Promotion is rejected. The SDK also replaced
its warmed container, failing the warm-bucket and container-stability gates.

The western attempt exhausted its total client deadline after a capacity wait and late warmup
compilation. Its final logs and corrected
diagnosis remain with [the western recovery](renderer-region-recovery-2026-09-05.md).

The model, revision, L4, eight CPU cores, 64 GiB RAM, dimensions, four steps, seeds, synthetic
warmups, six measured pairs, image/depth byte-equality gate and 25% median improvement gate remain
unchanged. The new experiment identity is `klein-region-20260905-c`; prior request IDs and attempt
markers are preserved. No automatic retry or region fallback is enabled.

## Existing image and bounded funding

Both previous deployment layouts identify `im-WtXer8GjRPdgMqWAAUSMwJ` as their image. The candidate
now uses `Image.from_id` and explicit runtime mounts. It cannot trigger package installation or a
weight bake; a missing image fails lookup. The manifest pins this identity, and the runtime checks
it before model initialization. The isolated import test rejects candidate image build steps and
still reproduces the original missing-module failure when that mount is removed.

The run reserves **14 × $0.39 + $0.50 setup = $5.96**. The smaller setup allowance depends on this
existing-image-only deployment, not an assumption that build cache will hit. Narrow-region pricing
remains 1.75× on GPU, CPU and memory. [Modal region selection](https://modal.com/docs/guide/region-selection).

The stopped western attempt's hold was reduced from $6.96 to $1.89, retaining its full $1.50 setup
and $0.39 for the sole dispatched operation. The reduction releases capacity only for operations
never dispatched; it does not declare billing settled. All other holds remain. Fresh workspace
usage was **$13.86137298**, including $0.03122589 attributed to the western SDK attempt. With the
new reservation, conservative projected exposure is **$34.85137298**, below the already approved
$35 stop. The $30 credit amount, $7 additional paid authorization and $2 reserve are unchanged.

The [evidence directory](../benchmarks/renderer-region-2026-09-05-c) retains the image provenance,
reconciliation, frozen draft and active manifest, authorization hashes, and deployment checks.
Private authorization and proxy credentials stayed outside the repository. While explicit
credential-transfer approval was pending, both idle deployments were stopped and the unused token
revoked. After the user approved the transfer, the same reviewed deployment was restored with a
fresh temporary token. Manifest expiry and the existing reservation passed recheck; metadata
lookup and unauthenticated HTTP rejection passed with zero tasks before dispatch. The C attempt
marker was consumed once, without retries.

## Measured result and cold-start diagnosis

| Verified artifact-ready latency | SDK | HTTP |
| --- | ---: | ---: |
| Median, six measured requests | 2.611 s | 2.200 s |
| p95 / maximum, six measured requests | 30.016 s | 2.402 s |
| Median, four pairs warm on both sides | 2.556 s | 2.200 s |
| Image inference median, those four pairs | 1.579 s | 1.595 s |

The four-pair warm subset is descriptive only; its 13.94% improvement does not replace the frozen
gate. With six observations per transport, nearest-rank p95 is the maximum. These measurements
end after validation and local artifact storage, not projector display. Byte equality preserves
the paired output; it does not establish story fidelity or visual quality.

SDK preparation took 263.849 seconds including startup and waiting. Its runtime reported 29.757
seconds of startup and 14.520 seconds of synthetic warmup. The subsequent HTTP preparation took
111.549 seconds, including 89.988 seconds of readiness waiting. That interval exceeds the
90-second idle window before the next SDK call. The measured SDK container differed from its
prewarm container; its first 128-token and 256-token bucket calls were cold, at 30.016 and 5.619
seconds artifact-ready. This is consistent with idle expiry, though lifecycle logs do not prove
the shutdown cause. HTTP kept one container and all six measured calls were warm.

The [diagnosis](../benchmarks/renderer-region-2026-09-05-c/diagnosis.json) retains these stage
measurements and container hashes. The original [summary](../benchmarks/renderer-region-2026-09-05-c/summary.json)
and journal remain unchanged and reproduce byte for byte from the pinned manifest and source.

## Product repair

Inspection found a separate readiness bug in the workbench: it started the renderer's remaining
90-second lifetime after waiting for the local planner. A planner taking 111.55 seconds could
therefore leave the interface showing ready until 201.55 seconds after the renderer response.
Reusing an already warm renderer extended its deadline the same way.

The interface now captures the deadline when the renderer response finishes, preserves that
deadline during reuse, and checks it after planning completes. Expired readiness returns to idle,
including when planning fails. It makes no automatic prewarm request. Two focused regression
cases cover delayed planning, warm reuse, expiry and unexpired controls; the previous code fails
the delayed-planning case. Independent probes also cover duplicate clicks and passage changes.
This repair changes the local workbench; the accepted Jetson service has not been redeployed.

## Cleanup and cost

The supervisor completed in 436.701 seconds and stopped both apps successfully. Independent
inventory confirmed both stopped with zero tasks and no containers. The temporary token was
revoked and its Mac and Jetson files removed. The accepted reader remained ready and its planner
process start was unchanged. External shutdown evidence lives in the
[cleanup receipt](../benchmarks/renderer-region-2026-09-05-c/cleanup-cost.json); the harness summary
cannot independently attest external app shutdown and is deliberately not rewritten.

The hourly report through September 6 attributes **$0.26456376** to C ($0.16924399 SDK,
$0.09531977 HTTP), bringing reported September workspace usage to **$14.12593674**. These are
provisional reported charges, not settled billing. The full $5.96 C hold remains; total retained
holds, including C, are $20.99. Updating the local ledger's reported floor produces **$35.11593674** of
conservative exposure, so further paid dispatch is blocked under the approved $35 stop. This
figure includes potentially overlapping charges and holds; it is not actual spending.

Warm inference is approximately 1.6 seconds on either transport. The next performance decision
is how to bound session startup and preserve useful warm time within an explicit session budget.
No transport promotion, automatic session prewarming or additional paid experiment is enabled.

## Verification and reproduction

All 891 Python tests, the anticipatory workbench, hand-interaction and readiness JavaScript
suites, scoped Ruff and diff checks pass. Independent adversarial review approved the readiness
repair. Deslop review found no unrelated changes. Benchmark reproduction verifies the stored
JPEG bytes as well as journal metadata and reproduces summary SHA-256
`83d9e3c395ccfb3126ffac4fee3e5a2d80cf9684961e31ca53466490f4a368e8`.

The benchmark implementation is pinned at `b48f377`. For offline reproduction, copy the retained
`.bookforge/renderer-region-20260905-c/results` directory to a fresh temporary directory and run
`scripts/benchmark_klein_region.py --aggregate-only` with that `--output`, the checked-in C
`--manifest`, and the retained private `--authorization` plus its `--authorization-sha256`.
Compare `recomputed-summary.json` byte for byte with the checked-in summary. The authorization
proof contains no proxy credential. The measured JPEGs are retained locally; a journal-only copy
is insufficient because aggregation verifies the actual artwork hashes. No cloud call is made.
