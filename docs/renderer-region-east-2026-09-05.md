# Eastern-region renderer comparison

The next frozen comparison requests AWS `us-east` and requires observed `us-east-1` for both SDK
and HTTP transports, with routing through `us-east`. The western attempt exhausted its total
client deadline after a capacity wait and late warmup compilation. Its final logs and corrected
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
Private authorization and proxy credentials remain outside the repository. Metadata lookup and
unauthenticated HTTP rejection passed with zero containers before dispatch. Generation still
requires the explicit credential-transfer approval requested by automatic approval review.
While that approval is pending, both idle deployments were stopped and the unused proxy token
revoked. No C generation request was dispatched and no C attempt marker was consumed. After
approval, restore the same reviewed deployment and issue a fresh temporary credential; recheck
the manifest expiry and existing reservation before dispatch.
