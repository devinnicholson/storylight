# Native GCP startup diagnostic — September 7, 2026

## V3 loader results

Build `9d357b30-bb1c-4ad2-972a-6ad1ec3ed2e8` and its one diagnostic child completed.
All four generation-pinned artifacts, totaling 204,025 bytes, were retrieved and
verified against provider hashes and child-output SHA-256 hashes. Both ordinary
mixed BF16/FP16/F32/I64 cases preserved names, values, shapes, strides, dtypes,
requires-grad flags and the destination's train/eval state exactly.

The six boundary checks explain why this loader needs a narrow integration:

| Boundary | Observed result |
| --- | --- |
| Eval source, train destination | Destination stays in training mode. |
| Scalar I64 buffer | Value survives; shape changes from `[]` to `[1]`. |
| Noncontiguous parameter | Packing refuses with `RuntimeError`. |
| Tied parameter | Strict assignment refuses with `ValueError`. |
| Wrong destination shape | Assignment silently replaces the declared shape. |
| Wrong destination dtype | Assignment silently replaces the declared dtype. |

These are real Torch 2.8 CPU observations with FlashPack 0.4.4, not model or GPU
qualification. The existing VAE contains an integer scalar, whereas the pinned
transformer headers contain 169 non-scalar BF16 tensors. Proceed with a checked
transformer-only candidate, explicit evaluation mode, and full offline tensor
verification. Keep the VAE, Qwen, depth and inference settings unchanged. The small
direct-depth import saving does not justify prioritizing that GPU experiment over
model loading; its reviewed correctness harness remains offline.

The [v3 summary](../benchmarks/gcp-klein-cpu-startup-v3-2026-09-07/summary.json)
retains every boundary outcome. No model weights were converted, no GPU was called,
and no rendering speed improvement or production promotion is claimed.

## Corrected tiny-loader follow-up

The separate [v3 plan](../benchmarks/gcp-klein-cpu-startup-v3-2026-09-07/plan.json)
limits work to **one tiny CPU FlashPack child**, with a 90-second probe deadline
inside one 600-second CPU8 build. It uses the same immutable image and audited wheel.
The only tensor-fixture change is the non-conflicting `half_weight` name. All eight
checks remain, including strict failure observations for scalar, alias, shape, dtype
and mode boundaries. Child failures preserve a traceback in the private stderr
artifact. The completed import trials are not repeated; the pinned depth adapter is
not used. No checkpoint is opened and no model conversion or GPU work is performed.

The separately announced allowance is $0.50: $0.17 build, $0.01 small source/evidence
storage and $0.32 margin. Earlier reserved holds total $26.5539592, including both CPU
attempts; none is released. The context contains 94,943 bytes. A successful build
still only means evidence export succeeded; examine each tensor check independently.
One attempt only, with no automatic retry or production promotion.

## Retained v2 results

Build `a17d69c0-c433-409d-85e8-4abe9813d152` completed and exported all twelve
generation-pinned artifacts, totaling 1,006,542 bytes. Provider MD5 hashes, child
SHA-256 hashes, source pins and parsed payloads were verified. Build success enabled
export; the diagnostic itself remains **incomplete** because its FlashPack child
returned exit code 1 with `KeyError`.

All four import children completed. Their selected depth-import times were:

| Order | Arm | Selected import seconds | Pipeline registry loaded |
| --- | --- | ---: | --- |
| 0 | Baseline | 1.709 | Yes |
| 1 | Direct-depth | 1.083 | No |
| 2 | Direct-depth | 1.089 | No |
| 3 | Baseline | 1.234 | Yes |

The adjacent-pair differences are 0.626 and 0.145 seconds. The latter is a modest
saving; it does not support a large image-generation gain. The first baseline's
common imports were much slower than the other three fresh processes, which shared
the build host and its page cache. Do not attribute whole-process differences to
the adapter or transfer CPU timings to GPU cold starts. No pretrained model was
loaded by these processes. The existing production runtime remains unchanged.

The synthetic fixture registers a parameter called `half`, but PyTorch 2.8 already
defines `Module.half()`. Its registration method raises `KeyError` before packing.
An extracted-method source check reproduces that collision; `half_weight` and the
other fixture names pass registration. This is a harness defect, not a FlashPack
compatibility result. A follow-up should repeat only the tiny tensor check with the
corrected name and preserve these failed outcomes.
[PyTorch 2.8 source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/module.py).

The v2 build ran for 337.866 seconds, with a CPU-rate estimate of $0.087845. This
is not an invoice; its $0.18 work/storage hold is unreconciled. The separate
[summary](../benchmarks/gcp-klein-cpu-startup-v2-2026-09-07/summary.json) retains the
full distinction between build status, import results and loader failure.

## First attempt and evidence repair

Build `c2afec5c-7346-4f2f-8939-7aa325a425c4` failed after emitting all five child
completion markers. It ran for 270.968 seconds from build start to finish; the CPU
rate estimate is $0.070452, not an invoice. Its actual submitted archive matched all
eight pinned files and retained the writable output-directory mode. The image digest
also matched. However, the failed step prevented artifact export, and the original
parent log did not retain child outcomes. The exact private output prefix had no
objects. No import timings, tensor outcomes or underlying child-failure cause can be
recovered from this attempt. Its $0.18 reserved work/storage hold remains unreconciled.

The separate [v2 plan](../benchmarks/gcp-klein-cpu-startup-v2-2026-09-07/plan.json)
authorizes one new $0.50 CPU attempt with the same fixed five-process workload. The
new config uses `allowFailure: true` so failed diagnostic steps can export evidence.
The parent retains exit status and file hashes before parsing child JSON, rejects
non-object results, and prints compact child outcomes plus its final structured result.
Raw traces stay in private artifacts. **Cloud Build success is not diagnostic success**;
the final result and step exit code must be checked separately. Cancellation can still
prevent export; missing evidence remains incomplete. No model or GPU work is added.
[Cloud Build schema](https://docs.cloud.google.com/build/docs/build-config-file-schema).

The new allowance again reserves $0.17 build plus $0.01 source/evidence storage, with
$0.32 margin. Prior phase holds are now $26.3739592, including the failed CPU attempt;
none is reused or released. The original plan and failed result stay unchanged.

## Original frozen scope

This separate CPU-only diagnostic isolates import costs before another GPU loading
experiment. The direct-device comparison stopped after P/Q/R: thirty requests
succeeded, but R's metrics stopped before the frozen release guard could pass. S
was not run and that phase remains incomplete.

The frozen [plan](../benchmarks/gcp-klein-cpu-startup-2026-09-07/plan.json) permits one
600-second E2_HIGHCPU_8 Cloud Build with 100 GB disk, using the exact original G image.
The source context is 96,014 bytes. Four fresh import processes run baseline,
candidate, candidate, baseline. The candidate loads the isolated direct-depth adapter
instead of the broad Transformers pipeline registry. A fifth process tests FlashPack
0.4.4 with tiny CPU tensors, including scalar, alias, dtype and shape boundaries.

No pretrained model is constructed, checkpoint opened, weight converted, package
installed, GPU allocated or image generated. The existing image contains model layers,
so pulling it can still transfer those layers. The trace is a CPU import diagnostic;
its timing cannot establish a GPU cold-start gain. Synthetic loader checks cannot
establish compatibility with the actual renderer. The production runtime is unchanged.

The separate announced allowance is **$0.50**: $0.17 for the single ten-minute build,
$0.01 for small source/evidence storage, and $0.32 margin. This uses the retained
eight-CPU rate of $0.0156/minute and the included first 100 GB disk. Shared model-image
retention stays with its earlier phase. The prior $26.1939592 in reserved phase holds
is unchanged and unreconciled. This is an execution allowance, not a platform hard
billing cap or an invoice. The latest existing billing-guard observation before this
diagnostic reports **$22.39 gross project cost at 00:15:15 UTC**, across the project.
[Cloud Build pricing](https://cloud.google.com/build/pricing).

The build exports only diagnostic results to a new prefix in the existing private
Cloud Build bucket. Source pins, build status and returned artifact hashes must be
verified. Any failed process or build ends this attempt without retry; retain partial
evidence. No GPU experiment or model conversion is funded by this allowance.
