# Native GCP startup diagnostic — September 7, 2026

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
