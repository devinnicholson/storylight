# Paired Klein prompt comparison

Trial `v` preserved subsecond prepared delivery and improved the missing fox
and original-seed chase direction, but failed the predeclared full-fact screen.
Neither independent reviewer passed all four candidate images. Production
routing remains unchanged.

Ten measured deliveries had a **0.504-second median and 0.660-second maximum**;
preparation took **75.972 seconds**. All twelve renders completed. These timings
cover the prepared renderer client, not microphone-to-projector latency or a
ready-from-cold service.

The [four matched pairs and original images](../benchmarks/prepared-gcp-2026-09-08/support/paired-v/gallery.html)
retain every result. Both reviewers agreed that the candidate depicts two foxes
at both seeds and fixes pursuit at the original cat seed. The fox count improves
at the original seed; both arms already show two foxes at the second seed. The
reviewers disagreed about joint lantern carrying, ribbon count and alley darkness:

| Reviewer | Baseline full-fact passes | Candidate full-fact passes | Paired wins / losses |
| --- | ---: | ---: | --- |
| Mill | 0/4 | 2/4 | 2 / 0 |
| Franklin | 1/4 | 2/4 | 1 / 0 |

Mill accepts carrying by one animal within the fox group, but considers the
brightly lit alley insufficiently certain. Franklin requires both foxes to
participate in carrying, but accepts the shaded alley. Mill also marks the
second-seed baseline's ribbon count uncertain; Franklin accepts one ribbon.
That ribbon judgment accounts for Mill's second-seed fox win. Their original ratings
remain unchanged. Combining their favorable judgments would falsely produce
an all-pass result. Both pass watercolor detail and gross depth alignment for
all eight images. Prior familiarity with `u` limits the label masking.

| Scene / seed | Baseline delivery | Candidate delivery |
| --- | ---: | ---: |
| Two foxes / 2026090801 | 0.645 s | 0.507 s |
| Two foxes / 2026091801 | 0.476 s | 0.660 s |
| Cat and mouse / 2026090802 | 0.466 s | 0.502 s |
| Cat and mouse / 2026091802 | 0.510 s | 0.639 s |

The candidate is slower in three of four pairs. This small comparison establishes
no wording-related speed gain. All four pairs have identical initial-noise
hashes. All eight warmup/control JPEGs match both `u` and `t`; the four original-seed
baseline master/depth JPEGs exactly reproduce `u`. The schedule change did not
change these reference outputs.

The rest of this document records the fixed design and execution boundary. The
[pre-review criteria](../benchmarks/prepared-gcp-2026-09-08/support/paired-v/paired-v-criteria-before-review.md),
both frozen reviews, arm mapping, image hashes and unblinded comparison are
retained separately from the cloud evidence archive.

## Fixed comparison

The baseline is the exact modern compiler prompt used in `u`. The candidate is
the exact handwritten prose retained in the
[tokenizer receipt](../benchmarks/prepared-gcp-2026-09-08/support/quality-u-next-tokenizer.json).
Both preserve the admitted facts and watercolor style. Candidate counts are
85 and 80 tokens; all four prompts occupy the existing 128-token bucket. This
does not establish a speed improvement from shorter wording.

| Scene | Original seed | Second seed fixed before dispatch |
| --- | ---: | ---: |
| Two silver foxes | 2026090801 | 2026091801 |
| Cat chasing a mouse | 2026090802 | 2026091802 |

For each scene, compare baseline/candidate at the original seed, then
candidate/baseline at the second seed. Four A/B pairs give eight comparison
renders. Keep the exact two legacy 128/256 warmups and two legacy control
repeats, for twelve renders total. After preparation, retain the 30-second idle
interval before the ten measured deliveries.

The manifest has ten unique cases: two warmup controls and eight comparisons.
Only `v` admits that shape; its measured order is the eight comparisons followed
by the two controls. Existing n–u schedules remain unchanged. This requires a
reviewed scheduling change, not a change to model weights, precision, resolution,
four-step inference, guidance, depth generation or the initial-noise function.
The complete model/runtime identity and all inherited image layers remain
subject to verification.

## Decision before seeing the images

Require matching initial-noise hashes within every same-seed A/B pair. Check
the legacy controls against their corresponding retained images and account
for any change before drawing conclusions. Retain every result; do not retry,
select a best seed or change the rubric after viewing an image.

Two reviewers independently inspect the eight full-resolution master/depth pairs
under shuffled neutral labels. Keep the arm/seed mapping separate until their
ratings are saved. The reviewers have seen earlier `u` images, which may allow
recognition of baseline images; this is a limitation of masking, not a fully
blind evaluation.

For the fox scene, require exactly two silver foxes carrying one blue lantern
in a forest, with one golden ribbon beside them. For the chase, require one cat
visibly pursuing one mouse in a dark alleyway. London remains omitted. Missing
or uncertain required facts fail. Independently require watercolor pigment
variation, paper texture, layered depth and a readable composition, with no
gross subject displacement in the depth map.

An encouraging screen requires all four candidate images to pass, no paired
factual regression, at least one improvement over its matched baseline, and
the existing ten-delivery speed screen (median ≤1 second, maximum ≤2 seconds).
If both arms fail, preserve that result. If both arms pass, this small screen
does not demonstrate a candidate advantage. Four candidates cannot establish
general reliability or justify production promotion.

The candidate bundles prose, plural agreement, repeated action wording and
subject-first ordering. Even a successful comparison cannot attribute an effect
to one component. Both paired delivery timings and inclusive preparation time
must be reported; neither measures speech-to-projector latency.

## Execution boundary

Use one new private service, `bookforge-klein-qualification-20260907-v`, with
the existing single-GPU resource profile and exclusive attempt identity. The
build remains bounded to 600 seconds, supervised work to 600 seconds, cleanup
to 60 seconds within the 660-second lifecycle, and release observation to 900 seconds.
The prepared lease lasts 300 seconds and permits at most twelve total renders.

Validate the exact profile, source snapshots and actual client inputs before
building. Upload only compiled case fields and the reviewed runtime context;
local source text and review mapping stay local. Preserve deletion, route and
release evidence. No production routing or fallback changes are included.

The prior eight reservations total $28. This one reservation reaches $31.50;
failed trials are not recycled. These are admission-accounting amounts, not
invoices or cloud-enforced spending limits. No further trial is authorized by
this approval.

## Closure and next work

The [archived run](../benchmarks/prepared-gcp-2026-09-08/v/README.md) passes
offline source, build, image, client, route and release verification. Terminal
deletion occurred at 18:00:01.490391 UTC; the release observer concluded
`historical_zero_and_current_absence` at 18:10:29 UTC. This does not prove billing
closure or reserve future GPU capacity. Final read-only inventory shows only
the two standing services and no ongoing build.

The approved $3.50 reservation brought the program total to $31.50, with nothing
unallocated. The latest delayed project-wide gross cost was $40.81, timestamped
18:01:57 UTC. It is not the cost of this trial.

The next experiment should clarify the ambiguous carrying and darkness criteria
before rendering, then test whether explicit spatial/action wording improves
them. The current style requests bright midtones alongside a dark alley; that
is a hypothesis to isolate, not an established cause. Preserve the watercolor
detail and fixed seeds. Any additional paid run needs a new allowance. The
production adapter remains inactive until factual quality and the complete
voice-to-display path pass their separate checks.
