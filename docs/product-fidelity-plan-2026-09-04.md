# Plan: a short story that projects faithfully

Status: the object-color repair passes the complete 512-case development gate: 38 exact candidate
contracts versus zero accepted, with no category-total regression. Complete-scene coverage and
ordered display integration continue before the frozen story's visual comparison. See the
[latest measurement](product-fidelity-color-repair-2026-09-04.md) and
[earlier implementation](product-fidelity-results-2026-09-04.md).

## Outcome

Deliver a six-page typed-story demonstration whose essential characters, counts, objects and
actions appear correctly and consistently on the projector. Use the existing workbench,
Jetson planner, renderer and playback runtime.

The first milestone improves scene fidelity. Microphone activation, physical hand tracking,
new models and renderer replacements follow separately.

## Starting point

- The typed-text generation and projection flow exists, with privacy checks, local storage,
  cache replay and progressive artwork/depth display.
- Both recorded 512-case live graph experiments produced zero accepted graphs. The accepted
  appliance remains on the four-slot planner.
- Subsequent deterministic repairs produce two graphs from 32 fixed training controls, one
  exact. They have not passed a new live-model gate.
- The baseline suite had 741 passing cases. Preserve its focus on public behavior and distinct
  consequential failures; do not rebuild exhaustive test matrices.

See [current construction and scoring evidence](live-scene-underlying-fixes-2026-09-04.md).

## 1. Freeze the story and its acceptance criteria

Author six connected, synthetic pages covering counts, actor/object binding, explicit passive
voice, spatial relationships, a transformation and event order. Give each page a short checklist
of visible required facts and forbidden mistakes. Freeze the text and checklist before tuning.
The story is a demonstration and engineering fixture, not an independent accuracy estimate.

For a transformation or sequence, specify the necessary before/after states. A single still
image with parallax cannot prove that an event happened over time. Use existing authored
playback states where possible; record an unmet requirement if they cannot express the story.

Capture the existing accepted planner's local contracts for these pages, with code, engine and
prompt identity. Measure planning separately from image generation and cached playback.

Deliverable: a small versioned story manifest with required facts, visual criteria and baseline
results. Do not change the pages later merely to make the candidate look better.

## 2. Fix one construction failure class

Start with explicit passive actor/object binding, as identified in the current diagnostic notes.
Use synthetic and public training examples to distinguish a visible action from reporting,
negation, hypothetical action and ambiguous references. Verify swapped-actor and swapped-object
examples fail rather than borrowing a nearby fact.

Extend the finite source-grounding grammar only where those controls establish a sound rule.
Reuse the current slot parser, graph validator, privacy gate and renderer compiler. A refusal
must preserve the accepted fallback and must not trigger another model request.

Deliverable: one bounded parser change with representative positive and adversarial checks.
Exit gate: the selected behavior survives graph construction and exact renderer-prompt proof;
unsafe variants refuse, and existing accepted-path behavior remains unchanged.

If the change does not pass, document the blocker and stop that hypothesis. Do not broaden the
grammar speculatively or start model training to bypass an unresolved construction defect.

## 3. Prove the improvement with the resident model

Freeze candidate code, evaluator and decision criteria before measurement. Start with a small
training/synthetic smoke on the Jetson. If useful graphs survive the complete path, run the
existing accepted-first benchmark on all 512 public development records, using one accepted
model response per record for both accepted and candidate comparison.

Compare both paths under the same current evaluator. Historical reports retain their original
scores and revisions. The development corpus has already been inspected during earlier work;
this run is a regression gate, not an untouched holdout. Do not open the hidden split or tune
against individual development failures within this iteration.

Required gates:

- Nonzero accepted graph coverage and a strict increase in exact final-contract passes and
  required-fact recall over the matched accepted path.
- No decrease in exact passes or required-fact recall within any previously passing category.
- Zero final privacy failures; report graph refusals and accepted-plan failures separately.
- Median local planning at or below 1.5 seconds. As a proposed additional limit, p95 may be
  at most 10% above the matched accepted path; freeze this limit before the run.
- Retain maximum latency, token usage, memory and thermal samples, request provenance and
  reproducible sanitized results. No automatic retries after interrupted requests.

Deliverable: an explicit advance/reject decision with category-level gains and losses.
Passing this stage permits a bounded visual comparison, not appliance promotion.

## 4. Check whether better contracts produce better pictures

Only after the live semantic gate passes, compare accepted and candidate contracts for the
six story pages. Use the same provider/model revision, seed, dimensions and generation settings.
Begin with one image per path per page: 12 images total. If temporal criteria require additional
assets, include their exact count in the scope before any calls.

Check current billing and calculate a maximum cost covering startup, idle time and generation.
Run within existing authorization; any additional spend requires an exact, reviewable request.
Disable paid retries and retain failures in the results. Do not reroll until an attractive image
appears. Scale temporary GPU workloads down after evidence capture.

Hide candidate labels and use the frozen fact checklists for human review. Score correctness,
projection legibility and continuity separately from visual appeal. Record preparation, generation,
download and first-display times separately. A text score alone cannot pass this gate.

Deliverable: paired projections, fact-level review and actual cost. All six candidate pages must
meet their essential visual criteria, with no essential-fact regression against the accepted
path. Any failure remains visible and determines the next bounded fix.

## 5. Rehearse the complete experience

Run the six-page story through the existing workbench and projector without developer intervention
during playback. Exercise first generation, cached replay and restart recovery; repeat cached
playback three times. Confirm that replay makes no cloud generation calls, stale events cannot
change the active page, verified assets survive restart, and page transitions do not blank the
projector unexpectedly.

Use a controlled offline rehearsal to check cached playback. Record limitations of the test
environment rather than claiming whole-device offline behavior from configuration alone.

Deliverable: an uninterrupted demonstration and a short record of correctness, display latency,
recovery behavior and remaining limitations. Promote the candidate only through the existing
reviewed deployment and rollback procedure after all prior gates pass.

## Work ownership and verification

Freeze interfaces and disjoint file ownership before parallel implementation:

| Owner | Responsibility |
| --- | --- |
| Root | Story criteria, shared integration, Jetson runs, cost scope, final review and promotion |
| Agent A | Bounded live-parser change and its focused checks |
| Agent B | Existing benchmark harness, sanitized evidence and reproducibility |
| Agent C | Independent adversarial review of grounding, privacy, fallback and evidence claims |

Review proposed failures before expanding implementation scope. Complete focused checks during
development, then the full Python and JavaScript suites, scoped lint, changed-artifact reproduction,
credential scanning and deslop before reviewed fast-forward pushes. Preserve user-owned outputs.

## Next action

Validate complete-scene reconstruction against the original model responses and integrate authored
display steps before rendering the whole story. Keep failed reports and the original story;
microphone-driven reading follows a faithful typed-story demonstration.
