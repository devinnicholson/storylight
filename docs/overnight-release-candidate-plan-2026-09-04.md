# Overnight release-candidate plan

Status: integrated opt-in candidate implemented and measured September 4, 2026. See the
[results and remaining gates](overnight-candidate-results-2026-09-04.md). Additional Modal
allowance: $10. The live provider and physical kiosk defaults remain unchanged.
Baseline: `a66427c`, 1,148 tests passing. Planning window: approximately six working hours;
finish an integrated, reviewable candidate before expanding experiments.

## Outcome

Make the real typed-story → private planner → generated artwork/depth → moving projector path
more faithful, faster, and demonstrably reliable. Deliver a candidate the user can compare with
the accepted route in the morning, not a collection of unconnected benchmarks.

The scene remains generated artwork animated with depth and local effects. Neither particle
motion nor parallax is to be described as generated video. Microphone work and physical camera
calibration are deferred until the user is available.

## Why this is the highest-value work

- The current planner can lose actor/object relationships before image generation. The latest
  production-instruction screen scored 16/20 lexical cases; lexical success is not visual accuracy.
- Compiled Klein produced better results in small synthetic comparisons at about 1.6–1.9 seconds
  per warm artwork/depth/encoding operation. It is not integrated into live routing yet, and
  restored first renders still take about eight seconds after separate model loading.

These were planning-time observations. Tonight's actual-Gemma validation found substantial
remaining fidelity failures and a roughly 22-second restored first render; the results report
supersedes the preliminary observations. The matched study compared full versus concise
contracts on Klein only. An accepted-provider comparison was deferred after the accuracy gate
failed, rather than spending more or promoting a failing candidate.
- The concise-contract comparison used authored slots. Its benefit has not yet been verified
  through actual Gemma output on fresh passages.
- Nsight has now confirmed existing CUDA Graph execution. Further work must follow measured
  bottlenecks, not repeat an optimization that is already enabled.
- MediaPipe works in a Mac browser fixture test. Its deployment is opt-in; the Jetson webcam
  and physical-performance gates remain open.

## Priority 1 — preserve the story, then qualify the renderer

1. Freeze a small original synthetic corpus before making changes: eight development passages,
   24 separate validation passages, and a separate privacy/adversarial set. Do not open or reuse
   the sealed training holdout. Include exact counts, ownership, negation, color, relative position,
   open/closed state, transformations, competing actors, and quiet non-magical scenes.
2. Record actual Gemma slots, local postprocessing, renderer contract, and output separately.
   Trace where requirements disappear; never credit deterministic repairs to model learning.
3. Fix supported, bounded contract/assembly defects first. A missing relation must not be
   recovered by sending the raw passage to the cloud. Any local recovery must still pass the
   complete privacy gate and must not invent requirements.
4. Compare accepted and candidate pipelines with matching passages, styles, seeds, and image
   dimensions where providers support them. Record unavoidable provider differences explicitly.
5. Review count/attribute/relationship requirements against actual images. Do not treat a
   Nemotron verdict or a keyword match as independent proof of correctness.

Deliverable: a versioned contract candidate, regression tests, a compact side-by-side visual
review, and a decision backed by fresh actual-model evidence. If a candidate fails validation,
retain the accepted route and publish the failure rather than tune against validation unnoticed.

## Priority 2 — make the candidate usable through the real UI

1. Add one optional Klein provider using the existing provider/job/asset contracts, not a second
   workbench or parallel application. Reuse the existing isolated runtime and pinned assets.
2. Persist verified compiler artifacts; separate scheduling, imports, model load, compile/cache
   restore, first inference, and warmed inference. Test the real 128/256 boundaries; qualify 512
   before accepting that shape, or fail clearly instead of truncating.
3. Provide explicit preparation with a finite idle lease and scale-to-zero cleanup. Page opening
   must not silently create an always-on GPU or an unbounded prewarm loop.
4. Measure browser submission through local planning, provider response, verified asset delivery,
   and first drawn scene. Retain the old scene during the transition. Test superseded requests,
   failure, retry, context loss, and warmup races without duplicate paid jobs.
5. Keep candidate activation opt-in and easy to roll back. Physical projection acceptance is
   still required before changing the default merely because a cloud microbenchmark is faster.

Provisional targets, not promises: warm uncached submit-to-first-drawn-scene median below five
seconds; no p95 regression against the matched accepted route; cached activation below 300 ms
before the existing blend; no blank transitions. Report actual results even when targets miss.

Deliverable: a single end-to-end candidate the user can try, plus cold/warm/cache timings and
an exact rollback path. If accuracy fails, the integration remains an isolated preview.

## Priority 3 — Nsight-guided work with a strict time box

- Analyze the existing trace first. Spend at most one hour on additional profiling/changes.
- If the Jetson is reachable, collect one bounded node-level steady-state capture to distinguish
  graph-internal GPU work from host launch/transfer overhead. Preserve the existing watchdog and
  restore hooks; no root settings, driver changes, reboots, or power-mode changes.
- Test at most two targeted changes against identical inputs and responses. Measure performance
  without profiling separately. Waiting in `cudaStreamSynchronize` is not automatically wasted
  time, and startup ranges do not describe steady-state inference.
- Keep only a repeatable material gain without semantic/privacy regressions. An honest negative
  result is sufficient; do not rebuild a TensorRT engine overnight just to force a speed claim.

Deliverable: kernel/host attribution and either a tested improvement or a specific rejected
experiment. This work cannot delay integration, regression tests, or cleanup.

## Priority 4 — make tomorrow's hand test straightforward

- Extend camera-free tests to worker initialization failure, camera removal, camera switching,
  projection changes, malformed calibration, repeated enable/disable, and long-running cleanup.
- Exercise hand events during scene replacement and planner pauses using clearly labeled fixture
  input. Compare interaction off/on at the same browser render resolution.
- Improve only necessary calibration and performance-status feedback. No additional gestures,
  object manipulation, microphone permissions, or cloud vision service.
- If no camera is connected, finish with a specific hardware checklist rather than marking
  tracking quality or Jetson frame pacing as accepted.

## Sequencing and unattended behavior

Suggested allocation: 30 minutes baseline/preflight, two hours semantics/renderer qualification,
two hours integration/end-to-end testing, up to one hour profiling and hand regression work,
and the final 30 minutes for full tests, cleanup, and the morning report. Cut lower-priority
experiments first if the main path needs more time.

Work can be separated into planner/contract, renderer integration, and regression/profiling
ownership if the user asks for parallel agents. No agents are started by this planning request.
Shared schemas and provider interfaces must be agreed before concurrent edits.

If Jetson connectivity fails, continue local code, captured-trace analysis, browser fixtures,
and Modal renderer qualification; do not keep retrying the same unreachable host. Do not substitute
authored slots and call them Gemma acceptance. If credentials or billing access block a paid job,
finish the relevant local work and record the exact missing step without changing account settings.

## Cost and safety

- Proposed additional Modal experiment allowance: up to $10, with no obligation to spend it.
  Check current rates and account state first, then bound GPU count, runtime, concurrency,
  CPU/memory, builds, and warm leases so the conservative estimate fits the allowance.
  Billing can lag; this is not represented as a provider-enforced dollar cap.
- One experimental GPU at a time; explicit bounded batches and no unlimited retries. Stop all
  experimental apps/tasks and warm leases at handoff, and report observed charges separately
  from estimates. Account usage from unrelated jobs is not attributed to this pass.
- No new GCP infrastructure, quota requests, billing changes, public endpoints, or expanded IAM.
  Preserve existing $150 alerts and $175 emergency-disconnect settings; they are not hard caps.
- Use original synthetic passages only. No private book text, microphone, or camera uploads.
- Preserve unrelated files and accepted model engines. Changes to the physical kiosk, camera
  permissions, or live provider default remain deferred; validated reversible fixes can use the
  existing deployment workflow and prior direct-push authorization.

## Morning handoff

1. One link to try the candidate and a short comparison with the accepted experience.
2. Fresh visual examples with the original synthetic prompt and measured requirement failures.
3. Before/after timings by boundary, including cold-start penalties and small-sample limitations.
4. Test counts, commits, accepted/rejected experiments, and rollback instructions.
5. Cloud usage and evidence that experimental GPUs/warm leases are stopped.
6. The minimum remaining user action: webcam connection, permission, and physical calibration.

## Host requirement and activation

This workspace and orchestration run on the Mac. Keep it powered, awake, network-connected, and
the Codex app running; keep the Jetson powered/networked if device benchmarks are desired. An
already-dispatched bounded cloud job may continue separately, but that does not keep local editing,
testing, orchestration, or follow-up alive if the Mac sleeps.

The OpenAI Docs skill was used to check the local-execution constraint against
[official scheduled-task documentation](https://learn.chatgpt.com/docs/automations?surface=app).
A goal or recurring continuation should be created only when the user asks to start that work;
this document does not itself start unattended execution or schedule a morning notification.
