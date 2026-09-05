# Test suite halving

Baseline: `450cd80173c82a3ad5be0b391987f2157dc69107`, after the
[initial cleanup](test-suite-sweep-2026-09-04.md).

The suite falls from **1,496 to 741 collected Python cases**, removing 755 (50.5%).
This deliberately reduces detailed coverage of secondary behavior as well as duplicate tests.
Production code, benchmark artifacts, pytest discovery, CI, and the two JavaScript suites are
unchanged. Cases were deleted, not skipped, excluded, or moved into loops.

| Area | Before | After | Removed |
| --- | ---: | ---: | ---: |
| Live planner, graphs, semantic regressions, TensorRT client | 558 | 250 | 308 |
| Training, checkpoints, cloud execution and evidence | 344 | 145 | 199 |
| Providers, anticipation, critic, deployment and browser bridges | 229 | 109 | 120 |
| Reader, core services, evaluation and remaining tools | 365 | 237 | 128 |
| Total | 1,496 | 741 | 755 |

Across `tests/test_*.py`, test functions fall from 1,050 to 612, files from 121 to 114,
and lines from 37,076 to 24,509: 12,567 fewer lines of test code.

## Retained coverage

- Reader progression, repeated words, partial transcripts, reset isolation, stale events and
  trusted-page matching; story compilation and local storage round trips.
- Successful live scene generation and cache replay; bounded model requests, accepted fallback,
  malformed output refusal, exact renderer proof and cache identity.
- Source grounding, entity bindings, counts, negation, event order, transformation and salience;
  representative name, printed-payload, injection and source-echo privacy failures.
- Private scenes blocked before renderer or critic calls; loopback API boundaries, consent,
  asset checksums, path traversal and immutable accepted scenes.
- Billing checks before execution, retained reservations after ambiguous failures, cancellation,
  reconciliation freshness, approval binding, checkpoint integrity and rollback checks.
- Public-only benchmark entrypoints, resumable journals, sanitized evidence, counterfactual
  scoring and separation of raw model output from graph and renderer evaluation.
- Both original JavaScript behavior suites through their existing Python bridge tests.

Independent reviews restored the API check that prevents a private fallback scene from reaching
the critic and the assertion that changed submission intent invalidates prior reconciliation.
Unused imports, fixtures and branches left by parameter reductions were removed.

## Deliberate omissions

Removed detailed planner repair and layout variants, repeated protocol/privacy combinations,
secondary API validation matrices, static packaging/default snapshots, historical training
compatibility variants, isolated optimizer/tensor shape cases, prewarm and cache timing
permutations, and detailed benchmark statistics tests.

Seven test modules were deleted: the anticipation simulator and benchmark, planner benchmark,
Nsight planner, GCP monitoring packaging, and JAX cache/rematerialization comparisons. Their
production tools remain available. Changes to those tools require focused validation; their
old dedicated unit coverage is no longer present.

The smaller suite retains representative product flows and critical failure checks. It does
not establish identical branch or mutation coverage. Hardware behavior, model accuracy,
cloud deployment and image fidelity still need their separate acceptance work.

## Maintaining the smaller suite

Add a test for a distinct product behavior or consequential failure mode. Extend an existing
scenario when it exercises the same contract, and remove superseded cases. Keep grammar and
protocol variants only where they follow different code paths or reproduce a distinct bug.
Do not freeze implementation strings, historical benchmark totals or arbitrary test counts.

Full corpus scoring and benchmark reproduction remain separate commands. Test count alone
does not demonstrate coverage, and this reduction is not an application performance claim.

## Verification

The complete `make test` run passed: **741 cases in 42.07 seconds**, including both JavaScript
bridges, with the existing Starlette/httpx deprecation warning. Scoped Ruff and
`git diff --check` passed. Independent reviews checked retained critical behavior, the seven
deleted modules and the documented omissions. No benchmark artifacts changed.

The local run used `UV_CACHE_DIR=/private/tmp/bookforge-test-sweep-uv-cache`, `UV_OFFLINE=1`
and `UV_NO_SYNC=1` to use the installed environment within the desktop sandbox. No paid model,
cloud or device experiment ran. This verifies local tests, not a GitHub CI run.
