# Test suite reduction

Baseline: `776d8d673e583b65f1079e5d485d051411bf193c`.
This sweep changes tests, with no production-code, CI-workflow, or benchmark-artifact changes.

| Measure | Before | After | Removed |
| --- | ---: | ---: | ---: |
| Collected Python cases | 1,720 | 1,496 | 224 |
| Python test functions | 1,205 | 1,050 | 155 |
| Python test files | 125 | 121 | 4 net |
| Python test-file lines | 40,122 | 37,076 | 3,046 net |

File and line counts cover `tests/test_*.py`. The two JavaScript behavior suites are unchanged.
No skip, collection exclusion, or loop replacing parameterized cases was added to reduce counts.

## What was removed

- Jetson packaging, CSS/HTML, source-order, historical hash, and pinned-value snapshots. The
  packaging/browser slice falls from 118 to 41 Python cases. Seven executable kiosk checks,
  configuration rollback, privacy, approval, terminal lineage, and asset-integrity tests remain.
- Thirty training/cloud tests that restated configuration values, patch text, or historical command
  shapes. Current checkpoint custody, separate billing parsers, concurrency, approval, and rollback
  behavior remain. Documented v2/v3 staging, cache/rematerialization comparisons, and clone workflows
  are still callable; their distinct executable guards were not retired with the historical runs.
- Repeated name/protocol and temporal/negation Cartesian combinations. The live/graph slice falls
  from 642 to 558 cases. Each distinct grammar family remains at the shared validation boundary,
  with representative protocol integration. Wrong bindings, negative facts, repeated-verb order,
  counted transformations, unsafe styles, and renderer-prompt proof retain regressions.
- Full training replay loops and archived hardware-output sweeps embedded inside a few unit tests.
  Synthetic semantic controls remain. Full aggregate scoring and reproduction stay available through
  the unchanged diagnostic scripts and their documented commands.
- Duplicate record serialization, manifest recomputation, fixture-schema, CLI-option, and default
  value checks. The retained manifest integration already executes metadata recomputation; API,
  source-grounding, privacy, and runtime-limit behavior remain tested.

The two diagnostics retain small tests that execute their actual report builders against one public
record, enforce the permitted split, and reject source/target/record-ID leakage. These replace the
expensive full-corpus wrappers without dropping the scripts' own privacy checks.

## Invocation and limits

`make test` and CI retain their existing Python entrypoint and two bridge tests that invoke the
standalone Node scripts. Node must be installed for those browser checks to run; the existing
bridges skip them when Node is absent. Both ran successfully locally. A proposed move to explicit
CI steps was withdrawn after GitHub rejected the workflow update for insufficient OAuth scope.
The final commit keeps the existing workflow and browser coverage unchanged.

This is a deliberate reduction in exhaustive combinations and static snapshots, not a proof of
identical mutation or branch coverage. Deployment spelling, UI copy/layout, historical pins, and
every protocol-by-grammar combination are no longer individually frozen by tests. Actual hardware,
cloud deployment, model accuracy, and image fidelity still require their separate acceptance work.

For future additions, prefer a public behavior and a distinct failure mode. Reuse existing fixtures
and shared validation cases; add a protocol-level case when that protocol changes the behavior.
Avoid tests that copy implementation strings, retest standard schema mechanics, or freeze arbitrary
test counts and historical benchmark totals.

## Verification

The complete `make test` run passed: 1,496 Python cases including both JavaScript bridges. The existing
Starlette/httpx deprecation warning remains. Scoped Ruff and `git diff --check` pass. Independent
reviews checked the deletion rationale, diagnostic privacy consolidation, and retained semantic
regressions. No production files, historical evidence, or user-owned output directories changed.

The local run used the existing environment with `UV_NO_SYNC=1` and `UV_OFFLINE=1`; its uv cache
was placed under `/private/tmp` because the desktop sandbox cannot write the default user cache.
This is local test execution, not evidence of a GitHub CI run or a measured application speedup.
