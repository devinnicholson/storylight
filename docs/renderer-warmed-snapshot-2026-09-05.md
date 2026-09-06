# Warmed renderer snapshot qualification

The [imports-only snapshot](renderer-cold-start-2026-09-05.md) restored twice with identical
images, but first rendering still took 10–13 seconds. Its cache loader installs compiled wrappers
without executing the model. This probe captures the unchanged renderer after its two token
buckets have each run twice, moving that first-execution work before snapshot creation.

This differs from the older failed full-transformer snapshot: it uses the existing baked weights,
regional `default` compilation and verified compiler cache. No precision, image size, model
revision, prompt, seed or compilation setting changes. Roughly 17 GiB of GPU state and compiler
subprocesses remain compatibility and restore-cost risks; successful import snapshots do not
prove this larger snapshot works.

## Frozen scope and gates

One capture and three sequential single-use requests are allowed. Capture loads the existing
runtime/cache and renders the two frozen public references twice. Every master and depth hash
must match before capture can complete. Only the warmed runtime is retained; returned image bytes
are discarded. Each restored container validates its current hardware, refreshes activation
identity and renders four reference images without reloading or recompiling the model.

Correctness requires three distinct containers, two later restores of the same observed capture,
all reference hashes identical, unchanged renderer identity, and no failed or ambiguous request.
The complete provider log must show restoration without a native fallback or retry. One permitted
capture may be insufficient for a different underlying worker type; that fails this bounded
qualification rather than allowing another attempt automatically.

Latency is a separate gate: the first 128-token and first 256-token render in each later restored
container must each take at most **2.5 seconds**, including depth and encoding. Median complete
four-render artifact-ready time across those two restored requests must be at most **15 seconds**.
Results report capture and restore separately. No physical-display or causal baseline improvement
is claimed, and the unresolved renderer fidelity gate still prevents production promotion.

Placement broadens to US regions without a fixed underlying cloud. The prior experiment recorded
explicit L4 capacity waits under fixed AWS/eastern constraints, while the appliance's current
candidate already permits broad placement. This change and the larger snapshot are qualified
together; their individual latency effects are not isolated by this test.

## Execution and funding

One L4, eight CPU cores and 64 GiB requested/maximum RAM remain fixed. The nonparametrized class
has minimum/buffer zero, maximum one, a two-second idle window, single-use containers, a
120-second startup timeout, a 60-second input timeout and no application retries. The client has
a 210-second per-input ceiling. A monotonic external watchdog starts before deployment and stops
the exact app after at most 240 seconds, with 60 seconds allowed for shutdown. Extra pools or
unexpected concurrency abort the run. There are no builds, public endpoints or appliance changes.

The **$1.15** reservation covers two resource slots for 300 seconds, four 30-second teardown
allowances, and $0.50 setup: **$1.09078880** at the conservative 1.75× regional rate. Broad US
selection currently uses a lower 1.15× premium.
[Modal region pricing](https://modal.com/docs/guide/region-selection).

Independently reviewed closeout bounds retain $1.15 for the completed memory comparison, $0.57 for
the CPU import audit and $1.05 for the imports-only snapshot. These release $1.20 without changing
any unrelated hold or the funding envelope. Full setup allowances remain included, and reported
charges are not treated as settled bills. At reported workspace usage $14.42443591, adding this
probe projects **$34.87443591** under the existing $35 stop. Fresh billing must still pass before
dispatch. The [evidence directory](../benchmarks/renderer-warmed-snapshot-2026-09-05) retains the
reconciliation and frozen authorization.

Local qualification passes 904 Python tests, all three JavaScript suites, scoped lint and
independent implementation review. Four focused tests cover capture-before-restore work, reference
bytes, current GPU identity, finite claims, cancellation, and separate correctness/latency failures.
The exact declaration also passes offline validation against Modal 1.5.5. Manifest and private
authorization are frozen; preflight made zero generation calls.
