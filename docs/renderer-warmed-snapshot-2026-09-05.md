# Warmed renderer snapshot qualification

The completed warmed-runtime trial is **rejected**. Its capture hook rendered and internally
hash-checked four reference images, then Modal failed to create the snapshot. No image payload
returned to the client, no snapshot restored, and neither correctness nor latency qualified.
The app is stopped with zero containers. The separate serial-compiler trial also failed
qualification: its 180-second deadline expired before any platform snapshot result or restore.
Both apps are stopped; accepted appliance routing and the visual-fidelity gate remain unchanged.

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

## Original execution and funding

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

The frozen authorization and original funding calculation remain in the
[evidence directory](../benchmarks/renderer-warmed-snapshot-2026-09-05). Before dispatch, local
qualification passed 904 Python tests, all three JavaScript suites, scoped lint and independent
review. Offline Modal 1.5.5 validation made no generation calls.

## Completed trial and stop verification

The [phase events](../benchmarks/renderer-warmed-snapshot-2026-09-05/phase-events.jsonl) show the
capture hook completed all four warmups and their internal master/depth checks. Imports took
12.452 seconds. The images were discarded inside the capture hook as designed; none were
independently downloaded or returned. The subsequent platform checkpoint failed, so the client
[summary](../benchmarks/renderer-warmed-snapshot-2026-09-05/summary.json) correctly records one
failed request, zero observed captures and zero restores. A completed capture hook is not a
successfully created platform snapshot.

Despite application `retries=0`, Modal started five replacement tasks after the first task failed.
The durable single-capture claim refused their heavy work before imports. This demonstrates that
application retry settings do not prevent platform replacement attempts. An emergency app stop
succeeded; the supervisor's later stop found it already stopped. The
[cleanup record](../benchmarks/renderer-warmed-snapshot-2026-09-05/cleanup-cost.json) verifies all
six observed tasks ended, the app is stopped, and no containers remain. The earlier client summary
could not verify external shutdown; the later cleanup record supplies that proof.

The app-attributed charge is provisionally **$0.04543025**, with reported workspace usage
**$14.46986616**. These are reported charges, not settled bills. The initial $1.15 hold stayed in
place at cleanup. Subsequent reviewed reconciliation bounds this closed run at a **$0.90 gross
ceiling**, retaining setup, the observed lifetime, an additional capture-worker allowance and
teardown allowances for the replacement tasks.

## Completed serial-compiler trial

The separate `bookforge-klein-serial-snapshot` candidate sets
`TORCHINDUCTOR_COMPILE_THREADS=1` after authorization and the single capture claim, before any
Torch or Diffusers import. It refuses an already imported Torch runtime and verifies effective
Inductor configuration after imports, immediately before compile, and after restore. Logs expose
only the previous environment category (`unset`, `one`, or `other`) and the checked value one.
The response schema, reference hashes, baked image, model, precision, regional `default` compile
mode, compiler cache, placement and resource limits remain unchanged.

Modal documents this setting as a mitigation for some Torch Compiler snapshot failures, not a
guarantee. [Modal memory snapshot limitations](https://modal.com/docs/guide/memory-snapshots).
In pinned Torch 2.8, one compiler thread bypasses persistent Inductor thread/process pools and
executes compile submissions synchronously. This is a concrete hypothesis for the failed
checkpoint; the platform error does not identify its cause, and transient native compiler
subprocesses may still exist.
[PyTorch 2.8 compiler implementation](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_inductor/async_compile.py).

The current baked image's recorded source lineage does not set this variable; its effective
container value was not captured in the failed trial. The older full-transformer snapshot did
already set one, but used `reduce-overhead` on a different image branch. Its failure neither tests
nor validates this current regional-compile/cache combination.

The serial trial retained one capture and at most three sequential requests, with the same
reference-output and two-restore latency gates. Its source-free logs verified compiler threads one
twice, with the original environment unset. The capture hook completed four internal reference
hash checks at 20:20:52 PDT. No activation or image payload returned. The
[client summary](../benchmarks/renderer-serial-snapshot-2026-09-05/summary.json) records one started,
one failed and zero unresolved requests, with zero observed restores.

The 180-second deadline expired without a native snapshot success, failure or restore message.
The [supervisor](../benchmarks/renderer-serial-snapshot-2026-09-05/supervisor.json) began stopping
the app at 180.035 seconds; shutdown succeeded and supervision ended at 181.040 seconds. The
supervisor correctly reports the experiment's timeout as failure. The immediate inventory
still showed one task during shutdown; subsequent final inventories verified the app stopped with
zero tasks and no containers, recorded in the
[cleanup receipt](../benchmarks/renderer-serial-snapshot-2026-09-05/cleanup-cost.json).
The continuous fatal-log guard remained active throughout. This
is a failed bounded qualification, **not proof that serial compilation is incompatible with
snapshots**: eventual checkpoint completion remains unknown.

The serial charge is provisionally **$0.04028130**, and reported workspace usage is
**$14.51014746**. Its original **$1.05 gross ceiling remains retained**. The accounting rule in the
[closed-run reconciliation](../benchmarks/renderer-serial-snapshot-2026-09-05/closed-run-reconciliation.json)
separates each gross ceiling into charges already included in the workspace report and a remaining
pending hold. Refreshing the floor and crediting the serial charge against that same gross ceiling
leaves the projection at **$34.96088685** under the unchanged $35 stop. This accounting update is
complete; these reports do not establish settled bills or release the gross allowance.
Credits are never subtracted again from a residual hold.

The practical path remains preserving useful warm-session lifetime and preparing the renderer
ahead of demand: warm inference is measured, whereas this warmed snapshot has not delivered an
image after restore. Explicit prewarm reuse already avoids duplicate work without extending its
original deadline. Automatic session prewarming and default promotion remain gated. Another
snapshot trial would require a separately bounded, longer qualification and funding review;
there is no automatic retry or further paid call from this result.
