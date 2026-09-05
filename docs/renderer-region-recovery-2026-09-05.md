# Regional renderer recovery

The corrected western-region attempt did not acquire an L4 within the client's 310-second result
deadline. It recorded one failed SDK warmup and no completed images. Both temporary applications
were stopped by the supervisor, each with return code 0; separate checks confirmed zero containers.
The proxy token was revoked and its private local and Jetson files removed. The accepted reader
and planner remained running.

This is an availability failure, not a measured speed or image-quality result. Provider logs
reported the requested AWS `us-west`, L4, and 64 GiB memory combination waiting for capacity.
The container inventory was empty during the wait. The corrected import layout passed locally,
but this attempt did not establish successful remote model initialization.

Evidence is retained under [renderer-region-2026-09-05-b](../benchmarks/renderer-region-2026-09-05-b):
the exact manifest, source hashes, journal, rejected summary, scheduling messages, supervisor,
shutdown/cost receipt and independent verification. The summary reproduces byte-for-byte. All
891 Python tests, both JavaScript suites and scoped lint passed; public coverage is unchanged.

## Budget reconciliation

The original ledger treated every planned operation as still possible after its app had stopped.
We reduced only two named holds using closed-run evidence, without claiming billing had settled:

- Previous latency comparison: $4.58 to $2.00. Both full app lifetimes plus 30 seconds of shutdown
  at maximum resource rates cost at most $0.42761856; adding the complete $1.50 setup allowance and
  rounding upward leaves $2.00.
- Failed first regional attempt: $6.96 to $1.89. Keep the complete $1.50 setup allowance and $0.39
  for its sole dispatched operation. Release capacity only for 13 operations never dispatched.

This freed $7.65 while preserving every other hold and ledger record. The fresh $6.96 western run
then projected $33.93014709 under the approved $35 workspace stop. Its complete reservation remains
held in this receipt. Actual workspace usage still reported $13.83014709 immediately after shutdown.
Reservations remain conservative future-charge allowances, not additional actual usage.

The next bounded hypothesis is a matched AWS eastern-region run. It must use a new identity and
fresh funding proof, preserve this attempt, and hold both transports to the same observed compute
region. No automatic region fallback or paid request retry is enabled.
