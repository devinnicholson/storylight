# Prepared Klein serving canary

This prospective worker exposes a compiled-prompt interface over the verified
prepared-session renderer. It is disabled until the parent task qualifies the
images, reviews the adapter, and authorizes a separate finite deployment. No
production routing, cloud lifecycle, or automatic scaling is implemented here.

`app.py` pins four byte-identical files from `renderer-session`: its application
is named `session_worker.py`, alongside `runtime_worker.py`, `session_state.py`,
and `noise.py`. The original endpoints remain unchanged in their original
directory. The serving module exports `PACK_PROOF_SHA256`, as required by the
unchanged packed loader. Runtime, model weights, compiler cache, profile, and
depth generation are inherited without changes. A future image must preserve
those parent layers and bind this wrapper in `manifest.json`; `session.json`
still pins the original runtime worker and lifecycle sources.

The two IAM-private routes are:

- `POST /v1/prewarm {session_id}`: claim one session ID, execute the original
  128/256 warmups, and return `{lease, model_load_seconds, warmup_seconds}`.
- `POST /v1/generate {session_id, instance_id, request_id, scene_id, prompt, seed}`:
  render one already-compiled prompt and return
  `{lease, request_id, scene_id, master_b64, depth_b64, metrics}`.

Session, instance, and request IDs are lowercase 32-character hexadecimal UUIDs.
The lease contains `schema_version`, `state`, `session_id`, `instance_id`,
`service`, `revision`, `identity`, `started_at`, `expires_at`, and `warmups`.
Warmups contain the actual bucket, seed, master hash, and depth hash. The lease
expires 300 seconds after preparation starts; repeated prewarming does not extend
it. `model_load_seconds` is the original factory/cache setup measurement;
`warmup_seconds` sums the two renderer `total_seconds` values, excluding model
loading and compilation. Neither replaces the adapter's inclusive preparation timer.

At most ten request claims and results are retained after the two warmups. A
completed exact request replay returns the stored response. Changed payloads,
pending claims, a different process/session, capacity exhaustion, and expiry
return 409. Rendering is serialized; failures and cancellation consume the claim
and invalidate the session. Cancellation or disconnect waits for the render
thread to exit before releasing its lock. The 180-second request timeout does
not pretend to terminate CUDA work; the external owner remains responsible for
the finite service lifecycle and deletion.

Only prompt, seed, and identifiers cross this interface—there are no transcript
or book fields. Prompts are bounded to 4,000 characters, and the worker repeats
the edge policy's email, phone, URL, and sensitive-keyword checks. Original-source
privacy and fact grounding still belong to the existing edge planner; these
pattern checks cannot prove them. Responses are bounded to 12 MiB each. No HTTP
retry, replacement worker, cache export, or paid fallback is added.

Run the CPU-only fake-worker checks with:

```sh
.venv/bin/python -m pytest -q experiments/prepared-klein-serving/test_app.py
.venv/bin/ruff check experiments/prepared-klein-serving
```

The tests cover real lifecycle composition, both warmups, stable request replay,
capacity, pending duplicates, cancel/disconnect ownership, expiry, sensitive-input
refusal, and source tampering. They do not establish GPU speed or image quality.
