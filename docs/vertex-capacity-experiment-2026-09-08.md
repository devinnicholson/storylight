# Vertex capacity and repeatability

## C protocol, frozen before dispatch

Keep the production global endpoint, model, TEXT+IMAGE payload, rich watercolor
style, HTTP/1.1 and default minimal thinking. Send twelve unfiltered requests:
the same six synthetic scenes as A/B, each twice with identical payload and seed.
Use seeds 7600–7605. Space starts at least 30 seconds apart. This is a diagnostic
A/A run, not an optimization qualification or a production throttle change.

Reserve $3 independently of earlier experiments. Twelve requests have a
conservative maximum token reservation of $1.48656 using the earlier 4096-output,
4000-input bound; twelve typical image components estimate $0.4032. Authentication
has a separate 30-second deadline, the paid loop has 600 seconds, and each request
has 90 seconds. Stop at the first error, with no retry or output-directory reuse.
Preserve response metadata and any Retry-After/request-ID/server-timing headers.

Record every dispatch and result, first request separately, total successful
delivery latency, dimensions, model version, usage, and decoded-pixel equality for
each A/A pair. Completing twelve calls only establishes success at this tested
pace and time; it cannot prove a reliable capacity ceiling or a production p95.
Pixel differences between identical unfiltered requests establish baseline
variation in this sample; they do not retroactively pass B's frozen exactness gate.

Read-only preflight: billing enabled, delayed gross project cost $35.73 at
2026-09-08 06:22:36 UTC, existing two Cloud Run services, existing GKE coordinator
at one replica and Nemotron at zero. All two pages of the Vertex Service Usage
inventory returned 366 metrics, with no exact `gemini-3.1-flash-lite-image` entry.
This absence alone does not prove unlimited capacity. No infrastructure changes.

Google's [exact-model specification](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/3-1-flash-lite-image)
lists global only and no fixed quota. [Standard PayGo](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/standard-paygo)
excludes this image model from spend-based usage tiers. Its text-model tier table
must not be applied to this renderer. [429 guidance](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/deploy/error-code-429)
recommends smoother traffic and gradual increases; a generic 429 does not identify
a fixed request limit. [Seed documentation](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/capabilities/content-generation-parameters)
describes best-effort rather than guaranteed determinism.

[Priority PayGo](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/priority-paygo)
does not list this image model. [Provisioned Throughput](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/provisioned-throughput/supported-models)
does support it, but a capacity commitment is distinct from this finite experiment.
No throughput purchase or speculative priority header is included.

```sh
.venv/bin/python scripts/benchmark_vertex_output.py \
  .bookforge/vertex-output-20260908-c --run --capacity --interval 30
```

## C outcome

Five images succeeded, then request six received an explicit 429 with no
Retry-After or retained request-ID/server-timing header. Both completed A/A pairs
had identical pixels and dimensions. Their latency nevertheless differed by
13.59% and 30.67%, demonstrating substantial timing variation without a treatment
change. The third scene has no completed repeat. Thirty-second spacing did not
eliminate failures in this sample. No optimization is qualified.

## D protocol, frozen before dispatch

Screen `gemini-3.1-flash-image` at 512 resolution, with MINIMAL thinking and an
explicit 4096-token output limit. Retain C's twelve-call schedule, prompts, style,
paired seeds, global endpoint, JPEG95, TEXT+IMAGE and 30-second pacing. This changes
model and resolution together: it screens a candidate product configuration, not
the isolated effect of either change. Production stays on Lite at its original
resolution. Do not assume the alternate model has independent capacity.

The [Flash Image specification](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/3-1-flash-image)
supports 512 output and normally permits 32,768 output tokens; the explicit cap is
necessary to bound this experiment. The [published global prices](https://cloud.google.com/gemini-enterprise-agent-platform/generative-ai/pricing)
are $0.50/M input, $3/M text/thinking and $60/M image output. Its 747-token 512 image
component estimates $0.04482 each. Reserve another $3: at most twelve calls with
4096 output tokens priced conservatively at the image rate plus 4000 input tokens
equals $2.97312. Keep C's deadlines and first-error stop. No retry or new service.

Prospective advancement screen: twelve successes, median artifact ready below
3 seconds and observed maximum below 5 seconds, with all six scene facts and rich
watercolor treatment passing assistant inspection. Advancement would justify a
matched comparison and human review, not automatic default-model promotion.
Retain wrong counts/actions as failures. Lower resolution is an explicit tradeoff;
do not describe smaller files as faster GPU inference. The refreshed delayed cost
notification remained $35.73 before dispatch.

```sh
.venv/bin/python scripts/benchmark_vertex_output.py \
  .bookforge/vertex-output-20260908-d --run --capacity --interval 30 --flash-512
```

## D outcome and direction

Two requests succeeded at 6.926 and 6.523 seconds, producing 688×384 JPEGs with
747 image tokens each. Request three returned explicit HTTP 429, again without a
retained recovery hint. The two repeated unfiltered images differed in pixels
and composition. Assistant inspection found the requested orange-cat/gray-mouse
chase and watercolor treatment in both. This single fact set does not qualify the
other five scenes, nor the lower-resolution projection quality.

D fails the speed, failure and coverage screen. Keep it experimental. The
response filter is not promoted either. A/B/C/D now establish that neither
15/30-second spacing nor this alternate model eliminated the observed shared
capacity failures. They do not identify the upstream cause or prove a fixed quota.

The next systems direction is bounded session preparation on the native GCP GPU
path, retaining a ready worker while reading. Earlier
[native GCP measurements](gcp-klein-cache-diagnostics-continuation-2026-09-07.md)
showed a 0.483-second median across eight later deliveries, but 76.011 seconds for
the first request, with image acceptance still incomplete. A useful next trial
must count preparation time and idle cost, exercise both token buckets before
declaring ready, and measure first-scene delivery after readiness and subsequent
requests. It must also retain warm-worker loss and quality failures. Merely moving
the startup outside a timer is not faster cold inference. No new GPU service was
created in C/D and no capacity reservation was purchased.

## Deployed recovery fix

Live process inspection confirmed `gcp_resilient`, the Lite model, no configured
Cloud Run URL, deferred fidelity, and a 300-second routing failure cooldown. An
explicit Vertex 429 previously opened that same five-minute circuit as all other
safe rejections. Later requests skipped Vertex without checking recovery.

The router now recognizes a typed rate-limit rejection and permits later requests
after the valid Retry-After delay, or 30 seconds if absent/invalid. Both integer
seconds and HTTP-date headers are supported; legitimate longer server delays are
preserved. This removes 270 seconds of default local exclusion. It does not make
upstream capacity available or automatically retry the failed generation. Other
failures retain the existing configured cooldown, and ambiguous transport failures
still suppress fallback and repeat billing. Prompt, model, field mask and image
settings remain unchanged in production.

The two installed Jetson files differed from repository HEAD only by comments and
a redundant catch-and-reraise. Their exact hashes were checked before replacement,
and backups retained at `/opt/bookforge/.voice-backups/rate-limit-20260908T064444Z`.
An initial local expected-hash typo caused the guard to stop before any write;
correcting it allowed the same reviewed bytes to install. Both modules imported
in the actual Python 3.12 environment. The allowlisted API restart succeeded, and
provider readiness selected the original Lite route over HTTP/1.1. That HEAD check
proves connectivity/configuration, not image capacity.

Rollback: restore both saved Python modules to the installed package under
`/opt/bookforge/.venv/lib/python3.12/site-packages/bookforge`, then run
`sudo -n /usr/local/sbin/bookforge-admin restart-api`.

## Evidence and cost

Retained source snapshots, manifests, journals, hashed response metadata and
summaries are in `benchmarks/vertex-output-2026-09-08/{c,d}`. Image bytes and full
read-only quota responses remain in local `.bookforge` directories. Reproduce each
summary with `scripts/benchmark_vertex_output.py` and its original local directory;
this verifies image hashes and decodes every completed pair again. C's provider
source corresponds to pre-change commit `f3d59c5`; D's source is the reviewed
rate-limit fix. Neither difference changes the benchmark's request body or its
direct-client error handling.

C's five successes estimate $0.168269; D's two estimate $0.089855. Their combined
successful-token estimate is $0.258124 across nine attempts, including two 429s.
Across A–D there were eighteen attempts and fourteen successful images, with a
$0.493700 successful-token estimate. These are not settled invoice totals. The
latest delayed project notification remained $35.73 at 06:44:19 UTC; reporting lag
prevents using that unchanged value as proof that C/D were free.

All 1,070 Python tests and six JavaScript suites passed. Independent reviews cleared
each paid launch and the narrow recovery change. Scoped lint, whitespace checks,
source-hash verification, benchmark reproduction and a credential scan complete
the closeout; no production speed or provider reliability qualification is claimed.
All 17 provider/router tests also passed against the actual installed Jetson
modules using mocked transports, with no paid requests.
