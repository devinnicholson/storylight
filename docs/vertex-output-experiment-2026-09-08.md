# Vertex output modality experiment

Compare the current `TEXT, IMAGE` response against `IMAGE` only on
`gemini-3.1-flash-lite-image`, using the global Vertex endpoint and the current
request builder. No product route, prompt, style, model, thinking setting, seed,
JPEG setting, or GPU deployment changes in this experiment.

The six fixed synthetic scenes cover action, jumping, counts, appearance, spatial
relations, and a negative constraint. Each receives two paired seeds. Alternate
ABBA and BAAB between scenes, for 24 sequential POSTs and 12 observations per arm.
Use one HTTP/1.1 client with the same one-connection limits as production. Token
acquisition is outside measurement. The first request is retained and reported
separately; no paid warmup is hidden before the run.

The primary timer ends after the full response is received, the completed image
is validated, and its file is written. Response-header and full-response timers
are separate. These are Mac-to-Vertex measurements; they do not include the
Jetson, browser, depth packaging, or projection. They are not isolated GPU kernel
times. Report paired changes, medians, observed maxima, failures, image dimensions,
response bytes, text output, model version, token usage, and image checksums.

Prospective promotion screen: at least 20% lower median, no higher observed maximum
or failure count, and no regression in reviewed scene facts or the restored rich
watercolor appearance. Review every returned image, retaining errors. This small
screen cannot establish p95, production reliability, or general visual fidelity.
Human visual acceptance remains distinct from assistant inspection.

Google's [exact-model Vertex example](https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/getting-started/intro_gemini_3_1_flash_lite_image_gen.ipynb)
uses image-only output. Minimal thinking is already the default, so both arms
leave it unchanged. The [model specification](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/3-1-flash-lite-image)
limits output to 4,096 tokens and 1K images. At the published image rate of $30 per
million tokens, 24 single 1,120-token images estimate $0.8064 before input and text.
Reserve $3 for this fixed run, conservatively pricing all 4,096 possible output
tokens at the image rate, plus up to 4,000 input tokens per request. This is a
finite request allowance, not a cloud billing cap. No retry follows an ambiguous
call, malformed image, refusal, or timeout. Per-request wall timeout is 90 seconds;
the paid-loop wall timeout is 600 seconds, after at most 30 seconds for authentication.
Existing output directories cannot be reused.

Read-only preflight found billing enabled, $35.44 gross cost in the latest delayed
notification (2026-09-08 06:00:38 UTC), the existing scene and emergency-stop Cloud
Run services, two GKE nodes, the coordinator at one replica, and Nemotron at zero.
No infrastructure was changed. The first workload inventory command could not
find the installed GKE authentication plugin; using its existing SDK path fixed
the read-only command.

Run only after the local harness checks and review:

```sh
.venv/bin/python scripts/benchmark_vertex_output.py \
  .bookforge/vertex-output-20260908-a --run
```

Recompute without provider calls:

```sh
.venv/bin/python scripts/benchmark_vertex_output.py \
  .bookforge/vertex-output-20260908-a
```

The exclusive output directory retains the exact manifest, source hashes,
dispatch/completion journal, metadata for every received JSON response, and
generated images. Inline image bytes are replaced by hashes in response metadata;
validated image files are retained separately. Credentials are never recorded.

## A: stopped burst and next hypothesis

Run A completed three requests, then stopped on an explicit HTTP 429
`RESOURCE_EXHAUSTED`. One paired seed took 3.873 seconds for TEXT+IMAGE and
3.653 seconds for IMAGE, a 5.69% descriptive difference. Both returned zero text,
1,120 image tokens, and identical decoded pixels. The remaining successful image
has no completed baseline partner. This incomplete screen cannot qualify either
arm; the 20% prospective gate is unchanged. Three successes estimate $0.100961
from reported input/image tokens. The failed attempt and full schedule remain
retained; nothing was retried.

The response was 2,677,222 bytes, including a 2,067,140-character thought signature
and a 456,943-byte JPEG (base64 encoded on the wire). The signature accounts for
about 77% of response bytes. It is used for later editing turns, which the current
stateless generation route does not perform. This makes response filtering a
separate, measurable transport hypothesis rather than a claim of faster inference.

## B: response filtering, frozen before dispatch

Google's [system parameters](https://docs.cloud.google.com/apis/docs/system-parameters)
document `X-Goog-FieldMask` for response filtering. Compare the current unfiltered
response with a mask preserving image bytes, text/thought flags, completion and
safety fields, usage, model identity, response ID, and creation time. Omit only
unneeded fields such as the opaque signature. Both arms request TEXT+IMAGE and
use identical bodies, paired seeds, model, style, and HTTP settings.

Use another fixed 24-call ABBA/BAAB schedule, spaced at least 15 seconds between
dispatches. This tests a smoother workload after the burst error; it does not
claim to reproduce burst capacity. The [429 guidance](https://cloud.google.com/vertex-ai/generative-ai/docs/error-code-429)
recommends traffic smoothing; the error alone does not identify a fixed quota.
The same 90-second request and 600-second paid-loop deadlines and first-error
stop apply. Reserve another $3 independently of A's held attempts.

Prospective B gate: at least 60% fewer response bytes, at least 5% lower median
artifact-ready latency, identical decoded RGB pixels for all paired seeds, no
higher observed maximum and no failures. This narrower latency threshold reflects
the measured 250–400 ms transfer tail; it does not change A's failed/incomplete
inference hypothesis. Image decoding/hashing adds local verification work to B's
artifact timer for both arms. No production change is included in B dispatch.

```sh
.venv/bin/python scripts/benchmark_vertex_output.py \
  .bookforge/vertex-output-20260908-b --run --field-mask --interval 15
```

## B result: smaller payload, not qualified

Four requests succeeded, then ordinal 4 returned explicit HTTP 429
`RESOURCE_EXHAUSTED`. Dispatch gaps were 15.001–15.003 seconds. The run stopped
without retry. This leaves two observations per arm, both for the cat/mouse scene;
the other five scenes were not completed.

| Measure | Unfiltered | Filtered |
| --- | ---: | ---: |
| Successful requests | 2 | 2 |
| Median artifact ready | 3.678 s | 3.103 s |
| Observed maximum | 4.097 s | 3.138 s |
| Total response bytes | 5,548,700 | 1,281,110 |
| Image output tokens per success | 1,120 | 1,120 |

The first baseline request took 4.097 seconds and remains included. The two paired
latency differences were 23.40% and 5.92%; the descriptive median difference was
15.65%. Response bytes fell 76.91% in aggregate. Neither latency comparison proves
a stable or causal speedup with this sample and the terminal capacity failure.

The endpoint accepted the response mask and omitted the large opaque signature
in both successful candidate responses. Both pairs retained 1376×768 dimensions.
Seed 7400 had identical decoded RGB pixels, but seed 7401 did not. Assistant image
inspection found the same rich watercolor cat/mouse composition in the latter
pair with visual differences; it does not establish general fidelity or human
acceptance. Equal request seeds did not ensure identical images in this run, and
these observations cannot establish whether the filter caused the variation.

**Decision: no production promotion.** Coverage is incomplete, the exact-pixel
gate failed for one pair, and the candidate received a 429. The experiment remains
available through the benchmark flag. The production provider, Jetson service,
model, and image style are unchanged.

Four B successes estimate $0.134615 from returned input/image tokens. Together
with A, seven successes estimate $0.235576. There were nine total attempts,
including two explicit 429s; this token estimate is not an invoice reconciliation.
The latest delayed project notification rose from $35.44 to $35.73 at
2026-09-08 06:22:36 UTC; its $0.29 difference includes other project activity and
cannot be attributed precisely to these runs. No GPU service was started or
infrastructure changed, so no experiment GPU teardown was needed.

Evidence is in `benchmarks/vertex-output-2026-09-08/{a,b}`. Each directory retains
the executed harness snapshot matching its manifest hash, exact request schedule,
journal, hashed response metadata, and offline summary. Original images remain
in the corresponding local `.bookforge` directories. Reproduce B's summary with
the current analyzer, which decodes the saved images again:

```sh
.venv/bin/python scripts/benchmark_vertex_output.py \
  .bookforge/vertex-output-20260908-b > /tmp/vertex-b-summary.json
cmp /tmp/vertex-b-summary.json benchmarks/vertex-output-2026-09-08/b/summary.json
```

The next useful experiment is a separately bounded capacity check before spending
on another full quality matrix: inspect model-specific quota/usage and compare a
sparser dispatch schedule. Do not infer a quota limit from the generic 429. A later
filter qualification also needs repeated unfiltered seeds to distinguish normal
output variation from a treatment effect. Keep the current failed pixel gate
recorded rather than relaxing it after seeing the result.

Validation: 1,059 Python tests and all six JavaScript suites passed. Scoped Ruff,
diff checks and the staged credential scan passed. Independent review reproduced the paired payload,
pixel, timing, usage and failure findings. The benchmark contains no automatic
paid retries and refuses to reuse an output directory.
