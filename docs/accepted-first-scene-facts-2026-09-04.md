# Accepted-first SceneFacts development gate

**Decision: reject promotion.** The frozen 512-record run completed without request failures,
but produced zero accepted graphs and no output improvement. Median final planning was 1.046
seconds, meeting the latency gate. Semantic improvement failed, so no paid image comparison ran.

The `tensorrt_accepted_graph` research backend makes one request with the unchanged accepted
four-slot prompt and 64-token cap. It attempts finite graph binding on that response and the
local source. A refused graph retains the same accepted wire fields without another model call.
The installed appliance remains on `tensorrt_slots`.

## Frozen hypothesis and gates

This experiment follows the rejected [hybrid-first comparison](live-scene-facts-gate-2026-09-04.md).
Removing dependence on generated hybrid IDs should allow useful deterministic recovery without
the cost of a second inference. This is a postprocessing hypothesis, not a model improvement.
Before measurement, eight synthetic plain-slot cases improved from six accepted graphs to eight
after bounded handling of explicit `as`/`while` clauses and directly stated result motion.
Simultaneity does not establish causality. The live adapter never imports corpus target derivation.

Promotion requires semantic improvement over the matched accepted renderer, zero final privacy
failures, no material regression in previously passing categories, and median local planning at
or below 1.5 seconds. Report p95, maximum, refusals, token counts, sampled memory and thermals.
Paid image comparison remains conditional on this local gate. Neither the hidden split nor cloud
inference is used for this experiment.

## Runtime behavior

Set `BOOKFORGE_LIVE_SCENE_PLANNER_BACKEND=tensorrt_accepted_graph` only for an explicit research
deployment. The accepted backend, prompt, wire schema, and cache identity remain unchanged;
this candidate has its own cache identity. Graphs survive persistent cache reads. The graph is
grounded and privacy checked through default-style page compilation before attachment. An unsafe
or invalid accepted response still refuses. The candidate cannot rescue an invalid accepted plan.

Safe visual styles longer than the graph compiler's 120-character bound reuse the accepted page
after graph, source, and style privacy validation. Unsafe styles and invalid graph facts still
reject. Incomplete or malformed model responses fail with value-free errors and no inference retry.

## Measurement and interpretation

The `--mode accepted_first` benchmark obtains exactly one accepted response per public development
record. It separately scores accepted raw slots, accepted renderer text, typed graph, and final
renderer text. Fallback hashes and token counts must match the accepted output. Per-case required
atom booleans expose losses and gains even when total recall is unchanged. Grading runs outside
all construction timers; graph construction includes accepted normalization and the graph helper.
Latency is HTTP inference plus measured local construction. It includes any resident-server
queueing inside the request, and excludes upstream reader queueing, image rendering, and
network transport to an image provider.

Code, dataset, prompts, request options, engine, installed wheel, environment, and hardware
snapshots are bound to the journal. Evidence retains hashes, scores, timings, and resource samples;
it does not retain source passages, raw model outputs, or target strings. Interrupted requests
are not retried. Aggregate reproduction does not rerun inference or regrade unavailable raw text.
The accepted renderer uses the candidate checkout's shared postprocessing, rather than replaying
the installed wheel's postprocessing. The accepted raw prompt and resident model are unchanged.

The scorer is unchanged. `scripts/diagnose_live_scene_facts_scoring.py` runs positive controls
using only public development targets. Exact raw target envelopes can pass the raw evaluator.
Typed exactness also compares representation: removed articles or a modifier stored as an
attribute instead of inside an entity label can fail. The lexical renderer screen can penalize
facts expressed in separate clauses. These controls describe scoring limits; they neither certify
candidate semantics nor replace the frozen gate with a more favorable score. No image fidelity
claim follows from a text score.

The positive controls at frozen revision `dd2e2645b051817f21d12f55a92f84ba44f6cc60` are:

| Control | Evaluated | Exact | Required atoms passed |
| --- | ---: | ---: | ---: |
| Exact target raw envelope | 512 | 512 | 2,560/2,560 |
| Literal target envelope as renderer text | 512 | 512 | 2,560/2,560 |
| Eligible typed target | 509 | 509 | 2,545/2,545 |
| Directly compiled typed target as renderer text | 383 | 310 | 1,842/1,915 |
| Live adapter given exact target slots | 30 | 0 | 90/150 |
| Compiled live adapter given exact target slots | 30 | 0 | 90/150 |

Three ambiguous target graphs refuse. Of the 509 eligible targets, 126 fail direct general-purpose
source grounding; the corpus target machinery has proof rules that the live compiler does not
use. All 73 missed atoms among the 383 compiled target prompts are ACTION atoms. Exact plain slots
still produce 482 adapter refusals: 358 ungrounded and 124 unsupported syntax. The 30 surviving
graphs each miss ACTION and MAGIC under the existing representation-sensitive scoring. These
controls reveal both limited live parsing coverage and scoring mismatches; they do not establish
a numerical upper bound for arbitrary learned outputs or explain every future model failure.
Construction refusals are unavailable graph outputs before scoring. Identical accepted/final
hashes prove unchanged output. Changed hashes and atom booleans identify scored gains or losses,
but cannot distinguish dropped facts from rephrasing after the underlying content is discarded.

GPU allocator peak is unavailable. The 100 ms sampler reports resident process RSS, system used
memory, and available thermal-sensor extrema; short peaks can be missed. Candidate source runs
from an isolated Jetson directory using the installed Python environment, without replacing the
wheel or changing service, engine, power mode, or environment settings.

## Reproduction

Use the frozen revision for both commands. Discover the resident PID before a new inference run;
the retained experiment uses PID 1504944. Omit `--aggregate-only` only for a separately authorized
new experiment with a new journal and fresh provenance.

```bash
.venv/bin/python scripts/diagnose_live_scene_facts_scoring.py \
  --output /tmp/accepted-first-scoring-controls.json
cmp /tmp/accepted-first-scoring-controls.json \
  benchmarks/accepted-first-scene-facts-2026-09-04/scoring-controls.json
.venv/bin/python scripts/benchmark_live_scene_facts.py \
  --mode accepted_first --endpoint http://127.0.0.1:11435 --model llm --memory-pid 1504944 \
  --provenance benchmarks/accepted-first-scene-facts-2026-09-04/provenance.json \
  --evidence benchmarks/accepted-first-scene-facts-2026-09-04/development.jsonl \
  --output /tmp/accepted-first-reproduced.json --aggregate-only
cmp /tmp/accepted-first-reproduced.json \
  benchmarks/accepted-first-scene-facts-2026-09-04/summary.json
```

The previous hybrid-first artifact still reproduces byte-for-byte using its own frozen revision,
`30c71fa`. Keep its code and evidence separate from this candidate's fingerprints.

## Results

All 512 public development requests completed without interruptions or transport failures.
There were 346 `adapter_refused` and 166 `parse_refused` results. Here `parse_refused` includes
accepted-plan validation failures; it does not mean that 166 raw envelopes were malformed.
All 512 raw outputs passed the raw schema screen. This mode retains the final helper outcome,
not the adapter's finer refusal enum, so the report cannot attribute every refusal to a grammar rule.

| Surface | Valid schema/contract | Exact cases | Privacy failures | Required atoms passed | p50 ms | p95 ms | Max ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Accepted raw | 512/512 | 0/512 | 0 | 1,399/2,560 | 1,033.4 | 1,221.0 | 1,362.3 |
| Accepted renderer | 346/512 | 0/512 | 0 | 741/2,560 | 1,045.9 | 1,229.6 | 1,369.5 |
| Graph candidate | 0/512 | 0/512 | 0 | 0/2,560 | 1,040.6 | 1,229.2 | 1,371.1 |
| Final renderer | 346/512 | 0/512 | 0 | 741/2,560 | 1,046.2 | 1,236.9 | 1,378.3 |

Output tokens are 28/35/40 at p50/p95/maximum on every surface: all refer to the same upstream
model request, including the graph row. No graph wire was emitted. The final path has 346 nonempty
contracts and 166 empty/refused results; this is not 512 successful render jobs. Zero graph privacy
failures is vacuous at zero graph coverage. All 512 accepted/final output hashes and token counts
match, including the empty results. Per-case and per-category atom gains, atom losses, exact
improvements, exact regressions, schema regressions, and privacy regressions are all zero.

Graph-helper construction took 6.717 ms median, 9.108 ms p95, and 18.644 ms maximum, excluding
grading. The matched final-minus-accepted timing should be read from per-case measurements;
subtracting two independent percentile values is not a percentile of overhead. This experiment
removes the previous second-inference fallback penalty, but supplies no richer scene facts.

Sampled TensorRT process RSS peaked at 4,017,385,472 bytes and system used memory at 6,835,621,888
bytes. There were 6,053 sample attempts, with thermal sensors spanning 48.906–68.343°C. The retained
sampler description says "paired cases" because the harness also supports the older mode; this
run made one HTTP request per case and compared four surfaces. GPU allocator peak is unavailable.

Evidence is in [`benchmarks/accepted-first-scene-facts-2026-09-04/`](../benchmarks/accepted-first-scene-facts-2026-09-04/).
The eight-case smoke has a separate journal and is excluded from all 512-case results.
The complete journal has 1,025 lines: one bound header and one start/result pair per record.
Both this aggregate and the scoring-control aggregate reproduce byte-for-byte.
Before/after engine and tokenizer hashes, installed modules, accepted prompt, environment hash,
planner configuration, power mode, and resident service identity match. Both services remain active,
the API reports ready, and the resident model remains `llm`. The prior device check's camera
presence/enumeration failure remains an open physical acceptance item; this typed-text run did
not re-test or repair the camera.

Verification passed 1,560 Python tests, both JavaScript suites, scoped Ruff, the previous public
target artifact, and historical hybrid aggregate reproduction. The only Python warning is the
external Starlette/httpx deprecation. The installed runtime is not promoted by publishing this
research implementation and evidence to main.

Before another inference experiment, use synthetic or public training fixtures to diagnose the
binding refusals and establish representation-compatible positive controls for semantic comparison.
Freeze any revised evaluator separately and retain these original scores. Current evidence does
not justify a paid image A/B, a model-training claim, or enabling either graph candidate on the
accepted appliance.
