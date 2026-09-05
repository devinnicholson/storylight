# Product fidelity: repaired candidate measurement

Frozen implementation: `d47895fea475cffb3b4c86c251e3610899e15624`.
Evaluator: `colored-references-renderer-proof-v3`.

The new seven-request smoke produced seven valid candidate contracts and six compiler-proved
graphs, compared with four valid accepted contracts. The final story page still uses fallback:
its model-selected result differs from the expected birds. The two-fox pages currently contain
only the selected focal actor, so graph construction does not pass their complete story checklists.

The matched development run completed all 512 public cases using one resident-model response per
case, with no retries. It compared the accepted path and opt-in graph path under the frozen evaluator.

| Measure | Accepted | Candidate |
| --- | ---: | ---: |
| Valid final contracts | 346 | 346 |
| Exact final contracts | 0 | 15 |
| Required facts retained | 956 / 2,560 | 970 / 2,560 |
| Final privacy failures | 0 | 0 |
| Median planning | 1,048.4 ms | 1,064.7 ms |
| p95 planning | 1,223.8 ms | 1,232.9 ms |
| Maximum planning | 1,365.3 ms | 1,375.6 ms |

The candidate accepted 43 graphs. The gate **rejects** it because actor/object selection loses
required facts: 67 to 64 out of 120 in that category, despite six new exact passes there. Every
other gate check passes. This blocks image generation and promotion while engineering continues.
No individual development failures are used for tuning this iteration; investigate the class with
synthetic and public training controls.

Accepted renderer text remains a lexical screen; candidate graph prompts require compiler proof.
These are contract scores, not learned-model improvement, image fidelity or physical projector
measurements. Privacy failure counts do not turn the 166 failed contract constructions into passes.

The unchanged fixed 32 training controls still produce two graphs, one exact. They expose remaining
finite-grammar and binding limits independently of the development run. The six-page story remains
frozen. Its transformation and event-order pages require visible ordered display states.

Evidence: `benchmarks/product-fidelity-gate-2026-09-04/`. Raw synthetic smoke responses remain in
owner-only archives on the Jetson; development journals contain sanitized measurements and hashes.
The candidate ran in an isolated directory, using the installed Python and resident loopback model.
No cloud inference, image generation, engine rebuild or appliance promotion ran.

Reproduce the aggregate with the frozen implementation:

```bash
PYTHONPATH=src /opt/bookforge/.venv/bin/python scripts/benchmark_live_scene_facts.py \
  --model llm --mode accepted_first --limit 512 --max-output-tokens 64 \
  --memory-pid 1504944 --provenance evidence/before/provenance.json \
  --evidence evidence/development.jsonl --output /tmp/development-reproduced.json \
  --aggregate-only
```

Subsequent explicit multi-actor and projector-navigation repairs are separate from this frozen
measurement. They must not inherit its measured accuracy or latency claims.

## Independent repair

Synthetic and public training controls reproduced an action-reference defect: the graph stored
the object's color but omitted it from the actor's action. “Raises blue lantern” became “raises
lantern,” losing action and actor/object checks. Revision `c210469` preserves the bound color.
The evaluator and decision criteria are unchanged.

Across 24 actor/object controls in the first 512 public training records, the repair retains the
same 20 graphs and four refusals. Graph facts rise from 74 to 86, against 58 accepted facts, with
no accepted-to-graph fact losses on those 20 paired graph cases. The synthetic case improves from
3/5 to 5/5 facts and becomes exact. These controls use supplied slots, not new model output, and
do not attribute individual development losses. A fresh frozen live measurement is required.
