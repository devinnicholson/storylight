# Live scene construction and scoring fixes

This increment diagnoses the causes behind the two rejected live graph experiments. It changes
deterministic construction, grounding, and evaluation; it does not train Gemma or establish new
live-model accuracy. The accepted appliance remains unchanged and graph backends remain opt-in.

## Diagnostic scope

Before editing, we froze the first 32 public training records in generator order and passed their
exact target slots through each pipeline stage. This is a small training diagnostic, not a held-out
accuracy estimate. The live parser receives only the source and slots; it never imports target
derivation. Synthetic positive/negative pairs test each change independently of corpus wording.
No hidden split or new 512-case development inference run is used for these fixes.

The initial diagnostic produced zero graphs. Twenty-eight inputs passed accepted wire, plan,
privacy, and renderer construction; four stopped at the accepted wire's privacy-preserving
rewrite. Those refusals are distinct from raw-envelope failures. Existing accepted rewriting
and its refusal behavior are preserved.

The original development journals and summaries remain unchanged. Reproduce them at their
recorded revisions, rather than applying the new evaluator to unavailable raw outputs. The new
evaluator identifies itself as `bound-descriptors-renderer-proof-v2`; new benchmark headers bind
that revision as well as source hashes.

## Construction and grounding

The result noun and its corresponding explicit source predicate now use the same source-bound
matching: for a rising ribbon in the source, both `ribbon` and `ribbon rises` are supported.
A directly attached spatial anchor is retained; an unrelated
actor's anchor is not pulled into the graph. Explicit `calling forth` and `causing` result clauses
must attach to the verified selected action. A small neutral-adverb set can occur between actor
and action. Tested reporting, hypothetical, negative, and ambiguous variants still refuse;
this finite grammar does not establish safety for arbitrary English constructions.

Transformation results now retain their explicit count through validation, wire serialization,
and renderer compilation. `result_count` is an optional strict integer from 1 to 12. A legacy
transformation wire remains `T|source|label|color|attributes`; an explicit count appends a sixth
field. Old five-field wires still decode identically. Older decoders cannot read the new counted
form, which remains inside the opt-in graph path. The count must belong to the transformed result,
not to its source or a nearby distractor.

Action and event validation now binds actor, verb, and object within the same clause. It rejects
an object borrowed from another actor's clause or from a nearby spatial relation. Temporal order
uses those complete event spans, so repeating a verb cannot make a reversed object sequence pass.
Shared-subject continuation is supported only after `and then`, `and afterward`, or
`and only afterward`, with a fully grounded prior event ending at the connector. An intervening
actor, negative, or unrelated noun cannot supply that proof.
The graph cache revisions change to avoid reusing pre-fix graph plans; the accepted cache identity
is unchanged.

## Semantic comparison

Semantic phrase matching ignores `a`, `an`, and `the` while preserving meaningful token order,
counts, and negation. Privacy matching retains its original tokens, including short protected
names that happen to be articles. Per-node canonical descriptors allow a modifier to move between
the label and an attribute field without changing the fact. Relationships and events reference
the complete canonical node identity, including role and count; descriptor normalization cannot
merge two distinct entities. Closed-world extra facts still fail exactness.

Typed renderer evidence must include `master_prompt`, `scene_facts`, and `visual_style`. The
evaluator recompiles those facts against the source and requires byte-for-byte equality with the
actual prompt before scoring typed semantics. A correct attached graph cannot excuse an incorrect
or missing prompt. Bare renderer text remains a lexical screen and is not interchangeable with
verified graph evidence or image fidelity. Relation scoring no longer credits disconnected words
belonging to the wrong subject.

The raw relation screen uses the declared slot and bounded relation aliases. A declared shortened
object descriptor can match only within the same bound relation; it cannot be borrowed from
another clause. Movement aliases distinguish upward from downward motion. Indirect holding paths
require an explicit intermediate object from the evaluation contract's allowed concepts. These
rules belong to the evaluator, not the live parser.

## Remaining limits

The live parser still has a finite predicate grammar. An attempted open-ended predicate extension
was removed after adversarial review showed it could treat promises or denials as visible action.
The remaining 30 training refusals have these first blockers: 14 predicate/complement grammar
cases, two shortened-anchor identity cases, ten participial/passive or referential clause cases,
and four extended-motion or relative-result clause cases. Fixing a first blocker may expose a
second one. Because these inputs are exact target slots, the refusals cannot be attributed to
missing model output.

These repairs do not justify another full model run or a paid image comparison yet. The next
construction work should isolate one class, such as explicit passive actor/object binding, with
frozen synthetic positive and negative controls before extending the grammar. Only after that gate
should a new, separately retained development run measure actual model output and latency.

## Reproduction

`scripts/diagnose_live_scene_training.py` writes only stage labels, scores, and hashes. Run it
with the frozen old source and the final source using the same diagnostic script to reproduce
the paired reports in `benchmarks/live-scene-underlying-fixes-2026-09-04/`:

```bash
PYTHONPATH=/absolute/path/to/9e5024b/src .venv/bin/python \
  scripts/diagnose_live_scene_training.py --output /tmp/training-before.json
.venv/bin/python scripts/diagnose_live_scene_training.py --output /tmp/training-after.json
```

The source tree and diagnostic script hashes are retained separately. These controls attribute
deterministic pipeline changes; they do not predict what the resident model will emit.

## Results and interpretation

On the fixed first 32 training controls, graph construction improves from 0/32 to 2/32.
Both graphs survive integration and verified renderer compilation; one is exact. Their atom score
is 8/10 **among the two valid graphs**, not across all 32 inputs. The final paths are two graphs,
26 accepted fallbacks, and four accepted-wire refusals. The accepted path remains 28 valid and
four refused before and after.

Regression verification also runs the existing public development target controls without model
inference. All 512 authored raw targets and literal wire renderings pass their lexical controls.
The live adapter accepts 54/512 authored targets, with 20 exact graphs. This measures deterministic
construction from supplied target slots; it does not replace either historical live-model result.

The corrected typed target score is 460/509, with 49 missed ACTION atoms: 21 containment cases
state only that the actor watches, without binding the watched object, and 28 foreground cases
do not prove the target's standing posture or extent across that region. The source generator
itself has those distinctions. The old scorer also synthesized standing from salience alone;
that shortcut is removed.
We retain the missing-action failures instead of inventing facts to recover the previous score.
The fixed 128-record training control independently exposes the same two issues: eight unbound
watching cases and eight unsupported standing/filling actions.

The newly versioned public target coverage report separates valid graph construction from exact
semantic coverage:

| Split | Inputs | Valid targets | Exact targets |
| --- | ---: | ---: | ---: |
| Training | 4,096 | 4,096 | 3,684 |
| Development | 512 | 509 | 460 |

The three development refusals remain the existing ambiguous repeated-label cases. The historical
coverage artifact and both live-run journals are preserved. Their old scores reproduce at the
recorded revisions; they must not be compared as though both evaluator versions were identical.

Reproduce the new regression aggregates separately:

```bash
.venv/bin/python scripts/diagnose_live_scene_facts_scoring.py --output /tmp/scoring-controls.json
.venv/bin/python scripts/benchmark_fidelity_graph_targets.py --token-budget 64 \
  --output /tmp/public-target-coverage.json
```

Compare each with its corresponding file in
`benchmarks/live-scene-underlying-fixes-2026-09-04/`. Native compiled renderer scores in the scoring
diagnostic remain lexical screens; the paired training diagnostic separately verifies the exact
compiled graph prompt. No image fidelity or new latency claim follows from either report.

## Verification

All 1,720 Python tests, both JavaScript suites, scoped Ruff, and `git diff --check` pass. The only
Python warning is the external Starlette/httpx deprecation. Independent adversarial reviews cover
the parser, grounding, evaluator, renderer proof, and the revised target-control accounting.
The deslop review retains the finite parser and removes the rejected speculative predicate path.

Both paired training reports and both new regression aggregates reproduce byte-for-byte.
The two historical live summaries, historical scoring controls, and historical public target
coverage also reproduce with their pinned source revisions. Credential-pattern and public
source/target/record-ID scans pass for the new evidence. Local runtime versions and artifact
checksums are retained in `verification.json` alongside the reports. No installed Jetson runtime,
accepted backend default, cloud workload, or user-owned output directory was changed.
