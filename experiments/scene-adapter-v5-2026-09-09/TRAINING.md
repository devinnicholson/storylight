# V5 training corpus

The training set contains **4,800 rows: 3,600 positive descriptions and 1,200 refusal controls**. There are 1,800 linked groups and 1,826 distinct positive wire graphs. A graph has at most two positive surface variants and belongs to only one group. The 26 additional graphs come from paired explicit-target versus neighboring-clause examples whose meanings intentionally differ.

| Part | Groups | Rows | Purpose |
|---|---:|---:|---|
| Six boundary contrast families | 600 | 2,400 | Two valid neighbors and two refusals per group |
| Twelve positive families | 1,200 | 2,400 | Two surface variants or closely related positive meanings per group |

The boundary families cover named actors, named places, unbound targets, unfinished corrections, clipped predicates and indistinguishable individuals with different roles. Valid identity neighbors use two distinct colors, retain the initial shared action, and assign each subsequent action to its explicit actor. Refusal examples retain their actual incomplete or private source; labels are literal `REFUSE`, never a repaired graph.

Positive families cover same-actor and different-actor temporal events, repeated-actor action retention, action and object negation with “without,” static descriptions, explicit relation targets, independent neighboring clauses, counted shared actions, object attachment, composition and actor-specific positive/negative scope. Temporal targets contain two distinct event references and an explicit order; they do not substitute arbitrary relations for events. Static targets have no invented `exist` or `stand` action. Negative facts occupy the typed negative field rather than disappearing into an action string.

In 26 linked groups, an intransitive clause such as “A watches while B digs” has no target relation, while the explicit variant “A watches B. B digs” does. Both actors' stated actions remain. A regression test verifies that adding the explicit relation to the intransitive source is rejected by the newly frozen grounding validator.

Sentence variation includes coordinated versus separately stated initial actions, subsequent actor mentions, clause reversal, before/then/afterward connectors, negative-first versus positive-first statements, several object-absence expressions, existential versus fragmentary statics, and fronted versus trailing settings. Material/object pairs use plausible collocations. The generator still combines authored constructions with nouns, counts and colors; **4,800 rows are not 4,800 independent natural scenes**. The `form` key includes predicate choices as well as construction variants and is not a count of independent linguistic templates. Passive voice, unrestricted pronouns, reverse “after” ordering and arbitrary open-ended prose are not established by this corpus.

The fixed V2 few-shot prompt is byte-identical. New training sources exclude the development/test author's reserved nominal lexicon and its plurals; historical nouns inside the unchanged few-shot prompt are an explicit exception. No independent development or test examples, labels or authoring files are read. `train-overlap.json` exports group, tuple, source and target hashes for the independent author to check before releasing development.

Every positive target passes the frozen schema, original-source grounding, privacy and wire roundtrip. Every target, including refusals, passes the fixed V3 grammar. The corpus is bound to `scene_facts.py` SHA `5f51b4e1a8a6a2d6df81319e67a461973223e911cc4e2d725a4a2b7b40866399`; this is the repaired relation-binding boundary, not the historical V4 validator. It does not establish validator completeness.

The pinned local Gemma tokenizer verified all 4,800 completion-prefix boundaries, decoded targets, 2,048-token context limits and all three grammar stop tokens. Maximum total length is 732 tokens and maximum completion length is 78 tokens. No truncation, model inference, GPU or network request is used for these checks.

Reproduce with the isolated pinned CPU environment:

```sh
PYTHONPATH=src /private/tmp/bookforge-v3-cpu/bin/python experiments/scene-adapter-v5-2026-09-09/generate_training.py
PYTHONPATH=src BOOKFORGE_V5_TOKENIZER_DIR=/private/tmp/bookforge-v3-tokenizer /private/tmp/bookforge-v3-cpu/bin/python experiments/scene-adapter-v5-2026-09-09/test_training.py
```

The independent development set and hidden test remain separately owned. This training preparation does not authorize cloud allocation, choose a checkpoint or promote a model. New data and any larger training budget must be reported together; this corpus alone cannot isolate the effect of one failure-family intervention.
