# Relation target binding repair

The V4 adapter attached `looks_at` to the porcupine in “A brown meerkat watches while a gray porcupine digs.” The grounding validator accepted this invented relationship because it found the predicate and both nouns without proving that the second noun was its target.

The repair requires a complete target noun phrase and checks clause, negation and intervening-entity boundaries. Coordinated objects remain valid; embedded nouns such as the owl in “a picture of an owl” do not become direct targets. The same checks cover symmetric and inverse relations, with a separate coordinated secondary target for `between`. This remains a bounded lexical validator, not a complete semantic parser.

Independent source review approved `scene_facts.py` at `5f51b4e1a8a6a2d6df81319e67a461973223e911cc4e2d725a4a2b7b40866399`. Validation: 1,223 Python tests, both JavaScript suites, Ruff and a focused style review passed. All 900 V4 training positives and 48 V4 screen gold descriptions still validate.

A separate replay of unchanged V4 raw predictions changes only the two repeated observations of the invented meerkat-to-porcupine relationship: both now fail grounding. Strict extraction remains 33/48 and correct refusal 15/16 for the V4 candidate. This is a validator improvement, not a learned-model gain. Historical V4 results remain untouched; the replay is retained under the V5 experiment.

The Jetson demo deployment and the best-demo checkpoint remain unchanged while baseline measurement and V5 experiments proceed.
