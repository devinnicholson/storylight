# Story smoke diagnostics

The v2 smoke distinguishes envelope parsing, accepted-plan construction, adapter refusal,
privacy, grounding and compiler proof. Evidence contains stage enums, bounded schema paths,
slot shapes and hashes. A successful integration stage may still be an accepted fallback;
only graph validity and compiler proof establish graph construction.

Optional `--private-responses` captures the seven frozen synthetic responses on the Jetson.
The archive must be a new 0600 file in an existing, owner-only 0700 directory outside the
checkout. Do not copy this archive into repository evidence or print its contents.

Use `--replay-private ARCHIVE --replay-evidence ORIGINAL_JOURNAL` with a fresh evidence/output
pair to reconstruct those responses after a code change. Replay verifies source, response,
request and provenance bindings, retains original failure counts, and records the current
implementation hash. It makes no model requests, carries no performance measurements and
cannot pass the development gate. Keep the original provenance argument when replaying.

V1 journals remain reproducible with their pinned implementation; v2 rejects incompatible
journals. Keep every failed result. A failed gate blocks rendering and promotion while
diagnosis and implementation continue.

Shared grounding now excludes quoted, reported, conditional and modal assertions. Independent
assertions separated by semicolons, or preceding a comma plus explicit conjunction, remain
available. This is a conservative finite grammar, not general discourse understanding. Only
graph cache identities change; the accepted four-slot path retains its cache identity.

Validation: 769 tests passed, including JavaScript bridges; after the final replay metadata
change, all 31 affected tests passed again. Scoped Ruff and whitespace checks passed.

## Observed failures and repair replay

The fresh seven-request run at `006025f` reproduced four accepted contracts and zero graphs.
Every response had one known terminal TensorRT control marker. The strict graph parser treated
its pipe as hybrid syntax; the legacy parser already removed the marker. The three accepted
construction failures were separately traced to the privacy phrase separator. These diagnoses
used only static code locations and delimiter/control-marker flags outside the device.

At `2e9836b`, the graph parser removes one terminal known marker and rejects embedded or repeated
markers. Offline reconstruction of the same seven responses now produces two compiler-proved
graphs: the transformation page and passive control. Page 1 passes the adapter but still fails
integration because accepted-wire construction fails. The three two-fox pages refuse ambiguous
binding, and the final page refuses result grounding. No new inference occurred during replay.

The adapter also rejects transformation antecedents whose explicit color or attributes differ
from the selected entity. Shared grounding conservatively refuses multiple explicit colors for
one entity label, closing borrowed action, event, relation and transformation claims. Distinct
colored characters still need a representation and binding extension before those pages can pass.

The diagnostic run's median inference was 891.2 ms and maximum/p95 1,128.6 ms. Engine, deployed
files, accepted prompt, planner configuration, power mode and resident process identity match
before and after. Its summary reproduces byte-for-byte with the pinned `006025f` implementation.
Evidence lives in `benchmarks/product-fidelity-diagnostics-2026-09-04`; private archives remain
on the Jetson. The marker and identity changes passed 782 tests and scoped lint. These results
establish construction repairs, not full story fidelity, image quality or promotion readiness.

Replay at `582db7b` produces three proved graphs and five valid candidate contracts from the
original responses. Page 1 now constructs independently from its validated graph when the
legacy wire fails. The accepted wire still behaves identically; graph failure preserves its
original fallback or error. Graph scaffolding retains the focal subject even when its actions
are typed events and a secondary subject has a simple action.

The adapter now supports explicitly flying result subjects, including their count, color and
spatial anchor. This repairs synthetic page-6 construction with correct result slots, but does
not repair the captured page-6 response: its selected result does not match the expected birds,
count or color. Do not turn that refusal into a semantic pass. An omitted second action also
remains omitted; constructing one graph does not establish the story's complete event order.

The next identity increment permits distinct explicit colors on repeated entity labels without
changing the wire fields. Grounding and rendering qualify references when the source contains
competing colors, including an unselected actor. Ambiguous bare references and swapped edges
refuse. The adapter currently selects one colored focal actor; the two-character checklists
still require both actors. The evaluator's flattened reference atoms retain colors for repeated
graph labels, with revision `colored-references-renderer-proof-v3`; historical scores stay pinned.

The combined increment passes 801 tests, including JavaScript bridges, scoped lint and whitespace
checks. A full-suite motion regression was repaired with an independent synthetic case: destination
objects must not be treated as competing actors. Public target coverage was regenerated across
4,608 public records: 4,605 eligible, 4,144 exact under the stricter evaluator and three ambiguous
refusals. These counts match a fresh reproduction at `9f449f8`; only the evaluator revision differs.
This remains deterministic coverage rather than live-model accuracy.
