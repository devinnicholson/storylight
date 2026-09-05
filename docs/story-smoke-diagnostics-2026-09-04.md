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
