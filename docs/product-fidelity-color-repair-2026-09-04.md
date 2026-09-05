# Product fidelity: object-color repair passes the live gate

Frozen revision: `c2104698b52240a398335f3a4cd5668df231e6a4`.
Evaluator: `colored-references-renderer-proof-v3`.

The matched run completed all 512 public development cases, with one accepted model request per
case, no retries and no interrupted requests. All frozen gate checks pass. This permits a bounded
visual comparison; it does not promote the candidate on the appliance.

| Measure | Accepted | Candidate |
| --- | ---: | ---: |
| Valid final contracts | 346 | 346 |
| Exact final contracts | 0 | 38 |
| Required facts retained | 956 / 2,560 | 1,003 / 2,560 |
| Final privacy failures | 0 | 0 |
| Median planning | 1,047.3 ms | 1,064.1 ms |
| p95 planning | 1,227.2 ms | 1,237.7 ms |
| Maximum planning | 1,364.4 ms | 1,375.0 ms |

Graph coverage remains 43/512. Category totals do not regress, although five individual cases still
lose six required-fact checks. The comparison gains 53 checks elsewhere, for a net gain of 47.
These limitations remain visible; passing this gate does not imply generally reliable story
understanding. The 166 failed accepted constructions remain failures.

The preceding rejected run retained 970 facts and 15 exact contracts. Its action references dropped
bound object colors. Independent synthetic and public training controls reproduced that defect;
the production repair preserves those colors without changing the scoring rules. The new run
retains the same model request, engine and evaluator. This is deterministic reconstruction
improvement, not a learned-model improvement.

The seven-request story smoke produces six compiler-proved graphs and one focal fallback. Page 6's
model-selected result still refuses graph construction, and focal graphs omit secondary actors.
The later complete-scene and authored-display implementation must be checked separately before it
can satisfy the frozen story's full checklists.

Peak sampled planner RSS was 4,002,385,920 bytes; peak sampled system use was 7,108,505,600 bytes.
Thermal readings ranged from 44.7 to 66.2 °C. GPU allocator peak was unavailable. Before/after
provenance confirms the installed code, engine, prompt, service process, power mode and environment
were unchanged. The sanitized aggregate reproduces byte-for-byte under its pinned revision.

Evidence lives in `benchmarks/product-fidelity-color-repair-2026-09-04/`. Raw synthetic responses
remain in the owner-only Jetson archive. No hidden split, cloud inference, image generation,
engine rebuild or appliance promotion was used.
