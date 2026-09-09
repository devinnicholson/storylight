# Broader cache confirmation result

The dynamic-prefill/compiled-static-decode candidate preserved **all 128 measured output token sequences** and reduced matched-pair resident extraction latency by **59.49%**. This confirms the earlier eight-probe result on the full, previously exposed V5 screen. The original model-quality gate still fails, and the live demo remains unchanged.

| Measurement | Dynamic | Compiled bridge |
| --- | ---: | ---: |
| Measured descriptions | 128 | 128 |
| Median resident extraction latency | 2,851.856 ms | 1,089.807 ms |
| Strict positive matches | 81/96 | 81/96 |
| Positive outputs passing schema and grounding/privacy checks | 96/96 | 96/96 |
| Literal refusals | 27/32 | 27/32 |
| Schema-valid unexpected refusal admissions | 5 | 5 |
| Unexpected admissions passing grounding/privacy checks | 0 | 0 |

All 260 calls completed, including four retained training warmups. The two warmup outputs also matched across modes. Independent verification replayed the actual tokenizer, EOS and grammar, checked the complete dispatch schedule and artifact pins, recomputed comparisons, and reviewed cache-transfer lengths, first tokens, tensor shapes and stable buffer addresses. Local scoring then used the unchanged validator and strict scorer with the private answer key; there were no score changes between modes.

These scores were computed anew for both merged arms. Their 27/32 literal refusals differ from the original unmerged V5 comparison's 25/32; this separate experiment does not isolate the cause of that difference or replace the frozen primary result. Five schema-valid unexpected admissions remain, even though the existing grounding/privacy checks block all five. Runtime parity is not model-quality acceptance.

The median of the 128 paired fractional reductions is 59.48516%; it is distinct from the ratio of the two overall medians. Cold costs remain substantial: the first compiled warmup took **110.416 seconds**, and the second took 294.433 ms. Dynamic warmups took 3.851 and 1.332 seconds. Warmups are excluded from the resident medians but retained in the evidence. The complete process took 625.442 seconds. The compiled trace observed CUDA graph launches and two compiled graphs; dynamic recorded neither. This does not mean every generation executes as a single CUDA graph.

The candidate preserves dynamic prefill and transfers its K/V history into reusable static buffers before compiled decoding. The preceding all-static approach changed a valid prompt into a refusal; the bridge avoids that observed regression in this screen. The underlying static-prefill numerical divergence remains unexplained.

This is one process and one repetition, with dynamic in source order followed by compiled bridge in reverse order, using the same checked merged BF16 base. Inputs span only 638–654 prompt tokens. All 128 descriptions were previously exposed, so this is a runtime parity confirmation rather than fresh held-out accuracy. Fresh local compiler directories do not establish cold driver or operating-system caches. Cache reset is checked by the pinned runtime; complete K/V tensors were not retained for offline comparison.

The measurements concern scene extraction on an L4, not diffusion image generation, microphone-to-display latency or Jetson execution. No production promotion follows from this result.

Evidence: `results/verified.json` contains the independent runtime audit; `results/diagnostic.json` contains the subsequent local scores; `results/gpu/cache-confirmation-01/` retains the complete run. Private answer-key bytes remain outside the repository and were never uploaded.
