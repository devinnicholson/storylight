# V5 results — quality and serving experiments complete

The selected V5 adapter improved strict scene extraction from **64/96 to 81/96** on the independent synthetic screen. It failed the frozen promotion gate: literal refusals fell from 28/32 to 25/32, with five schema-valid unexpected admissions (ten repeated observations). All five were blocked by the existing grounding/privacy validation. The best live demo remains unchanged.

## Independent comparison

Each adapter answered the same 128 cases twice. All 512 calls completed and passed offline prompt, token, EOS and grammar replay. Both adapters repeated their exact token sequences on all 128 cases. These are authored synthetic cases, not an estimate of real traffic accuracy.

| Measurement | V4 adapter | V5 adapter |
| --- | ---: | ---: |
| Strict positive cases correct in both repetitions | 64/96 | 81/96 |
| Literal refusals correct in both repetitions | 28/32 | 25/32 |
| Schema-valid unexpected admission observations | 0 | 10 |
| Median resident extraction latency | 2,264.308 ms | 2,871.552 ms |

The overall latency difference includes different answers and output lengths. Across the 180 pairs with identical token sequences, the median new-minus-old difference was only 0.691 ms. Training alone did not accelerate matched-output inference. See `results/report.json` and `results/paired-verified.json`.

A separate route table was frozen before joining the answer key. It retained all 31 existing-demo outputs and considered the adapter only on the 97 unclassified unsupported-description responses. Sixty-seven candidate outputs passed the unchanged schema/grounding checks in both repetitions; 62 were strict positive matches. These are potential recoveries on this captured screen, not a deployed fallback, refusal credit for HTTP 422, or a measured end-to-end success rate. Assistant review found that five nonexact candidate recoveries retain readable source meaning, without changing their strict scores. The existing demo also has four excluded-object polarity problems, one omitted event and one typed actor error in this screen; the hypothetical fallback preserves those existing outputs. See `results/SEMANTIC-DIAGNOSTIC.md`.

## Completed pilot comparison

Both recipes used the same 4,800-row ordering, first 1,200 examples, base revision, rank, optimizer and independent 256-row development set. These are single training runs, not a repeated-seed comparison.

| Measurement | Q/V | All text-decoder linears |
| --- | ---: | ---: |
| Trainable parameters | 2,678,784 | 26,165,248 |
| Target modules | 50 | 275 |
| Development loss, step 400 | 0.04892646 | 0.05038651 |
| Development loss, step 800 | 0.03614932 | 0.03594880 |
| Development loss, step 1,200 | 0.02565008 | 0.04637847 |
| Selected checkpoint | 1,200 | 800 |
| Mean training compute per step | 724.98 ms | 995.85 ms |
| Complete process wall time | 1,097.59 s | 1,475.64 s |

Development loss selected the recipe; it is not extraction accuracy. Training compute timing excludes development and checkpointing and is not inference latency. The selection receipt is `results/recipe-selection.json`; both complete runs and independent verification receipts are retained under `results/`. Adapter weight files remain ignored and local; a Git clone alone does not contain the bytes needed for weight replay.

## Completed full run

The fresh Q/V run completed all 4,800 examples. Development losses were 0.03301520 at step 1,200, **0.02929925 at step 2,400**, and 0.03636755 at step 4,800. The frozen rule selected step 2,400. All checkpoints, training rows, development records and the original pilot selection were independently verified. The selected adapter hash is `8acf87bd…980123`; full pins are in `results/full-training-verified.json`.

The four-call inference preflight passed exact tokenizer and grammar replay with both adapters. It used exposed training inputs and provides compatibility evidence, not quality or a fair speed comparison. The first call incurred a substantial cold-call cost.

The actual demo accepted 31/128 synthetic preparation requests and rejected 97 with an unclassified unsupported-description response. Median inclusive HTTP time was 63.696 ms. No images or sessions were created, and source/process identities stayed unchanged. These are admission and timing observations; the separate diagnostic is retained in `results/demo-hybrid-diagnostic.json`.

## Verification findings

PEFT 0.20 shortened the broader adapter's 275 module paths into 207 suffixes when saving its configuration. The initial offline verifier rejected that representation. Independent checks against all 1,596 modules in the pinned architecture established exactly the intended 275 matches, with all 550 expected tensors and no additional targets. The verifier now checks this complete resolution. No training code, data or checkpoint bytes were changed.

The frozen paired runtime still requires full target names. The selected Q/V recipe has those names; these results do not establish support for loading the broader recipe through that runtime.

The fresh full run has the pilot's exact initial adapter hash and training order, but its numerical optimization trajectory differs. The first loss matches; the first recorded gradient norm differs, and loss differs at step 2. The trainer fixes seeds without enforcing deterministic CUDA execution. The initiating cause is not established. This campaign cannot claim exact training reproduction or infer that the variation is harmless from configuration checks alone.

## Remaining measurements

Adapter merging was measured on fixed training inputs. The first run produced identical tokens in all 12 measured pairs and reduced matched-pair resident latency by 12.88% (median latencies 2,332.905 ms unmerged and 2,033.790 ms merged). The reverse-order run confirmed 13.20% matched-pair reduction (medians 2,319.566 ms unmerged and 2,003.089 ms merged), again with all 12 token sequences identical. Prefix logits differ after BF16 merging, so limited greedy-token parity is not general numerical or semantic equivalence.

The 54-call cache benchmark completed on eight training probes. Resident medians were 3,777.903 ms dynamic, 3,379.626 ms static/eager and 1,216.455 ms static/compiled. Fourteen of 16 measured static/compiled outputs match the dynamic tokens; the matched subset improved by 63.21%. Static/eager was 2.45% slower on its matched subset, despite its lower unmatched overall median. Both static modes refuse the valid “One blue penguin walks without running” prompt; dynamic decoding preserves it. Static/eager and static/compiled match each other on all 16 measured outputs. The eight-probe parity gate therefore fails.

The first compiled generation took 114.245 seconds, excluding about 3.403 seconds model loading and 3.573 seconds verified merging. The retained one-decode-step trace observed a CUDA graph launch, with two compiled graphs and no recorded graph breaks. This is real compilation evidence but not a claim that every request or full generation is one CUDA graph. The reverse-order run with fresh local compiler directories confirmed 63.62% matched-pair reduction and repeated the same one-prompt refusal regression. Its first compiled generation took 114.549 seconds; compiled resident median was 1,212.101 ms versus 3,813.543 ms dynamic. All 54 outputs and the compilation trace were independently verified. Fresh local compiler directories do not establish cold driver or operating-system caches. These measurements concern learned scene extraction on an L4, not diffusion generation, Jetson latency or microphone-to-display timing.

## Resources

The single L4 allocation in `us-east1-b` had a five-hour provider deletion deadline. Official public rates for its VM, 100 GiB balanced disk and external IPv4 total approximately **$0.8723/hour**, excluding transfer. Five hours would be about **$4.36**, before transfer and excluding earlier allocations. This is a list-rate estimate, not an invoice or remaining-credit balance. Region-bound source evidence is in `results/cloud-07/pricing-reference.json`.

The allocation was deleted after all required artifacts were downloaded. The VM, boot disk and dedicated firewall rules were verified absent at 22:40:52 UTC. Creation through confirmed absence spans 4.16 hours; the public list-rate VM/disk/IPv4 estimate is $3.63, excluding transfer, earlier allocations and other workloads. This is not an invoice or remaining-credit balance. See `results/cloud-07/lifecycle-estimate.json`.

## Follow-up diagnostics

The separate 64-call exposed-case decoding experiment found only 4.73% matched-token improvement without grammar enforcement, with one malformed excluded-object graph. Both arms scored 20/24 strict positives and 8/8 literal refusals on this selected subset; grammar enforcement preserved valid typed output for all 24 positives while the unconstrained arm preserved 23. These are exposed cases and cannot reverse the failed primary gate. The report is in `../scene-decoding-feasibility-2026-09-09/RESULTS.md`.

A 16-prefill diagnostic located the static-cache refusal at the first token. The grammar masks match exactly; seven prompts choose the same first token under both caches, while the “without running” prompt changes. Its maximum full-logit difference is 22.5625, compared with 0.4375–0.8125 on the others. The mechanism behind this divergence is not established. A separately reviewed dynamic-to-static cache transfer passes CPU K/V-history tests around the sliding-window boundary; the eager full-generation experiment then preserved all eight token sequences, including the formerly refused scene. All 16 streams passed independent token/grammar replay. The separate 54-call compiled version also preserved every token sequence, including warmups. Across 16 measured pairs it reduced latency by 60.95%; medians were 3,659.998 ms dynamic and 1,428.220 ms compiled transfer. Its first compiled preparation took 111.349 seconds, and the trace confirmed CUDA graph execution. The broader 128-case runtime confirmation passed independent verification under `../scene-cache-confirmation-2026-09-09/`: all 128 measured token pairs matched, and median paired latency fell 59.49% (medians 2,851.86 ms dynamic and 1,089.81 ms compiled transfer). These checks do not yet establish a deployable fix.

The final new-process repeat reused compiler artifacts and preserved all 54 prior outputs. Its first compiled preparation fell only from 111.35 to 103.45 seconds (7.09%). Two AOTAutograd and two FX graph cache misses remained; disk-cache reuse did not remove preparation cost. This is not a whole-service cold-start measurement. See `results/cache-bridge-warm-01-independent-review.json`.
