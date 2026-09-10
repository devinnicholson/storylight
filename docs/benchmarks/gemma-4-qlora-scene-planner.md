# Gemma 4 QLoRA: 81 of 96 strict scene extractions

NF4 QLoRA moved Gemma 4 E2B from 64 to 81 exact positive scene extractions on a fixed 96-case screen. The gain survived two repetitions per adapter, and all 512 comparison calls passed offline replay of their token streams through the grammar and end-of-sequence rules. The experiment showed that a small open model could learn Storylight's typed scene language from 4,800 synthetic training rows on one NVIDIA L4.

| Measurement | Earlier adapter | Selected adapter |
| --- | ---: | ---: |
| Strict positives correct in both repetitions | 64/96 | 81/96 |
| Literal refusals correct in both repetitions | 28/32 | 25/32 |
| Median resident extraction latency | 2,264.308 ms | 2,871.552 ms |
| Schema-valid unexpected admission observations | 0 | 10 |

The positive score rose by 17 cases, from 66.7 to 84.4 percent. Five refusal cases were admitted in both repetitions, producing the ten observations in the table. Storylight's existing grounding and privacy validators blocked all five before they could become render requests. The adapter stayed out of production because the frozen gate required refusal quality to hold.

## Training an open model for a narrow contract

The base model was `google/gemma-4-E2B-it`. Training used NF4 quantization with rank-16 LoRA, which kept the base weights compressed and trained a small set of adapter matrices. The corpus contained 3,600 positive descriptions and 1,200 literal refusal controls. It covered 1,800 linked groups and 1,826 distinct positive wire graphs, with at most two positive surface forms attached to a graph.

The target is a compact typed representation of scene facts rather than prose. Training examples exercise actor and object binding, ordered events, negation, explicit relation targets, counts, and privacy boundaries. Completion strings must satisfy a fixed grammar, then pass Pydantic schema validation and source-grounding checks. Refusal labels remain literal `REFUSE`; the data generator never repairs an incomplete or private description into a plausible scene.

The full run consumed all 4,800 rows. Development loss selected step 2,400, where loss reached 0.02929925 before rising to 0.03636755 at step 4,800. A separately authored 256-row development set selected the recipe and checkpoint before the comparison screen was scored. The 128-case screen contained 96 positive cases and 32 refusal cases, and each adapter answered every case twice.

The selected Q/V recipe trained 2,678,784 parameters across 50 target modules. An earlier pilot compared it with adapters over all text-decoder linear layers, which exposed 26,165,248 trainable parameters across 275 targets. The wider adapter reached a slightly lower development loss at step 800, yet its loss degraded by step 1,200 while compute per step rose from 724.98 to 995.85 ms. Q/V targeting gave the more stable pilot and became the full-run recipe.

## What the result established

The model gained 17 exact positive cases without a larger base model or a managed training API. More of the source sentence survived into a machine-checkable graph, and local validators contained the refusal regression. That division of labor is useful on constrained hardware: the model proposes a structured interpretation, while deterministic code controls privacy, grounding, and admission to the expensive renderer.

The latency increase between adapters cannot be read as a pure model-speed regression because answer lengths changed. Among 180 observations with identical token sequences, the median new-minus-old difference was 0.691 ms. The measured quality change came from output behavior, not a different serving stack.

## What we learned and what is next

Small adapters can move exact extraction substantially, though positive accuracy and refusal discipline did not move together. Future training should increase hard refusal coverage, especially unfinished corrections and identity-sensitive descriptions, then evaluate once on unseen authored inputs before any promotion decision. The five unexpected admissions remain the clearest error set.

Quantized export and Jetson execution are separate engineering problems. The next device study needs the selected checkpoint, grammar support, peak-memory telemetry, and exact output comparison against the L4 reference. A faster or smaller runtime cannot compensate for a failed refusal gate.

Evidence: [`research/results.json`](../../research/results.json), [`V5 results`](../../experiments/scene-adapter-v5-2026-09-09/RESULTS.md), and [`training protocol`](../../experiments/scene-adapter-v5-2026-09-09/TRAINING.md).
