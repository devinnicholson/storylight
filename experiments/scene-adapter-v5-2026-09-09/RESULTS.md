# V5 results — training in progress

The two pilots completed, and the frozen development-only selector chose Q/V. A fresh 4,800-step Q/V run is active. Independent test accuracy, serving speed and product integration are not established yet. The live demo and best-build checkpoint remain unchanged.

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

## Verification findings

PEFT 0.20 shortened the broader adapter's 275 module paths into 207 suffixes when saving its configuration. The initial offline verifier rejected that representation. Independent checks against all 1,596 modules in the pinned architecture established exactly the intended 275 matches, with all 550 expected tensors and no additional targets. The verifier now checks this complete resolution. No training code, data or checkpoint bytes were changed.

The frozen paired runtime still requires full target names. The selected Q/V recipe has those names; these results do not establish support for loading the broader recipe through that runtime.

The fresh full run has the pilot's exact initial adapter hash and training order, but its numerical optimization trajectory differs. The first loss matches; the first recorded gradient norm differs, and loss differs at step 2. The trainer fixes seeds without enforcing deterministic CUDA execution. The initiating cause is not established. This campaign cannot claim exact training reproduction or infer that the variation is harmless from configuration checks alone.

## Remaining measurements

After full training and checkpoint verification, run the fixed 128-case comparison against V4, twice per arm: 512 calls. Keep test labels local. Report strict positive accuracy, literal refusals, unexpected admissions and repeatability separately. Compare the actual demo parser in a separate preparation-only capture; an unexplained HTTP 422 receives no refusal credit.

Then benchmark verified adapter merging and dynamic/eager versus static/eager versus static/compiled decoding on fixed training inputs. Retain cold compilation cost, token parity, memory, failures and actual compiler/profiler evidence. These measurements concern learned scene extraction, not diffusion generation or microphone-to-display timing.

## Resources

The single L4 allocation in `us-east1-b` has a five-hour provider deletion deadline. Official public rates for its VM, 100 GiB balanced disk and external IPv4 total approximately **$0.8723/hour**, excluding transfer. Five hours would be about **$4.36**, before transfer and excluding earlier allocations. This is a list-rate estimate, not an invoice or remaining-credit balance. Region-bound source evidence is in `results/cloud-07/pricing-reference.json`.

The experiment remains within the user's $100 authorization. Final lifecycle and cleanup receipts will determine the retained usage estimate.
