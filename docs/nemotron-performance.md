# Nemotron and next-page latency experiments

The [subsequent performance and accuracy pass](performance-accuracy-2026-09-03.md) records the
clean-node cached-engine startup, failed image-blind/short-question experiments, and accepted
repair-contract and response-integrity fixes. Existing Nemotron acceptance is not reliable evidence
of action/count/spatial fidelity; the stronger prompts were not promoted.

## Accepted transport changes

The next-page panel now uses a bounded 20-second long poll rather than sleeping for 1.5 seconds
between readiness checks. GKE waits on the candidate's existing completion event. A timeout returns
the current state without cancelling or resubmitting generation. The Jetson still verifies scene
identity, asset hashes, and decoded dimensions before staging; showing a staged scene remains
local-only and requires the user's explicit action. Cache hits skip an unnecessary status request.

The NIM client expires idle connections after four seconds. The pinned NIM 1.3.1 server explicitly
sets `TIMEOUT_KEEP_ALIVE = 5`. The previous 300-second client lifetime encountered a stale socket
through `kubectl port-forward`, terminating the forward during an alternating-client experiment.
The four-second setting completed the subsequent 26-call run without transport errors. There is
still no automatic inference retry: an ambiguous failure must not silently duplicate paid work.

## Review experiments

The regression fixture is an already-generated synthetic illustration, not a reader photograph:
`benchmarks/fixtures/fox-lantern-regression.jpg`, SHA-256
`fcc298275a6a37473e614af7bab8cdc700edf28035d4be0469e542f03624235a`.
Visual inspection shows one fox to the left of a lantern, with nothing held in its mouth.

Six contracts test two matching descriptions and four mismatches: carrying the lantern, two foxes,
a dragon/castle instead of the forest, and reversed left/right placement. Each policy gets a
separate warmup and two alternating-order repeats. The measured HTTP latency includes the private
port-forward; it is not isolated GPU kernel time. Twelve samples per policy on one image are not
evidence of general accuracy or production p95.

| Run | Outcome |
| --- | --- |
| `nemotron-critic-paired-2026-09-03-v1.json` | Interrupted by the stale connection; not a valid performance or quality comparison |
| `nemotron-critic-paired-2026-09-03-v2.json` | Baseline median 5,556 ms, 6/12 correct; compact action prompt 4,763 ms, 4/12 correct; zero transport errors |
| `nemotron-critic-paired-2026-09-03-v3.json` | Aborted warmup through a forward to the terminated Pod; no usable inference result |
| `nemotron-critic-paired-2026-09-03-v4.json` | Baseline median 5,554 ms, 6/12 correct; observation-first 4,485 ms, 8/12 correct; zero transport errors |

The compact action prompt is **not promoted**: its roughly 14% median speed improvement came with
more false acceptances. Neither policy reliably rejected the carrying/count/position errors. An
image-only observation probe took 3,036 ms (38 output tokens) and correctly reported one fox left
of the lantern with nothing in its mouth. This motivates an observation-first experiment, not a
claim that the visual quality gate is solved. Observation-first reduced median latency by 19.2%
and fixed the count mismatch in both repeats, without changing either positive result. It still
missed carrying and reversed left/right placement, so it remains experimental rather than being
silently promoted as a solution to the known fidelity regression.

Run the current paired experiment against an already-running private NIM:

```bash
.venv/bin/python -m bookforge.nemotron_critic_benchmark \
  --base-url http://127.0.0.1:18083 \
  --image benchmarks/fixtures/fox-lantern-regression.jpg \
  --output /tmp/bookforge-new-critic-experiment.json \
  --repeats 2 \
  --authorization I_AUTHORIZE_BOUNDED_GPU_REVIEW_OF_THIS_GENERATED_IMAGE
```

The output path must not exist. The image checksum and repeat count are bounded before inference;
the CLI requires explicit authorization. A transport error stops the remaining measured calls.
The harness does not start, resize, or stop a GPU, so the operator must retain the deployment's
watchdog and explicitly scale it to zero after the supervised run.

## Persistent TensorRT engine experiment

[NVIDIA's version-matched guide](https://docs.nvidia.com/nim/vision-language-models/1.3.1/fine-tune-model.html)
documents `NIM_CUSTOM_MODEL_NAME` for caching locally built models. The installed pinned runtime
also uses it for its original buildable model. The first supervised run saved both engines on the
existing 80 GiB PVC: the vision build took 92.20 seconds and the language-engine build 277.86 seconds.
Pod creation to readiness was 954 seconds, including scheduling, image pull, imports, building,
writing the cache, and startup. The engine cache consumed approximately 17 GiB beyond the weights.

Automatic selection of the newly cached profile failed because NIM omitted
`visual_engine/vision_processor.py` from that profile. The workaround selects the original NVIDIA
buildable profile explicitly and supplies the named cache. The installed runtime's existing
`get_vision_jit_engine` and `get_jit_trtllm_engine` paths then reuse the saved engines while retaining
the original vision processor. No NVIDIA source or model weights are patched.

The corrected restart passed: Pod creation to readiness was **234 seconds** on the existing node.
Container start to readiness decreased from **631 to 229 seconds** (63.7%) across the build and
reuse runs. Both engine-reuse paths appeared in the logs, and all 12 baseline verdicts were
identical before and after the restart. This is a same-node restart test, not clean-node cold-start
or production p95 evidence. The 92.20-second vision build and 277.86-second language build did not
repeat. A new node still needs provisioning and the large image pull.

The configuration helper refuses to modify a running NIM or a different image, GPU, context length,
batch size, or cache volume:

```bash
BOOKFORGE_NIM_CACHE_APPLY=I_UNDERSTAND_THIS_CONFIGURES_THE_STOPPED_NIM \
  bash infra/gcp/gke/configure-nemotron-cache.sh
```

It does not scale up a GPU. Keep this cache name tied to the pinned image, L4, BF16, 2048-token
context and batch-one configuration. Revalidate and use a new cache name when changing those
inputs. The original buildable profile is
`308eb4483d24f2e7f53a75d669cacd83539f3415167733accfea1a5efe6986aa`.

`NIM_ENABLE_KV_CACHE_REUSE=1` was also tested. This exact multimodal runtime disables it before
constructing its TensorRT KV cache. The helper removes the inactive flag; no prefix-cache speedup
is claimed.

Next experiments should separate clean-node provisioning/image-pull time, cached engine loading,
first-request CUDA extension compilation, and steady-state inference. In particular, a warm-node
restart must not be presented as a clean-node cold start. Persistent CUDA extension caches and
quantized Nemotron profiles still require independent correctness and restart tests.

## Full-flow verification

Story: preparing a typed next page flows from the workbench through local Gemma, private GKE,
the renderer and Nemotron, back to hash-verified local assets without changing the projection.

| Boundary | Evidence |
| --- | --- |
| Browser → Jetson | Actual workbench POST returned 202 for a synthetic golden-paper-boat passage |
| Jetson → GKE | Final CPU image `sha256:8da6b2f103669794f949d53200663c352283977d02f96e038be5d8ee2c6e3afc`; deployed wait contract verified |
| Completion → browser | Two bounded `?wait_seconds=20` requests returned 200; no 1.5-second polling sleeps |
| Download → local cache | Stage POST returned 200; `assets_verified=true` |
| Response → UI | “Verified on this device” and enabled “Show prepared scene”; projection was not switched |

The live sample planned in 1,635 ms, rendered in **26,107 ms on the cold renderer path**, and reviewed
in 5,345 ms. The long poll removes notification delay; it does not make a cold renderer instantaneous.
Cloud Run logs confirmed a new instance at 23:44:10Z followed by pipeline loading and a successful
24.746-second HTTP request. The browser console had no errors or warnings.
No physical projector timing is claimed. The final local suite passed **1,070 tests**. The GPU was
scaled to zero after verification, with regional L4 quota usage confirmed at zero; the CPU
coordinator and persistent cache remain provisioned. Full startup identities, source hashes, and
HTTP evidence are in `benchmarks/nemotron-cache-optimization-2026-09-03.json`.
